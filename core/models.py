"""Speech-recognition model library: models live in <app>/models/asr/<folder>/.

Users can add a model by
  - pasting a Hugging Face link or repo id (downloaded automatically, only the files needed), or
  - copying a model folder into models/asr/ themselves.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Callable

import paths
from engines import KIND_LABELS, UnsupportedModel, detect_kind, files_needed, read_config
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ASR_DIR = os.path.join(paths.MODEL_DIR, "asr")
META = "meetingscribe.json"
# small (0.24 GB), runs on any CPU, and was as accurate as the best GPU model in our tests
DEFAULT_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SPK_REPO = "speechbrain/spkrec-ecapa-voxceleb"
SPK_FILES = ["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt",
             "label_encoder.txt"]
os.makedirs(ASR_DIR, exist_ok=True)

FRIENDLY_NAMES = {
    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17": "SenseVoice 中英日韓粵",
    "csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en": "Paraformer 中英串流",
    "MediaTek-Research/Breeze-ASR-25": "Breeze-ASR-25 台灣華語",
    "openai/whisper-large-v3-turbo": "Whisper large-v3-turbo",
    "nvidia/nemotron-3.5-asr-streaming-0.6b": "Nemotron 3.5 串流 0.6B",
}


def _display_name(repo_or_folder: str) -> str:
    repo = repo_or_folder.replace("__", "/")
    return FRIENDLY_NAMES.get(repo, repo.split("/")[-1])


def _is_sensevoice(path: str) -> bool:
    try:
        with open(os.path.join(path, "tokens.txt"), encoding="utf-8") as f:
            return "<|withitn|>" in f.read()
    except OSError:
        return False

Progress = Callable[[str, int, int], None]  # (message, done_bytes, total_bytes)


@dataclass
class ModelInfo:
    id: str  # folder name
    path: str
    kind: str
    name: str
    repo: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}　·　{KIND_LABELS.get(self.kind, self.kind)}"


# ---------------------------------------------------------------------------------------------- library
def list_models() -> list[ModelInfo]:
    out = []
    for d in sorted(os.listdir(ASR_DIR)):
        path = os.path.join(ASR_DIR, d)
        if not os.path.isdir(path) or d.startswith("."):
            continue
        meta = _read_meta(path)
        try:
            kind = meta.get("kind") or detect_kind(os.listdir(path), read_config(path), d)
        except UnsupportedModel:
            continue
        if meta.get("incomplete"):
            continue
        if kind == "sherpa-offline-paraformer" and _is_sensevoice(path):
            kind = "sherpa-sensevoice"
        out.append(ModelInfo(d, path, kind, meta.get("name") or _display_name(d), meta.get("repo", "")))
    return out


def get_model(model_id: str) -> ModelInfo | None:
    return next((m for m in list_models() if m.id == model_id), None)


def remove_model(model_id: str):
    path = os.path.join(ASR_DIR, os.path.basename(model_id))
    if os.path.isdir(path):
        shutil.rmtree(path)


def _read_meta(path: str) -> dict:
    try:
        with open(os.path.join(path, META), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_meta(path: str, **meta):
    with open(os.path.join(path, META), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------- Hugging Face
def parse_repo(text: str) -> str:
    """Accepts 'org/name', 'https://huggingface.co/org/name', '.../org/name/tree/main', 'hf.co/org/name'."""
    t = text.strip()
    m = re.search(r"(?:huggingface\.co|hf\.co)/([\w.\-]+/[\w.\-]+)", t)
    if m:
        return m.group(1)
    if re.fullmatch(r"[\w.\-]+/[\w.\-]+", t):
        return t
    raise ValueError("請輸入 Hugging Face 模型網址，例如 https://huggingface.co/openai/whisper-large-v3-turbo")


@dataclass
class Plan:
    repo: str
    kind: str
    files: list[str]
    size: int
    folder: str


def plan_download(text: str) -> Plan:
    """Look at a repo without downloading the weights: which engine, which files, how big."""
    repo = parse_repo(text)
    info = HfApi().model_info(repo, files_metadata=True)
    names = [s.rfilename for s in info.siblings]
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    config = None
    if "config.json" in names:
        with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
    kind = detect_kind(names, config, repo)
    files = files_needed(kind, names)
    return Plan(repo, kind, files, sum(sizes.get(f, 0) for f in files), repo.replace("/", "__"))


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _download(repo: str, files: list[str], local_dir: str | None, total: int, label: str, progress: Progress,
              cancel: threading.Event | None = None, retries: int = 20):
    """snapshot_download with automatic retry (resumes partial files) and a progress callback."""
    watch = local_dir or os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"))
    start = _dir_size(watch) if local_dir is None else 0
    done = threading.Event()

    def poll():
        while not done.wait(1.0):
            progress(label, min(total, _dir_size(watch) - start), total)

    threading.Thread(target=poll, daemon=True).start()
    try:
        for attempt in range(1, retries + 1):
            if cancel and cancel.is_set():
                raise InterruptedError("已取消")
            try:
                snapshot_download(repo, allow_patterns=files, local_dir=local_dir)
                return
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError(f"下載失敗（已重試 {retries} 次）：{e}") from e
                wait = min(30, 3 * attempt)
                progress(f"網路中斷，{wait} 秒後自動續傳（第 {attempt} 次）", 0, 0)
                time.sleep(wait)
    finally:
        done.set()


def install(plan: Plan, progress: Progress, cancel: threading.Event | None = None) -> ModelInfo:
    path = os.path.join(ASR_DIR, plan.folder)
    os.makedirs(path, exist_ok=True)
    name = _display_name(plan.repo)
    _write_meta(path, repo=plan.repo, kind=plan.kind, name=name, incomplete=True)
    _download(plan.repo, plan.files, path, plan.size, f"下載 {name}", progress, cancel)
    shutil.rmtree(os.path.join(path, ".cache"), ignore_errors=True)
    _write_meta(path, repo=plan.repo, kind=plan.kind, name=name)
    return get_model(plan.folder)


# ---------------------------------------------------------------------------------------------- first run
def speaker_model_present() -> bool:
    try:
        snapshot_download(SPK_REPO, allow_patterns=SPK_FILES, local_files_only=True)
        return True
    except Exception:
        return False


def ensure_first_run(progress: Progress):
    """Voiceprint + speaker-change models always; the default speech model only if the library is empty."""
    from turns import SEG_FILE, SEG_REPO, model_path

    if not speaker_model_present():
        _download(SPK_REPO, SPK_FILES, None, 90_000_000, "下載聲紋模型", progress)
    from diarize import CLUSTER_FILE, CLUSTER_REPO, cluster_model_path

    cl = cluster_model_path()
    if not os.path.exists(cl):
        _download(CLUSTER_REPO, [CLUSTER_FILE], os.path.dirname(cl), 101_400_000, "下載分群聲紋模型", progress)
        shutil.rmtree(os.path.join(os.path.dirname(cl), ".cache"), ignore_errors=True)
    seg = model_path(paths.MODEL_DIR)
    if not os.path.exists(seg):
        _download(SEG_REPO, [SEG_FILE], os.path.dirname(seg), 1_600_000, "下載換人偵測模型", progress)
        shutil.rmtree(os.path.join(os.path.dirname(seg), ".cache"), ignore_errors=True)
    if not list_models():
        plan = plan_download(DEFAULT_REPO)
        install(plan, progress)
