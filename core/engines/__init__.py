"""Pluggable speech-recognition engines.

Every engine turns 16 kHz mono audio of one utterance into text while it is being spoken:
    engine = load_engine(model, device)
    stream = engine.new_stream(language, latency, on_text)   # on_text(str) gets the running text
    stream.feed(audio) ...; final = stream.finish()

Which engine a model needs is detected from its files (see detect_kind).
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000

# generic language codes used by the UI
LANGUAGES = [("中文", "zh"), ("自動偵測", "auto"), ("English", "en"), ("日本語", "ja"), ("한국어", "ko")]

KIND_LABELS = {
    "nemotron": "NVIDIA 串流",
    "whisper": "Whisper",
    "sherpa-online-paraformer": "Paraformer 串流",
    "sherpa-online-transducer": "Zipformer 串流",
    "sherpa-sensevoice": "SenseVoice",
    "sherpa-offline-paraformer": "Paraformer",
    "sherpa-offline-transducer": "Zipformer",
}


class UnsupportedModel(Exception):
    pass


# ---------------------------------------------------------------------------------------------- detection
def detect_kind(files: list[str], config: dict | None, hint: str) -> str:
    """Decide which engine runs a model from its top-level file names (+ config.json, + folder/repo name)."""
    top = [f for f in files if "/" not in f]
    hint = hint.lower()
    if config is not None:
        mt = str(config.get("model_type", "")).lower()
        if mt == "whisper":
            return "whisper"
        if "nemotron" in mt and "asr" in mt:
            return "nemotron"
        raise UnsupportedModel(f"不支援的模型類型：{mt or '未知'}")
    onnx = [f for f in top if f.endswith(".onnx")]
    if "tokens.txt" in top and onnx:
        has = lambda role: any(f.startswith(role) for f in onnx)  # noqa: E731
        if has("joiner"):
            return "sherpa-online-transducer" if "streaming" in hint or "online" in hint else "sherpa-offline-transducer"
        if has("encoder") and has("decoder"):
            return "sherpa-online-paraformer"
        if has("model"):
            return "sherpa-sensevoice" if "sense" in hint else "sherpa-offline-paraformer"
    raise UnsupportedModel("找不到可辨識的模型檔（需要 config.json + safetensors，或 tokens.txt + .onnx）")


def files_needed(kind: str, files: list[str]) -> list[str]:
    """Only the files inference needs (repos often also hold training checkpoints worth many GB)."""
    top = [f for f in files if "/" not in f]
    if kind in ("nemotron", "whisper"):
        keep = [f for f in top if f.endswith((".json", ".txt", ".model", ".tiktoken"))
                and not f.startswith(("README", "optimizer", "scheduler", "trainer_state", "training_args"))]
        weights = [f for f in top if f.endswith(".safetensors")] or \
                  [f for f in top if f.startswith("pytorch_model") and f.endswith(".bin")]
        return keep + weights
    roles = {"sherpa-online-paraformer": ["encoder", "decoder"], "sherpa-online-transducer": ["encoder", "decoder", "joiner"],
             "sherpa-offline-transducer": ["encoder", "decoder", "joiner"], "sherpa-sensevoice": ["model"],
             "sherpa-offline-paraformer": ["model"]}[kind]
    return ["tokens.txt"] + [pick_onnx(top, r) for r in roles]


def pick_onnx(files: list[str], role: str) -> str:
    cands = sorted(f for f in files if f.startswith(role) and f.endswith(".onnx"))
    int8 = [f for f in cands if ".int8." in f]
    if not cands:
        raise UnsupportedModel(f"缺少 {role}*.onnx")
    return (int8 or cands)[0]  # int8 runs much faster on CPU with near-identical accuracy


def read_config(path: str) -> dict | None:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------------------------- base classes
class Engine:
    kind = ""
    streaming = True

    def __init__(self, path: str, device: str):
        self.path, self.device = path, device
        self.lock = threading.Lock()

    def latency_options(self) -> list[tuple[str, int]]:
        raise NotImplementedError

    def default_latency(self) -> int:
        return self.latency_options()[0][1]

    def latency_ms(self, value: int) -> int:
        raise NotImplementedError

    def new_stream(self, language: str, latency: int, on_text: Callable[[str], None]):
        raise NotImplementedError

    def close(self):
        pass


class PseudoStream:
    """For models that only transcribe whole clips: re-transcribe the growing utterance every `interval` s."""

    def __init__(self, transcribe: Callable[[np.ndarray], str], interval: float, on_text: Callable[[str], None]):
        self.transcribe, self.interval, self.on_text = transcribe, interval, on_text
        self.buf = np.zeros(0, dtype=np.float32)
        self.text = ""
        self.cond = threading.Condition()
        self.closed = False
        self.decoded_len = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def feed(self, audio: np.ndarray):
        with self.cond:
            self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
            self.cond.notify_all()

    def _loop(self):
        while True:
            with self.cond:
                while not self.closed and len(self.buf) - self.decoded_len < self.interval * SAMPLE_RATE:
                    self.cond.wait(0.1)
                if self.closed:
                    return
                audio = self.buf.copy()
            self.decoded_len = len(audio)
            if len(audio) >= SAMPLE_RATE * 0.5:
                t = time.perf_counter()
                text = self.transcribe(audio).strip()
                if text != self.text and not self.closed:
                    self.text = text
                    self.on_text(text)
                # if decoding is slower than the interval, back off instead of falling further behind
                self.interval = max(self.interval, (time.perf_counter() - t) * 1.2)

    def abort(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.thread.join(30)

    def finish(self, timeout: float = 30.0) -> str:
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.thread.join(timeout)
        if len(self.buf) < SAMPLE_RATE * 0.3:
            return self.text
        self.text = self.transcribe(self.buf).strip()
        return self.text


PSEUDO_LATENCY = [("每 0.5 秒更新", 500), ("每 1 秒更新", 1000), ("每 2 秒更新（省資源）", 2000)]


# ---------------------------------------------------------------------------------------------- factory
def load_engine(kind: str, path: str, device: str) -> Engine:
    if kind == "nemotron":
        from .nemotron import NemotronEngine as E
    elif kind == "whisper":
        from .whisper import WhisperEngine as E
    elif kind.startswith("sherpa-"):
        from .sherpa import SherpaEngine as E
    else:
        raise UnsupportedModel(kind)
    e = E(path, device)
    e.kind = kind
    return e
