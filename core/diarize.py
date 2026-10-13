"""Speaker identification: voice-activity detection + ECAPA voiceprints + online clustering."""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from silero_vad import VADIterator, load_silero_vad
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from paths import MODEL_DIR, PROFILES_PATH
VAD_WINDOW = 512  # silero expects 512 samples @16kHz


class SpeakerEncoder:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(MODEL_DIR, "ecapa"),
            run_opts={"device": "cuda:0" if device == "cuda" else device},
            local_strategy=LocalStrategy.COPY,
        )
        self.lock = threading.Lock()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 1600:  # under 0.1 s: pad, the model needs a few frames
            audio = np.pad(audio, (0, 1600 - len(audio)))
        with self.lock:
            for attempt in range(3):
                try:
                    with torch.inference_mode():
                        wav = torch.from_numpy(audio.astype(np.float32))[None].to(self.model.device)
                        emb = self.model.encode_batch(wav).squeeze()
                    return F.normalize(emb, dim=0).cpu().numpy()
                except RuntimeError as e:  # GPU hiccup (other apps on the GPU, driver reset...)
                    print(f"[voiceprint] {type(e).__name__}: {str(e)[:120]}", flush=True)
                    if self.device == "cpu" or attempt == 1:
                        self._to_cpu()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
            raise RuntimeError("voiceprint failed on CPU too")

    def _to_cpu(self):
        """Keep the meeting going on the CPU if the GPU keeps failing."""
        if self.device != "cpu":
            print("[voiceprint] falling back to CPU", flush=True)
            self.model.to("cpu")
            self.model.device = "cpu"
            self.device = "cpu"


def new_vad() -> VADIterator:
    return VADIterator(
        load_silero_vad(), threshold=0.5, sampling_rate=16000, min_silence_duration_ms=600, speech_pad_ms=120
    )


class SpeakerRegistry:
    """Online clustering of utterance voiceprints into speakers."""

    MIN_RELIABLE_SEC = 1.0  # shorter utterances never create speakers nor update centroids

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.speakers: dict[str, dict] = {}
        self._next = 1
        for p in load_profiles():
            sid = self._new_id()
            self.speakers[sid] = {
                "id": sid, "name": p["name"], "sum": np.array(p["embedding"], dtype=np.float32),
                "count": 1, "enrolled": True, "active": False,
            }

    def _new_id(self) -> str:
        sid = f"S{self._next}"
        self._next += 1
        return sid

    def _scores(self, emb: np.ndarray) -> list[tuple[float, str]]:
        out = []
        for sid, s in self.speakers.items():
            c = s["sum"] / (np.linalg.norm(s["sum"]) + 1e-9)
            out.append((float(np.dot(c, emb)), sid))
        return sorted(out, reverse=True)

    def predict(self, emb: np.ndarray) -> str | None:
        scores = self._scores(emb)
        if scores and scores[0][0] >= self.threshold:
            return scores[0][1]
        return None

    def assign(self, emb: np.ndarray, duration: float, last_speaker: str | None) -> tuple[str, float]:
        scores = self._scores(emb)
        reliable = duration >= self.MIN_RELIABLE_SEC
        if scores and scores[0][0] >= self.threshold:
            sid, score = scores[0][1], scores[0][0]
        elif not reliable and (last_speaker or scores):
            # Too short to trust: keep the conversation flowing with the most likely speaker.
            sid = last_speaker if last_speaker in self.speakers else scores[0][1]
            score = 0.0
        else:
            sid, score = self._new_id(), 1.0
            n = sum(1 for s in self.speakers.values() if not s["enrolled"]) + 1
            self.speakers[sid] = {
                "id": sid, "name": f"說話者 {n}", "sum": np.zeros_like(emb), "count": 0,
                "enrolled": False, "active": True,
            }
        s = self.speakers[sid]
        s["active"] = True
        if reliable:
            s["sum"] = s["sum"] + emb * min(duration, 10.0)
            s["count"] += 1
        return sid, score

    def similarity(self, emb: np.ndarray, sid: str) -> float:
        s = self.speakers.get(sid)
        if s is None or s["count"] <= 0:
            return -1.0
        return float(np.dot(s["sum"] / (np.linalg.norm(s["sum"]) + 1e-9), emb))

    def best(self, emb: np.ndarray, exclude: str | None = None) -> tuple[str | None, float]:
        """Closest speaker that has a voiceprint (optionally ignoring one)."""
        for score, sid in self._scores(emb):
            if sid != exclude and self.speakers[sid]["count"] > 0:
                return sid, score
        return None, -1.0

    def move(self, emb: np.ndarray, duration: float, src: str, dst: str):
        """A sentence turned out to be someone else's: move its voiceprint contribution between speakers."""
        if duration >= self.MIN_RELIABLE_SEC:
            w = min(duration, 10.0)
            if src in self.speakers and self.speakers[src]["count"] > 0:
                s = self.speakers[src]
                s["count"] -= 1
                s["sum"] = s["sum"] - emb * w if s["count"] > 0 else np.zeros_like(s["sum"])
            if dst in self.speakers:
                self.speakers[dst]["sum"] = self.speakers[dst]["sum"] + emb * w
                self.speakers[dst]["count"] += 1
        if dst in self.speakers:
            self.speakers[dst]["active"] = True

    def new_speaker(self, emb: np.ndarray) -> str:
        sid = self._new_id()
        n = sum(1 for s in self.speakers.values() if not s["enrolled"]) + 1
        self.speakers[sid] = {"id": sid, "name": f"說話者 {n}", "sum": emb.astype(np.float32).copy(), "count": 1,
                              "enrolled": False, "active": True}
        return sid

    def add(self, sid: str, emb: np.ndarray, weight: float):
        s = self.speakers[sid]
        s["sum"] = s["sum"] + emb * weight
        s["count"] += 1
        s["active"] = True

    def set_active(self, used: set[str]):
        for sid, s in self.speakers.items():
            s["active"] = sid in used

    def rename(self, sid: str, name: str):
        if sid in self.speakers and name.strip():
            self.speakers[sid]["name"] = name.strip()[:40]

    def merge(self, src: str, dst: str):
        if src in self.speakers and dst in self.speakers and src != dst:
            a, b = self.speakers.pop(src), self.speakers[dst]
            b["sum"] = b["sum"] + a["sum"]
            b["count"] += a["count"]

    def enroll(self, sid: str):
        """Remember this voiceprint across meetings."""
        s = self.speakers.get(sid)
        if not s or not s["sum"].any():
            return False
        profiles = [p for p in load_profiles() if p["name"] != s["name"]]
        c = s["sum"] / (np.linalg.norm(s["sum"]) + 1e-9)
        profiles.append({"name": s["name"], "embedding": c.round(6).tolist()})
        with open(PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False)
        s["enrolled"] = True
        return True

    def public(self) -> list[dict]:
        return [
            {"id": s["id"], "name": s["name"], "enrolled": s["enrolled"]}
            for s in self.speakers.values() if s["active"]
        ]


def load_profiles() -> list[dict]:
    try:
        with open(PROFILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def forget_profile(name: str):
    profiles = [p for p in load_profiles() if p["name"] != name]
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False)


CLUSTER_REPO = "csukuangfj/speaker-embedding-models"
CLUSTER_FILE = "nemo_en_titanet_large.onnx"  # NVIDIA TitaNet-Large (CC-BY-4.0), ONNX export by k2-fsa


class ClusterEncoder:
    """Stronger (slower, CPU) voiceprint used only for the periodic whole-meeting re-clustering."""

    def __init__(self, model_path: str, threads: int = 2):
        import sherpa_onnx as so

        self.ex = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=threads))
        self.lock = threading.Lock()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self.lock:
            s = self.ex.create_stream()
            s.accept_waveform(16000, audio.astype(np.float32))
            s.input_finished()
            v = np.array(self.ex.compute(s), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def cluster_model_path() -> str:
    from paths import MODEL_DIR

    return os.path.join(MODEL_DIR, "speaker", CLUSTER_FILE)
