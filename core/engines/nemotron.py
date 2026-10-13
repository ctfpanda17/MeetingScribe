"""Engine for NVIDIA nemotron streaming ASR (cache-aware FastConformer-RNNT), e.g. nemotron-3.5-asr-streaming-0.6b.

One `UtteranceStream` = one call to `model.generate` that consumes audio as it arrives. Audio is pushed
with `feed()`; partial text is reported through `on_text` while the person is still speaking.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.generation.streamers import BaseStreamer

from . import SAMPLE_RATE, Engine

LANG = {"zh": "zh-CN", "en": "en-US", "ja": "ja-JP", "ko": "ko-KR", "auto": "auto"}


class NemotronEngine(Engine):
    """Holds the model once; creates per-utterance streams. Latency value = lookahead tokens."""

    def __init__(self, path: str, device: str = "cuda"):
        super().__init__(path, device)
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForRNNT.from_pretrained(
            path, device_map=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32
        )
        self.model.eval()
        self._proc_lock = threading.Lock()  # processor's chunk sizes depend on shared lookahead state

    def latency_options(self):
        ms = self.processor.supported_streaming_latencies_ms  # {lookahead: ms}
        out = []
        for la in sorted(ms):
            tag = "（最快）" if la == min(ms) else "（最準）" if la == max(ms) else "（建議）" if la == 3 else ""
            out.append((f"{ms[la]} ms{tag}", la))
        return out

    def default_latency(self):
        values = [v for _, v in self.latency_options()]
        return 3 if 3 in values else values[len(values) // 2]

    def latency_ms(self, value):
        return self.chunk_params(value)["latency_ms"]

    def new_stream(self, language, latency, on_text):
        return UtteranceStream(self, LANG.get(language, language), latency, on_text)

    def close(self):
        del self.model
        torch.cuda.empty_cache()

    def chunk_params(self, lookahead: int) -> dict:
        with self._proc_lock:
            p = self.processor
            p.set_num_lookahead_tokens(lookahead)
            return {
                "lookahead": lookahead,
                "first_samples": p.num_samples_first_audio_chunk,
                "first_mel": p.num_mel_frames_first_audio_chunk,
                "chunk_samples": p.num_samples_per_audio_chunk,
                "chunk_mel": p.num_mel_frames_per_audio_chunk,
                "hop": p.feature_extractor.hop_length,
                "n_fft": p.feature_extractor.n_fft,
                "latency_ms": p.streaming_latency_ms,
            }

    def features(self, audio: np.ndarray, first: bool, language: str, lookahead: int):
        with self._proc_lock:
            self.processor.set_num_lookahead_tokens(lookahead)
            inputs = self.processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=first,
                language=language,
                return_tensors="pt",
            )
        return inputs.to(self.model.device, dtype=self.model.dtype)


class _TokenStreamer(BaseStreamer):
    """Receives token ids from generate() and reports the decoded running text."""

    def __init__(self, tokenizer, on_text: Callable[[str], None]):
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.ids: list[int] = []
        self.text = ""

    def put(self, value):
        if value.dim() > 1:
            value = value[0]
        self.ids.extend(value.tolist())
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True).strip()
        if text != self.text:
            self.text = text
            try:
                self.on_text(text)
            except Exception:  # a UI hiccup must never kill the recognition thread
                pass

    def end(self):
        pass


class UtteranceStream:
    def __init__(self, engine: NemotronEngine, language: str, lookahead: int, on_text: Callable[[str], None]):
        self.engine = engine
        self.language = language
        self.lookahead = lookahead
        self.cp = engine.chunk_params(lookahead)
        self.buf = np.zeros(0, dtype=np.float32)
        self.cond = threading.Condition()
        self.closed = False
        self.streamer = _TokenStreamer(engine.processor.tokenizer, on_text)
        self.thread: threading.Thread | None = None
        self.error: Exception | None = None

    @property
    def text(self) -> str:
        return self.streamer.text

    # -- producer side ---------------------------------------------------------------------------------
    def feed(self, audio: np.ndarray):
        with self.cond:
            self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
            self.cond.notify_all()
        if self.thread is None and len(self.buf) >= self.cp["first_samples"]:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def finish(self, timeout: float = 30.0) -> str:
        """Flush remaining audio (padding with silence so the lookahead is satisfied) and return final text."""
        if self.thread is None:
            if len(self.buf) == 0:
                return ""
            self.feed(np.zeros(self.cp["first_samples"], dtype=np.float32))
        pad = np.zeros(max(SAMPLE_RATE, self.cp["chunk_samples"] * 3) + self.cp["n_fft"], dtype=np.float32)
        with self.cond:
            self.buf = np.concatenate([self.buf, pad])
            self.closed = True
            self.cond.notify_all()
        self.thread.join(timeout)
        if self.error:
            raise self.error
        return self.streamer.text

    def abort(self):
        """Stop decoding now (no flush): the audio generator simply ends."""
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        if self.thread is not None:
            self.thread.join(30)

    # -- consumer side (runs inside generate) ------------------------------------------------------------
    def _wait_for(self, n: int) -> bool:
        with self.cond:
            while len(self.buf) < n and not self.closed:
                self.cond.wait()
            return len(self.buf) >= n

    def _run(self):
        try:
            cp, eng = self.cp, self.engine
            first = eng.features(self.buf[: cp["first_samples"]], True, self.language, self.lookahead)

            def gen():
                yield first.input_features[:, : cp["first_mel"], :]
                mel_idx = cp["first_mel"]
                start = mel_idx * cp["hop"] - cp["n_fft"] // 2
                while self._wait_for(start + cp["chunk_samples"]):
                    seg = self.buf[start : start + cp["chunk_samples"]]
                    yield eng.features(seg, False, self.language, self.lookahead).input_features
                    mel_idx += cp["chunk_mel"]
                    start = mel_idx * cp["hop"] - cp["n_fft"] // 2

            with torch.inference_mode(), eng.lock:
                eng.model.generate(**{**first, "input_features": gen(), "streamer": self.streamer})
        except Exception as e:  # surfaced in finish()
            self.error = e
