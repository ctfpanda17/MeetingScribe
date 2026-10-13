"""Engine for sherpa-onnx models (CPU): streaming Paraformer / Zipformer, and SenseVoice / offline models.

Model folders come from k2-fsa / csukuangfj on Hugging Face and contain tokens.txt plus .onnx files.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import sherpa_onnx as so

from . import PSEUDO_LATENCY, SAMPLE_RATE, Engine, PseudoStream, pick_onnx

THREADS = int(os.environ.get("MEETINGSCRIBE_ASR_THREADS", 0)) or max(2, min(8, (os.cpu_count() or 4) // 2))
SENSEVOICE_LANG = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "auto": "auto"}


class SherpaEngine(Engine):
    def __init__(self, path: str, device: str = "cpu"):
        super().__init__(path, device)
        self.files = os.listdir(path)
        self.tokens = os.path.join(path, "tokens.txt")
        self._offline: dict[str, so.OfflineRecognizer] = {}
        self.online = None

    def _f(self, role: str) -> str:
        return os.path.join(self.path, pick_onnx(self.files, role))

    @property
    def streaming(self):
        return self.kind.startswith("sherpa-online")

    def _build(self):
        """Create the recognizer lazily (kind is set by load_engine after __init__)."""
        if self.streaming and self.online is None:
            if self.kind == "sherpa-online-paraformer":
                self.online = so.OnlineRecognizer.from_paraformer(
                    tokens=self.tokens, encoder=self._f("encoder"), decoder=self._f("decoder"), num_threads=THREADS)
            else:
                self.online = so.OnlineRecognizer.from_transducer(
                    tokens=self.tokens, encoder=self._f("encoder"), decoder=self._f("decoder"),
                    joiner=self._f("joiner"), num_threads=THREADS)

    def _offline_for(self, language: str) -> so.OfflineRecognizer:
        key = language if self.kind == "sherpa-sensevoice" else "_"
        if key not in self._offline:
            if self.kind == "sherpa-sensevoice":
                rec = so.OfflineRecognizer.from_sense_voice(
                    model=self._f("model"), tokens=self.tokens, num_threads=THREADS,
                    language=SENSEVOICE_LANG.get(language, "auto"), use_itn=True)
            elif self.kind == "sherpa-offline-paraformer":
                rec = so.OfflineRecognizer.from_paraformer(paraformer=self._f("model"), tokens=self.tokens,
                                                           num_threads=THREADS)
            else:
                rec = so.OfflineRecognizer.from_transducer(
                    encoder=self._f("encoder"), decoder=self._f("decoder"), joiner=self._f("joiner"),
                    tokens=self.tokens, num_threads=THREADS)
            self._offline[key] = rec
        return self._offline[key]

    def latency_options(self):
        if self.streaming:
            return [("即時", 0)]
        return PSEUDO_LATENCY

    def default_latency(self):
        return 0 if self.streaming else 500

    def latency_ms(self, value):
        return 200 if self.streaming else int(value)

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        # the whole call is serialised: feature extraction is not safe to run on two threads at once
        # (live partial updates and a sentence's final pass overlap constantly during a meeting)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with self.lock:
            rec = self._offline_for(language)
            s = rec.create_stream()
            s.accept_waveform(SAMPLE_RATE, audio)
            rec.decode_stream(s)
            return s.result.text

    def new_stream(self, language, latency, on_text):
        if self.streaming:
            self._build()
            return OnlineStream(self, on_text)
        self._offline_for(language)  # load before the first words arrive
        return PseudoStream(lambda a: self.transcribe(a, language), latency / 1000, on_text)


class OnlineStream:
    TAIL = np.zeros(int(0.66 * SAMPLE_RATE), dtype=np.float32)  # lets the model flush its last chunk

    def __init__(self, engine: SherpaEngine, on_text):
        self.engine, self.on_text = engine, on_text
        self.rec = engine.online
        self.stream = self.rec.create_stream()
        self.text = ""
        self.lock = threading.Lock()

    def _decode(self):
        with self.engine.lock:
            while self.rec.is_ready(self.stream):
                self.rec.decode_stream(self.stream)
            text = self.rec.get_result(self.stream)
        text = text if isinstance(text, str) else text.text
        if text != self.text:
            self.text = text
            try:
                self.on_text(text)
            except Exception:
                pass

    def feed(self, audio: np.ndarray):
        with self.lock:
            self.stream.accept_waveform(SAMPLE_RATE, audio.astype(np.float32))
            self._decode()

    def abort(self):
        pass

    def finish(self, timeout: float = 0) -> str:
        with self.lock:
            self.stream.accept_waveform(SAMPLE_RATE, self.TAIL)
            self.stream.input_finished()
            self._decode()
        return self.text
