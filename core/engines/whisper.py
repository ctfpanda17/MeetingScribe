"""Engine for Whisper-family models (openai/whisper-*, MediaTek-Research/Breeze-ASR-25, other fine-tunes).

Whisper only transcribes whole clips, so live text comes from re-transcribing the current utterance
every update interval (PseudoStream). Utterances are cut at 25 s, inside Whisper's 30 s window.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from . import PSEUDO_LATENCY, SAMPLE_RATE, Engine, PseudoStream


class WhisperEngine(Engine):
    streaming = False

    def __init__(self, path: str, device: str = "cuda"):
        super().__init__(path, device)
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path, dtype=self.dtype).to(device)
        self.model.eval()

    def latency_options(self):
        return PSEUDO_LATENCY

    def default_latency(self):
        return 1000

    def latency_ms(self, value):
        return int(value)

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        kwargs = {"task": "transcribe"}
        if language != "auto":
            kwargs["language"] = language
        with self.lock, torch.inference_mode():
            ids = self.model.generate(feats.to(self.device, dtype=self.dtype), max_new_tokens=220, **kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def new_stream(self, language, latency, on_text):
        return PseudoStream(lambda a: self.transcribe(a, language), latency / 1000, on_text)

    def close(self):
        del self.model
        torch.cuda.empty_cache()
