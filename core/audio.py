"""Live audio capture on Windows: microphone and/or system playback (WASAPI loopback), mixed to 16 kHz mono."""
from __future__ import annotations

import collections
import threading
import time
from typing import Callable

import numpy as np
import pyaudiowpatch as pa
import soxr

RATE = 16000
TICK = 0.1  # emit 100 ms packets
PREBUFFER = 0.2  # jitter buffer before mixing starts
MAX_BUFFER = 1.0  # drop the oldest audio if a source runs ahead
CUSHION = 0.3  # each source keeps this much audio in reserve; devices deliver in irregular bursts


def _wasapi(p: pa.PyAudio) -> int:
    return p.get_host_api_info_by_type(pa.paWASAPI)["index"]


def list_microphones() -> list[dict]:
    p = pa.PyAudio()
    try:
        api = _wasapi(p)
        default = p.get_host_api_info_by_index(api)["defaultInputDevice"]
        out = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] == api and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice"):
                out.append({"index": i, "name": d["name"], "default": i == default})
        out.sort(key=lambda d: not d["default"])
        return out
    finally:
        p.terminate()


def list_speakers() -> list[dict]:
    """Playback devices whose sound can be captured (WASAPI loopback) — where Meet / Teams audio comes out."""
    p = pa.PyAudio()
    try:
        try:
            default = p.get_default_wasapi_loopback()["index"]
        except Exception:
            default = None
        out = [{"index": d["index"], "name": d["name"].replace(" [Loopback]", ""), "default": d["index"] == default}
               for d in p.get_loopback_device_info_generator()]
        out.sort(key=lambda d: not d["default"])
        return out
    finally:
        p.terminate()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


class _Source:
    def __init__(self, p: pa.PyAudio, info: dict):
        self.channels = max(1, min(int(info["maxInputChannels"]), 2))
        self.rate = int(info["defaultSampleRate"])
        self.resampler = soxr.ResampleStream(self.rate, RATE, 1, dtype="float32")
        self.buf: collections.deque[np.ndarray] = collections.deque()
        self.size = 0
        self.primed = False  # enough audio buffered to read steadily without gaps
        self.last_arrival = 0.0
        self.lock = threading.Lock()
        self.stream = p.open(
            format=pa.paFloat32, channels=self.channels, rate=self.rate, input=True,
            input_device_index=info["index"], frames_per_buffer=int(self.rate * 0.02),
            stream_callback=self._cb,
        )

    def _cb(self, data, frames, time_info, status):
        x = np.frombuffer(data, dtype=np.float32)
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(1)
        y = self.resampler.resample_chunk(x)
        with self.lock:
            self.buf.append(y)
            self.size += len(y)
            self.last_arrival = time.perf_counter()
            while self.size > RATE * MAX_BUFFER:
                self.size -= len(self.buf.popleft())
        return None, pa.paContinue

    def take(self, n: int) -> np.ndarray:
        """Next n samples. Waits (silence) until CUSHION is buffered, then reads steadily; after an underrun it
        buffers again instead of punching silence into the middle of words. A leftover tail with no new audio
        coming (playback stopped) is flushed."""
        out = np.zeros(n, dtype=np.float32)
        got = 0
        with self.lock:
            idle = time.perf_counter() - self.last_arrival > 0.15
            if not self.primed:
                if self.size >= CUSHION * RATE:
                    self.primed = True
                elif not (idle and self.size):
                    return out
            if self.primed and self.size < n:
                self.primed = False  # underrun: rebuffer before reading on
            while got < n and self.buf:
                head = self.buf[0]
                k = min(n - got, len(head))
                out[got : got + k] = head[:k]
                got += k
                if k == len(head):
                    self.buf.popleft()
                else:
                    self.buf[0] = head[k:]
            self.size -= got
        return out

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass


class LiveCapture:
    """mode: 'mic' | 'system' | 'both'. Calls on_audio(float32[1600]) every 100 ms from a worker thread.

    anti_echo: in 'both' mode, when the remote side is talking (system audio active) the microphone is turned
    down, so meeting audio coming out of loudspeakers and back into the mic is not transcribed twice.
    """

    ECHO_HOLD = 0.4  # keep ducking this long after remote speech stops (room echo tail)

    def __init__(self, mode: str, mic_index: int | None, on_audio: Callable[[np.ndarray], None],
                 speaker_index: int | None = None, anti_echo: bool = True):
        self.on_audio = on_audio
        self.anti_echo = anti_echo
        self.mic: _Source | None = None
        self.system: _Source | None = None
        self.p = pa.PyAudio()
        self.sources: list[_Source] = []
        self.running = False
        try:
            if mode in ("mic", "both"):
                idx = mic_index if mic_index is not None else self.p.get_host_api_info_by_index(
                    _wasapi(self.p))["defaultInputDevice"]
                self.mic = _Source(self.p, self.p.get_device_info_by_index(idx))
                self.sources.append(self.mic)
            if mode in ("system", "both"):
                info = (self.p.get_device_info_by_index(speaker_index) if speaker_index is not None
                        else self.p.get_default_wasapi_loopback())
                self.system = _Source(self.p, info)
                self.sources.append(self.system)
        except Exception:
            self.close()
            raise

    def start(self):
        self.running = True
        for s in self.sources:
            s.stream.start_stream()
        threading.Thread(target=self._mix, daemon=True).start()

    def _mix(self):
        time.sleep(PREBUFFER)
        n = int(RATE * TICK)
        nxt = time.perf_counter()
        remote_until = 0.0
        gain = 1.0
        while self.running:
            if self.mic and self.system:
                mic, sysa = self.mic.take(n), self.system.take(n)
                now = time.perf_counter()
                if _rms(sysa) > 0.005:
                    remote_until = now + self.ECHO_HOLD
                talking_over = _rms(mic) > 2.5 * max(_rms(sysa), 0.004)
                target = 0.08 if (self.anti_echo and now < remote_until and not talking_over) else 1.0
                ramp = np.linspace(gain, target, n, dtype=np.float32)  # avoid clicks
                gain = target
                mixed = np.clip(mic * ramp + sysa, -1.0, 1.0)
                levels = (_rms(mic * ramp), _rms(sysa))  # who is louder: me (mic) or the meeting (speakers)
            else:
                mixed = self.sources[0].take(n)
                levels = None
            self.on_audio(mixed.astype(np.float32), levels)
            nxt += TICK
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def close(self):
        self.running = False
        for s in self.sources:
            s.close()
        self.sources = []
        try:
            self.p.terminate()
        except Exception:
            pass


class FileFeeder:
    """Plays an audio file into the pipeline faster than real time (for recordings made elsewhere)."""

    def __init__(self, path: str, on_audio: Callable[[np.ndarray], None], on_progress: Callable[[float, float], None],
                 on_done: Callable[[], None], speed: float = 5.0):
        import librosa

        self.audio, _ = librosa.load(path, sr=RATE, mono=True)
        self.on_audio, self.on_progress, self.on_done, self.speed = on_audio, on_progress, on_done, speed
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        n = int(RATE * TICK)
        total = len(self.audio) / RATE
        for i in range(0, len(self.audio), n):
            if not self.running:
                return
            self.on_audio(self.audio[i : i + n].astype(np.float32))
            self.on_progress(min(total, (i + n) / RATE), total)
            time.sleep(TICK / self.speed)
        self.running = False
        self.on_done()

    def close(self):
        self.running = False
