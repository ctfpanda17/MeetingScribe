"""Who-is-speaking hints from outside the audio: the meeting app (via the browser extension) and the microphone.

The meeting platform knows who is talking because every remote participant has their own microphone stream;
the app also knows when the local microphone (not the computer's speakers) carries the speech. Hints are kept
as a timeline of names (meeting time, seconds) and fused with the voice-based diarization in session.py.
"""
from __future__ import annotations

import threading
import time

import numpy as np

RES = 0.1  # seconds per hint frame
ME = "__me__"  # the local microphone


class SpeakerHints:
    def __init__(self):
        self.lock = threading.Lock()
        self.tracks: dict[str, np.ndarray] = {}
        self.n = 0
        self.wall0: float | None = None  # wall-clock time of meeting time 0 (set from the audio clock)
        self._active: dict[str, float] = {}  # platform names currently speaking -> since (meeting time)
        self.platform = ""
        self.last_event = 0.0

    # ------------------------------------------------------------------ recording
    def _track(self, name: str, upto: int) -> np.ndarray:
        if upto > self.n:
            grow = upto - self.n + int(60 / RES)
            for k in self.tracks:
                self.tracks[k] = np.concatenate([self.tracks[k], np.zeros(grow, bool)])
            self.n += grow
        if name not in self.tracks:
            self.tracks[name] = np.zeros(self.n, bool)
        return self.tracks[name]

    def add(self, name: str, t0: float, t1: float):
        if t1 <= t0 or not name:
            return
        a, b = int(max(0.0, t0) / RES), int(np.ceil(t1 / RES))
        with self.lock:
            self._track(name, b)[a:b] = True

    def to_meeting_time(self, wall: float, latency: float) -> float | None:
        """Wall-clock (time.time()) -> meeting time. `latency`: how long after the speech the cue appears."""
        if self.wall0 is None:
            return None
        return wall - self.wall0 - latency

    def platform_update(self, speaking: list[str], wall: float, latency: tuple[float, float], platform: str = ""):
        """The meeting app says these people are speaking now (an empty list = nobody).
        latency: (how late the "speaking" cue appears, how late it disappears) in seconds."""
        self.platform, self.last_event = platform or self.platform, time.time()
        t_on = self.to_meeting_time(wall, latency[0])
        t_off = self.to_meeting_time(wall, latency[1])
        if t_on is None:
            return
        now = set(n.strip()[:60] for n in speaking if n and n.strip())
        for name in list(self._active):
            if name not in now:
                self.add(name, self._active.pop(name), t_off)
        for name in now:
            self._active.setdefault(name, t_on)

    def flush(self, upto: float):
        """Close still-open platform intervals up to `upto` (meeting time), keeping them open."""
        for name, since in list(self._active.items()):
            if upto > since:
                self.add(name, since, upto)

    # ------------------------------------------------------------------ queries
    def coverage(self, t0: float, t1: float) -> dict[str, float]:
        """Seconds each name was flagged as speaking within [t0, t1)."""
        a, b = int(max(0.0, t0) / RES), int(np.ceil(max(t0, t1) / RES))
        with self.lock:
            return {k: float(v[a:b].sum()) * RES for k, v in self.tracks.items() if v[a:b].any()}

    def frames(self, t0: float, t1: float) -> dict[str, np.ndarray]:
        a, b = int(max(0.0, t0) / RES), int(np.ceil(max(t0, t1) / RES))
        with self.lock:
            out = {}
            for k, v in self.tracks.items():
                seg = v[a:b]
                if len(seg) < b - a:
                    seg = np.pad(seg, (0, b - a - len(seg)))
                out[k] = seg.copy()
            return out

    @property
    def connected(self) -> bool:
        return time.time() - self.last_event < 10
