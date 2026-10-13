"""Speaker-change detection inside continuous speech (no pause needed).

pyannote segmentation-3.0 (ONNX) labels every ~17 ms frame with which of up to 3 local voices is active,
including overlaps. A change point is where one voice's run ends and another's begins; for interruptions
(overlap) the change is placed where the new voice starts.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import onnxruntime as ort

SR = 16000
FRAME_SHIFT = 270  # samples per output frame
RECEPTIVE = 991
# powerset classes -> active local speakers (3 speakers, at most 2 at once)
POWERSET = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]

SEG_REPO = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
SEG_FILE = "model.int8.onnx"


class ChangeDetector:
    MIN_RUN = 0.45  # a voice must hold the floor this long to count as a turn (filters coughs, "嗯")

    def __init__(self, model_path: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.lock = threading.Lock()

    def activity(self, audio: np.ndarray) -> np.ndarray:
        """(frames, 3) boolean activity of local speakers."""
        with self.lock:
            y = self.sess.run(None, {self.inp: audio.astype(np.float32)[None, None, :]})[0][0]
        cls = y.argmax(-1)
        act = np.zeros((len(cls), 3), dtype=bool)
        for i, c in enumerate(cls):
            for s in POWERSET[c]:
                act[i, s] = True
        return act

    @staticmethod
    def frame_to_sample(i: int) -> int:
        return i * FRAME_SHIFT + RECEPTIVE // 2

    def change_points(self, audio: np.ndarray, min_run: float | None = None) -> list[tuple[int, int]]:
        """(new_voice_starts, old_voice_stops) sample positions within `audio`, one pair per change of speaker.

        Without overlap both are about equal; when someone interrupts, the old voice stops after the new one starts.
        """
        if len(audio) < SR:
            return []
        act = self.activity(audio)
        # dominant voice per frame: the single active one; during overlap, the one that started most recently
        dom = np.full(len(act), -1)
        last = -1
        for i, a in enumerate(act):
            on = np.flatnonzero(a)
            if len(on) == 1:
                last = on[0]
            elif len(on) == 2:
                started = [s for s in on if i == 0 or not act[i - 1, s]]
                last = started[0] if started else (last if last in on else on[0])
            dom[i] = last if len(on) else -1
        # runs of the same dominant voice; drop runs shorter than MIN_RUN
        min_frames = int((min_run or self.MIN_RUN) * SR / FRAME_SHIFT)
        runs = []  # [speaker, start_frame, end_frame]
        for i, s in enumerate(dom):
            if s < 0:
                continue
            if runs and runs[-1][0] == s:
                runs[-1][2] = i
            else:
                runs.append([s, i, i])
        runs = [r for r in runs if r[2] - r[1] + 1 >= min_frames]
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        out = []
        for prev, cur in zip(merged, merged[1:]):
            # start of the new voice, moved back to where it first became active (overlap onset)
            j = cur[1]
            while j > prev[1] and act[j - 1, cur[0]]:
                j -= 1
            k = prev[2]  # last frame where the old voice is still active
            while k + 1 < len(act) and act[k + 1, prev[0]]:
                k += 1
            out.append((self.frame_to_sample(j), self.frame_to_sample(k + 1)))
        return out


def model_path(models_dir: str) -> str:
    return os.path.join(models_dir, "segmentation", SEG_FILE)
