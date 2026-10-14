"""One live meeting: audio in -> utterances (VAD) -> streaming text + speaker label -> events out."""
from __future__ import annotations

import datetime as dt
import json
import os
import queue
import re
import struct
import threading
import time
from typing import Callable

import numpy as np
import torch

from engines import SAMPLE_RATE, Engine
from diarize import VAD_WINDOW, SpeakerEncoder, SpeakerRegistry, new_vad
from hints import ME, SpeakerHints
from streamdiar import StreamingDiarizer

from paths import TRANSCRIPT_DIR

MAX_UTTERANCE_SEC = 25.0  # force a cut during very long monologues
SHORT_FOREIGN_SEC = 3.5  # auto language: English-only sentences up to this long in a Chinese meeting are re-checked
PREROLL_SEC = 0.3
PROVISIONAL_AT_SEC = 1.6  # guess the speaker this far into an utterance

# who-speaks-when comes from streaming diarization (see streamdiar.py)
TURN_CHECK_EVERY = 0.5  # seconds between live checks for a change of speaker inside the current sentence
LIVE_SETTLE = 1.5  # the diarization of the last moments is still provisional; act on audio older than this
TURN_MIN_RUN = 0.3  # a speaker must hold the floor this long to get a sentence of their own

# background correction: re-check finished sentences once more audio (and voiceprint history) is available
REFINE_EVERY = 6.0  # seconds between passes
RECLUSTER_EVERY = 15.0  # seconds of meeting between whole-meeting re-clusterings
REFINE_LOOKBACK = 90.0  # only sentences that ended within this many seconds (the whole meeting at the end)
REFINE_SETTLE = 2.0  # wait this long after a sentence ends before re-checking it
REFINE_MIN_PIECE = 0.35  # shortest piece whose voiceprint is trusted at all
REFINE_MARGIN = 0.10  # another speaker must match this much better before a label changes
REFINE_MAX_RELABELS = 4  # never flip one sentence back and forth more than this
# short audio gives weaker voiceprints: the right speaker still wins, but with lower absolute scores
SHORT_FLOOR = 0.30  # for guessed sentences and hidden short turns: best match must reach this
GUESSED_MARGIN = 0.05  # ...and beat the guessed speaker by this much
# hints from the meeting app / microphone
# (cue appears, cue disappears) this long after the speech starts / stops. Teams tiles measured Oct 2026 with a
# scripted dialogue through virtual audio cables: +0.22 s / +0.68 s.
HINT_LATENCY = {"captions": (1.0, 1.0), "tile": (0.25, 0.65)}
CAPTURE_LATENCY = 0.25  # audio reaches the app this long after it was spoken
MIC_DOMINANT = 1.5  # mic louder than the computer's sound by this factor (and audible) = the local user speaks
HINT_MIN_SECONDS = 2.0  # hinted speech needed before a voice is given a name
HINT_SHARE = 0.7  # ...and the share of its hinted speech that must carry that name
PUBLIC_KEYS = ("seg", "start", "end", "speaker", "score", "text", "corrected", "guessed")


class AudioLog:
    """The whole meeting's audio (16-bit, ~115 MB per hour) so finished sentences can be re-examined."""

    def __init__(self):
        self.buf = np.zeros(SAMPLE_RATE * 60, dtype=np.int16)
        self.n = 0
        self.lock = threading.Lock()

    def append(self, x: np.ndarray):
        with self.lock:
            if self.n + len(x) > len(self.buf):
                self.buf = np.concatenate([self.buf, np.zeros(max(len(self.buf), len(x)), dtype=np.int16)])
            self.buf[self.n:self.n + len(x)] = (np.clip(x, -1, 1) * 32767).astype(np.int16)
            self.n += len(x)

    def slice(self, a: int, b: int) -> np.ndarray:
        with self.lock:
            return self.buf[max(0, a):min(b, self.n)].astype(np.float32) / 32767

    @property
    def seconds(self) -> float:
        return self.n / SAMPLE_RATE


class WavWriter:
    """16-bit mono WAV whose header is kept up to date while recording, so the file stays playable
    even if the app is closed or killed before the meeting is finished."""

    HEADER_EVERY = 5.0  # seconds between header updates

    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.bytes = 0
        self.last_header = 0.0
        self._header()

    def _header(self):
        pos = self.f.tell()
        self.f.seek(0)
        self.f.write(struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + self.bytes, b"WAVE", b"fmt ", 16, 1, 1,
                                 SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b"data", self.bytes))
        self.f.seek(max(pos, 44))
        self.f.flush()
        self.last_header = time.monotonic()

    def write(self, x: np.ndarray):
        if self.f.closed:
            return
        data = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        self.f.write(data)
        self.bytes += len(data)
        if time.monotonic() - self.last_header >= self.HEADER_EVERY:
            self._header()

    def close(self):
        if not self.f.closed:
            self._header()
            self.f.close()


try:
    from opencc import OpenCC

    _S2T = OpenCC("s2twp")
except Exception:  # pragma: no cover
    _S2T = None

_CJK = r"[㐀-鿿豈-﫿]"


_AVATAR = [re.compile(p, re.I) for p in (r"\s*(的)?\s*(個人資料圖片|个人资料图片|大頭貼|头像|頭像)[。.]?$",
                                         r"^(profile picture|avatar|photo) (of|for)\s+",
                                         r"['’]s (profile picture|avatar|photo)\.?$", r"\s*(profile picture|avatar)\.?$")]


def person_name(label: str) -> str:
    """Meeting apps label video-less tiles with the avatar ("Kevin Lee 的個人資料圖片。"): keep the name only.
    (The extension does this too; repeated here for older extension versions.)"""
    n = re.sub(r"\s+", " ", label or "").strip()
    for p in _AVATAR:
        n = p.sub("", n)
    return n.strip()[:60]


def tidy(text: str, traditional: bool, final: bool = False) -> str:
    if re.search(_CJK, text):
        text = re.sub(r"\s*,\s*", "，", text)
        text = re.sub(r"\s*\?\s*", "？", text)
        text = re.sub(r"\s*!\s*", "！", text)
        text = re.sub(r"\s*;\s*", "；", text)
        text = re.sub(r"\s*:\s*", "：", text)
        text = re.sub(rf"(?<={_CJK})\s+(?={_CJK})", "", text)
        text = re.sub(r"(?<=[。，？！；：])\s+", "", text)
        if traditional and _S2T:
            text = _S2T.convert(text)
        if final and re.search(rf"{_CJK}$", text):
            text += "。"
    return text.strip()


class MeetingSession:
    def __init__(self, engine: Engine, encoder: SpeakerEncoder, emit: Callable[[dict], None], cfg: dict,
                 turns=None):
        self.engine, self.encoder, self.emit = engine, encoder, emit
        self.turns = turns  # turns.ChangeDetector; None = no diarization (split at pauses, voiceprint per sentence)
        self.refine = bool(cfg.get("refine", True))
        self.hints = SpeakerHints()
        self.me_name = (cfg.get("me_name") or "我").strip()[:40]
        self.platform_self = ""  # my own name inside the meeting app, if the extension knows it
        self.pushed = 0
        self.name_to_sids: dict[str, list[str]] = {}
        self.named: dict[str, str] = {}  # meeting-app name -> speaker id (the meeting app is the authority)
        self.language = (cfg.get("language") or "auto").split("-")[0]
        self.lang_votes = {"zh": 0, "en": 0}  # auto mode: language of the clearly identifiable sentences so far
        self.latency = int(cfg.get("latency", engine.default_latency()))
        self.traditional = bool(cfg.get("traditional", True))
        self.title = (cfg.get("title") or "").strip() or "會議紀錄"
        self.registry = SpeakerRegistry(float(cfg.get("threshold", 0.55)))
        self.vad = new_vad()

        self.stopped = False
        self.audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
        self.pending = np.zeros(0, dtype=np.float32)
        self.preroll = np.zeros(0, dtype=np.float32)
        self.samples_seen = 0

        self.stream = None
        self.utt_audio: list[np.ndarray] = []
        self.utt_start = 0.0
        self.utt_id = 0
        self.provisional: str | None = None
        self.next_turn_check = 0.0
        self.last_speaker: str | None = None
        self.segments: list[dict] = []

        self.started_at = dt.datetime.now()
        self.base_path = os.path.join(TRANSCRIPT_DIR, self.started_at.strftime("%Y%m%d_%H%M%S"))
        self.lock = threading.Lock()  # guards registry/segments vs. control messages
        self.id_lock = threading.Lock()
        self._last_id = 0
        self.log = AudioLog()
        self.seg_emb: dict[int, tuple[np.ndarray, float]] = {}  # seg -> (voiceprint, voiced seconds)
        self.wav = None
        if cfg.get("save_audio", True):
            self.wav = WavWriter(self.base_path + ".wav")
        self.registry.lock = self.lock
        self.diar = StreamingDiarizer(turns, encoder.embed, self.registry,
                                      cluster_embed=cfg.get("cluster_embed")) if turns is not None else None
        self.last_recluster = 0.0
        self.refine_stop = threading.Event()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.refiner = threading.Thread(target=self._refine_loop, daemon=True)
        self.refiner.start()

    def _new_id(self) -> int:
        with self.id_lock:
            self._last_id += 1
            return self._last_id

    # ---------------------------------------------------------------- input
    def push_audio(self, pcm: np.ndarray, levels: tuple[float, float] | None = None):
        """levels: (microphone rms, computer-sound rms) of this packet when both are captured."""
        t0 = self.pushed / SAMPLE_RATE
        self.pushed += len(pcm)
        if self.hints.wall0 is None:
            self.hints.wall0 = time.time() - CAPTURE_LATENCY - self.pushed / SAMPLE_RATE
        if levels is not None:
            mic, comp = levels
            if mic > 0.01 and mic > MIC_DOMINANT * comp:
                self.hints.add(ME, t0, self.pushed / SAMPLE_RATE)
        self.audio_q.put(pcm)

    def platform_event(self, event: dict):
        """From the browser extension: who the meeting app shows as speaking right now."""
        if event.get("self"):
            self.platform_self = person_name(event["self"])
        names = [person_name(n) for n in event.get("speaking", [])]
        names = [ME if (self.platform_self and n == self.platform_self) else n for n in names if n]
        self.hints.platform_update(names, event.get("t") or time.time(),
                                   HINT_LATENCY.get(event.get("source", ""), (0.4, 0.6)), event.get("platform", ""))

    def stop(self, timeout: float | None = None):
        """Finish: transcribe everything still queued, run the end-of-meeting pass, save.
        Waits until done unless a timeout is given (e.g. when the window is closing)."""
        if not self.stopped:
            self.stopped = True
            self.audio_q.put(None)
        self.worker.join(timeout=timeout)

    # ---------------------------------------------------------------- worker
    def _loop(self):
        try:
            while True:
                pcm = self.audio_q.get()
                if pcm is None:
                    break
                self.pending = np.concatenate([self.pending, pcm])
                while len(self.pending) >= VAD_WINDOW:
                    win, self.pending = self.pending[:VAD_WINDOW], self.pending[VAD_WINDOW:]
                    self._process_window(win)
            if self.wav is not None:  # all audio is in: finish the recording before the (long) final pass
                self.wav.close()
            if self.stream is not None:
                self._end_utterance()
            self.refine_stop.set()
            self.refiner.join(timeout=120)
            if self.refine:
                self._emit("整理整場會議…")
                self._refine(full=True)
            self.save()
            self.emit({"type": "stopped", "files": self.files()})
        except Exception as e:  # keep the UI informed instead of dying silently
            if self.wav is not None:
                self.wav.close()
            self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            raise

    def _process_window(self, win: np.ndarray):
        t = self.samples_seen / SAMPLE_RATE
        self.samples_seen += len(win)
        self.log.append(win)
        if self.wav is not None:
            self.wav.write(win)
        if self.diar is not None and self.log.seconds >= self.diar.next_t + self.diar.STEP:
            try:
                self.diar.update(self.log.slice, self.log.seconds)
            except Exception as e:  # a failed speaker analysis must never stop the transcript
                print(f"[diarization] skipped a window: {type(e).__name__}: {e}", flush=True)
                self.diar.next_t = max(self.diar.next_t, self.log.seconds - self.diar.STEP)
        ev = self.vad(torch.from_numpy(win))

        if self.stream is None:
            if ev and "start" in ev:
                self._begin_utterance(t)
                self._feed(win)
            else:
                keep = int(PREROLL_SEC * SAMPLE_RATE)
                self.preroll = np.concatenate([self.preroll, win])[-keep:]
            return

        self._feed(win)
        dur = self._utt_len()
        if self.diar is None and self.provisional is None and dur >= PROVISIONAL_AT_SEC:
            emb = self.encoder.embed(np.concatenate(self.utt_audio))
            with self.lock:
                self.provisional = self.registry.predict(emb) or "?"
            self._emit_partial(self.stream.text)
        if ev and "end" in ev:
            self._end_utterance()
        elif self.diar is not None and dur >= self.next_turn_check:
            self.next_turn_check = dur + TURN_CHECK_EVERY
            self._live_speaker()
        elif dur >= MAX_UTTERANCE_SEC:
            self._end_utterance()
            self._begin_utterance(self.samples_seen / SAMPLE_RATE)

    def _utt_len(self) -> float:
        return sum(len(a) for a in self.utt_audio) / SAMPLE_RATE

    def _feed(self, audio: np.ndarray):
        self.utt_audio.append(audio)
        self.stream.feed(audio)

    def _begin_utterance(self, t: float):
        self.utt_id = self._new_id()
        uid = self.utt_id
        self.utt_start = max(0.0, t - len(self.preroll) / SAMPLE_RATE)
        self.provisional = None
        self.next_turn_check = TURN_CHECK_EVERY
        self.utt_audio = []
        self.stream = self.engine.new_stream(
            self._lang(), self.latency, lambda txt, uid=uid: self._on_partial(uid, txt)
        )
        if len(self.preroll):
            self._feed(self.preroll)
            self.preroll = np.zeros(0, dtype=np.float32)

    def _on_partial(self, uid: int, text: str):
        if uid == self.utt_id:
            self._emit_partial(text)

    def _emit_partial(self, text: str):
        self.emit({
            "type": "partial", "seg": self.utt_id, "start": round(self.utt_start, 2),
            "speaker": self.provisional if self.provisional not in (None, "?") else None,
            "text": tidy(text, self.traditional),
        })

    def _end_utterance(self):
        stream, self.stream = self.stream, None
        audio = np.concatenate(self.utt_audio) if self.utt_audio else np.zeros(0, np.float32)
        self._finalize(self.utt_id, self.utt_start, audio, stream.finish())

    # ---------------------------------------------------------------- speaker turns without pauses
    def _live_speaker(self):
        """Show who is talking now, and close the sentence when the diarization says someone else took over."""
        now = self.log.seconds
        lab, _ = self.diar.label(max(self.utt_start, now - 3.0), now - 0.3)
        if lab and lab != self.provisional:
            self.provisional = lab
            self._emit_partial(self.stream.text)
        runs = self.diar.turns(self.utt_start, now - LIVE_SETTLE, min_run=TURN_MIN_RUN)
        if len(runs) >= 2 and runs[1][0] - self.utt_start >= TURN_MIN_RUN:
            audio = np.concatenate(self.utt_audio)
            cp = _snap_cut(audio, int((runs[1][0] - self.utt_start) * SAMPLE_RATE))
            self._split_at(cp, audio, runs[0][2], runs[1][2])

    def _split_at(self, cp: int, audio: np.ndarray, old_speaker: str, new_speaker: str):
        """Close the current sentence at sample `cp` (old speaker) and continue live with the new speaker."""
        old_id, old_start = self.utt_id, self.utt_start
        self.stream.abort()
        old_end = min(len(audio), cp + int(0.15 * SAMPLE_RATE))  # a little slack so the last word isn't clipped
        self._finalize(old_id, old_start, audio[:cp], self._transcribe_now(audio[:old_end]), speaker=old_speaker)
        self.utt_id = self._new_id()
        uid = self.utt_id
        self.utt_start = old_start + cp / SAMPLE_RATE
        self.utt_audio = []
        self.next_turn_check = TURN_CHECK_EVERY
        self.provisional = new_speaker
        self.stream = self.engine.new_stream(self._lang(), self.latency, lambda txt: self._on_partial(uid, txt))
        self._feed(audio[cp:])

    def _transcribe_now(self, audio: np.ndarray, language: str | None = None) -> str:
        """Decode already-recorded audio as fast as possible (no live partials needed)."""
        if self.engine.streaming:
            latency = max(v for _, v in self.engine.latency_options())  # biggest chunks: fewest steps, most accurate
        else:
            latency = 10 ** 7  # no intermediate re-transcriptions, just the final pass
        stream = self.engine.new_stream(language or self._lang(), latency, lambda _t: None)
        stream.feed(audio)
        return stream.finish()

    def _zh_meeting(self) -> bool:
        zh, en = self.lang_votes["zh"], self.lang_votes["en"]
        return zh >= 3 and zh >= 4 * en

    def _lang(self) -> str:
        """Language for decoding. With "auto", once the meeting is clearly held in Chinese it is decoded as
        Chinese: auto-detection turns short sounds and Chinese words into English ("呃" -> "Oh.", "還是" ->
        "還 think"), while Chinese mode still writes English terms (AI, meet, OK) correctly."""
        if self.language == "auto" and self._zh_meeting():
            return "zh"
        return self.language

    def _fix_language(self, audio: np.ndarray, raw_text: str) -> str:
        """Auto language: count which language the sentences are in; a short English-only sentence in a
        Chinese meeting is decoded again as Chinese (the first sentences are decoded before the vote is clear)."""
        if self.language != "auto":
            return raw_text
        cjk = len(re.findall(_CJK, raw_text))
        latin = len(re.findall(r"[A-Za-z]+", raw_text))
        if cjk >= 6:
            self.lang_votes["zh"] += 1
        elif latin >= 4 and not cjk:
            self.lang_votes["en"] += 1  # real English speech still comes out as English in Chinese mode
        if cjk or not latin or len(audio) > SHORT_FOREIGN_SEC * SAMPLE_RATE or not self._zh_meeting():
            return raw_text
        return self._transcribe_now(audio, "zh") or raw_text

    def _finalize(self, seg_id: int, start: float, audio: np.ndarray, raw_text: str, speaker: str | None = None):
        text = tidy(self._fix_language(audio, raw_text), self.traditional, final=True)
        if re.fullmatch(r"[呃額]+[。，？！]*", text):  # a bare hesitation sound is not something anyone said
            text = ""
        end = start + len(audio) / SAMPLE_RATE
        if not text:
            self.emit({"type": "discard", "seg": seg_id})
            return
        score = 1.0
        if speaker is None and self.diar is not None:
            speaker, score = self.diar.label(start, end)  # refined later by the background pass
        speaker = self._hinted_speaker(start, end) or speaker
        if speaker is None:  # no diarization yet (first seconds) or disabled: voiceprint of this sentence alone
            voiced = _trim_silence(audio)
            emb = self.encoder.embed(voiced if len(voiced) >= SAMPLE_RATE * 0.4 else audio)
        with self.lock:
            if speaker is None:
                sid, score = self.registry.assign(emb, len(voiced) / SAMPLE_RATE, self.last_speaker)
                self.seg_emb[seg_id] = (emb, len(voiced) / SAMPLE_RATE)
            else:
                sid = speaker
                self.registry.speakers[sid]["active"] = True
            self.last_speaker = sid
            seg = {"seg": seg_id, "start": round(start, 2), "end": round(end, 2),
                   "speaker": sid, "score": round(score, 3), "text": text,
                   "guessed": score == 0.0}  # too short to judge: went to the previous speaker
            self.segments.append(seg)
            speakers = self.registry.public()
        self.emit({"type": "final", **seg})
        self.emit({"type": "speakers", "speakers": speakers})
        self.save()

    # ---------------------------------------------------------------- background correction
    def public_segments(self, locked: bool = False) -> list[dict]:
        def grab():
            return [{k: s[k] for k in PUBLIC_KEYS if k in s} for s in self.segments]
        if locked:
            return grab()
        with self.lock:
            return grab()

    def _emit(self, message: str):
        self.emit({"type": "status", "message": message})

    def _refine_loop(self):
        while not self.refine_stop.wait(REFINE_EVERY):
            if not self.refine:
                continue
            try:
                self._refine(full=False)
            except Exception as e:  # corrections are best-effort; never break the live transcript
                print("[refine] error:", type(e).__name__, e, flush=True)

    def _refine(self, full: bool):
        now = self.log.seconds
        with self.lock:
            cands = [s["seg"] for s in self.segments
                     if full or (now - REFINE_LOOKBACK <= s["end"] <= now - REFINE_SETTLE)]
        fixed = 0
        if self.diar is not None:
            if full:
                self.diar.update(self.log.slice, self.log.seconds)
            if full or now - self.last_recluster >= RECLUSTER_EVERY:
                self.last_recluster = now
                if self.diar.recluster():  # who-is-who re-decided from the whole meeting: re-check everything
                    with self.lock:
                        cands = [s["seg"] for s in self.segments if s["end"] <= now - REFINE_SETTLE or full]
            for seg_id in cands:
                fixed += self._apply_diarization(seg_id, final=full)
            fixed += self._apply_hints(now)
        else:
            for seg_id in cands:
                fixed += self._relabel(seg_id)
        if fixed:
            with self.lock:
                self.registry.set_active({s["speaker"] for s in self.segments})
                speakers = self.registry.public()
            self.emit({"type": "segments", "segments": self.public_segments()})
            self.emit({"type": "speakers", "speakers": speakers})
            self.emit({"type": "toast", "message": f"已自動校正 {fixed} 處說話者"})
            self.save()

    def _seg(self, seg_id: int) -> dict | None:
        return next((s for s in self.segments if s["seg"] == seg_id), None)

    def _relabel(self, seg_id: int) -> int:
        """With more voiceprint history now, does this sentence clearly belong to someone else?"""
        with self.lock:
            s = self._seg(seg_id)
            if not s or s.get("locked") or s.get("relabels", 0) >= REFINE_MAX_RELABELS or seg_id not in self.seg_emb:
                return 0
            emb, dur = self.seg_emb[seg_id]
            if dur < REFINE_MIN_PIECE:
                return 0
            cur = s["speaker"]
            best, best_sim = self.registry.best(emb, exclude=cur)
            cur_sim = self.registry.similarity(emb, cur)
            floor, margin = (SHORT_FLOOR, GUESSED_MARGIN) if s.get("guessed") else (self.registry.threshold, REFINE_MARGIN)
            if best is None or best_sim < floor or best_sim - cur_sim < margin:
                return 0
            s["guessed"] = False
            self.registry.move(emb, dur, cur, best)
            s["speaker"], s["corrected"] = best, True
            s["relabels"] = s.get("relabels", 0) + 1
            if self.last_speaker == cur and self.segments and self.segments[-1] is s:
                self.last_speaker = best
            return 1

    # ---------------------------------------------------------------- meeting-app / microphone hints
    def _hint_name(self, t0: float, t1: float) -> str | None:
        """The name the hints give to [t0, t1), if one clearly dominates it."""
        cov = self.hints.coverage(t0, t1)
        if not cov:
            return None
        name, secs = max(cov.items(), key=lambda x: x[1])
        rest = sum(cov.values()) - secs
        if secs >= 0.6 * max(t1 - t0, 0.1) and rest <= 0.3 * secs:
            return name
        return None

    def _hinted_speaker(self, t0: float, t1: float) -> str | None:
        """The meeting app (or the local mic) clearly says who spoke in [t0, t1): that person's speaker id.
        Works even when the voice-based diarization has no answer (e.g. heavily processed meeting audio)."""
        name = self._hint_name(t0, t1)
        if not name:
            return None
        sids = self.name_to_sids.get(name)
        if sids and len(sids) > 1:
            return None  # one name, several voices (a shared room microphone): let the voices decide
        return self._sid_for_name(name)

    def _display_name(self, name: str) -> str:
        return (self.platform_self or self.me_name) if name == ME else name

    def _sid_for_name(self, name: str) -> str:
        """Speaker id that represents a meeting-app name, created on first use."""
        sid = self.named.get(name)
        if sid in self.registry.speakers:
            return sid
        mapped = self.name_to_sids.get(name)
        if mapped and mapped[0] in self.registry.speakers:
            sid = mapped[0]
        else:
            sid = self.registry._new_id()
            self.registry.speakers[sid] = {"id": sid, "name": self._display_name(name),
                                           "sum": np.zeros(192, np.float32), "count": 0,
                                           "enrolled": False, "active": True, "named": True}
        self.named[name] = sid
        spk = self.registry.speakers[sid]
        if not spk.get("user_named"):
            spk["name"] = self._display_name(name)
        return sid

    def _apply_hints(self, now: float) -> int:
        """Use the meeting app's / microphone's "who is speaking" to name voices and correct sentences."""
        self.hints.flush(now)
        if not self.hints.tracks or now < 2:
            return 0
        step = 5  # diarization frames are 0.02 s, hints 0.1 s
        labels = self.diar.frame_labels(0, now)[::step]
        hint = self.hints.frames(0, len(labels) * 0.1)
        M: dict[str, dict[str, float]] = {}
        for name, track in hint.items():
            n = min(len(track), len(labels))
            for i in np.flatnonzero(track[:n]):
                sid = labels[i]
                if sid:
                    M.setdefault(sid, {}).setdefault(name, 0.0)
                    M[sid][name] += 0.1
        name_to: dict[str, list[tuple[str, float]]] = {}
        for sid, row in M.items():
            name, secs = max(row.items(), key=lambda x: x[1])
            if secs >= HINT_MIN_SECONDS and secs >= HINT_SHARE * sum(row.values()):
                name_to.setdefault(name, []).append((sid, secs))
        fixed = 0
        with self.lock:
            # a remote participant has their own microphone: several voices with one name = one person split in two
            for name, lst in list(name_to.items()):
                if name == ME or len(lst) < 2:
                    continue
                lst.sort(key=lambda x: -x[1])
                keep = lst[0][0]
                for sid, _ in lst[1:]:
                    together = self.diar.cooc.get(frozenset((keep, sid)), 0)
                    if together > 0.3 * min(self.diar.present.get(keep, 1), self.diar.present.get(sid, 1)):
                        continue  # heard together a lot: really two people sharing a microphone (meeting room)
                    self.registry.merge(sid, keep)
                    self.diar.merge(sid, keep)
                    for s in self.segments:
                        if s["speaker"] == sid and not s.get("locked"):
                            s["speaker"], s["corrected"] = keep, True
                            fixed += 1
                name_to[name] = [x for x in lst if x[0] in self.registry.speakers]
            # names (a name typed by the user always wins)
            self.name_to_sids = {n: [sid for sid, _ in sorted(lst, key=lambda x: -x[1])] for n, lst in name_to.items()}
            # a voice the diarization found that belongs to a name already in use: same person, join them
            for name, sids in self.name_to_sids.items():
                have = self.named.get(name)
                if len(sids) == 1 and have and have != sids[0] and have in self.registry.speakers:
                    voice = sids[0]
                    if self.registry.speakers[have]["count"] == 0:  # name-only speaker: keep id, take the voiceprint
                        self.registry.merge(voice, have)
                        self.diar.merge(voice, have)
                    else:
                        self.registry.merge(voice, have)
                        self.diar.merge(voice, have)
                    for s in self.segments:
                        if s["speaker"] == voice and not s.get("locked"):
                            s["speaker"], s["corrected"] = have, True
                            fixed += 1
                    self.name_to_sids[name] = [have]
            for name, sids in self.name_to_sids.items():
                if len(sids) == 1:
                    self.named.setdefault(name, sids[0])
            for name, sids in self.name_to_sids.items():
                base = self._display_name(name)
                for i, sid in enumerate(sids):
                    spk = self.registry.speakers.get(sid)
                    want = base if len(sids) == 1 else f"{base} ({i + 1})"
                    if spk and not spk.get("user_named") and spk["name"] != want:
                        spk["name"] = want
                        fixed += 1
            # sentences the meeting app clearly attributes to someone else
            for s in self.segments:
                if s.get("locked"):
                    continue
                target = self._hinted_speaker(s["start"], s["end"])
                if target and target != s["speaker"]:
                    s["speaker"], s["corrected"], s["guessed"] = target, True, False
                    fixed += 1
        return fixed

    def _apply_diarization(self, seg_id: int, final: bool = False) -> int:
        """Compare a finished sentence with the settled who-spoke-when: relabel it, or split it at speaker changes."""
        with self.lock:
            s = self._seg(seg_id)
            if not s or s.get("locked") or s.get("edited"):
                return 0
            start, end, owner = s["start"], s["end"], s["speaker"]
        runs = self.diar.turns(start, end, min_run=TURN_MIN_RUN)
        if not runs:
            if owner in self.diar.retired:  # its speaker was absorbed by someone else: never leave a ghost
                lab, _ = self.diar.label(start - 3.0, end + 3.0)
                with self.lock:
                    s = self._seg(seg_id)
                    if s is not None:
                        i = self.segments.index(s)
                        lab = lab or next((x["speaker"] for x in reversed(self.segments[:i])
                                           if x["speaker"] not in self.diar.retired), None)
                        if lab and lab != owner:
                            s["speaker"], s["corrected"] = lab, True
                            return 1
            return 0
        if len(runs) == 1:
            lab = runs[0][2]
            _, share = self.diar.label(start, end)
            with self.lock:
                s = self._seg(seg_id)
                if not s:
                    return 0
                s["guessed"] = share < 0.6
                retired = owner in self.diar.retired  # its speaker no longer exists: must move, whatever the count
                # during the meeting a sentence may change at most REFINE_MAX_RELABELS times (no flicker);
                # the end-of-meeting pass always applies the final answer
                if lab == owner or (s.get("relabels", 0) >= REFINE_MAX_RELABELS and not retired and not final):
                    return 0
                s["speaker"], s["corrected"] = lab, True
                s["relabels"] = s.get("relabels", 0) + 1
                return 1
        # several speakers inside one sentence: split it and re-transcribe each part
        a = int(start * SAMPLE_RATE)
        audio = self.log.slice(a, int(end * SAMPLE_RATE))
        bounds = [0] + [_snap_cut(audio, int((r[0] - start) * SAMPLE_RATE)) for r in runs[1:]] + [len(audio)]
        groups = [(r[2], x, y) for r, x, y in zip(runs, bounds, bounds[1:])]
        slack = int(0.15 * SAMPLE_RATE)
        texts = [tidy(self._transcribe_now(audio[x:min(len(audio), y + slack)]), self.traditional, final=True)
                 for _, x, y in groups]
        with self.lock:
            s = self._seg(seg_id)
            if not s or s.get("edited") or s.get("locked"):
                return 0
            i = self.segments.index(s)
            new = []
            for k, ((lab, x, y), text) in enumerate(zip(groups, texts)):
                if not text:
                    continue
                seg = {"seg": seg_id if not new else self._new_id(), "start": round(start + x / SAMPLE_RATE, 2),
                       "end": round(start + y / SAMPLE_RATE, 2), "speaker": lab, "score": 1.0, "text": text}
                if lab != owner:
                    seg["corrected"] = True
                new.append(seg)
            if len(new) < 2:
                if new and owner in self.diar.retired:
                    s["speaker"], s["corrected"] = new[0]["speaker"], True
                    return 1
                return 0
            self.segments[i:i + 1] = new
        return len(new) - 1

    # ---------------------------------------------------------------- control
    def control(self, msg: dict):
        kind = msg.get("type")
        with self.lock:
            if kind == "rename":
                self.registry.rename(msg["speaker"], msg["name"])
                if msg["speaker"] in self.registry.speakers:
                    self.registry.speakers[msg["speaker"]]["user_named"] = True
            elif kind == "merge":
                self.registry.merge(msg["from"], msg["into"])
                if self.diar is not None:
                    self.diar.merge(msg["from"], msg["into"])
                for s in self.segments:
                    if s["speaker"] == msg["from"]:
                        s["speaker"] = msg["into"]
                if self.last_speaker == msg["from"]:
                    self.last_speaker = msg["into"]
            elif kind == "reassign":
                for s in self.segments:
                    if s["seg"] == msg["seg"]:
                        s["speaker"] = msg["speaker"]
                        s["locked"] = True
                        s["guessed"] = s["corrected"] = False
            elif kind == "edit":
                for s in self.segments:
                    if s["seg"] == msg["seg"]:
                        s["text"] = str(msg["text"])[:5000]
                        s["edited"] = True
            elif kind == "threshold":
                self.registry.threshold = float(msg["value"])
            elif kind == "enroll":
                ok = self.registry.enroll(msg["speaker"])
                self.emit({"type": "toast", "message": "已儲存聲紋，下次會議會自動認出" if ok else "聲紋資料不足，請讓此人多說幾句"})
            elif kind == "title":
                self.title = str(msg.get("title") or "會議紀錄")[:100]
            speakers = self.registry.public()
        self.emit({"type": "speakers", "speakers": speakers})
        if kind in ("merge", "reassign", "edit"):
            self.emit({"type": "segments", "segments": self.public_segments()})
        self.save()

    # ---------------------------------------------------------------- persistence
    def files(self) -> dict:
        return {"md": self.base_path + ".md", "json": self.base_path + ".json", "wav": self.base_path + ".wav"}

    def save(self):
        with self.lock:
            names = {s["id"]: s["name"] for s in self.registry.speakers.values()}
            segs = [dict(s, name=names.get(s["speaker"], s["speaker"])) for s in self.public_segments(locked=True)]
            title = self.title
        if not segs:
            return
        lines = [f"# {title}", "", f"- 日期：{self.started_at:%Y-%m-%d %H:%M}",
                 f"- 與會者：{'、'.join(dict.fromkeys(s['name'] for s in segs))}", "", "---", ""]
        for start, name, text in paragraphs(segs, lambda s: s["name"]):
            lines.append(f"**[{fmt_ts(start)}] {name}**：{text}  ")
        with open(self.base_path + ".md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(self.base_path + ".json", "w", encoding="utf-8") as f:
            json.dump({"title": title, "started_at": self.started_at.isoformat(), "segments": segs},
                      f, ensure_ascii=False, indent=1)


PARA_GAP = 2.0  # sentences of one speaker closer than this are written as one paragraph
PARA_MAX = 60.0  # ...up to this long, so timestamps stay useful


def paragraphs(segs: list[dict], name_of: Callable[[dict], str]) -> list[tuple[float, str, str]]:
    """(start, speaker name, text) per paragraph: the recognizer cuts at every short pause, which reads as a
    stutter of one-line fragments, so a speaker's consecutive sentences are joined for the written record."""
    out: list[list] = []
    last_end = -1e9
    for s in segs:
        name = name_of(s)
        if out and out[-1][1] == name and s["start"] - last_end < PARA_GAP and s["end"] - out[-1][0] < PARA_MAX:
            prev = out[-1][2]
            sep = "" if re.search(rf"[{_CJK[1:-1]}。，？！；：]$", prev) and re.match(_CJK, s["text"]) else " "
            out[-1][2] = prev + sep + s["text"]
        else:
            out.append([s["start"], name, s["text"]])
        last_end = s["end"]
    return [tuple(p) for p in out]


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _snap_cut(audio: np.ndarray, cut: int, reach: float = 0.3, frame: int = 320) -> int:
    """Move a cut to the quietest 20 ms within +-reach seconds, i.e. between words rather than inside one."""
    a, b = max(0, cut - int(reach * SAMPLE_RATE)), min(len(audio), cut + int(reach * SAMPLE_RATE))
    n = (b - a) // frame
    if n < 2:
        return cut
    e = (audio[a:a + n * frame].reshape(n, frame) ** 2).mean(1)
    return a + int(np.argmin(e)) * frame + frame // 2


def _trim_silence(audio: np.ndarray, frame: int = 320, min_pause: float = 0.25) -> np.ndarray:
    """Remove pauses (quiet stretches longer than `min_pause`) so voiceprints come from actual speech.

    Short dips inside words are kept: dropping every quiet frame would throw away soft syllables and
    leave too little of a short reply like "好的" to recognise the voice.
    """
    n = len(audio) // frame
    if n < 3:
        return audio
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    quiet = rms <= max(rms.max() * 0.02, 1e-4)  # about -34 dB below the loudest frame
    keep = ~quiet
    run_min = max(1, int(min_pause * SAMPLE_RATE / frame))
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i < run_min and i > 0 and j < n:  # brief dip between voiced frames: keep it
                keep[i:j] = True
            i = j
        else:
            i += 1
    return frames[keep].reshape(-1) if keep.any() else audio
