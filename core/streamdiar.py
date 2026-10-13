"""Streaming speaker diarization (who speaks when), in the spirit of diart (Coria et al., 2021).

Every STEP seconds the newest WINDOW seconds are analysed:
  1. pyannote segmentation labels each ~17 ms frame with up to 3 *local* voices (overlap-aware). Inside one
     window it compares voices side by side, which is far more reliable for short replies ("嗯", "對")
     than a voiceprint of the reply alone.
  2. Each local voice gets one voiceprint from ALL of its clean speech in the window (usually seconds).
  3. Local voices are matched to the meeting's speakers with the constraint that two voices the segmentation
     separated in the same window can't be the same person (Hungarian assignment). A voice that can't be
     matched becomes a new speaker.
  4. Every moment is covered by WINDOW/STEP windows; its speaker is the majority vote.
Windows are kept, so when the set of speakers changes (a new one appears, two get merged) recent windows
can be re-linked and the timeline corrected retroactively.
"""
from __future__ import annotations

import itertools
import threading

import numpy as np
from scipy.optimize import linear_sum_assignment

from cluster import merge_with_constraints, nme_sc
from turns import FRAME_SHIFT, RECEPTIVE, ChangeDetector

SR = 16000
RES = 0.02  # timeline resolution (s)


class _Window:
    __slots__ = ("start", "act", "locals", "assign", "a", "b", "fi", "cl", "masks")

    def __init__(self, start, act, locals_, a, b, fi):
        self.start, self.act, self.locals, self.a, self.b, self.fi = start, act, locals_, a, b, fi
        self.assign: dict[int, str] = {}


class StreamingDiarizer:
    WINDOW = 8.0
    STEP = 0.5
    MIN_SHORT = 0.15  # local voices with less clean speech get no voiceprint (matched by elimination only)
    MIN_LOCAL = 0.5  # clean speech for a voiceprint that is trusted on its own
    NEW_SPEAKER_MIN = 1.0  # clean speech needed to start a new speaker
    MATCH = 0.40  # similarity to link a local voice to a known speaker (while < 2 people are known)
    MATCH_LATER = 0.50  # ...once several people are known; in-between voices stay unassigned (other windows vote)
    LEARN_SIM = 0.55  # only confident matches update a speaker's voiceprint (no contamination)
    MATCH_SHORT = 0.25  # ...for a local voice with little audio
    NEW_BELOW = 0.55  # an unmatched voice at least this unlike every known speaker becomes a new one
    # once two or more people are known, a newcomer needs clearer evidence (laughs and changes of tone
    # otherwise spawn phantom speakers)
    NEW_SPEAKER_MIN_LATER = 1.5
    NEW_BELOW_LATER = 0.50
    PIECE_GAP = 0.5  # a local voice interrupted this long is checked piece by piece...
    SPLIT_SIM = 0.30  # ...and split if two of its pieces sound this different (segmentation merged two people)
    MERGE_SIM = 0.60  # two speakers this similar...
    MERGE_COOC = 0.10  # ...heard as two voices together in at most this share of their windows: one person

    def __init__(self, detector: ChangeDetector, embed, registry, cluster_embed=None):
        """embed: fast voiceprint for live matching. cluster_embed: optional stronger voiceprint, used only for
        the periodic whole-meeting re-clustering (computed for every RECLUSTER_STRIDE-th window)."""
        self.det, self.embed, self.reg = detector, embed, registry
        self.cluster_embed = cluster_embed
        self.n_windows = 0
        self.votes: dict[str, np.ndarray] = {}  # speaker -> votes per RES frame
        self.speech = np.zeros(0, dtype=np.float32)  # windows that heard someone speaking at each frame
        self.seen = np.zeros(0, dtype=np.float32)  # windows that covered each frame
        self.windows: list[_Window] = []
        self.cooc: dict[frozenset, int] = {}  # pairs of speakers heard as different voices in one window
        self.present: dict[str, int] = {}  # windows in which each speaker was clearly heard
        self.next_t = 0.0
        self.retired: set[str] = set()  # speakers absorbed by others at the last re-clustering
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ timeline queries
    def _grow(self, n: int):
        if len(self.seen) < n:
            extra = n - len(self.seen) + int(60 / RES)
            self.seen = np.concatenate([self.seen, np.zeros(extra, np.float32)])
            self.speech = np.concatenate([self.speech, np.zeros(extra, np.float32)])
            for k in self.votes:
                self.votes[k] = np.concatenate([self.votes[k], np.zeros(extra, np.float32)])

    def _vote(self, sid: str) -> np.ndarray:
        if sid not in self.votes:
            self.votes[sid] = np.zeros(len(self.seen), np.float32)
        return self.votes[sid]

    def label(self, t0: float, t1: float) -> tuple[str | None, float]:
        """Majority speaker over [t0, t1) and the share of votes it got."""
        a, b = int(t0 / RES), max(int(t0 / RES) + 1, int(t1 / RES))
        with self.lock:
            tot = {k: float(v[a:b].sum()) for k, v in self.votes.items()}
        if not tot or max(tot.values()) <= 0:
            return None, 0.0
        best = max(tot, key=tot.get)
        return best, tot[best] / (sum(tot.values()) + 1e-9)

    def frame_labels(self, t0: float, t1: float) -> list[str | None]:
        """Per-RES-frame majority speaker (None = nobody / undecided)."""
        a, b = int(t0 / RES), int(t1 / RES)
        with self.lock:
            keys = [k for k in self.votes if self.votes[k].any()]
            if not keys or b <= a:
                return [None] * max(0, b - a)
            m = np.stack([self.votes[k][a:b] for k in keys])
            sp = self.speech[a:b]
            seen = self.seen[a:b]
        if m.shape[1] < b - a:  # beyond what has been processed
            pad = b - a - m.shape[1]
            m = np.pad(m, ((0, 0), (0, pad)))
            sp, seen = np.pad(sp, (0, pad)), np.pad(seen, (0, pad))
        best = m.argmax(0)
        ok = (seen > 0) & (sp >= 0.5 * seen) & (m.max(0) > 0)
        return [keys[i] if o else None for i, o in zip(best, ok)]

    def turns(self, t0: float, t1: float, min_run: float = 0.3) -> list[tuple[float, float, str]]:
        """Speaker runs inside [t0, t1); runs shorter than min_run and gaps are absorbed by neighbours."""
        labs = self.frame_labels(t0, t1)
        runs = []
        for i, lab in enumerate(labs):
            if lab is None:
                continue
            if runs and runs[-1][2] == lab:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1, lab])
        while len(runs) > 1:
            j = min(range(len(runs)), key=lambda x: runs[x][1] - runs[x][0])
            if (runs[j][1] - runs[j][0]) * RES >= min_run:
                break
            if j == 0:
                nb = 1
            elif j == len(runs) - 1:
                nb = j - 1
            else:
                nb = j - 1 if runs[j - 1][1] - runs[j - 1][0] >= runs[j + 1][1] - runs[j + 1][0] else j + 1
            runs[nb][0], runs[nb][1] = min(runs[nb][0], runs[j][0]), max(runs[nb][1], runs[j][1])
            del runs[j]
            merged = []
            for r in runs:
                if merged and merged[-1][2] == r[2]:
                    merged[-1][1] = r[1]
                else:
                    merged.append(r)
            runs = merged
        # a speaker's run reaches until the next speaker starts (pauses belong to the one who was talking)
        out = []
        for i, (a, b, lab) in enumerate(runs):
            start = 0 if i == 0 else a
            end = len(labs) if i == len(runs) - 1 else runs[i + 1][0]
            out.append((t0 + start * RES, t0 + end * RES, lab))
        return out

    # ------------------------------------------------------------------ processing
    def update(self, log_slice, now: float):
        """Process every STEP up to `now` (seconds of audio available). log_slice(a, b) -> float32 samples."""
        while self.next_t + self.STEP <= now:
            self.next_t += self.STEP
            end = self.next_t
            start = max(0.0, end - self.WINDOW)
            if end - start < 2.0:
                continue
            self._process(log_slice(int(start * SR), int(end * SR)), start)

    def _process(self, audio: np.ndarray, start: float):
        act = self.det.activity(audio)  # (frames, 3)
        nf = len(act)
        single = act.sum(1) == 1
        sampled = self.n_windows % self.RECLUSTER_STRIDE == 0
        self.n_windows += 1

        def clean_audio(mask):
            idx = np.flatnonzero(mask & single)
            return np.concatenate([audio[i * FRAME_SHIFT:(i + 1) * FRAME_SHIFT] for i in idx]) if len(idx) else None

        def dur_of(mask):
            return (mask & single).sum() * FRAME_SHIFT / SR

        masks: list[np.ndarray] = []
        for k in range(act.shape[1]):
            if act[:, k].any():
                masks.extend(self._split_local(act[:, k], single, audio))
        locals_ = []
        self_cluster = {}
        for key, mask in enumerate(masks):
            dur = dur_of(mask)
            emb = None
            if dur >= self.MIN_SHORT:
                x = clean_audio(mask)
                emb = self.embed(x)
                if sampled and dur >= 1.0:
                    self_cluster[key] = self.cluster_embed(x) if self.cluster_embed else emb
            locals_.append((key, dur, emb))
        a = int(start / RES)
        b = int((start + len(audio) / SR) / RES)
        fi = np.clip(((np.arange(a, b) * RES - start) * SR - RECEPTIVE // 2) / FRAME_SHIFT, 0, nf - 1).astype(int)
        w = _Window(start, act, locals_, a, b, fi)
        w.cl = self_cluster
        w.masks = masks
        durs = {k: d for k, d, _ in locals_}
        with self._reg_lock(), self.lock:  # lock order everywhere: registry/session first, then timeline
            w.assign = self._link(locals_, learn=True)
            self.windows.append(w)
            self._grow(b + 1)
            self.seen[a:b] += 1
            self.speech[a:b] += act.any(1)[fi]
            self._apply(w, +1)
            heard = sorted({sid for k, sid in w.assign.items() if durs[k] >= self.MIN_LOCAL})
            for sid in heard:
                self.present[sid] = self.present.get(sid, 0) + 1
            for x, y in itertools.combinations(heard, 2):
                key = frozenset((x, y))
                self.cooc[key] = self.cooc.get(key, 0) + 1

    def _apply(self, w: _Window, sign: int):
        # the segmentation is least reliable at the edges of its window (no context on one side): weight votes
        # with a Hamming window so every moment is decided mostly by windows where it sits in the middle
        n = w.b - w.a
        weight = 0.54 - 0.46 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)
        for k, sid in w.assign.items():
            self._vote(sid)[w.a:w.b] += sign * w.masks[k][w.fi] * weight

    def _split_local(self, track: np.ndarray, single: np.ndarray, audio: np.ndarray) -> list[np.ndarray]:
        """The segmentation sometimes gives two people separated by a pause the same local voice. Check the
        pieces of a local voice (split at pauses) and separate them if two of them sound clearly different."""
        on = np.flatnonzero(track)
        gap = int(self.PIECE_GAP * SR / FRAME_SHIFT)
        cuts = np.flatnonzero(np.diff(on) > gap)
        if not len(cuts):
            return [track]
        ranges = [(on[s], on[e] + 1) for s, e in zip(np.r_[0, cuts + 1], np.r_[cuts, len(on) - 1])]
        pieces = []
        for f0, f1 in ranges:
            m = np.zeros_like(track)
            m[f0:f1] = track[f0:f1]
            d = (m & single).sum() * FRAME_SHIFT / SR
            e = None
            if d >= 1.0:
                idx = np.flatnonzero(m & single)
                e = self.embed(np.concatenate([audio[i * FRAME_SHIFT:(i + 1) * FRAME_SHIFT] for i in idx]))
            pieces.append((f0, m, e))
        longs = [p for p in pieces if p[2] is not None]
        if len(longs) < 2:
            return [track]
        sims = [(float(a[2] @ b[2]), i, j) for i, a in enumerate(longs) for j, b in enumerate(longs) if i < j]
        worst, i, j = min(sims)
        if worst >= self.SPLIT_SIM:
            return [track]
        seeds = (longs[i][2], longs[j][2])
        groups = [np.zeros_like(track), np.zeros_like(track)]
        last = 0
        for f0, m, e in pieces:
            g = last if e is None else int(e @ seeds[1] > e @ seeds[0])  # short pieces follow the previous one
            groups[g] |= m
            last = g
        return [g for g in groups if g.any()]

    def _reg_lock(self):
        return getattr(self.reg, "lock", None) or _NullLock()

    def _link(self, locals_, learn: bool) -> dict[int, str]:
        """Match one window's local voices to meeting speakers (all distinct), creating new ones when needed."""
        reg = self.reg
        known = [sid for sid, s in reg.speakers.items() if s["count"] > 0]
        longs = sorted([(k, d, e) for k, d, e in locals_ if e is not None and d >= self.MIN_LOCAL], key=lambda x: -x[1])
        shorts = [(k, d, e) for k, d, e in locals_ if e is None or d < self.MIN_LOCAL]
        assign: dict[int, str] = {}
        match = self.MATCH if len(known) < 2 else self.MATCH_LATER
        created: set[str] = set()
        if known and longs:
            sim = np.array([[reg.similarity(e, sid) for sid in known] for _, _, e in longs])
            rows, cols = linear_sum_assignment(-sim)
            for r, c in zip(rows, cols):
                if sim[r, c] >= match:
                    assign[longs[r][0]] = known[c]
        for k, d, e in longs:
            if k in assign:
                continue
            free = [sid for sid in known if sid not in assign.values()]
            best_free = max(free, key=lambda x: reg.similarity(e, x), default=None)
            if best_free is not None and reg.similarity(e, best_free) >= match:
                assign[k] = best_free
            elif learn and self._new_ok(d, e, known, free):
                # nobody matches -- or the segmentation hears another voice while every known speaker
                # is already taken in this window: a new person
                sid = reg.new_speaker(e)
                known.append(sid)
                created.add(sid)
                assign[k] = sid
        # short voices: the segmentation says they differ from the other voices here, so choose among the
        # speakers NOT present in this window (with two people that is simply the other one)
        for k, d, e in shorts:
            free = [sid for sid in known if sid not in assign.values()]
            if len(free) == 1 and assign:
                assign[k] = free[0]
            elif free and e is not None:
                sid = max(free, key=lambda x: reg.similarity(e, x))
                if reg.similarity(e, sid) >= self.MATCH_SHORT:
                    assign[k] = sid
        if learn:
            for k, d, e in longs:
                sid = assign.get(k)
                if sid and sid not in created and d >= 1.5 and reg.similarity(e, sid) >= self.LEARN_SIM:
                    reg.add(sid, e, d * self.STEP / self.WINDOW)  # each second is seen WINDOW/STEP times
        return assign

    def _new_ok(self, d: float, e: np.ndarray, known: list[str], free: list[str]) -> bool:
        best = max((self.reg.similarity(e, sid) for sid in known), default=-1.0)
        if len(known) < 2:  # telling the first two people apart: trust the segmentation's "different voice"
            return d >= self.NEW_SPEAKER_MIN and (not free or best < self.NEW_BELOW)
        return d >= self.NEW_SPEAKER_MIN_LATER and best < self.NEW_BELOW_LATER

    # ------------------------------------------------------------------ whole-meeting re-clustering
    RECLUSTER_STRIDE = 4  # use every 4th window (2 s apart) ...
    RECLUSTER_MAX_ITEMS = 800  # ...and at most this many voiceprints (keeps it ~1 s even for long meetings)

    def recluster(self) -> bool:
        """Re-decide who is who from ALL voiceprints so far (auto-tuned spectral clustering + merging),
        keep speaker ids (and the names users gave them) where clusters match, and rebuild the timeline."""
        with self.lock:
            wins = self.windows[::self.RECLUSTER_STRIDE]
        stride = self.RECLUSTER_STRIDE
        while len(wins) * 2 > self.RECLUSTER_MAX_ITEMS:
            stride *= 2
            wins = self.windows[::stride]
        items = [(w, k, d, e) for w in wins for k, d, e in w.locals if k in getattr(w, "cl", {})]
        if len(items) < 6:
            return False
        E = np.array([x[0].cl[x[1]] for x in items])  # clustering voiceprints
        live = np.array([x[3] for x in items])  # live voiceprints: the registry stays in this space
        durs = np.array([x[2] for x in items]) * (stride * self.STEP / self.WINDOW)  # undo window overlap
        wid = np.array([id(x[0]) for x in items])
        lab = merge_with_constraints(E, durs, nme_sc(E, max_speakers=12), wid)
        with self._reg_lock(), self.lock:
            reg = self.reg
            # keep ids: match new clusters to the speakers those voices were assigned to until now
            old = sorted({x[0].assign.get(x[1]) for x in items} - {None})
            k = int(lab.max()) + 1
            C = np.zeros((k, max(1, len(old))))
            for (w, kk, d, e), c in zip(items, lab):
                sid = w.assign.get(kk)
                if sid in old:
                    C[c, old.index(sid)] += d
            ids = {}
            if old:
                for c, j in zip(*linear_sum_assignment(-C)):
                    if C[c, j] > 0:
                        ids[c] = old[j]
            for c in range(k):
                if c not in ids:
                    v = live[lab == c].T @ durs[lab == c]
                    ids[c] = reg.new_speaker(v / (np.linalg.norm(v) + 1e-9))
            keep = set(ids.values())
            for sid, spk in reg.speakers.items():
                if sid in keep:
                    c = next(c for c, x in ids.items() if x == sid)
                    spk["sum"] = (live[lab == c].T @ durs[lab == c]).astype(np.float32)
                    spk["count"] = int(np.sum(lab == c))
                elif not spk.get("enrolled"):
                    spk["count"], spk["sum"] = 0, np.zeros_like(spk["sum"])  # retired: absorbed by others
            # rebuild the whole timeline with the new speakers
            for v in self.votes.values():
                v[:] = 0
            for w in self.windows:
                w.assign = self._link(w.locals, learn=False)
                self._apply(w, +1)
            self.retired = {sid for sid, spk in reg.speakers.items() if spk["count"] == 0 and not spk.get("enrolled")}
        return True

    # ------------------------------------------------------------------ corrections
    def relink(self, t0: float, t1: float) -> int:
        """Re-match the windows overlapping [t0, t1) with today's speakers; returns windows that changed."""
        changed = 0
        with self._reg_lock(), self.lock:
            for w in self.windows:
                if w.start + self.WINDOW < t0 or w.start >= t1:
                    continue
                new = self._link(w.locals, learn=False)
                if new != w.assign:
                    self._apply(w, -1)
                    w.assign = new
                    self._apply(w, +1)
                    changed += 1
        return changed

    def merge(self, src: str, dst: str):
        with self.lock:
            if src in self.votes:
                v = self.votes.pop(src)
                self.votes[dst] = self.votes.get(dst, np.zeros_like(v)) + v
            for w in self.windows:
                w.assign = {k: (dst if s == src else s) for k, s in w.assign.items()}
            self.present[dst] = self.present.get(dst, 0) + self.present.pop(src, 0)
            for key in list(self.cooc):
                if src in key:
                    n = self.cooc.pop(key)
                    other = next(iter(key - {src}), None)
                    if other and other != dst:
                        nk = frozenset((other, dst))
                        self.cooc[nk] = self.cooc.get(nk, 0) + n

    def merge_candidates(self) -> list[tuple[str, str]]:
        """(src, dst) pairs that sound alike and were (almost) never heard as two voices together.

        Two real people in a conversation share most windows; one person the segmentation split in two
        shares only the odd window where it confused itself.
        """
        reg = self.reg
        known = [sid for sid, s in reg.speakers.items() if s["count"] > 0 and not s.get("enrolled")]
        out = []
        for x, y in itertools.combinations(known, 2):
            together = self.cooc.get(frozenset((x, y)), 0)
            if together > self.MERGE_COOC * min(self.present.get(x, 0), self.present.get(y, 0)):
                continue
            sx, sy = reg.speakers[x]["sum"], reg.speakers[y]["sum"]
            sim = float(np.dot(sx, sy) / (np.linalg.norm(sx) * np.linalg.norm(sy) + 1e-9))
            if sim >= self.MERGE_SIM:
                out.append((y, x) if reg.speakers[x]["count"] >= reg.speakers[y]["count"] else (x, y))
        return out


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
