"""Collect per-window local voiceprints like the live diarizer does, cluster them with NME-SC, compare to reference."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np, soundfile as sf, paths
from scipy.optimize import linear_sum_assignment
from diarize import SpeakerEncoder, SpeakerRegistry
from turns import ChangeDetector, model_path, FRAME_SHIFT, RECEPTIVE
from streamdiar import StreamingDiarizer
from cluster import nme_sc, merge_with_constraints
enc = SpeakerEncoder("cuda"); det = ChangeDetector(model_path(paths.MODEL_DIR))
import sherpa_onnx as so
ALT = os.environ.get("ALT")  # re-embed the clustering items with another model
if ALT:
    ex = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=ALT, num_threads=4))
    def alt_embed(x):
        st = ex.create_stream(); st.accept_waveform(16000, x); st.input_finished()
        v = np.array(ex.compute(st)); return v / np.linalg.norm(v)
for clip, refp in [("tests/podcast_5min.wav", "tests/podcast_5min.wav.labels.npy"), ("tests/kk_5min.wav", "tests/kk_5min.wav.labels2.npy"),
                   ("tests/meeting_fast.wav", None), ("tests/meeting_zh.wav", None), ("tests/meeting_early.wav", None)]:
    audio, _ = sf.read(clip, dtype="float32")
    reg = SpeakerRegistry(0.5); reg.speakers = {}
    d = StreamingDiarizer(det, enc.embed, reg)
    d.update(lambda a, b: audio[a:b], len(audio) / 16000)
    items = []  # (emb, window, local k, dur, window index)
    for wi, w in enumerate(d.windows):
        if wi % 4: continue  # 2 s apart: the cannot-link evidence needs many windows
        for k, dur, e in w.locals:
            if e is not None and dur >= 1.0:
                if ALT:
                    fr = np.flatnonzero(w.act[:, k] & (w.act.sum(1) == 1)); a0 = int(w.start * 16000)
                    e = alt_embed(np.concatenate([audio[a0 + i * FRAME_SHIFT:a0 + (i + 1) * FRAME_SHIFT] for i in fr]))
                items.append((e, w, k, dur, wi))
    E = np.array([x[0] for x in items])
    t = time.time(); lab0 = nme_sc(E, max_speakers=12)
    lab = merge_with_constraints(E, np.array([x[3] for x in items]) * 0.25, lab0, np.array([x[4] for x in items]))
    ms = (time.time() - t) * 1000
    print(f"   NME-SC first: {lab0.max()+1} clusters -> after constrained merge: {lab.max()+1}")
    print(f"{os.path.basename(clip):22s} pieces={len(E):4d} clusters={lab.max()+1} ({ms:.0f} ms) sizes={np.bincount(lab).tolist()}")
    if refp:
        ref = np.load(refp)[0]
        truth = []
        for (e, w, k, _, _) in items:
            fr = np.flatnonzero(w.act[:, k] & (w.act.sum(1) == 1))
            ts = ((w.start + (fr * FRAME_SHIFT + RECEPTIVE // 2) / 16000) * 10).astype(int)
            r = ref[np.clip(ts, 0, len(ref) - 1)]; r = r[r >= 0]
            truth.append(np.bincount(r).argmax() if len(r) else -1)
        truth = np.array(truth); m = truth >= 0
        C = np.array([[np.sum((lab[m] == i) & (truth[m] == j)) for j in range(truth.max() + 1)] for i in range(lab.max() + 1)])
        ri, ci = linear_sum_assignment(-C)
        # pair statistics on the NME-SC clusters: who is really who, similarity, co-occurrence
        W = np.array([x[4] for x in items]); D = np.array([x[3] for x in items])
        cl = sorted(set(lab0)); maj = {c: np.bincount(truth[(lab0 == c) & m], minlength=2).argmax() if np.any((lab0 == c) & m) else -1 for c in cl}
        cent = {c: (E[lab0 == c].T @ D[lab0 == c]) / np.linalg.norm(E[lab0 == c].T @ D[lab0 == c]) for c in cl}
        pres = {c: len(set(W[lab0 == c])) for c in cl}
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                tog = sum(1 for w in set(W) if a in lab0[W == w] and b in lab0[W == w])
                same = "SAME person" if maj[a] == maj[b] else "different  "
                print(f"      c{a}(ref{maj[a]},{pres[a]}w) c{b}(ref{maj[b]},{pres[b]}w) {same} sim={cent[a] @ cent[b]:.2f} together={tog} ratio={tog / min(pres[a], pres[b]):.2f}")
        print(f"   pieces in the right cluster: {C[ri, ci].sum() / m.sum() * 100:.1f}%   (confusion rows=clusters cols=ref) {C.tolist()}")
