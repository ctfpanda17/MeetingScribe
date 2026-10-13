"""Offline 2-speaker diarization of a clip with several embedding models. Keep only models that really find two
speakers, report their pairwise agreement, and save the consensus (frames where they agree) as the reference."""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np, soundfile as sf, sherpa_onnx as so, paths
from turns import model_path
audio, _ = sf.read(sys.argv[1], dtype="float32")
step = 0.1; n = int(len(audio) / 16000 / step)
labels = {}
for emb in sorted(os.listdir("tests/spk")):
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=model_path(paths.MODEL_DIR))),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=f"tests/spk/{emb}"),
        clustering=so.FastClusteringConfig(num_clusters=2), min_duration_on=0.2, min_duration_off=0.3)
    res = so.OfflineSpeakerDiarization(cfg).process(audio).sort_by_start_time()
    lab = np.full(n, -1)
    for r in res:
        sl = slice(int(r.start / step), int(r.end / step))
        lab[sl] = np.where((lab[sl] >= 0) & (lab[sl] != r.speaker), -2, r.speaker)  # -2 = both talking
    share = [np.sum(lab == k) / 10 for k in (0, 1)]
    ok = min(share) > 0.1 * sum(share)
    print(f"{emb[:58]:58s} speaker seconds {share[0]:6.1f} / {share[1]:6.1f}  {'two speakers' if ok else 'COLLAPSED to one -> ignored'}")
    if ok: labels[emb] = lab
names = list(labels)
def align(la, lb):
    m = (la >= 0) & (lb >= 0)
    return (lb, np.mean(la[m] == lb[m])) if np.mean(la[m] == lb[m]) >= 0.5 else (np.where(lb >= 0, 1 - lb, lb), np.mean(la[m] == 1 - lb[m]))
print("\npairwise agreement (speech frames, best mapping):")
for a, b in itertools.combinations(names, 2):
    print(f"  {a[:34]:34s} vs {b[:34]:34s}: {align(labels[a], labels[b])[1]*100:.1f}%")
base = labels[names[0]]
aligned = [base] + [align(base, labels[k])[0] for k in names[1:]]
stack = np.stack(aligned)
cons = np.where((stack == stack[0]).all(0), stack[0], -3)  # -3 = models disagree (not scored)
cons[(stack == -2).any(0)] = -2
v, c = np.unique(cons, return_counts=True)
print("\nconsensus reference seconds:", {int(k): round(x / 10, 1) for k, x in zip(v, c)}, "(-3 disagree, -2 overlap, -1 silence)")
np.save(sys.argv[1] + ".labels.npy", cons[None])
