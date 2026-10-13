"""Reference for a clip with N speakers: two independent embedding models, forced to N clusters; keep agreement."""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np, soundfile as sf, sherpa_onnx as so, paths
from scipy.optimize import linear_sum_assignment
from turns import model_path
clip, N = sys.argv[1], int(sys.argv[2])
audio, _ = sf.read(clip, dtype="float32"); step = 0.1; n = int(len(audio) / 16000 / step)
labs = []
for emb in ["3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx", "nemo_en_titanet_large.onnx"]:
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=model_path(paths.MODEL_DIR))),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=f"tests/spk/{emb}"),
        clustering=so.FastClusteringConfig(num_clusters=N), min_duration_on=0.2, min_duration_off=0.3)
    lab = np.full(n, -1)
    for r in so.OfflineSpeakerDiarization(cfg).process(audio).sort_by_start_time():
        sl = slice(int(r.start / step), int(r.end / step))
        lab[sl] = np.where((lab[sl] >= 0) & (lab[sl] != r.speaker), -2, r.speaker)
    labs.append(lab)
    print(emb[:30], "seconds per speaker:", [round(np.sum(lab == k) / 10, 1) for k in range(N)])
a, b = labs
m = (a >= 0) & (b >= 0)
C = np.array([[np.sum((a[m] == i) & (b[m] == j)) for j in range(N)] for i in range(N)])
ri, ci = linear_sum_assignment(-C)
mp = dict(zip(ci, ri)); b2 = np.array([mp.get(x, x) if x >= 0 else x for x in b])
agree = np.mean(a[m] == b2[m]); print(f"agreement N={N}: {agree*100:.1f}%")
cons = np.where(a == b2, a, -3); cons[(a == -2) | (b == -2)] = -2
print("consensus seconds:", {int(k): round(c / 10, 1) for k, c in zip(*np.unique(cons, return_counts=True))})
np.save(f"{clip}.labels{N}.npy", cons[None])
