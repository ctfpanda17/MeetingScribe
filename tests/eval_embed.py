"""How well does each voiceprint model separate the two podcast hosts, by turn length? Simulate online clustering."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np, soundfile as sf, sherpa_onnx as so
from diarize import SpeakerEncoder, SpeakerRegistry
from session import _trim_silence
clip = sys.argv[1]
audio, _ = sf.read(clip, dtype="float32")
ref = np.load(clip + ".labels.npy")[0]; step = 0.1
turns = []  # (start_s, end_s, speaker)
i = 0
while i < len(ref):
    j = i
    while j < len(ref) and ref[j] == ref[i]: j += 1
    if ref[i] >= 0: turns.append((i * step, j * step, int(ref[i])))
    i = j
print(f"{len(turns)} reference turns; durations: <1s {sum(e-s<1 for s,e,_ in turns)}, 1-3s {sum(1<=e-s<3 for s,e,_ in turns)}, >=3s {sum(e-s>=3 for s,e,_ in turns)}")

ecapa = SpeakerEncoder("cuda")
def sherpa_embedder(path):
    ex = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=path, num_threads=4))
    def f(x):
        s = ex.create_stream(); s.accept_waveform(16000, x); s.input_finished()
        v = np.array(ex.compute(s)); return v / np.linalg.norm(v)
    return f
models = {"ECAPA (目前)": ecapa.embed,
          "CAM++ zh": sherpa_embedder("tests/spk/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
          "CAM++ zh-en adv": sherpa_embedder("tests/spk/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"),
          "WeSpeaker cnceleb": sherpa_embedder("tests/spk/wespeaker_zh_cnceleb_resnet34_LM.onnx")}
for name, emb in models.items():
    E, L, D = [], [], []
    for s, e, spk in turns:
        x = _trim_silence(audio[int(s * 16000):int(e * 16000)])
        if len(x) < 0.4 * 16000: continue
        E.append(emb(x)); L.append(spk); D.append(len(x) / 16000)
    E, L, D = np.array(E), np.array(L), np.array(D)
    # centroid of each true speaker from its long turns (like a settled registry)
    cents = [E[(L == k) & (D >= 3)].mean(0) for k in (0, 1)]
    cents = [c / np.linalg.norm(c) for c in cents]
    own = np.array([E[i] @ cents[L[i]] for i in range(len(E))]); other = np.array([E[i] @ cents[1 - L[i]] for i in range(len(E))])
    print(f"\n## {name}")
    print(f"   two hosts' voiceprints similarity: {cents[0] @ cents[1]:.2f}")
    for lo, hi in [(0.4, 1), (1, 2), (2, 4), (4, 99)]:
        m = (D >= lo) & (D < hi)
        if m.any():
            print(f"   turns {lo}-{hi}s n={m.sum():3d}: own {own[m].mean():.2f} other {other[m].mean():.2f}  correct-side {np.mean(own[m] > other[m])*100:.0f}%")
    # online clustering like the app does, for several thresholds
    for thr in (0.3, 0.4, 0.5, 0.55, 0.6):
        reg = SpeakerRegistry(thr); reg.speakers = {}  # no saved voiceprints
        assigned = []
        last = None
        for e, d in zip(E, D):
            sid, _ = reg.assign(e, d, last); last = sid; assigned.append(sid)
        # purity: map each found speaker to its majority true speaker
        correct = 0
        for sid in set(assigned):
            idx = [i for i, a in enumerate(assigned) if a == sid]
            correct += max(np.sum(L[idx] == 0), np.sum(L[idx] == 1))
        print(f"   online threshold {thr}: speakers found {len(set(assigned))}, turns correct {correct / len(L) * 100:.0f}%")
