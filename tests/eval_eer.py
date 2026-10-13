"""Equal error rate of same-vs-different speaker decisions for several voiceprint models, on real references."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np, soundfile as sf, sherpa_onnx as so
from diarize import SpeakerEncoder
from session import _trim_silence
def turns_of(ref, min_len):
    out, i = [], 0
    while i < len(ref):
        j = i
        while j < len(ref) and ref[j] == ref[i]: j += 1
        if ref[i] >= 0 and (j - i) / 10 >= min_len: out.append((i / 10, j / 10, int(ref[i])))
        i = j
    return out
def sherpa(path):
    ex = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=path, num_threads=4))
    def f(x):
        s = ex.create_stream(); s.accept_waveform(16000, x); s.input_finished()
        v = np.array(ex.compute(s)); return v / np.linalg.norm(v)
    return f
ecapa = SpeakerEncoder("cuda")
models = {"ECAPA (now)": ecapa.embed,
          "ERes2NetV2 zh": sherpa("tests/spk/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"),
          "TitaNet-L": sherpa("tests/spk/nemo_en_titanet_large.onnx"),
          "CAM++ zh": sherpa("tests/spk_unused/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")}
samples = [("tests/podcast_5min.wav", "tests/podcast_5min.wav.labels.npy"), ("tests/kk_5min.wav", "tests/kk_5min.wav.labels2.npy")]
import time
for name, emb in models.items():
    res = []
    for clip, refp in samples:
        audio, _ = sf.read(clip, dtype="float32"); ref = np.load(refp)[0]
        for L in (1.0, 2.5):
            T = turns_of(ref, L)
            # chop long turns into L-second pieces so every piece has the same length
            pieces = []
            for s, e, k in T:
                t = s
                while t + L <= e: pieces.append((t, t + L, k)); t += L
            t0 = time.time()
            E = np.array([emb(_trim_silence(audio[int(a * 16000):int(b * 16000)])) for a, b, _ in pieces]); K = np.array([p[2] for p in pieces])
            ms = (time.time() - t0) / len(pieces) * 1000
            S = E @ E.T; iu = np.triu_indices(len(E), 1); same = K[iu[0]] == K[iu[1]]; sc = S[iu]
            ths = np.linspace(-0.2, 1, 241)
            far = np.array([np.mean(sc[~same] >= t) for t in ths]); frr = np.array([np.mean(sc[same] < t) for t in ths])
            i = np.argmin(np.abs(far - frr))
            res.append(f"{os.path.basename(clip)[:7]} {L}s: EER {(far[i]+frr[i])/2*100:4.1f}% @ {ths[i]:.2f}")
    print(f"{name:14s} {ms:5.0f} ms/piece | " + " | ".join(res))
