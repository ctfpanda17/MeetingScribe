"""Streaming diarization vs the offline 2-speaker reference: speakers found, frame & turn accuracy."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import numpy as np, soundfile as sf, paths
from diarize import SpeakerEncoder, SpeakerRegistry
from turns import ChangeDetector, model_path
from streamdiar import StreamingDiarizer, RES
clip = sys.argv[1]
audio, _ = sf.read(clip, dtype="float32")
ref = np.load(clip + ".labels.npy")[0] if os.path.exists(clip + ".labels.npy") else None
enc = SpeakerEncoder("cuda"); reg = SpeakerRegistry(0.55); reg.speakers = {}
for k, v in (a.split("=") for a in sys.argv[2:]): setattr(StreamingDiarizer, k, float(v))
d = StreamingDiarizer(ChangeDetector(model_path(paths.MODEL_DIR)), enc.embed, reg)
DELAY = 2.0
live = {}  # frame(0.1s) -> label known DELAY s later
t = time.time()
for now in np.arange(0.5, len(audio) / 16000 + 0.01, 0.5):
    d.update(lambda a, b: audio[a:b], now)
    t0 = now - DELAY - 0.5
    if t0 >= 0:
        for i, lab in enumerate(d.frame_labels(t0, t0 + 0.5)[::5]):
            live[int(round((t0 + i * 0.1) / 0.1))] = lab
cost = time.time() - t
final = d.frame_labels(0, len(audio) / 16000)[::5]
print(f"processing: {cost:.1f}s for {len(audio)/16000:.0f}s audio ({cost/(len(audio)/16000)*100:.1f}% of real time)")
print("speakers created:", [s["name"] for s in reg.speakers.values()])
def score(labs, name):
    labs = list(labs)
    n = min(len(labs), len(ref))
    pairs = [(labs[i], ref[i]) for i in range(n) if ref[i] >= 0 and labs[i] is not None]
    cover = len(pairs) / max(1, sum(ref[:n] >= 0))
    correct = 0
    for sid in set(p[0] for p in pairs):
        r = [p[1] for p in pairs if p[0] == sid]
        correct += max(r.count(0), r.count(1))
    # turn-level: majority label per reference turn
    i, tc, tn = 0, 0, 0
    mapping = {sid: max((0, 1), key=lambda k: sum(1 for p in pairs if p[0] == sid and p[1] == k)) for sid in set(p[0] for p in pairs)}
    short_c = short_n = 0
    while i < n:
        j = i
        while j < n and ref[j] == ref[i]: j += 1
        if ref[i] >= 0:
            got = [mapping.get(labs[k]) for k in range(i, j) if labs[k] is not None]
            if got:
                ok = max(set(got), key=got.count) == ref[i]
                tn += 1; tc += ok
                if (j - i) * 0.1 < 1.0: short_n += 1; short_c += ok
        i = j
    ov = np.mean(ref[:n] == -2) * 100
    print(f"  {name} [overlap {ov:.0f}% of time not scored]: frames correct {correct/max(1,len(pairs))*100:.1f}% (covered {cover*100:.0f}%) | turns correct {tc}/{tn} | short(<1s) turns {short_c}/{short_n}")
if ref is not None:
    score(final, "final (after all windows)")
    score([live.get(i) for i in range(len(ref))], f"live ({DELAY:.0f}s after)")
