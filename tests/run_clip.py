"""Feed a clip in real time through the full pipeline and print the final transcript (after any corrections)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import soundfile as sf
import numpy as np
import models as M, paths
from engines import load_engine
from diarize import SpeakerEncoder
from session import MeetingSession
from turns import ChangeDetector, model_path

clip, model = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "sense-voice")
m = next(m for m in M.list_models() if model in m.id)
eng = load_engine(m.kind, m.path, "cpu" if m.kind.startswith("sherpa") else "cuda")
segs, notes = {}, []
lags = []
def emit(e):
    if e["type"] == "final":
        segs[e["seg"]] = dict(e)
        if "t0" in globals(): lags.append(time.time() - (t0 + e["end"]))
    elif e["type"] == "segments": segs.clear(); segs.update({s["seg"]: dict(s) for s in e["segments"]})
    elif e["type"] == "discard": segs.pop(e["seg"], None)
    elif e["type"] == "toast": notes.append(e["message"])
from diarize import ClusterEncoder, cluster_model_path
s = MeetingSession(eng, SpeakerEncoder("cuda"), emit, {"language": "zh", "latency": eng.default_latency(),
                   "cluster_embed": None if os.environ.get("NOTITANET") else ClusterEncoder(cluster_model_path()).embed},
                   turns=None if os.environ.get("NODIAR") else ChangeDetector(model_path(paths.MODEL_DIR)))
audio, _ = sf.read(clip, dtype="float32")
SIM = os.environ.get("SIMHINTS")  # reference labels -> fake "meeting app says X is speaking" (late and imprecise)
if SIM:
    ref_h = np.load(SIM)[0]
    names = ["主持人", "來賓", "第三位"]
    i = 0
    while i < len(ref_h):
        j = i
        while j < len(ref_h) and ref_h[j] == ref_h[i]: j += 1
        if ref_h[i] >= 0 and (j - i) >= 5:  # platforms miss very short bursts
            jitter = 0.3
            s.hints.add(names[ref_h[i]], i / 10 + jitter, j / 10 + jitter)
        i = j
SIMNAME = os.environ.get("SIMNAME")  # "name:t0-t1,t0-t1,..." fake meeting-app intervals
if SIMNAME:
    nm, spans = SIMNAME.split(":")
    for sp in spans.split(","):
        x, y = map(float, sp.split("-")); s.hints.add(nm, x, y)
t0 = time.time()
FAST = os.environ.get("FAST")  # feed as fast as possible (the live logic follows audio time, not wall time)
for i in range(0, len(audio), 1600):
    s.push_audio(audio[i:i + 1600], (0.1, 0.0) if os.environ.get("FAKELEVELS") else None)
    if not FAST: time.sleep(max(0, t0 + (i + 1600) / 16000 - time.time()))
if lags and not FAST:
    print(f"  LAG: text final {np.median(lags):.2f}s (median) / {np.percentile(lags, 90):.2f}s (90%) after the sentence ended")
import copy
live_snapshot = copy.deepcopy(segs)  # what the screen showed during the meeting (before the end-of-meeting pass)
s.stop()
ids = {}
for seg in sorted(segs.values(), key=lambda x: x["start"]):
    who = ids.setdefault(seg["speaker"], f"P{len(ids) + 1}")
    print(f"  {seg['start']:6.2f}-{seg['end']:6.2f}  {who}{' (校正)' if seg.get('corrected') else ''}  {seg['text']}")
for n in notes[-3:]: print("  note:", n)
import numpy as np
ref_path = os.environ.get("REF")
def score(segs, title):
    """Time-weighted accuracy vs a reference (0.1 s labels; <0 = silence/overlap/undecided, not scored).
    Found speakers are matched one-to-one to reference speakers (Hungarian); extra speakers count as wrong."""
    from scipy.optimize import linear_sum_assignment
    print(f"  --- {title}")
    ref = np.load(ref_path)[0][: int(len(audio) / 1600)]
    rows = []
    for seg in sorted(segs.values(), key=lambda x: x["start"]):
        r = ref[int(seg["start"] * 10):int(seg["end"] * 10)]; r = r[r >= 0]
        if len(r): rows.append((seg["speaker"], r))
    if not rows:
        print("  SCORE: no transcript lines!"); return
    found = sorted({x[0] for x in rows}); refs = sorted({int(v) for _, r in rows for v in r})
    cover = np.array([[sum(np.sum(r == k) for s_, r in rows if s_ == f) for k in refs] for f in found])
    fi, ki = linear_sum_assignment(-cover)
    mapping = {found[i]: refs[j] for i, j in zip(fi, ki)}
    for sid in found:
        allr = np.concatenate([r for s_, r in rows if s_ == sid])
        share = " / ".join(f"ref{k} {np.mean(allr == k) * 100:.0f}%" for k in refs)
        tag = f"= reference speaker {mapping[sid]}" if sid in mapping else "= EXTRA speaker"
        print(f"  MAP {sid} {tag}: {len(allr) / 10:.1f}s ({share})")
    ok = tot = wrong = mixed = 0
    for sid, r in rows:
        good = np.sum(r == mapping[sid]) if sid in mapping else 0
        ok += good; tot += len(r); wrong += good < 0.5 * len(r)
        mixed += max(np.mean(r == k) for k in refs) < 0.8
    print(f"  SCORE: right speaker {ok / tot * 100:.1f}% of speech | lines mostly wrong {wrong}/{len(rows)} | "
          f"lines mixing speakers {mixed}/{len(rows)} | speakers found {len(found)} (reference {len(refs)})")
if ref_path:
    score(live_snapshot, "LIVE (on screen during the meeting)")
    score(segs, "FINAL (after the end-of-meeting pass)")
if os.environ.get("DIAG") and s.diar is not None:
    import itertools
    reg, d = s.registry, s.diar
    known = [k for k, v in reg.speakers.items() if v["count"] > 0]
    used = {seg["speaker"] for seg in segs.values()}
    for k in known:
        print(f"  DIAG {k} present={d.present.get(k,0)} count={reg.speakers[k]['count']} in_transcript={k in used}")
    for x, y in itertools.combinations(known, 2):
        sx, sy = reg.speakers[x]["sum"], reg.speakers[y]["sum"]
        print(f"  DIAG {x}-{y} sim={float(np.dot(sx,sy)/(np.linalg.norm(sx)*np.linalg.norm(sy))):.2f} together={d.cooc.get(frozenset((x,y)),0)}")
if os.environ.get("NAMES"):
    print("  NAMES:", {k: v["name"] for k, v in s.registry.speakers.items() if v.get("active")})
    print("  name map:", s.name_to_sids)
