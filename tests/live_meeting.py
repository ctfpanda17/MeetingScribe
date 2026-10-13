"""Headless live run: capture a meeting (mic + meeting sound), take speaker names from the browser extension,
print the transcript with names. Usage: live_meeting.py <mic name part> <speaker name part> <seconds>"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import models as M, paths
from engines import load_engine
from diarize import SpeakerEncoder, ClusterEncoder, cluster_model_path
from session import MeetingSession
from turns import ChangeDetector, model_path
from audio import LiveCapture, list_microphones, list_speakers
from bridge import Bridge
mic_part, spk_part, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
mic = next(d["index"] for d in list_microphones() if mic_part in d["name"])
spk = next(d["index"] for d in list_speakers() if spk_part in d["name"])
m = next(m for m in M.list_models() if "sense-voice" in m.id)
eng = load_engine(m.kind, m.path, "cpu")
segs, events = {}, []
T0 = [None]
def emit(e):
    if os.environ.get("MEETINGSCRIBE_DEBUG") and e["type"] not in ("partial",): print("  [event]", e["type"], flush=True)
    if e["type"] == "final":
        segs[e["seg"]] = dict(e)
        print(f"  [final {e['start']:.2f}-{e['end']:.2f}] {e['text']}", flush=True)
    if e["type"] == "segments":
        print(f"  [segments update: {len(e['segments'])} lines]", flush=True)
    elif e["type"] == "segments": segs.clear(); segs.update({x["seg"]: dict(x) for x in e["segments"]})
    elif e["type"] == "discard": segs.pop(e["seg"], None)
s = MeetingSession(eng, SpeakerEncoder("cuda"), emit, {"language": "zh", "latency": eng.default_latency(),
                   "cluster_embed": None if os.environ.get("NOTITANET") else ClusterEncoder(cluster_model_path()).embed, "save_audio": True},
                   turns=None if os.environ.get("NODIAR") else ChangeDetector(model_path(paths.MODEL_DIR)))
br = Bridge(lambda ev: (events.append(ev), s.platform_event(ev)), lambda: True)
print("bridge:", br.start(), flush=True)
cap = LiveCapture("both", mic, s.push_audio, speaker_index=spk, anti_echo=False)
cap.start(); print("RECORDING", flush=True)
time.sleep(secs)
cap.close(); s.stop(); br.stop()
names = {k: v["name"] for k, v in s.registry.speakers.items()}
print(f"extension events: {len(events)}; names seen: {sorted({n for e in events for n in e['speaking']})}; self: {s.platform_self!r}")
for seg in sorted(segs.values(), key=lambda x: x["start"]):
    print(f"  {seg['start']:6.2f}-{seg['end']:6.2f}  {names.get(seg['speaker'], seg['speaker']):<16} {seg['text']}")
print("WAV:", s.files()["wav"])
if os.environ.get("MEETINGSCRIBE_DUMP"):
    import glob, soundfile as sf
    print("re-decode with the same engine, after the meeting:")
    for f in sorted(glob.glob(os.path.join(os.environ["MEETINGSCRIBE_DUMP"], "buf_*.wav"))):
        b, _ = sf.read(f, dtype="float32")
        print("  ", os.path.basename(f), eng.transcribe(b, "zh"))
