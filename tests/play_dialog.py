"""Play a two-person dialogue into the meeting through virtual audio cables, logging exact start/end times.

Person A -> "喇叭 (Sonic Studio Virtual Mixer)"   -> mic of the user's Teams participant
Person B -> "喇叭 (Steam Streaming Microphone)"   -> mic of the guest tab
"""
import json, os, sys, time
import librosa, numpy as np, pyaudiowpatch as pa

ROUTE = {"A": os.environ.get("ROUTE_A", "Sonic Studio Virtual Mixer"), "B": "Steam Streaming Microphone"}
SCRIPT = [("A", "a2")] if "--one" in sys.argv else ([("A", c) for c in ("a1", "a2", "a3", "a4", "b1", "b2")] if "--aonly" in sys.argv else [("A", "a1"), ("B", "b1"), ("A", "a2"), ("B", "y_short"), ("A", "a3"), ("B", "b2"), ("A", "a4")])
GAP = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0].isdigit() else 1.5

p = pa.PyAudio()
api = p.get_host_api_info_by_type(pa.paWASAPI)["index"]
dev = {}
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    for who, key in ROUTE.items():
        if d["hostApi"] == api and d["maxOutputChannels"] > 0 and key in d["name"] and not d.get("isLoopbackDevice") and d["name"].startswith("喇叭"):
            dev[who] = d
streams = {}
for who, d in dev.items():
    sr = int(d["defaultSampleRate"]); ch = min(2, d["maxOutputChannels"])
    streams[who] = (p.open(format=pa.paFloat32, channels=ch, rate=sr, output=True, output_device_index=d["index"]), sr, ch)
log = []
for who, clip in SCRIPT:
    s, sr, ch = streams[who]
    x, _ = librosa.load(f"tests/{clip}.wav", sr=sr)
    e = np.abs(x); on = np.flatnonzero(e > e.max() * 0.02); x = x[on[0]:on[-1] + 1]
    data = np.repeat((x * 0.8).astype(np.float32)[:, None], ch, 1).ravel().tobytes()
    t0 = time.time(); s.write(data); t1 = time.time()
    log.append({"who": who, "clip": clip, "start": t0, "end": t1})
    print(f"{who} {clip:8s} {t0:.2f}-{t1:.2f} ({t1 - t0:.1f}s)", flush=True)
    time.sleep(GAP)
json.dump(log, open("tests/play_log.json", "w"))
for s, _, _ in streams.values(): s.close()
p.terminate()
