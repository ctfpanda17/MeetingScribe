import sys, time, numpy as np, pyaudiowpatch as pa
key = sys.argv[1]
p = pa.PyAudio(); api = p.get_host_api_info_by_type(pa.paWASAPI)["index"]
d = next(p.get_device_info_by_index(i) for i in range(p.get_device_count()) if p.get_device_info_by_index(i)["hostApi"] == api and key in p.get_device_info_by_index(i)["name"] and p.get_device_info_by_index(i)["maxOutputChannels"] > 0 and not p.get_device_info_by_index(i).get("isLoopbackDevice"))
sr = int(d["defaultSampleRate"]); ch = min(2, d["maxOutputChannels"])
s = p.open(format=pa.paFloat32, channels=ch, rate=sr, output=True, output_device_index=d["index"])
t = np.arange(sr * int(sys.argv[2] if len(sys.argv) > 2 else 40)) / sr; tone = (float(sys.argv[3] if len(sys.argv) > 3 else 0.05) * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
s.write(np.repeat(tone[:, None], ch, 1).ravel().tobytes()); s.close(); p.terminate()
