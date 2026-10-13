"""Run every installed model on the same clip through the full live pipeline; report CER, speakers, timing."""
import sys, os, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import soundfile as sf, torch
import models as M
from engines import load_engine
from diarize import SpeakerEncoder
from session import MeetingSession
from turns import ChangeDetector, model_path
import paths
from opencc import OpenCC

REF = ["大家好今天的會議主要討論下一季的產品規劃我們先從預算開始", "好的我這邊整理了上一季的數據行銷費用比預期多了百分之十五",
       "那我們是不是應該把部分預算轉到研發", "我同意不過研發那邊需要先提出具體的時程表", "沒問題我下週三前會把報告寄給大家"]
TRUE_SPK = [0, 1, 0, 2, 1]
cc = OpenCC("s2twp")
norm = lambda t: re.sub(r"[^\w]", "", cc.convert(t)).replace("資料", "數據")  # s2twp maps 数据->資料

def cer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / len(ref)

audio, _ = sf.read(sys.argv[1], dtype="float32")
enc = SpeakerEncoder("cuda")
turns = None if os.environ.get("NO_TURNS") else ChangeDetector(model_path(paths.MODEL_DIR))
only = sys.argv[2:]
for m in M.list_models():
    if only and not any(o in m.id for o in only): continue
    dev = "cpu" if m.kind.startswith("sherpa") else "cuda"
    t = time.time(); eng = load_engine(m.kind, m.path, dev); load = time.time() - t
    finals, partial_t = [], {}
    def emit(e):
        if e["type"] == "partial" and e["seg"] not in partial_t: partial_t[e["seg"]] = time.time()
        if e["type"] == "final": finals.append((time.time(), e))
    s = MeetingSession(eng, enc, emit, {"language": "zh", "latency": eng.default_latency(), "traditional": True}, turns=turns)
    t0 = time.time()
    for i in range(0, len(audio), 1600):
        s.push_audio(audio[i:i + 1600]); time.sleep(max(0, t0 + (i + 1600) / 16000 - time.time()))
    s.stop()
    hyp = "".join(e["text"] for _, e in finals)
    c = cer("".join(REF), norm(hyp))
    # delay between the end of each sentence's audio and its final text appearing
    lag = [ft - (t0 + e["end"]) for ft, e in finals]
    first = [partial_t[e["seg"]] - (t0 + e["start"]) for _, e in finals if e["seg"] in partial_t]
    ids = {}; spk = [ids.setdefault(e["speaker"], len(ids)) for _, e in finals]
    print(f"\n### {m.label}  (load {load:.1f}s, {dev})")
    print(f"CER {c*100:.1f}%  | sentences {len(finals)}/5 | speakers {'OK' if spk == TRUE_SPK else spk}"
          f" | first words after {sum(first)/max(1,len(first)):.2f}s | final after sentence end {sum(lag)/max(1,len(lag)):.2f}s")
    for _, e in finals: print("   ", e["text"])
    eng.close(); del eng; torch.cuda.empty_cache()
