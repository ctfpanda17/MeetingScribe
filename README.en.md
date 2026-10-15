# MeetingScribe (會議記錄助手)

[繁體中文](README.md) | **English**

MeetingScribe turns what everyone says into text live, during the meeting, and automatically tells speakers apart by their voices. Everything runs on your own computer; no audio is uploaded anywhere.

- Speech recognition: choose or add your own model (SenseVoice by default; Whisper, Breeze-ASR-25, NVIDIA nemotron, Paraformer and others are also supported)
- Speaker separation: pyannote speaker-change detection + ECAPA / TitaNet voiceprints + streaming diarization (the number of speakers is detected automatically). With the Chrome / Edge extension, the participant names shown by Meet / Teams are used directly
- Chinese output is converted to Traditional Chinese automatically (OpenCC), with full-width punctuation fixed

> The app's interface is in Traditional Chinese. Below, UI labels are quoted in Chinese with an English translation so you can find them on screen.

## For users: how to use it

1. Download `MeetingScribe-<version>-win64.7z` from the [latest release](https://github.com/ctfpanda17/MeetingScribe/releases/latest), extract it with [7-Zip](https://www.7-zip.org/) (Windows 11 can also extract it directly), and run **`MeetingScribe.exe`** in the extracted folder
2. On first launch, the default model is downloaded automatically (about 0.5 GB, needs internet; interrupted downloads resume automatically). After that it works offline
3. Choose an audio source (音源):
   - **線上會議 Meet / Teams（對方 + 我）** — online meeting (others + me): the default. Other participants are captured from the computer's audio output, your voice from the microphone
   - **線上會議（只記對方）** — online meeting (others only): your microphone is not recorded
   - **實體會議（麥克風）** — in-person meeting (microphone): everyone in a room shares one microphone
   - **匯入錄音檔** — import a recording: transcribe an existing file afterwards
4. Click **● 開始記錄** (Start recording). Text appears live (grey italic = still being spoken, normal = the sentence is final). The ☾ / ☀ button at the top right switches between dark and light themes
5. During or after the meeting you can:
   - Rename "說話者 1" (Speaker 1) to a real name in the right panel (press Enter)
   - One person split into two speakers → click "⋯" next to the name → "合併到…" (Merge into…)
   - A sentence attributed to the wrong person → click the speaker name in the table and pick the right one
   - A misrecognized word → double-click the text to edit it, Enter to finish
   - Click "記住" (Remember) to save someone's voiceprint so they are recognized automatically next time
6. The transcript is **saved automatically as you go** to the `會議紀錄\` (meeting records) folder inside the app folder (Markdown + JSON). Exports (TXT / Markdown / JSON) are saved to the same folder

### Changing the speech recognition model

The "辨識模型" (recognition model) menu at the top switches between installed models. There are two ways to add a new one:

1. **Paste a URL to download it**: click "＋ 新增模型" (Add model), paste a Hugging Face URL (e.g. `https://huggingface.co/MediaTek-Research/Breeze-ASR-25`), click "檢查" (Check) to see the model type and size, then "下載並加入" (Download and add). Only the files needed for recognition are downloaded; you can close the window and the download continues in the background.
2. **Drop in a folder yourself**: copy the whole model folder into `models\asr\` (any folder name), then click "⋯ → 重新整理模型清單" (Refresh model list) next to the model menu.

Supported formats:

| Type | Files needed | Examples | Hardware |
|---|---|---|---|
| Whisper family (incl. fine-tunes) | config.json + model.safetensors + tokenizer | Breeze-ASR-25, whisper-large-v3-turbo | NVIDIA GPU recommended |
| NVIDIA nemotron streaming | config.json + model.safetensors | nemotron-3.5-asr-streaming-0.6b | NVIDIA GPU recommended |
| sherpa-onnx | tokens.txt + .onnx | SenseVoice, streaming Paraformer, Zipformer | CPU is enough |

Results on the same Chinese meeting recording (3 speakers, 5 sentences; every model separated the speakers correctly):

| Model | Wrong characters | End of sentence → final text | Hardware |
|---|---|---|---|
| SenseVoice (default) | 0 | ~0.2 s | CPU |
| Breeze-ASR-25 | 0 | ~1.1 s | GPU |
| Nemotron 3.5 streaming | 2 (行銷→行消, 時程→時辰) | ~0.3 s | GPU |
| Streaming Paraformer | 4, and no punctuation | almost instant | CPU |

### Using it with Google Meet / Microsoft Teams

Meet (browser), Teams (desktop or web), Zoom and LINE all work. The app records "the sound the computer plays", so no plug-in is required and no bot has to join the meeting.

1. Join the Meet / Teams meeting first
2. Open MeetingScribe and choose the audio source "線上會議 Meet / Teams（對方 + 我）"
3. **會議聲音** (meeting audio): choose the device you listen to the meeting on — your headphones if you wear them, your speakers otherwise (★ marks the system default)
4. **麥克風** (microphone): choose the same microphone you use in Meet / Teams (what you say is labeled as you automatically)
5. If you use loudspeakers, tick "外放喇叭防回音" (speaker echo suppression); with headphones you can untick it
6. Click "開始記錄" (Start recording)

#### Label the transcript with participant names (Chrome / Edge extension)

When you use Google Meet or Teams in the browser, you can add the "會議記錄助手 連接器" (MeetingScribe Connector) extension. It sends the meeting app's "who is speaking now" indicator to the local app, so every remote participant is labeled with their name in the meeting, and it is also used to correct speaker assignments.

- Install: open `chrome://extensions` in Chrome or `edge://extensions` in Edge → turn on "Developer mode" → "Load unpacked" → select the `extension` folder inside the app folder (see `安裝說明.txt` in that folder)
- Once connected, the app shows "🔗 Meet 已連接" or "🔗 Teams 已連接" (connected) at the top
- Data only goes to this computer (127.0.0.1), and only the extension can send it — ordinary web pages cannot
- When several people in a meeting room share one microphone, the meeting app only sees one participant, so voice separation tells them apart (e.g. "會議室 (1)", "會議室 (2)")
- Who decides: remote participants are labeled by **the speaker the meeting app shows** (their name in the meeting); you are identified by **your microphone**; voice separation is only used for shared meeting-room microphones or when the extension is not installed
- Tested on **Teams web** (October 2026, playing a dialogue through a virtual audio cable): the speaking indicator appears about 0.2 s after speech starts and disappears about 0.7 s after it ends, which the app compensates for; every line of the transcript was labeled with the right participant name. **Google Meet has not been tested yet**; if its page structure differs, adjustments may be needed
- If the extension does nothing: go to `chrome://extensions` (or `edge://extensions`) → "Details" → "Site access", set it to "On all sites" or at least allow `teams.microsoft.com` / `meet.google.com`, then reload the meeting tab

> Reminder: tell participants and get their consent before recording or transcribing a meeting.

### System requirements

- Windows 10 / 11 (64-bit)
- The default model (SenseVoice) runs in real time on the CPU; Whisper / Breeze / nemotron models are best with an NVIDIA GPU (6 GB+ VRAM)
- Disk space: about 3.3 GB for the app, plus models (0.25–3 GB each)

### Where files are stored

Everything stays inside the app's own folder; nothing is written elsewhere on the computer (you can move or delete the whole folder):

```
MeetingScribe\
  MeetingScribe.exe
  extension\   ← Chrome / Edge extension
  會議紀錄\    ← meeting transcripts (auto-save + exports)
  models\asr\  ← speech recognition models (one folder per model; add or remove freely)
  models\      ← also holds the voiceprint models
  data\        ← settings, remembered voiceprints, log file app.log
```

Put the folder somewhere you can write to (e.g. Desktop, Downloads, drive D:), not in C:\Program Files.

### How speakers are told apart

The app does not match voiceprints one sentence at a time (voiceprints of short sentences are too unreliable). It uses streaming speaker diarization instead:

1. **Speaker-change detection** (pyannote segmentation 3.0): every 0.5 s it analyzes the last 8 s, comparing voices side by side within that window, and marks which voice is speaking every ~17 ms — including interruptions, overlaps and short replies like "嗯" (mm) or "對" (right).
2. **Live mapping to people**: each voice's voiceprint (ECAPA) is computed from everything it said in those 8 s and mapped to a speaker in the meeting; voices separated within the same window cannot be the same person. Each moment is decided by votes from several windows.
3. **Whole-meeting re-clustering**: every 15 s, in the background, all voice segments of the meeting are clustered again (TitaNet voiceprints + self-tuning spectral clustering). This determines how many people there are, merges one person who was split into two, and goes back to correct earlier sentences.

Tested on real shows (scored where two independent reference models agree):

| Sample | Actual speakers | Time with correct speaker (as shown live during the meeting) |
|---|---|---|
| Podcast (two male voices, fast back-and-forth) | 2 | ~99% |
| Interview show (3 people, emotional, frequent interruptions) | 3 | ~97% |
| Previous approach (per-sentence voiceprint matching) | | ~51% (and wrong speaker count) |

Limitations:
- The speaker shown during the meeting is the "current best guess" and may occasionally include an extra speaker; background correction and the end-of-meeting pass fix this automatically.
- Someone whose first words are only half a second long (e.g. "好的", "OK") gives too little voice information and may be assigned to someone else.
- When two people talk over each other for a long time, the text of the overlapping part may still be inaccurate.

### Background correction

Live decisions can only use what has been heard "so far", so every 6 s the app re-checks recently finished sentences (within the last 90 s) in the background, and reviews the whole meeting when it ends:

- **Short interjections hidden inside a sentence**: e.g. someone says "對啊" (yeah) in the middle of another person's sentence — it is split out and labeled with the right person.
- **Sentences attributed to the wrong person**: early in a meeting there are few voice samples; once more voice has accumulated, clearly wrong sentences are corrected automatically.
- Corrected sentences are tagged "已校正" (corrected); sentences too short to tell who said them (e.g. just "好的") are assigned to the previous speaker and tagged "待確認" (needs review) — click the name to change it.
- Speakers or text you changed by hand are never touched by the app.
- The meeting audio is also saved as `會議紀錄\<time>.wav` (about 115 MB per hour; can be turned off in the right panel).

| Test case | Result |
|---|---|
| Someone says a 0.6 s "對啊" in the middle of someone else's speech | Missed live; correctly split out and labeled after background correction |
| Someone's first words are a 0.5 s "好的" | Cannot be decided reliably (voice scores are almost equal); tagged "待確認" for you to confirm |

### Known compatibility issue: ASUS Sonic Studio

In testing, as soon as any program plays sound to the **"Sonic Studio Virtual Mixer"** virtual device, computations in **other programs** on the computer get corrupted (garbage speech recognition, GPU computation errors), most likely a problem with that audio driver. Playing to normal speakers is not affected.
If you use Sonic Studio to route "computer audio" into a microphone, use another virtual audio cable instead (e.g. VB-Audio Virtual Cable), or update / remove Sonic Studio.

### When speaker separation is wrong

- One person split into two: click "⋯" next to the name → "合併到…" (Merge into…).
- A sentence labeled with the wrong person: click the speaker name in the table and pick the right one; sentences you changed by hand are never touched again.
- "自動說話者分離" (automatic speaker diarization) can be turned off in the right panel to use simple mode (split sentences on pauses, match voiceprints per sentence); simple mode shows a "分辨靈敏度" (sensitivity) slider.

## For developers

```
core/       recognition core (no UI)
  engines/    speech recognition engines (nemotron / whisper / sherpa-onnx), chosen from the model's files
  models.py   model library: list, download from Hugging Face (only needed files, auto-resume)
  diarize.py  VAD, voiceprints, speaker clustering
  turns.py    speaker-change detection (pyannote segmentation)
  streamdiar.py streaming speaker diarization (who speaks when)
  cluster.py  whole-meeting re-clustering (NME-SC spectral clustering + merging)
  hints.py    speaker hints (mic vs. system audio, meeting-app events)
  bridge.py   local server receiving events from the browser extension (127.0.0.1)
  session.py  one meeting: audio → sentences → text + speaker → auto-save
  audio.py    microphone / system audio (WASAPI loopback) capture and mixing
  paths.py    storage locations (all inside the app folder)
app/
  main.py     desktop UI (PySide6)
  theme.py    light / dark themes
  widgets.py  avatars, chat-style transcript cards
  model_dialog.py  "add model" dialog
extension/    Chrome / Edge extension (Manifest V3)
tests/        test scripts and test audio
```

Run from source:

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app\main.py
```

Build the exe: run `build.bat`; the output is in `dist\MeetingScribe\`. Compress the whole folder to share it.

## License

The code in this project is released under the [MIT License](LICENSE).

Models downloaded by the app at runtime have their own licenses; please check them before use, e.g. SenseVoice (FunAudioLLM), Breeze-ASR-25 (MediaTek Research), NVIDIA nemotron / TitaNet (NVIDIA), pyannote segmentation 3.0 (MIT), SpeechBrain ECAPA (Apache 2.0).
