# PyInstaller spec — run build.bat
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [("app/icon.ico", ".")], [], []
for pkg in ("speechbrain", "silero_vad", "opencc", "hyperpyyaml", "pyaudiowpatch", "soxr", "hf_xet", "sherpa_onnx", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
hiddenimports += collect_submodules("transformers.models.nemotron3_5_asr")
hiddenimports += collect_submodules("transformers.models.auto")
hiddenimports += collect_submodules("transformers.models.whisper")
hiddenimports += ["diarize", "session", "audio", "paths", "models", "engines", "engines.nemotron", "engines.whisper",
                  "engines.sherpa", "turns", "streamdiar", "cluster", "hints", "bridge", "theme",
                  "widgets", "model_dialog", "scipy.optimize"]
for pkg in ("transformers", "tokenizers", "huggingface_hub", "safetensors", "torch", "numpy", "tqdm", "regex",
            "requests", "packaging", "filelock", "pyyaml", "accelerate", "speechbrain", "librosa", "scipy",
            "soundfile", "sentencepiece"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

a = Analysis(
    ["app/main.py"],
    pathex=["core", "app"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "IPython", "jupyter", "notebook", "tkinter", "tensorflow", "pandas", "torchvision",
              "fastapi", "uvicorn", "PyInstaller", "yt_dlp", "imageio_ffmpeg", "pyflakes", "curl_cffi",
              "websockets", "Crypto", "mutagen", "brotli", "secretstorage"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MeetingScribe", icon="app/icon.ico",
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="MeetingScribe", upx=False)
