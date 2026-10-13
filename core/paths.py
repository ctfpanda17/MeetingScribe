"""Where the app keeps its files: everything stays inside the app's own folder (portable).

- running from source: the project folder (…/meetting)
- packaged .exe:       the folder that contains MeetingScribe.exe
"""
import os
import sys

APP_NAME = "MeetingScribe"

if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRANSCRIPT_DIR = os.path.join(APP_DIR, "會議紀錄")  # auto-saved transcripts and exports
DATA_DIR = os.path.join(APP_DIR, "data")  # settings, voiceprints, log
MODEL_DIR = os.path.join(APP_DIR, "models")  # downloaded speech models
PROFILES_PATH = os.path.join(DATA_DIR, "voiceprints.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
LOG_PATH = os.path.join(DATA_DIR, "app.log")

for d in (TRANSCRIPT_DIR, DATA_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# Hugging Face downloads (the ASR model) go into the app folder too, not the user's cache.
os.environ.setdefault("HF_HOME", os.path.join(MODEL_DIR, "huggingface"))
