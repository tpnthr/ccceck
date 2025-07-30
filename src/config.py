import json
import pathlib
from pathlib import Path

import whisperx
from optuna.terminator.improvement.emmr import torch

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
APP_NAME = config.get("app_name", "speech2text")
VERSION = config.get("version", "0.0.1")
ALLOW_SHUTDOWN = config.get("allow_shutdown", False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "large-v3"
LANGUAGE_DEFAULT = "pl"
ALIGN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
MAX_PAUSE = 1.5  # Max Pause of the fraze

ASR_MODEL = whisperx.load_model(WHISPER_MODEL, DEVICE)
ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(language_code="pl", device=DEVICE, model_name=ALIGN_MODEL_NAME)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

DATA_INPUT_DIR = pathlib.Path("data/input")
DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR = pathlib.Path("data/output")
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_TEMP_DIR = pathlib.Path("data/temp")
DATA_TEMP_DIR.mkdir(parents=True, exist_ok=True)


HF_TOKEN = "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"  # for gated alignment if needed
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32