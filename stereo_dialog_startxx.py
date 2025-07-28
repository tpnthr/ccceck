#!/usr/bin/env python3
import os
import tempfile
import subprocess
import shutil
import time
import urllib.parse
from typing import List, Optional, Tuple, Dict
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import whisperx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "large-v3"
ALIGN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
MAX_PAUSE = 1.5  # максимальная пауза в секундах

app = FastAPI()

ASR_MODEL = whisperx.load_model(WHISPER_MODEL, DEVICE)
ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(language_code="pl", device=DEVICE, model_name=ALIGN_MODEL_NAME)

class TranscribeRequest(BaseModel):
    input: str


def split_stereo(path: str) -> Tuple[str, str]:
    left = tempfile.NamedTemporaryFile(suffix="_left.wav", delete=False)
    right = tempfile.NamedTemporaryFile(suffix="_right.wav", delete=False)
    left.close()
    right.close()
    subprocess.run(["ffmpeg", "-y", "-i", path, "-map_channel", "0.0.0", "-ac", "1", "-ar", "16000", left.name])
    subprocess.run(["ffmpeg", "-y", "-i", path, "-map_channel", "0.0.1", "-ac", "1", "-ar", "16000", right.name])
    return left.name, right.name


def transcribe_channel(path: str) -> List[Dict]:
    result = ASR_MODEL.transcribe(path)
    aligned = whisperx.align(result["segments"], ALIGN_MODEL, ALIGN_META, path, device=DEVICE)
    return aligned["word_segments"]


def group_words(words: List[Dict]) -> List[Dict]:
    grouped, current = [], {"speaker": None, "start": None, "end": None, "text": []}

    for w in sorted(words, key=lambda x: x["start"]):
        spk, start, end, txt = w["speaker"], w["start"], w["end"], w["word"]
        if current["speaker"] != spk or (current["end"] and start - current["end"] > MAX_PAUSE):
            if current["text"]:
                grouped.append(current)
            current = {"speaker": spk, "start": start, "end": end, "text": [txt]}
        else:
            current["end"] = end
            current["text"].append(txt)

    if current["text"]:
        grouped.append(current)

    return grouped


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_file = req.input
    tmp_files = []

    try:
        left_path, right_path = split_stereo(audio_file)
        tmp_files.extend([left_path, right_path])

        left_words = transcribe_channel(left_path)
        right_words = transcribe_channel(right_path)

        for w in left_words:
            w["speaker"] = "client"
        for w in right_words:
            w["speaker"] = "caller"

        all_words = left_words + right_words
        grouped_dialogue = group_words(all_words)

        for r in grouped_dialogue:
            r["text"] = " ".join(r["text"])

        return {"ok": True, "dialogue": grouped_dialogue}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        for f in tmp_files:
            os.unlink(f)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
