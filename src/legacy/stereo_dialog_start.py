#!/usr/bin/env python3
"""stereo_dialog_start.py
FastAPI service for transcribing stereo call recordings.
Each channel is processed separately and labeled as ``caller`` (left) and
``client`` (right). Models are loaded once on startup and kept in memory.
"""

import os
import tempfile
import pathlib
import subprocess
import shutil
import time
import urllib.parse
from typing import List, Optional, Tuple, Dict

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn

# ----------------- CONFIGURABLE CONSTANTS -----------------
HF_TOKEN = "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"  # for gated alignment if needed
LANGUAGE_DEFAULT = "pl"
WHISPER_MODEL = "large-v3"
COMPUTE_TYPE = "float16"
ALIGN_MODEL_CANDIDATES = [
    "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "facebook/wav2vec2-large-xlsr-53-polish",
    "mbien/wav2vec2-large-xlsr-polish",
]
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
# -----------------------------------------------------------

# -------- Logging helpers --------

def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

def err(msg: str) -> None:
    print(f"[ERR ] {msg}", flush=True)

# -------- Heavy imports --------
import torch
import whisperx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL = None
ALIGN_MODEL = None
ALIGN_META = None

# ------------- FastAPI models -------------

class TranscribeRequest(BaseModel):
    input: str
    language: Optional[str] = None
    need_alignment: Optional[bool] = None
    return_srt: Optional[bool] = False
    return_vtt: Optional[bool] = False


class BatchRequest(BaseModel):
    inputs: List[str]
    language: Optional[str] = None
    need_alignment: Optional[bool] = None
    return_srt: Optional[bool] = False
    return_vtt: Optional[bool] = False

# -------- Utility functions --------

def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = pathlib.Path(parsed.path).name or "audio.wav"
    if not any(name.lower().endswith(ext) for ext in AUDIO_EXTS):
        name += ".wav"
    fd, tmp_path = tempfile.mkstemp(prefix="whx_", suffix=pathlib.Path(name).suffix)
    os.close(fd)
    log(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tmp_path


def ensure_wav_mono16k(path: str) -> str:
    if shutil.which("ffmpeg") is None:
        return path
    out_path = pathlib.Path(tempfile.gettempdir()) / (pathlib.Path(path).stem + "_16k.wav")
    if out_path.exists():
        return str(out_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_path)


def gen_srt(segments: List[Dict]) -> str:
    lines = []
    for idx, s in enumerate(segments, start=1):
        def ts(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            sec = t % 60
            return f"{h:02}:{m:02}:{sec:06.3f}".replace(".", ",")
        lines.append(f"{idx}\n{ts(s['start'])} --> {ts(s['end'])}\n{s.get('speaker','')} {s['text'].strip()}\n")
    return "\n".join(lines)


def gen_vtt(segments: List[Dict]) -> str:
    lines = ["WEBVTT\n"]
    for s in segments:
        def ts(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            sec = t % 60
            return f"{h:02}:{m:02}:{sec:06.3f}"
        lines.append(f"{ts(s['start'])} --> {ts(s['end'])}\n{s.get('speaker','')} {s['text'].strip()}\n")
    return "\n".join(lines)


def format_segments(segments: List[Dict]) -> str:
    lines = []
    for s in segments:
        spk = s.get("speaker") or "SPK"
        lines.append(f"[{s['start']:7.2f} - {s['end']:7.2f}] {spk}: {s['text']}")
    return "\n".join(lines)

# -------- Stereo helpers --------

def split_stereo(path: str) -> Tuple[str, str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
    left = tempfile.NamedTemporaryFile(suffix="_left.wav", delete=False)
    right = tempfile.NamedTemporaryFile(suffix="_right.wav", delete=False)
    left.close(); right.close()
    cmd_left = ["ffmpeg", "-y", "-i", path, "-map_channel", "0.0.0", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", left.name]
    cmd_right = ["ffmpeg", "-y", "-i", path, "-map_channel", "0.0.1", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", right.name]
    subprocess.run(cmd_left, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd_right, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return left.name, right.name


def transcribe_channel(path: str, language: str, need_alignment: bool) -> Dict:
    norm = ensure_wav_mono16k(path)
    asr_res = ASR_MODEL.transcribe(norm, language=language)
    if need_alignment and ALIGN_MODEL is not None:
        aligned = whisperx.align(asr_res["segments"], ALIGN_MODEL, ALIGN_META, norm, device=DEVICE)
    else:
        aligned = {"segments": asr_res["segments"], "word_segments": []}
    return aligned


def transcribe_stereo(inp: str, language: str, need_alignment: bool, want_srt: bool, want_vtt: bool) -> Dict:
    t0 = time.time()
    tmp_files: List[str] = []
    try:
        local = download_url(inp) if is_url(inp) else inp
        if is_url(inp):
            tmp_files.append(local)
        if not os.path.isfile(local):
            raise FileNotFoundError(f"Input not found: {local}")

        left, right = split_stereo(local)
        tmp_files.extend([left, right])

        left_a = transcribe_channel(left, language, need_alignment)
        right_a = transcribe_channel(right, language, need_alignment)

        for s in left_a["segments"]:
            s["speaker"] = "client"
        for w in left_a.get("word_segments", []):
            w["speaker"] = "client"
        for s in right_a["segments"]:
            s["speaker"] = "caller"
        for w in right_a.get("word_segments", []):
            w["speaker"] = "caller"

        segments = left_a["segments"] + right_a["segments"]
        segments.sort(key=lambda s: s["start"])
        words = left_a.get("word_segments", []) + right_a.get("word_segments", [])
        words.sort(key=lambda w: w["start"])

        seg_out = [
            {"start": float(s["start"]), "end": float(s["end"]), "speaker": s.get("speaker"), "text": s.get("text", "").strip()} for s in segments
        ]
        words_out = [
            {"word": w.get("word"), "start": float(w["start"]), "end": float(w["end"]), "speaker": w.get("speaker")} for w in words
        ]

        resp = {
            "ok": True,
            "input": inp,
            "segments": seg_out,
            "words": words_out,
            "aligned": bool(need_alignment and ALIGN_MODEL is not None),
            "diarized": False,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        if want_srt:
            resp["srt"] = gen_srt(seg_out)
        if want_vtt:
            resp["vtt"] = gen_vtt(seg_out)
        return resp
    except Exception as e:
        return {"ok": False, "input": inp, "error": str(e)}
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except Exception:
                pass

# -------- FastAPI app --------

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": WHISPER_MODEL}


@app.post("/transcribe", response_class=PlainTextResponse)
def api_transcribe(req: TranscribeRequest):
    lang = req.language or LANGUAGE_DEFAULT
    need_align = req.need_alignment if req.need_alignment is not None else bool(ALIGN_MODEL)
    res = transcribe_stereo(req.input, lang, need_align, req.return_srt, req.return_vtt)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error"))
    if req.return_srt:
        return res.get("srt", "")
    if req.return_vtt:
        return res.get("vtt", "")
    return format_segments(res.get("segments", []))


@app.post("/batch_transcribe")
def api_batch(req: BatchRequest):
    lang = req.language or LANGUAGE_DEFAULT
    need_align = req.need_alignment if req.need_alignment is not None else bool(ALIGN_MODEL)
    results = []
    for item in req.inputs:
        results.append(transcribe_stereo(item, lang, need_align, req.return_srt, req.return_vtt))
    return {"results": results}

ALLOW_SHUTDOWN = False  # set via CLI flag

@app.post("/shutdown")
def shutdown():
    if not ALLOW_SHUTDOWN:
        return {"ok": False, "error": "Shutdown not enabled"}
    log("Shutdown requested …")
    import threading, sys as _sys
    threading.Timer(0.5, lambda: _sys.exit(0)).start()
    return {"ok": True, "message": "Service exiting"}


# -------- Model load and warmup --------

def load_models(need_alignment: bool, language: str) -> None:
    global ASR_MODEL, ALIGN_MODEL, ALIGN_META
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={DEVICE}")
    log(f"Loading WhisperX ASR '{WHISPER_MODEL}' (compute_type={COMPUTE_TYPE})")
    ASR_MODEL = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)

    if need_alignment:
        loaded = False
        for cand in ALIGN_MODEL_CANDIDATES:
            try:
                log(f"Trying alignment model: {cand}")
                ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(language_code=language, device=DEVICE, model_name=cand)
                log(f"Loaded alignment model: {cand}")
                loaded = True
                break
            except Exception as e:
                warn(f"Alignment model failed: {cand} -> {e}")
        if not loaded:
            warn("All alignment candidates failed; proceeding WITHOUT alignment.")
            ALIGN_MODEL = None
            ALIGN_META = None

    # Warmup with second of silence
    try:
        import numpy as np, soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sr = 16000
        sf.write(tmp.name, np.zeros(sr, dtype="float32"), sr)
        transcribe_stereo(tmp.name, language, bool(ALIGN_MODEL), False, False)
        tmp.close()
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        log("Warmup complete.")
    except Exception as e:
        warn(f"Warmup skipped: {e}")


# -------- Main entry --------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--language", default=LANGUAGE_DEFAULT)
    parser.add_argument("--no-alignment", action="store_true", help="Disable alignment load")
    parser.add_argument("--allow-shutdown", action="store_true", help="Allow POST /shutdown")
    args = parser.parse_args()

    global ALLOW_SHUTDOWN
    ALLOW_SHUTDOWN = args.allow_shutdown

    load_models(not args.no_alignment, args.language)
    log("Models loaded & resident. Starting HTTP service.")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
