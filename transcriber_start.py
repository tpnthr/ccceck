#!/usr/bin/env python3
"""
whisperx_service.py
Start a local HTTP microservice that keeps WhisperX large-v3 + (optional) alignment + (optional) pyannote diarization in GPU VRAM.

Endpoints:
  POST /transcribe        -> single file/URL
  POST /batch_transcribe  -> list of files/URLs
  GET  /health            -> quick health check
  POST /shutdown          -> graceful stop (disabled by default; enable via --allow-shutdown)

Request JSON (single):
{
  "input": "http://.../audio.wav" | "/path/to/file.wav",
  "language": "pl",                  (optional override, default set at startup)
  "need_alignment": true|false,      (default from startup flag)
  "need_diarization": true|false,    (default from startup flag)
  "return_srt": true|false,          (optional)
  "return_vtt": true|false           (optional)
}

Response JSON (success):
{
  "ok": true,
  "input": "...",
  "segments": [...],
  "words": [...],                 (empty if alignment disabled)
  "aligned": true|false,
  "diarized": true|false,
  "elapsed_sec": 12.345,
  "srt": "....",                  (only if requested)
  "vtt": "...."
}

Error:
{
  "ok": false,
  "input": "...",
  "error": "message"
}
"""

import os, sys, time, json, tempfile, pathlib, subprocess, shutil, urllib.parse
from typing import List, Optional
import requests
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

# ----------------- CONFIGURABLE CONSTANTS -----------------
HF_TOKEN = "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"   # <--- PUT YOUR HUGGING FACE TOKEN HERE (pyannote gated)
LANGUAGE_DEFAULT   = "pl"
WHISPER_MODEL      = "large-v3"
COMPUTE_TYPE       = "float16"   # change to "int8" if VRAM tight
ALIGN_MODEL_CANDIDATES = [
    "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "facebook/wav2vec2-large-xlsr-53-polish",
    "mbien/wav2vec2-large-xlsr-polish",
]
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
# -----------------------------------------------------------

# -------- Logging helpers --------
def log(msg):  print(f"[INFO] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)
def err(msg):  print(f"[ERR ] {msg}",  flush=True)

# -------- Heavy imports (only once) --------
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Globals filled in startup
ASR_MODEL = None
ALIGN_MODEL = None
ALIGN_META = None
DIAR_PIPE = None

# ------------- FastAPI models -------------
class TranscribeRequest(BaseModel):
    input: str
    language: Optional[str] = None
    need_alignment: Optional[bool] = None
    need_diarization: Optional[bool] = None
    return_srt: Optional[bool] = False
    return_vtt: Optional[bool] = False

class BatchRequest(BaseModel):
    inputs: List[str]
    language: Optional[str] = None
    need_alignment: Optional[bool] = None
    need_diarization: Optional[bool] = None
    return_srt: Optional[bool] = False
    return_vtt: Optional[bool] = False

# ------------- Utility functions -------------

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
        "ffmpeg", "-y",
        "-i", path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_path)

def gen_srt(segments):
    lines = []
    for idx, s in enumerate(segments, start=1):
        start = s["start"]; end = s["end"]
        def ts(t):
            h = int(t//3600); m = int((t%3600)//60); sec = t%60
            return f"{h:02}:{m:02}:{sec:06.3f}".replace(".", ",")
        speaker = s.get("speaker") or ""
        text = s["text"].strip()
        lines.append(f"{idx}\n{ts(start)} --> {ts(end)}\n{speaker} {text}\n")
    return "\n".join(lines)

def gen_vtt(segments):
    lines = ["WEBVTT\n"]
    for s in segments:
        start = s["start"]; end = s["end"]
        def ts(t):
            h = int(t//3600); m = int((t%3600)//60); sec = t%60
            return f"{h:02}:{m:02}:{sec:06.3f}"
        speaker = s.get("speaker") or ""
        text = s["text"].strip()
        lines.append(f"{ts(start)} --> {ts(end)}\n{speaker} {text}\n")
    return "\n".join(lines)

def transcribe_one(
    inp: str,
    language: str,
    need_alignment: bool,
    need_diarization: bool,
    want_srt: bool,
    want_vtt: bool,
):
    t0 = time.time()
    tmp_files = []
    try:
        local = download_url(inp) if is_url(inp) else inp
        if is_url(inp):
            tmp_files.append(local)
        if not os.path.isfile(local):
            raise FileNotFoundError(f"Input not found: {local}")
        norm = ensure_wav_mono16k(local)
        if norm != local:
            tmp_files.append(norm)

        asr_res = ASR_MODEL.transcribe(norm, language=language)

        # Alignment
        if need_alignment and ALIGN_MODEL is not None:
            aligned = whisperx.align(
                asr_res["segments"],
                ALIGN_MODEL,
                ALIGN_META,
                norm,
                device=DEVICE
            )
        else:
            aligned = {"segments": asr_res["segments"], "word_segments": []}

        # Diarization
        if need_diarization and DIAR_PIPE is not None:
            diar_res = DIAR_PIPE(norm)
            merged = whisperx.assign_word_speakers(diar_res, aligned)
        else:
            merged = aligned

        segments_out = [{
            "start": float(s["start"]),
            "end": float(s["end"]),
            "speaker": s.get("speaker"),
            "text": s.get("text", "").strip()
        } for s in merged["segments"]]

        words_out = [{
            "word": w.get("word"),
            "start": float(w["start"]),
            "end": float(w["end"]),
            "speaker": w.get("speaker")
        } for w in merged.get("word_segments", [])]

        resp = {
            "ok": True,
            "input": inp,
            "segments": segments_out,
            "words": words_out,
            "aligned": bool(need_alignment and ALIGN_MODEL is not None),
            "diarized": bool(need_diarization and DIAR_PIPE is not None),
            "elapsed_sec": round(time.time() - t0, 3)
        }
        if want_srt:
            resp["srt"] = gen_srt(segments_out)
        if want_vtt:
            resp["vtt"] = gen_vtt(segments_out)
        return resp
    except Exception as e:
        return {"ok": False, "input": inp, "error": str(e)}
    finally:
        for f in tmp_files:
            try: os.remove(f)
            except: pass

# ------------- FastAPI App -------------

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": WHISPER_MODEL}

@app.post("/transcribe")
def api_transcribe(req: TranscribeRequest):
    lang = req.language or LANGUAGE_DEFAULT
    need_align = req.need_alignment if req.need_alignment is not None else bool(ALIGN_MODEL)
    need_diar  = req.need_diarization if req.need_diarization is not None else (DIAR_PIPE is not None)
    return transcribe_one(
        req.input,
        lang,
        need_align,
        need_diar,
        req.return_srt,
        req.return_vtt
    )

@app.post("/batch_transcribe")
def api_batch(req: BatchRequest):
    lang = req.language or LANGUAGE_DEFAULT
    need_align = req.need_alignment if req.need_alignment is not None else bool(ALIGN_MODEL)
    need_diar  = req.need_diarization if req.need_diarization is not None else (DIAR_PIPE is not None)
    results = []
    for item in req.inputs:
        results.append(
            transcribe_one(
                item, lang, need_align, need_diar,
                req.return_srt, req.return_vtt
            )
        )
    return {"results": results}

ALLOW_SHUTDOWN = False  # set via CLI flag

@app.post("/shutdown")
def shutdown():
    if not ALLOW_SHUTDOWN:
        return {"ok": False, "error": "Shutdown not enabled"}
    log("Shutdown requested …")
    # Use a delayed exit so response is sent
    import threading, sys as _sys
    threading.Timer(0.5, lambda: _sys.exit(0)).start()
    return {"ok": True, "message": "Service exiting"}

# ------------- Startup sequence -------------
def load_models(need_alignment: bool, need_diarization: bool, language: str):
    global ASR_MODEL, ALIGN_MODEL, ALIGN_META, DIAR_PIPE
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={DEVICE}")
    log(f"Loading WhisperX ASR '{WHISPER_MODEL}' (compute_type={COMPUTE_TYPE})")
    ASR_MODEL = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)

    if need_alignment:
        loaded = False
        for cand in ALIGN_MODEL_CANDIDATES:
            try:
                log(f"Trying alignment model: {cand}")
                ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(
                    language_code=language,
                    device=DEVICE,
                    model_name=cand
                )
                log(f"Loaded alignment model: {cand}")
                loaded = True
                break
            except Exception as e:
                warn(f"Alignment model failed: {cand} -> {e}")
        if not loaded:
            warn("All alignment candidates failed; proceeding WITHOUT alignment.")
            ALIGN_MODEL = None
            ALIGN_META = None

    if need_diarization:
        try:
            if HF_TOKEN.startswith("XXXXX"):
                warn("HF_TOKEN placeholder; diarization will fail. Add real token.")
            log("Loading pyannote speaker diarization pipeline (gated)")
            DIAR_PIPE = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
            log("Diarization pipeline loaded.")
        except Exception as e:
            warn(f"Diarization pipeline failed: {e}")
            DIAR_PIPE = None

    # Warmup (OPTIONAL) – small random noise second to trigger kernels
    try:
        import numpy as np, soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sr = 16000
        sf.write(tmp.name, np.zeros(sr, dtype="float32"), sr)
        transcribe_one(tmp.name, language, bool(ALIGN_MODEL), bool(DIAR_PIPE), False, False)
        tmp.close()
        try: os.remove(tmp.name)
        except: pass
        log("Warmup complete.")
    except Exception as e:
        warn(f"Warmup skipped: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--language", default=LANGUAGE_DEFAULT)
    parser.add_argument("--no-alignment", action="store_true", help="Disable alignment load")
    parser.add_argument("--no-diarization", action="store_true", help="Disable diarization load")
    parser.add_argument("--allow-shutdown", action="store_true", help="Allow POST /shutdown")
    args = parser.parse_args()

    global ALLOW_SHUTDOWN
    ALLOW_SHUTDOWN = args.allow_shutdown

    load_models(
        need_alignment=not args.no_alignment,
        need_diarization=not args.no_diarization,
        language=args.language
    )
    log("Models loaded & resident. Starting HTTP service.")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)

if __name__ == "__main__":
    main()
