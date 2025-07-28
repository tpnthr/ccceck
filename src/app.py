import os
from contextlib import asynccontextmanager

import torch
import uvicorn
import whisperx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from models.transcribe import transcribe_channel
from schemas.transcribe import TranscribeRequest
from utils import logger
from utils.format import group_words
from utils.logger import configure_logging, logger
from utils.sound import split_stereo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "large-v3"
LANGUAGE_DEFAULT = "pl"
ALIGN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
MAX_PAUSE = 1.5  # Max Pause of the fraze
ALLOW_SHUTDOWN = True

ASR_MODEL = whisperx.load_model(WHISPER_MODEL, DEVICE)
ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(language_code="pl", device=DEVICE, model_name=ALIGN_MODEL_NAME)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

# Enable better GPU support
torch.backends.cuda.matmul.allow_tf32 = True

configure_logging()

app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # logger.info(f"{request.method} {request.url.path} - FROM - {request.client.host}")
    try:
        response = await call_next(request)
        logger.info(
            f"FROM - {request.client.host} - STATUS {response.status_code} - {request.method} {request.url.path}"
        )
        return response
    except Exception as e:
        error_message = str(e)
        short_message = (
            error_message.split(":")[1] if ":" in error_message else error_message
        )

        logger.exception(
            f"Request failed: {request.method} {request.url.path} from {request.client.host} - "
            f"Error: {error_message}"
        )

        # Create a JSON response with an appropriate status code
        error_response = JSONResponse(
            content={
                "detail": "Internal Server Error",
                "error": error_message,
                "message": short_message,  # Quick summary of the error
            },
            status_code=500,
        )
        return error_response


@app.get("/")
async def root():
    return {"message": f"Welcome to speech2text v1.0.1!"}


@app.get("/health")
def health():
    return {"success": True, "device": DEVICE, "model": WHISPER_MODEL}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_file = req.input
    tmp_files = []

    try:
        left_path, right_path = split_stereo(audio_file)
        tmp_files.extend([left_path, right_path])

        # left_words = transcribe_channel(left_path, language="pl")
        left_words = transcribe_channel(left_path)
        right_words = transcribe_channel(right_path)

        for w in left_words:
            w["speaker"] = "client"
        for w in right_words:
            w["speaker"] = "agent"

        all_words = left_words + right_words
        grouped_dialogue = group_words(all_words)

        for r in grouped_dialogue:
            r["text"] = " ".join(r["text"])

        return {"success": True, "dialogue": grouped_dialogue}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        for f in tmp_files:
            os.unlink(f)


@app.post("/shutdown")
def shutdown():
    if not ALLOW_SHUTDOWN:
        return {"ok": False, "error": "Shutdown not enabled"}
    logger.info("Shutdown requested …")
    import threading, sys as _sys
    threading.Timer(0.5, lambda: _sys.exit(0)).start()
    return {"success": True, "message": "Service exiting"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
