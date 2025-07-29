from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import APP_NAME, VERSION, DEVICE, WHISPER_MODEL, ALLOW_SHUTDOWN
from utils import logger
from utils.logger import configure_logging, logger

# Enable better GPU support
# torch.backends.cuda.matmul.allow_tf32 = True

configure_logging()

app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("App starting...")
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


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.get("/")
async def root():
    return {"message": f"Welcome to {APP_NAME} v{VERSION}!"}


@app.get("/health")
def health():
    return {"success": True, "device": DEVICE, "model": WHISPER_MODEL}


# @app.post("/transcribe")
# def transcribe(req: TranscribeRequest):
#     audio_file = req.input
#     tmp_files = []
#
#     try:
#         left_path, right_path = split_stereo(audio_file)
#         tmp_files.extend([left_path, right_path])
#
#         # left_words = transcribe_channel(left_path, language="pl")
#         left_words = transcribe_channel(left_path)
#         right_words = transcribe_channel(right_path)
#
#         for w in left_words:
#             w["speaker"] = "client"
#         for w in right_words:
#             w["speaker"] = "agent"
#
#         all_words = left_words + right_words
#         grouped_dialogue = group_words(all_words)
#
#         for r in grouped_dialogue:
#             r["text"] = " ".join(r["text"])
#
#         return {"success": True, "dialogue": grouped_dialogue}
#
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     finally:
#         for f in tmp_files:
#             os.unlink(f)


@app.post("/shutdown")
def shutdown():
    if not ALLOW_SHUTDOWN:
        return {"success": False, "error": "Shutdown not enabled"}
    logger.info("Shutdown requested …")
    import threading, sys as _sys
    threading.Timer(0.5, lambda: _sys.exit(0)).start()
    return {"success": True, "message": "Service exiting"}


from routes.stereo import router as stereo_router

app.include_router(stereo_router, prefix="/stereo", tags=["Stereo"])

from routes.mono import router as mono_router
app.include_router(mono_router, prefix="/mono", tags=["Mono"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
