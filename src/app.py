import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

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


@app.post("/shutdown")
async def shutdown():
    if not ALLOW_SHUTDOWN:
        return {"success": False, "error": "Shutdown not enabled"}
    logger.info("Shutdown requested …")

    import threading
    import sys
    def exit_process():
        sys.exit(0)

    threading.Timer(1, exit_process).start()
    return {"success": True, "message": "Service shutting down"}


@app.get("/log/{log_filename}", response_class=PlainTextResponse)
async def view_log(log_filename: str):
    """
    Returns the content of a log file as plain text.
    """
    log_path = os.path.join("logs", f"{log_filename}.log")

    # Ensure the file exists and is inside the LOG_DIR
    if not os.path.isfile(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    with open(log_path, "r", encoding="utf-8") as file:
        log_content = file.read()

    return PlainTextResponse(content=log_content)


@app.delete("/log/{log_filename}")
async def clear_log(log_filename: str):
    """
    Clears the content of a specified log file.
    """
    log_path = os.path.join("logs", f"{log_filename}.log")

    # Ensure the file exists
    if not os.path.isfile(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    # Clear the log file
    with open(log_path, "w", encoding="utf-8") as file:
        file.write("")

    return {
        "success": True,
        "detail": f"Log file '{log_filename}.log' has been cleared.",
    }


from routes.stereo import router as stereo_router

app.include_router(stereo_router, prefix="/stereo", tags=["Stereo"])

from routes.mono import router as mono_router

app.include_router(mono_router, prefix="/mono", tags=["Mono"])

from routes.file import router as file_router

app.include_router(file_router, prefix="/file", tags=["File"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
