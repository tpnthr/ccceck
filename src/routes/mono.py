from fastapi import APIRouter, HTTPException

from models.transcribe import transcribe_channel
from schemas.transcribe import TranscribeRequest
from utils.file import prepare_audio_input, save_transcription_text
from loguru import logger

# Create the stereo router
router = APIRouter()

@router.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_file = prepare_audio_input(req.input)
    try:
        words = transcribe_channel(audio_file)
        transcript_text = " ".join([w["word"] for w in words])  # or w["text"] depending on your dictionary keys

        output_path = save_transcription_text(transcript_text, audio_file)
        return {"success": True, "transcription": transcript_text, "transcript_file": output_path}
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))