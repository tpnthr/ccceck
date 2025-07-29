import pathlib
import shutil

from fastapi import APIRouter, HTTPException

from models.transcribe import transcribe_channel
from schemas.transcribe import TranscribeRequest
from utils.file import save_transcription_text, DATA_TEMP_DIR, prepare_audio_input
from utils.format import group_words
from utils.sound import split_stereo
from utils.logger import logger

# Create the stereo router
router = APIRouter()

# Assuming TranscribeRequest is defined and imported already
@router.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_file = prepare_audio_input(req.input)
    tmp_files = []
    try:
        left_path, right_path = split_stereo(audio_file)

        left_words = transcribe_channel(left_path)
        right_words = transcribe_channel(right_path)

        # Add speaker tags
        for w in left_words:
            w["speaker"] = "client"
        for w in right_words:
            w["speaker"] = "agent"

        all_words = left_words + right_words

        # Group words into dialogue
        grouped_dialogue = group_words(all_words)

        # Join grouped text lines for output file
        transcript_text = "\n".join(
            [f'{r["speaker"]}: {" ".join(r["text"])}' for r in grouped_dialogue]
        )

        output_path = save_transcription_text(transcript_text, audio_file)

        return {"success": True, "dialogue": grouped_dialogue, "transcript_file": output_path}

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        for f in tmp_files:
            try:
                pathlib.Path(f).unlink()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="File not found")


@router.post("/transcribe/dialog")
def transcribe_dialog(req: TranscribeRequest):
    audio_file = prepare_audio_input(req.input)
    tmp_files = []
    try:
        left_path, right_path = split_stereo(audio_file)
        left_path = shutil.move(left_path, DATA_TEMP_DIR / pathlib.Path(left_path).name)
        right_path = shutil.move(right_path, DATA_TEMP_DIR / pathlib.Path(right_path).name)
        tmp_files.extend([left_path, right_path])

        left_words = transcribe_channel(left_path)
        right_words = transcribe_channel(right_path)

        for w in left_words:
            w["speaker"] = "client"
        for w in right_words:
            w["speaker"] = "agent"

        all_words = left_words + right_words
        grouped_dialogue = group_words(all_words)

        # Join the entire dialogue text as well
        full_text = " ".join([w["word"] for w in all_words])

        transcript_text = "\n".join(
            [f'{r["speaker"]}: {" ".join(r["text"])}' for r in grouped_dialogue]
        )

        output_path = save_transcription_text(transcript_text, audio_file)

        return {
            "success": True,
            "word_by_word": all_words,
            "dialogue": grouped_dialogue,
            "transcript_text": transcript_text,
            "transcript_file": output_path,
        }
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        for f in tmp_files:
            try:
                pathlib.Path(f).unlink()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="File not found")