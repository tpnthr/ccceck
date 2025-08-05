import pathlib
import shutil

from fastapi import APIRouter, HTTPException

from models.transcribe import transcribe_channel
from schemas.transcribe import TranscribeRequest
from utils.file import save_transcription_text, DATA_TEMP_DIR, prepare_audio_input
from utils.format import group_words, render_stereo_dialogue_lines
from utils.sound import split_stereo
from loguru import logger

# Create the stereo router
router = APIRouter()

# Assuming TranscribeRequest is defined and imported already
@router.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_file = prepare_audio_input(req.input)
    tmp_files = []
    try:
        left_path, right_path = split_stereo(audio_file)
        left_words = transcribe_channel(left_path, language=req.language, needs_alignment=req.need_alignment)
        right_words = transcribe_channel(right_path, language=req.language, needs_alignment=req.need_alignment)

        if req.label_speakers:
            for w in left_words:
                w["speaker"] = "client"
            for w in right_words:
                w["speaker"] = "agent"
        else:
            for w in left_words:
                w["speaker"] = "speaker1"
            for w in right_words:
                w["speaker"] = "speaker2"

        all_words = left_words + right_words
        grouped_dialogue = group_words(all_words)
        dialog_lines = render_stereo_dialogue_lines(grouped_dialogue)
        dialog_lines = "\n".join(dialog_lines)
        output_path = save_transcription_text(dialog_lines, audio_file)
        return {
            "success": True,
            "json": grouped_dialogue,
            "dialog": dialog_lines,
            "transcript_file": output_path
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


@router.post("/transcribe/word-by-word")
def transcribe_dialog(req: TranscribeRequest):
    audio_file = prepare_audio_input(req.input)
    tmp_files = []
    try:
        left_path, right_path = split_stereo(audio_file)
        left_path = shutil.move(left_path, DATA_TEMP_DIR / pathlib.Path(left_path).name)
        right_path = shutil.move(right_path, DATA_TEMP_DIR / pathlib.Path(right_path).name)
        tmp_files.extend([str(left_path), str(right_path)])

        left_words = transcribe_channel(str(left_path), language=req.language, needs_alignment=req.need_alignment)
        right_words = transcribe_channel(str(right_path), language=req.language, needs_alignment=req.need_alignment)

        if req.label_speakers:
            for w in left_words:
                w["speaker"] = "client"
            for w in right_words:
                w["speaker"] = "agent"
        else:
            for w in left_words:
                w["speaker"] = "speaker1"
            for w in right_words:
                w["speaker"] = "speaker2"

        all_words = left_words + right_words
        grouped_dialogue = group_words(all_words)

        dialog_lines = render_stereo_dialogue_lines(grouped_dialogue)
        dialog_lines = "\n".join(dialog_lines)
        output_path = save_transcription_text(dialog_lines, audio_file)
        return {
            "success": True,
            "json": grouped_dialogue,
            "words": all_words,
            "dialog": dialog_lines,
            "transcript_file": output_path
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