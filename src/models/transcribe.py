import os
import tempfile
from typing import List, Dict

import whisperx

from config import ALIGN_MODEL, ALIGN_META, DEVICE, ASR_MODEL
from loguru import logger


def transcribe_channel(path: str) -> List[Dict]:
    result = ASR_MODEL.transcribe(path)
    audio_np = whisperx.load_audio(path)  # load audio as np.ndarray for alignment
    logger.info("Type: {}, Shape: {}", type(audio_np), audio_np.shape if hasattr(audio_np, "shape") else None)
    aligned = whisperx.align(result["segments"], ALIGN_MODEL, ALIGN_META, audio_np, device=DEVICE)
    return aligned["word_segments"]
