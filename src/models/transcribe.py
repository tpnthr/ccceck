import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import whisperx

from config import ALIGN_MODEL, ALIGN_META, DEVICE, ASR_MODEL
from loguru import logger


# def transcribe_channel(path: str) -> List[Dict]:
#     result = ASR_MODEL.transcribe(path)
#     audio_np = whisperx.load_audio(path)  # load audio as np.ndarray for alignment
#     logger.info("Type: {}, Shape: {}", type(audio_np), audio_np.shape if hasattr(audio_np, "shape") else None)
#     aligned = whisperx.align(result["segments"], ALIGN_MODEL, ALIGN_META, audio_np, device=DEVICE)
#     return aligned["word_segments"]

def transcribe_channel(path: str, needs_alignment: bool = True, language: str = "en") -> List[Dict]:
    # Pass language as an argument if your model supports it (check your model docs)
    result = ASR_MODEL.transcribe(path, language=language)
    if needs_alignment:
        audio_np = whisperx.load_audio(path)  # load audio as np.ndarray for alignment
        logger.info("Type: {}, Shape: {}", type(audio_np), audio_np.shape if hasattr(audio_np, "shape") else None)
        aligned = whisperx.align(result["segments"], ALIGN_MODEL, ALIGN_META, audio_np, device=DEVICE)
        return aligned["word_segments"]
    else:
        # Return just segment-level data if alignment isn't needed
        return result["segments"]

def parallel_transcribe(paths, needs_alignment=True, language="pl"):
    results = []
    with ThreadPoolExecutor(max_workers=len(paths)) as executor:
        futures = [executor.submit(transcribe_channel, path, needs_alignment, language) for path in paths]
        for future in futures:
            results.append(future.result())
    return results