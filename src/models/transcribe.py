import os
import tempfile
from typing import List, Dict

import whisperx

from app import ASR_MODEL, ALIGN_MODEL, ALIGN_META, DEVICE
from utils.format import gen_srt, gen_vtt
from utils.logging import log, warn


def transcribe_channel(path: str) -> List[Dict]:
    result = ASR_MODEL.transcribe(path)
    aligned = whisperx.align(result["segments"], ALIGN_MODEL, ALIGN_META, path, device=DEVICE)
    return aligned["word_segments"]
