#!/usr/bin/env python3
"""Split stereo call recordings into two mono channels and produce a combined
dialogue transcript using WhisperX models from ``transcriber_main.py``.

Each channel is transcribed separately and labeled as ``caller`` (left) and
``client`` (right). Segments from both channels are merged by start time.
"""

import os
import sys
import tempfile
import subprocess
import pathlib
import shutil
import time
from typing import Tuple, List, Dict

import transcriber_main as tm  # reuse loaded models and helpers


def split_stereo(path: str) -> Tuple[str, str]:
    """Return paths to temp mono wavs for left and right channels."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
    left = tempfile.NamedTemporaryFile(suffix="_left.wav", delete=False)
    right = tempfile.NamedTemporaryFile(suffix="_right.wav", delete=False)
    left.close(); right.close()
    cmd_left = [
        "ffmpeg", "-y", "-i", path,
        "-map_channel", "0.0.0",
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        left.name,
    ]
    cmd_right = [
        "ffmpeg", "-y", "-i", path,
        "-map_channel", "0.0.1",
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        right.name,
    ]
    subprocess.run(cmd_left, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd_right, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return left.name, right.name


def transcribe_channel(path: str) -> Dict:
    """Transcribe one mono channel using ASR and optional alignment."""
    norm = tm.ensure_wav_mono16k(path)  # already mono/16k but keeps behaviour
    asr_result = tm.asr_model.transcribe(norm, language=tm.LANGUAGE)
    if tm.align_model is not None:
        aligned = tm.whisperx.align(
            asr_result["segments"], tm.align_model, tm.align_meta, norm, device=tm.DEVICE
        )
    else:
        aligned = {"segments": asr_result["segments"], "word_segments": []}
    return aligned


def transcribe_stereo_dialog(inp: str) -> Dict:
    """Return merged transcript for stereo call."""
    t0 = time.time()
    temp_files: List[str] = []
    try:
        local = tm.download_url(inp) if tm.is_url(inp) else inp
        if tm.is_url(inp):
            temp_files.append(local)
        if not os.path.isfile(local):
            raise FileNotFoundError(f"Missing file: {local}")
        left, right = split_stereo(local)
        temp_files.extend([left, right])

        left_aligned = transcribe_channel(left)
        right_aligned = transcribe_channel(right)

        for s in left_aligned["segments"]:
            s["speaker"] = "caller"
        for w in left_aligned.get("word_segments", []):
            w["speaker"] = "caller"
        for s in right_aligned["segments"]:
            s["speaker"] = "client"
        for w in right_aligned.get("word_segments", []):
            w["speaker"] = "client"

        segments = left_aligned["segments"] + right_aligned["segments"]
        segments.sort(key=lambda s: s["start"])
        words = left_aligned.get("word_segments", []) + right_aligned.get("word_segments", [])
        words.sort(key=lambda w: w["start"])

        return {
            "ok": True,
            "segments": [
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "speaker": s.get("speaker"),
                    "text": s.get("text", "").strip(),
                }
                for s in segments
            ],
            "words": [
                {
                    "word": w.get("word"),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker": w.get("speaker"),
                }
                for w in words
            ],
            "aligned": tm.align_model is not None,
            "diarized": False,
            "elapsed_sec": round(time.time() - t0, 3),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: stereo_dialog.py <stereo_audio_file>")
        sys.exit(1)
    res = transcribe_stereo_dialog(sys.argv[1])
    if res.get("ok"):
        print(tm.format_segments(res["segments"]))
    else:
        tm.err(res.get("error"))
