import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Tuple

from config import DATA_TEMP_DIR


def ensure_wav_mono16k(path: str) -> str:
    if shutil.which("ffmpeg") is None:
        return path
    out_path = pathlib.Path(tempfile.gettempdir()) / (pathlib.Path(path).stem + "_16k.wav")
    if out_path.exists():
        return str(out_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_path)


def split_stereo(path: str) -> Tuple[str, str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")

    left = tempfile.NamedTemporaryFile(dir=str(DATA_TEMP_DIR), suffix="_left.wav", delete=False)
    right = tempfile.NamedTemporaryFile(dir=str(DATA_TEMP_DIR), suffix="_right.wav", delete=False)
    left.close()
    right.close()

    # Use pan audio filter to extract left and right channels to mono, 16kHz, 16-bit files
    cmd_left = [
        "ffmpeg",
        "-y",
        "-i", path,
        "-af", "pan=mono|c0=c0",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        left.name
    ]
    cmd_right = [
        "ffmpeg",
        "-y",
        "-i", path,
        "-af", "pan=mono|c0=c1",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        right.name
    ]

    proc_left = subprocess.run(cmd_left, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc_right = subprocess.run(cmd_right, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc_left.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for left channel: {proc_left.stderr.decode()}")

    if proc_right.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for right channel: {proc_right.stderr.decode()}")

    # Check generated file sizes
    if os.path.getsize(left.name) == 0:
        raise RuntimeError("Left channel output file is empty after splitting")

    if os.path.getsize(right.name) == 0:
        raise RuntimeError("Right channel output file is empty after splitting")

    return left.name, right.name
