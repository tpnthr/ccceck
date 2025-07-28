import pathlib
import shutil
import subprocess
import tempfile
from typing import Tuple


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
    left = tempfile.NamedTemporaryFile(suffix="_left.wav", delete=False)
    right = tempfile.NamedTemporaryFile(suffix="_right.wav", delete=False)
    left.close(); right.close()
    cmd_left = ["ffmpeg", "-y", "-i", path, "-map_channel", "0.0.0", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", left.name]
    cmd_right = ["ffmpeg", "-y", "-i", path, "-map_channel", "0.0.1", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", right.name]
    subprocess.run(cmd_left, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd_right, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return left.name, right.name