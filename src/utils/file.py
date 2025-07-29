import pathlib
import urllib

import requests

from config import AUDIO_EXTENSIONS


def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = pathlib.Path(parsed.path).name or "audio.wav"

    if not any(name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
        name += ".wav"

    target_file = DATA_INPUT_DIR / name

    # If already exists, optionally skip or overwrite (your choice, here we overwrite)
    # Could add logic to check and rename if file exists (adding suffix) if needed

    print(f"Downloading {url} into {target_file}")

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(target_file, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    return str(target_file)


DATA_INPUT_DIR = pathlib.Path("data/input")
DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_audio_input(input_str: str) -> str:
    if is_url(input_str):
        # Download file into data/input and return path
        local_path = download_url(input_str)
        # Move from temp location to data/input
        target_path = DATA_INPUT_DIR / pathlib.Path(local_path).name
        pathlib.Path(local_path).rename(target_path)
        return str(target_path)
    else:
        return input_str

import shutil

DATA_TEMP_DIR = pathlib.Path("data/temp")
DATA_TEMP_DIR.mkdir(parents=True, exist_ok=True)

def move_to_temp(file_path: str) -> str:
    p = pathlib.Path(file_path)
    target = DATA_TEMP_DIR / p.name
    shutil.move(file_path, target)
    return str(target)

DATA_OUTPUT_DIR = pathlib.Path("data/output")
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_transcription_text(output_text: str, input_file_path: str):
    input_path = pathlib.Path(input_file_path)
    output_file = DATA_OUTPUT_DIR / (input_path.stem + ".txt")
    output_file.write_text(output_text, encoding="utf-8")
    return str(output_file)
