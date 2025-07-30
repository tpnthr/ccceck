import pathlib
import urllib

import requests

from config import AUDIO_EXTENSIONS, DATA_TEMP_DIR, DATA_OUTPUT_DIR, DATA_INPUT_DIR


def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)

    # Try to get the filename from the query parameter
    filename = None
    if 'filename' in qs and qs['filename']:
        # Use the last part after any slashes
        filename = pathlib.Path(qs['filename'][0]).name

    # If not in query, or query param is empty, fall back to parsed path's name
    if not filename:
        filename = pathlib.Path(parsed.path).name or "audio.wav"

    # Ensure it ends with a known audio extension
    if not any(filename.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
        filename += ".wav"

    target_file = DATA_INPUT_DIR / filename
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

def move_to_temp(file_path: str) -> str:
    p = pathlib.Path(file_path)
    target = DATA_TEMP_DIR / p.name
    shutil.move(file_path, target)
    return str(target)

def save_transcription_text(output_text: str, input_file_path: str):
    input_path = pathlib.Path(input_file_path)
    output_file = DATA_OUTPUT_DIR / (input_path.stem + ".txt")
    output_file.write_text(output_text, encoding="utf-8")
    return str(output_file)
