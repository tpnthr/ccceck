import os
import pathlib
import tempfile
import urllib

import requests

from app import AUDIO_EXTENSIONS


def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = pathlib.Path(parsed.path).name or "audio.wav"
    if not any(name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
        name += ".wav"
    fd, tmp_path = tempfile.mkstemp(prefix="whx_", suffix=pathlib.Path(name).suffix)
    os.close(fd)
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tmp_path
