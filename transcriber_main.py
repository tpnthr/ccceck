
#!/usr/bin/env python3
"""
whisperx_run_once.py
Load WhisperX large-v3 (Polish) + alignment + pyannote diarization ONCE into GPU VRAM.
Process many audio inputs (local files or HTTP/HTTPS URLs) inside the SAME Python process.
Supports: batch args, stdin list, or directory watch, optional JSONL output.

Usage examples:
  python whisperx_run_once.py a.wav b.wav https://host/audio.wav
  python whisperx_run_once.py --stdin            # then paste one URL/path per line (Ctrl+D to run)
  python whisperx_run_once.py --watch /incoming --out transcripts.jsonl
"""

import os, sys, time, json, argparse, pathlib, tempfile, subprocess, shutil, urllib.parse
import requests

# ------------- HARD-CODE YOUR (NEW / ROTATED) HUGGING FACE TOKEN HERE -------------
HF_TOKEN = "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"        # <- replace with real token (needed ONLY for gated pyannote diarization)
# ----------------------------------------------------------------------------------

LANGUAGE       = "pl"        # Force Polish (faster, skips language detection)
WHISPER_MODEL  = "large-v3"
COMPUTE_TYPE   = "float16"   # If VRAM tight, set to "int8" or use smaller model
ALIGN_MODELS   = [
    # Ordered fallbacks (all public). First one that downloads wins.
    "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "facebook/wav2vec2-large-xlsr-53-polish",
    "mbien/wav2vec2-large-xlsr-polish",
]
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

def log(msg):  print(f"[INFO] {msg}", file=sys.stderr, flush=True)
def warn(msg): print(f"[WARN] {msg}", file=sys.stderr, flush=True)
def err(msg):  print(f"[ERR ] {msg}",  file=sys.stderr, flush=True)

# --- Torch / WhisperX imports & model load (ONE TIME) ---
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={DEVICE}")
log(f"Loading WhisperX ASR '{WHISPER_MODEL}' (compute_type={COMPUTE_TYPE})")
asr_model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)

# Alignment: try ordered list; strip bad placeholder token so public models don't 401
if HF_TOKEN in {"XXXXXXXX", "", "XXXXX"}:
    # avoid sending placeholder token for public alignment repos
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

align_model = None
align_meta  = None
for name in ALIGN_MODELS:
    try:
        log(f"Trying alignment model: {name}")
        align_model, align_meta = whisperx.load_align_model(
            language_code=LANGUAGE, device=DEVICE, model_name=name
        )
        log(f"Loaded alignment model: {name}")
        break
    except Exception as e:
        warn(f"Alignment model failed: {name} -> {e}")
if align_model is None:
    warn("All alignment model choices failed; continuing WITHOUT forced alignment "
         "(word-level timings less precise).")

# Diarization (needs gated pyannote token). We pass token directly, not via global env.
log("Loading pyannote speaker diarization pipeline (gated)")
try:
    diar_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    log("Diarization pipeline loaded.")
except Exception as e:
    diar_pipeline = None
    warn(f"Diarization pipeline failed to load: {e}. Continuing without diarization.")

log("Models resident in GPU VRAM (keep this process alive to reuse).")

# ----------------- Helper functions -----------------
def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))

def download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = pathlib.Path(parsed.path).name or "audio.wav"
    if not any(name.lower().endswith(ext) for ext in AUDIO_EXTS):
        name += ".wav"
    fd, tmp_path = tempfile.mkstemp(prefix="whx_", suffix=pathlib.Path(name).suffix)
    os.close(fd)
    log(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tmp_path

def ensure_wav_mono16k(path: str) -> str:
    if shutil.which("ffmpeg") is None:
        return path
    out_path = pathlib.Path(tempfile.gettempdir()) / (pathlib.Path(path).stem + "_16k.wav")
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_path)

def transcribe_one(inp: str):
    t0 = time.time()
    temp_files = []
    try:
        local = download_url(inp) if is_url(inp) else inp
        if is_url(inp):
            temp_files.append(local)
        if not os.path.isfile(local):
            raise FileNotFoundError(f"Missing file: {local}")
        norm = ensure_wav_mono16k(local)
        if norm != local:
            temp_files.append(norm)

        # ASR
        asr_result = asr_model.transcribe(norm, language=LANGUAGE)

        # Alignment (optional)
        if align_model is not None:
            aligned = whisperx.align(asr_result["segments"], align_model, align_meta, norm, device=DEVICE)
        else:
            aligned = {"segments": asr_result["segments"], "word_segments": []}

        # Diarization (optional)
        if diar_pipeline is not None:
            diar = diar_pipeline(norm)
            merged = whisperx.assign_word_speakers(diar, aligned)
        else:
            merged = aligned

        segments_out = [{
            "start": float(s["start"]),
            "end": float(s["end"]),
            "speaker": s.get("speaker"),
            "text": s.get("text", "").strip()
        } for s in merged["segments"]]

        words_out = [{
            "word": w.get("word"),
            "start": float(w["start"]),
            "end": float(w["end"]),
            "speaker": w.get("speaker")
        } for w in merged.get("word_segments", [])]

        return {
            "ok": True,
            "segments": segments_out,
            "words": words_out,
            "elapsed_sec": round(time.time() - t0, 3),
            "aligned": align_model is not None,
            "diarized": diar_pipeline is not None
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        for f in temp_files:
            try: os.remove(f)
            except: pass

def format_segments(segments):
    lines = []
    for s in segments:
        spk = s.get("speaker") or "SPK"
        lines.append(f"[{s['start']:7.2f} - {s['end']:7.2f}] {spk}: {s['text']}")
    return "\n".join(lines)

def handle_list(items, out_jsonl=None):
    out_fh = open(out_jsonl, "a", encoding="utf-8") if out_jsonl else None
    for inp in items:
        inp = inp.strip()
        if not inp:
            continue
        print(f"\n=== {inp} ===")
        res = transcribe_one(inp)
        if res.get("ok"):
            print(format_segments(res["segments"]))
            print(f"(elapsed {res['elapsed_sec']} s; aligned={res['aligned']} diarized={res['diarized']})")
            if out_fh:
                out_fh.write(json.dumps({"input": inp, **res}, ensure_ascii=False) + "\n")
                out_fh.flush()
        else:
            err(res.get("error"))
    if out_fh:
        out_fh.close()

def stdin_mode(out_jsonl=None):
    log("Reading inputs from stdin (one path/URL per line)...")
    items = [line.rstrip("\n") for line in sys.stdin]
    handle_list(items, out_jsonl=out_jsonl)

def watch_mode(directory: str, out_jsonl=None, poll=5):
    log(f"Watching '{directory}' (poll {poll}s) for new audio files...")
    seen = set()
    while True:
        for p in pathlib.Path(directory).glob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p not in seen:
                seen.add(p)
                handle_list([str(p)], out_jsonl=out_jsonl)
        time.sleep(poll)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="Audio files or HTTP/HTTPS URLs")
    ap.add_argument("--stdin", action="store_true", help="Read inputs from stdin")
    ap.add_argument("--watch", metavar="DIR", help="Watch directory for new files")
    ap.add_argument("--out", help="Append JSONL output file")
    ap.add_argument("--poll", type=int, default=5, help="Directory watch poll interval seconds")
    return ap.parse_args()

def main():
    args = parse_args()
    if sum(bool(x) for x in [args.stdin, args.watch]) > 1:
        err("Choose only one of --stdin or --watch.")
        sys.exit(1)
    if args.watch:
        watch_mode(args.watch, out_jsonl=args.out, poll=args.poll)
    elif args.stdin:
        stdin_mode(out_jsonl=args.out)
    else:
        if not args.inputs:
            err("No inputs given. Provide files/URLs or use --stdin/--watch.")
            sys.exit(1)
        handle_list(args.inputs, out_jsonl=args.out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Exiting; GPU memory freed.")
