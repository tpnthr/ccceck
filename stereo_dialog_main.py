#!/usr/bin/env python3
"""stereo_dialog_main.py
Batch transcription of stereo call recordings using WhisperX.
Models are loaded once at startup and reused for all inputs.
"""

import os
import sys
import time
import json
import argparse
import pathlib

import stereo_dialog_start as sd

# Load models once
sd.load_models(need_alignment=True, language=sd.LANGUAGE_DEFAULT)


def format_segments(segments):
    lines = []
    for s in segments:
        spk = s.get("speaker") or "SPK"
        lines.append(f"[{s['start']:7.2f} - {s['end']:7.2f}] {spk}: {s['text']}")
    return "\n".join(lines)


def transcribe_one(inp: str):
    res = sd.transcribe_stereo(
        inp,
        language=sd.LANGUAGE_DEFAULT,
        need_alignment=sd.ALIGN_MODEL is not None,
        want_srt=False,
        want_vtt=False,
    )
    return res


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
            print(
                f"(elapsed {res['elapsed_sec']} s; aligned={res['aligned']})"
            )
            if out_fh:
                out_fh.write(json.dumps({"input": inp, **res}, ensure_ascii=False) + "\n")
                out_fh.flush()
        else:
            sd.err(res.get("error"))
    if out_fh:
        out_fh.close()


def stdin_mode(out_jsonl=None):
    sd.log("Reading inputs from stdin (one path/URL per line)...")
    items = [line.rstrip("\n") for line in sys.stdin]
    handle_list(items, out_jsonl=out_jsonl)


def watch_mode(directory: str, out_jsonl=None, poll=5):
    sd.log(f"Watching '{directory}' (poll {poll}s) for new audio files...")
    seen = set()
    while True:
        for p in pathlib.Path(directory).glob("*"):
            if p.is_file() and p.suffix.lower() in sd.AUDIO_EXTS and p not in seen:
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
        sd.err("Choose only one of --stdin or --watch.")
        sys.exit(1)
    if args.watch:
        watch_mode(args.watch, out_jsonl=args.out, poll=args.poll)
    elif args.stdin:
        stdin_mode(out_jsonl=args.out)
    else:
        if not args.inputs:
            sd.err("No inputs given. Provide files/URLs or use --stdin/--watch.")
            sys.exit(1)
        handle_list(args.inputs, out_jsonl=args.out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sd.log("Exiting; GPU memory freed.")
