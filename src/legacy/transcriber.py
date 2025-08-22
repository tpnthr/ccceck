import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Enable better GPU support
torch.backends.cuda.matmul.allow_tf32 = True

# Model setup
model_id = "openai/whisper-large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model {model_id} to {device} with dtype {torch_dtype}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
    chunk_length_s=5,
    stride_length_s=1,
    device=0 if torch.cuda.is_available() else -1,
)

# Divide the file by channels
def divide_by_channels(file):
    print(f"🎧 Dividing stereo file: {file}")
    data, sr = sf.read(file)
    if data.ndim != 2 or data.shape[1] != 2:
        print("❌ Audio must be stereo")
        sys.exit(1)
    base_name = os.path.splitext(os.path.basename(file))[0]
    chan1_path = os.path.join(TEMP_DIR, f"{base_name}_left.wav")
    chan2_path = os.path.join(TEMP_DIR, f"{base_name}_right.wav")
    sf.write(chan1_path, data[:, 0], sr)
    sf.write(chan2_path, data[:, 1], sr)
    return chan1_path, chan2_path


# Get the file from internet
def get_file_from_url(url) -> str:
    print(f"🌐 Downloading file from: {url}")
    filename = os.path.basename(url).split("?")[0]
    if not filename.endswith(".wav"):
        filename += ".wav"
    file_path = os.path.join(TEMP_DIR, filename)

    r = requests.get(url, allow_redirects=True)
    with open(file_path, "wb") as f:
        f.write(r.content)
    return os.path.abspath(file_path)


def transcribe_segments(audio_path):
    print(f"🔊 Transcribing: {audio_path}")
    output = pipe(audio_path)
    return output.get('chunks', [])  # List of dicts: {'text', 'timestamp'}


def merge_transcripts(left_segments, right_segments, left_label="Customer", right_label="Agent"):
    all_segments = []
    for seg in left_segments:
        all_segments.append({
            "timestamp": seg['timestamp'],
            "speaker": left_label,
            "text": seg['text'].strip()
        })
    for seg in right_segments:
        all_segments.append({
            "timestamp": seg['timestamp'],
            "speaker": right_label,
            "text": seg['text'].strip()
        })
    # Sort all segments by start time
    all_segments.sort(key=lambda x: x['timestamp'][0])
    return all_segments


def save_merged_transcript(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start, end = seg['timestamp']
            start_str = f"{start:.2f}" if start is not None else "??"
            end_str = f"{end:.2f}" if end is not None else "??"
            line = f"[{start_str}–{end_str}] {seg['speaker']}: {seg['text']}\n"
            f.write(line)
    print(f"✅ Merged transcript saved: {output_path}")

# Transcribe a file using the pipeline
def transcribe(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Dividing by channels")
    chan1, chan2 = divide_by_channels(file_path)
    print(f"Analysing every channel")
    customer_segments = transcribe_segments(chan1) # left: customer
    agent_segments = transcribe_segments(chan2)    # right: agent
    print(f"Merging transcriptions")
    merged = merge_transcripts(customer_segments, agent_segments, left_label="Customer", right_label="Agent")
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_merged_transcript.txt")
    save_merged_transcript(merged, output_path)

def trascribe_concurrent(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Dividing by channels")
    chan1, chan2 = divide_by_channels(file_path)
    print(f"Analysing channels")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_left = executor.submit(transcribe_segments, chan1)
        future_right = executor.submit(transcribe_segments, chan2)
        customer_segments = future_left.result()
        agent_segments = future_right.result()

    print(f"Merging transcriptions")
    merged = merge_transcripts(customer_segments, agent_segments, left_label="Customer", right_label="Agent")
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_merged_transcript.txt")
    save_merged_transcript(merged, output_path)


def main(argument, path):
    # Get the File from Internet
    if argument == 'url':
        # Get File
        file_path = get_file_from_url(path)
        transcribe(file_path)
    elif argument == 'path':
        if not os.path.isfile(path):
            print(f"❌ File not found: {path}")
            return
        transcribe(path)
    elif argument == '-r':
        if not os.path.isdir(path):
            print(f"❌ Directory not found: {path}")
            return
        print(f"📁 Scanning directory: {path}")
        for file in os.listdir(path):
            if file.lower().endswith((".wav", ".mp3", ".flac")):
                full_path = os.path.join(path, file)
                if os.path.isfile(full_path):
                    print(f"---\n📄 Found audio file: {file}")
                    transcribe(file)


if __name__ == "__main__":
    try:
        mode = sys.argv[1].lower()
        path = sys.argv[2]

        if mode not in {"url", "path", "-r"}:
            raise ValueError(f"Unknown command line argument: {mode}")
        main(mode, path)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("📖 USAGE:\n"
              "  python transcriber.py url <audio_url>\n"
              "  python transcriber.py path <file_path>\n"
              "  python transcriber.py -r <directory_path>\n")
        sys.exit(1)
