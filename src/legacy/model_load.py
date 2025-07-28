import torch
torch.backends.cuda.matmul.allow_tf32 = True  # Huuuge boost to the GPU processing
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "bardsai/whisper-large-v2-pl-v2"

print(f"Loading {model_id}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
if device == "cuda":
    print(f"{model_id} loaded and ready on {device} - GPU")
else:
    print(f"{model_id} loaded and ready on {device} (WARNING - CUDA was not Available)")
processor = AutoProcessor.from_pretrained(model_id)
