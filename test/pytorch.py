import torch
# Check PyTorch version
print("PyTorch version:", torch.__version__)
# Check CUDA version PyTorch was built/installed with
print("PyTorch linked CUDA version:", torch.version.cuda)
# Check if a GPU is available and which device(s) are detected
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
