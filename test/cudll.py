import os
import ctypes

# Test if the DLL is found
dll_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cudnn_ops_infer64_8.dll"
print(os.path.exists(dll_path))  # Should be True

# Try loading the DLL manually
try:
    ctypes.CDLL(dll_path)
    print("Loaded cuDNN DLL successfully")
except Exception as e:
    print("DLL load error:", e)
