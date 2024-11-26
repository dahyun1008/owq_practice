import torch
import torchvision
import torchaudio
import peft

print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"torchaudio version: {torchaudio.__version__}")
print(f"Peft version: {peft.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Accelerate version: {accelerate.__version__}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

