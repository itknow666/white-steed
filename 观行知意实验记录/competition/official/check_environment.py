import sys
import torch, torchvision, numpy, PIL
from timm.utils import ModelEmaV3
from raw_model import RawModel
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("NumPy:", numpy.__version__, "Pillow:", PIL.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Check the NVIDIA driver and the selected GPU instance.")
x = torch.ones(4, device="cuda")
assert x.sum().item() == 4
print("GPU:", torch.cuda.get_device_name(0))
print("Environment ready.")
