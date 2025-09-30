# test_env.py
import torch
import numpy as np
import pandas as pd

print("PyTorch version:", torch.__version__)
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)

if torch.cuda.is_available():
    print("CUDA disponible ✅")
    print("Nombre GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA NO disponible ❌")
