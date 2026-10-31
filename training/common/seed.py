"""
training/common/seed.py
Deterministic Random Seed Management for Sovereign Training.
"""
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic_cuda: bool = False):
    """Sets random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
