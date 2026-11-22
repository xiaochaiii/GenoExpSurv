"""Reproducible random seed configuration."""
import random
import numpy as np
import torch

def set_seed(seed: int = 11, threads: int = 2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if threads > 0:
        torch.set_num_threads(threads)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass
