import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The following two lines are often used for determinism,
        # but can impact performance.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False