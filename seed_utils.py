import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch, including CUDA.

    Args:
        seed: Random seed used by all libraries.
        deterministic: Whether to enable deterministic PyTorch operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older PyTorch versions do not provide this API.
        pass
    except RuntimeError:
        # Some operations may not support deterministic execution.
        pass
