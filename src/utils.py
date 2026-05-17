import logging
import os
import random
import warnings

import numpy as np
from rich.logging import RichHandler
import torch
from tqdm import TqdmExperimentalWarning


def setup_logging() -> None:
    """Set up logging configuration."""
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                show_level=True,
                show_path=False,
                log_time_format="%H:%M:%S",
                markup=True,
                rich_tracebacks=True,
            )
        ],
    )


def set_seed(seed: int) -> None:
    """Set random seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str) -> torch.device:
    """Get torch device with specific name."""
    if "cuda" in device_name and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(device_name)
