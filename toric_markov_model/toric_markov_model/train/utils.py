"""Training helpers (schedulers, clipping, logging)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def grad_clip(parameters, max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm))
