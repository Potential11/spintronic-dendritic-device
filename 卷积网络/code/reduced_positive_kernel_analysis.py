from __future__ import annotations


import csv


from dataclasses import dataclass


from pathlib import Path


from typing import Dict, Iterable, List, Tuple


import matplotlib.pyplot as plt


import torch


from torch import nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torchvision import datasets, transforms


from dendritic_batch import DendriticConv2d


def reduced_positive_kernel_bank(out_channels: int = 24) -> torch.Tensor:
    """Return 24 kernels made by repeating 3 positive kernel types."""
    base = torch.tensor(
        [
            [[0.15, 1.00, 0.15], [0.15, 1.00, 0.15], [0.15, 1.00, 0.15]],
            [[0.15, 0.15, 0.15], [1.00, 1.00, 1.00], [0.15, 0.15, 0.15]],
            [[0.20, 0.20, 0.20], [0.20, 1.20, 0.20], [0.20, 0.20, 0.20]],
        ],
        dtype=torch.float32,
    )
    repeats = (out_channels + len(base) - 1) // len(base)
    kernels = base.repeat(repeats, 1, 1)[:out_channels]
    return kernels.unsqueeze(1)

