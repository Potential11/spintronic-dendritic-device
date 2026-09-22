from __future__ import annotations


import argparse


import csv


from dataclasses import dataclass


from pathlib import Path


from typing import Dict, Iterable, List, Tuple


import matplotlib.pyplot as plt


import torch


from torch import nn


from torch.utils.data import DataLoader, TensorDataset


from torchvision import datasets, transforms


from dendritic_batch import DendriticConv2d


def original_base_kernels() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.15, 1.00, 0.15], [0.15, 1.00, 0.15], [0.15, 1.00, 0.15]],
            [[0.15, 0.15, 0.15], [1.00, 1.00, 1.00], [0.15, 0.15, 0.15]],
            [[0.20, 0.20, 0.20], [0.20, 1.20, 0.20], [0.20, 0.20, 0.20]],
        ],
        dtype=torch.float32,
    )


def repeat_kernel_bank(base: torch.Tensor, out_channels: int = 24) -> torch.Tensor:
    repeats = (out_channels + len(base) - 1) // len(base)
    return base.repeat(repeats, 1, 1)[:out_channels].unsqueeze(1)


def compress_ratio_preserve_sum(base: torch.Tensor, max_ratio: float = 3.0) -> torch.Tensor:
    """Raise small values until max/min <= max_ratio, then preserve kernel sum."""
    compressed = base.clone()
    for idx in range(compressed.shape[0]):
        kernel = compressed[idx]
        min_allowed = float(kernel.max()) / max_ratio
        clipped = kernel.clamp_min(min_allowed)
        compressed[idx] = clipped * (kernel.sum() / clipped.sum())
    return compressed


def compress_ratio_preserve_max(base: torch.Tensor, max_ratio: float = 3.0) -> torch.Tensor:
    """Raise small values until max/min <= max_ratio while keeping the largest value."""
    compressed = base.clone()
    for idx in range(compressed.shape[0]):
        kernel = compressed[idx]
        min_allowed = float(kernel.max()) / max_ratio
        compressed[idx] = kernel.clamp_min(min_allowed)
    return compressed


def device_range_kernel(base: torch.Tensor, low: float = 0.35, high: float = 1.05) -> torch.Tensor:
    """Map each kernel linearly into a fixed 3x device range."""
    mapped = base.clone()
    for idx in range(mapped.shape[0]):
        kernel = mapped[idx]
        span = kernel.max() - kernel.min()
        if float(span) == 0.0:
            mapped[idx] = torch.full_like(kernel, low)
        else:
            mapped[idx] = low + (kernel - kernel.min()) / span * (high - low)
    return mapped


def scaled_kernel(base: torch.Tensor, scale: float) -> torch.Tensor:
    return base * scale


def kernel_variants() -> Dict[str, torch.Tensor]:
    base = original_base_kernels()
    return {
        "original_ratio8": base,
        "scaled_0p75_ratio8": scaled_kernel(base, 0.75),
        "scaled_1p25_ratio8": scaled_kernel(base, 1.25),
        "ratio3_preserve_sum": compress_ratio_preserve_sum(base, 3.0),
        "ratio3_preserve_max": compress_ratio_preserve_max(base, 3.0),
        "ratio3_device_0p35_1p05": device_range_kernel(base, 0.35, 1.05),
    }

