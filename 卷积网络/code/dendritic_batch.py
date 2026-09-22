from __future__ import annotations


import argparse


import csv


from dataclasses import dataclass


from pathlib import Path


from typing import Iterable, Tuple


import torch


import torch.nn.functional as F


from torch import nn


from torch.utils.data import DataLoader, TensorDataset


from torchvision import datasets, transforms


def dendritic_sigmoid(
    x: torch.Tensor,
    start: float = 16.0,
    saturation: float = 25.0,
    steepness: float = 5.0,
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> torch.Tensor:
    """Shifted sigmoid-like response with configurable output range."""
    z = ((x - start) / (saturation - start)).clamp(0.0, 1.0)
    lower = torch.sigmoid(torch.as_tensor(-0.5 * steepness, dtype=x.dtype, device=x.device))
    upper = torch.sigmoid(torch.as_tensor(0.5 * steepness, dtype=x.dtype, device=x.device))
    y = torch.sigmoid(steepness * (z - 0.5))
    normalized = (y - lower) / (upper - lower)
    return output_min + (output_max - output_min) * normalized


class DendriticConv2d(nn.Module):
    """2D dendritic feature extractor with nonlinear row integration."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        *,
        trainable_device_weights: bool,
        current_gain: float = 3.0,
        row_bias_init: float = 15.5,
        sigmoid_start: float = 16.0,
        sigmoid_saturation: float = 25.0,
        steepness: float = 5.0,
        output_min: float = 0.0,
        output_max: float = 1.0,
        center_row_mix: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.current_gain = current_gain
        self.sigmoid_start = sigmoid_start
        self.sigmoid_saturation = sigmoid_saturation
        self.steepness = steepness
        self.output_min = output_min
        self.output_max = output_max

        init_weights = self._make_initial_weights(
            out_channels, in_channels, kernel_size, seed=seed
        )
        if trainable_device_weights:
            # softplus(weight_param) keeps the device conductance non-negative.
            self.weight_param = nn.Parameter(torch.log(torch.expm1(init_weights)))
            self.register_buffer("fixed_weights", torch.empty(0), persistent=False)
        else:
            self.register_buffer("fixed_weights", init_weights)
            self.weight_param = None

        self.row_bias = nn.Parameter(
            torch.full((out_channels, kernel_size), row_bias_init, dtype=torch.float32),
            requires_grad=trainable_device_weights,
        )
        if center_row_mix and kernel_size == 3:
            row_mix = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32).repeat(
                out_channels, 1
            )
        else:
            row_mix = torch.full((out_channels, kernel_size), 1.0 / kernel_size, dtype=torch.float32)
        self.row_mix = nn.Parameter(row_mix)
        self.output_bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

    @staticmethod
    def _make_initial_weights(
        out_channels: int,
        in_channels: int,
        kernel_size: int,
        *,
        seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        weights = torch.rand(
            out_channels, in_channels, kernel_size, kernel_size, generator=generator
        )
        weights = 0.35 + 0.85 * weights

        if in_channels == 1 and kernel_size == 3 and out_channels >= 8:
            patterns = torch.tensor(
                [
                    [[0.15, 1.00, 0.15], [0.15, 1.00, 0.15], [0.15, 1.00, 0.15]],
                    [[0.15, 0.15, 0.15], [1.00, 1.00, 1.00], [0.15, 0.15, 0.15]],
                    [[1.00, 0.15, 0.15], [0.15, 1.00, 0.15], [0.15, 0.15, 1.00]],
                    [[0.15, 0.15, 1.00], [0.15, 1.00, 0.15], [1.00, 0.15, 0.15]],
                    [[0.20, 0.20, 0.20], [0.20, 1.20, 0.20], [0.20, 0.20, 0.20]],
                    [[0.75, 0.75, 0.75], [0.75, 0.75, 0.75], [0.75, 0.75, 0.75]],
                    [[1.10, 0.20, 1.10], [0.20, 0.20, 0.20], [1.10, 0.20, 1.10]],
                    [[0.20, 1.10, 0.20], [1.10, 0.20, 1.10], [0.20, 1.10, 0.20]],
                ],
                dtype=torch.float32,
            )
            weights[:8, 0] = patterns
        return weights

    @property
    def device_weights(self) -> torch.Tensor:
        if self.weight_param is None:
            return self.fixed_weights
        return F.softplus(self.weight_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        out_height = height + 2 * self.padding - self.kernel_size + 1
        out_width = width + 2 * self.padding - self.kernel_size + 1
        patches = patches.view(
            batch_size,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            out_height * out_width,
        )

        row_sum = torch.einsum("bcrvl,ocrv->borl", patches, self.device_weights)
        drive = self.row_bias.unsqueeze(0).unsqueeze(-1) + self.current_gain * row_sum
        branch = dendritic_sigmoid(
            drive,
            start=self.sigmoid_start,
            saturation=self.sigmoid_saturation,
            steepness=self.steepness,
            output_min=self.output_min,
            output_max=self.output_max,
        )
        output = (branch * self.row_mix.unsqueeze(0).unsqueeze(-1)).sum(dim=2)
        output = output + self.output_bias.view(1, -1, 1)
        return output.view(batch_size, self.out_channels, out_height, out_width)

