from __future__ import annotations


import argparse


import csv


from dataclasses import asdict, dataclass, field


from pathlib import Path


from typing import Dict, Iterable, List, Tuple


import matplotlib.pyplot as plt


import torch


from torch import nn


from torch.utils.data import DataLoader, TensorDataset


from torchvision import datasets, transforms


from dendritic_batch import DendriticConv2d


from kernel_value_range_sweep import kernel_variants, repeat_kernel_bank


from reduced_positive_kernel_analysis import reduced_positive_kernel_bank


@dataclass
class Config:
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    out_dir: Path = Path(__file__).resolve().parent.parent / "results" / "epoch1_batch_accuracy_ratio3"
    batch_size: int = 512
    eval_batch_size: int = 512
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    kernel_variant: str = "ratio3_preserve_sum"
    num_workers: int = 2
    conv_out_channels: int = 24
    classifier_hidden_dims: Tuple[int, ...] = field(default_factory=lambda: (512, 256, 128))
    classifier_dropout: float = 0.10
    num_classes: int = 10
    dendritic_current_gain: float = 3.5
    dendritic_row_bias_init: float = 15.0
    dendritic_sigmoid_start: float = 16.0
    dendritic_sigmoid_saturation: float = 22.0
    dendritic_steepness: float = 3.0
    dendritic_output_min: float = 0.0
    dendritic_output_max: float = 4.0
    dendritic_center_row_mix: bool = True
    dendritic_seed: int = 42


def get_kernel_bank(variant: str, out_channels: int = 24) -> torch.Tensor:
    if variant == "original_ratio8":
        return reduced_positive_kernel_bank(out_channels)
    variants = kernel_variants()
    if variant not in variants:
        raise ValueError(f"Unknown kernel variant: {variant}")
    return repeat_kernel_bank(variants[variant], out_channels)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StrongHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        hidden_dims: Tuple[int, ...],
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        dims = (in_features, *hidden_dims, num_classes)
        layers: List[nn.Module] = [nn.Flatten()]
        for idx in range(len(dims) - 2):
            in_dim = dims[idx]
            out_dim = dims[idx + 1]
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.SiLU(inplace=True),
                ]
            )
            if dropout > 0.0 and idx < len(dims) - 3:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrongFC(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(0.1307).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(0.3081).view(1, 1, 1, 1))
        self.classifier = StrongHead(
            28 * 28,
            hidden_dims=cfg.classifier_hidden_dims,
            dropout=cfg.classifier_dropout,
            num_classes=cfg.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.classifier(x)


class StrongReducedConv(nn.Module):
    def __init__(self, kernel_bank: torch.Tensor, cfg: Config, *, trainable_conv: bool) -> None:
        super().__init__()
        out_channels = cfg.conv_out_channels
        self.conv = nn.Conv2d(1, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv.weight.data.copy_(kernel_bank)
        self.conv.bias.data.zero_()
        for param in self.conv.parameters():
            param.requires_grad = trainable_conv
        self.features = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = StrongHead(
            out_channels * 14 * 14,
            hidden_dims=cfg.classifier_hidden_dims,
            dropout=cfg.classifier_dropout,
            num_classes=cfg.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class StrongDendritic(nn.Module):
    def __init__(self, kernel_bank: torch.Tensor, cfg: Config) -> None:
        super().__init__()
        out_channels = cfg.conv_out_channels
        self.dendritic = DendriticConv2d(
            1,
            out_channels,
            kernel_size=3,
            padding=1,
            trainable_device_weights=False,
            current_gain=cfg.dendritic_current_gain,
            row_bias_init=cfg.dendritic_row_bias_init,
            sigmoid_start=cfg.dendritic_sigmoid_start,
            sigmoid_saturation=cfg.dendritic_sigmoid_saturation,
            steepness=cfg.dendritic_steepness,
            output_min=cfg.dendritic_output_min,
            output_max=cfg.dendritic_output_max,
            center_row_mix=cfg.dendritic_center_row_mix,
            seed=cfg.dendritic_seed,
        )
        self.dendritic.fixed_weights.copy_(kernel_bank)
        self.features = nn.Sequential(
            self.dendritic,
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = StrongHead(
            out_channels * 14 * 14,
            hidden_dims=cfg.classifier_hidden_dims,
            dropout=cfg.classifier_dropout,
            num_classes=cfg.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, cfg: Config) -> nn.Module:
    kernel_bank = get_kernel_bank(cfg.kernel_variant, cfg.conv_out_channels)
    if name == "fc":
        return StrongFC(cfg)
    if name == "trainable_conv":
        return StrongReducedConv(kernel_bank, cfg, trainable_conv=True)
    if name == "dendritic_fixed":
        return StrongDendritic(kernel_bank, cfg)
    raise ValueError(f"Unknown model: {name}")


def get_train_loader(cfg: Config) -> DataLoader:
    train_ds = datasets.MNIST(root=cfg.data_dir, train=True, transform=transforms.ToTensor(), download=True)
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )

