"""Per-batch noisy accuracy during epoch 1 for the strong ratio-3 experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from strong_frontend_noise_experiment import Config, build_model, get_train_loader, set_seed

# 可调参数
DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # MNIST 数据集缓存目录。
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "epoch1_batch_accuracy_ratio3"  # PNG 和 CSV 输出目录。
KERNEL_VARIANT = "ratio3_preserve_sum"  # 卷积/树突前端使用的卷积核版本。
TRAIN_EPOCHS = 1  # 训练轮数；这个脚本默认只看第 1 个 epoch。
TRAIN_BATCH_SIZE = 512 # 训练时的 batch 大小。
TRAIN_LR = 1e-3  # AdamW 学习率。
TRAIN_WEIGHT_DECAY = 1e-4  # AdamW 权重衰减。
SEED = 42  # 随机种子。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 训练和评估设备。
NUM_WORKERS = 2  # 训练 DataLoader 的 worker 数量。
CONV_OUT_CHANNELS = 24  # 卷积/树突前端输出通道数。
CLASSIFIER_HIDDEN_DIMS = (512, 256, 128)  # 分类头每一层隐藏维度。
CLASSIFIER_DROPOUT = 0.10  # 分类头 dropout；设为 0 可关闭。
NUM_CLASSES = 10  # 分类类别数，MNIST 默认 10。
DENDRITIC_CURRENT_GAIN = 3.5  # 树突行求和后的电流增益 g。
DENDRITIC_ROW_BIAS_INIT = 15.0  # 树突分支偏置初值 b。
DENDRITIC_SIGMOID_START = 16.0  # 类 sigmoid 开始明显抬升的位置。
DENDRITIC_SIGMOID_SATURATION = 22.0  # 类 sigmoid 接近饱和的位置。
DENDRITIC_STEEPNESS = 3.0  # 类 sigmoid 过渡陡峭程度。
DENDRITIC_OUTPUT_MIN = 0.0  # 树突分支输出下界。
DENDRITIC_OUTPUT_MAX = 4.0  # 树突分支输出上界。
DENDRITIC_CENTER_ROW_MIX = True  # 是否使用 [0.25, 0.50, 0.25] 的行混合。
DENDRITIC_SEED = 42  # 树突器件内部初始化随机种子。
EVAL_SUBSET = 500  # 固定测试子集大小，用于每次 batch 后重复评估。
EVAL_BATCH_SIZE = 500  # 评估固定测试子集时的 batch 大小。
EVAL_EVERY = 3  # 每隔多少个训练 batch 评估一次。
PER_BATCH_GAUSSIAN_LEVELS = [0.00, 0.20, 0.25]  # per-batch 图里使用的 Gaussian 噪声强度。
PER_BATCH_SALT_PEPPER_LEVELS = [0.00, 0.10, 0.20]  # per-batch 图里使用的椒盐噪声强度。
SWEEP_GAUSSIAN_LEVELS =[0.00,0.025, 0.05,0.075, 0.10,0.125, 0.15,0.175, 0.20,0.225, 0.25, ]  # 第三张扫描图使用的 Gaussian 噪声强度。
MODEL_ORDER = ["fc", "trainable_conv", "dendritic_fixed"]  # 训练和绘图时的模型顺序。
REPRESENTATIVE_PANELS = [  # 汇总图里显示的噪声面板。
    ("gaussian", 0.00, "Clean"),
    ("gaussian", 0.20, "Gaussian 0.20"),
    ("gaussian", 0.25, "Gaussian 0.25"),
    ("salt_pepper", 0.10, "Salt-pepper 0.10"),
    ("salt_pepper", 0.20, "Salt-pepper 0.20"),
]
SUMMARY_GAUSSIAN_LEVEL = 0.25  # 单独第二张训练曲线图对应的 Gaussian 强度。
PRINT_EVERY = 60  # 训练过程中每隔多少个 batch 打印一次中间结果。
PLOT_DPI = 200  # 导出 PNG 分辨率。
GENERATE_PER_BATCH_PLOTS = True  # 是否额外生成旧版 per-batch 图。
GENERATE_SUMMARY_PLOTS = True  # 是否生成你要的三张摘要图。


@dataclass
class BatchEvalConfig:
    out_dir: Path = OUT_DIR
    eval_subset: int = EVAL_SUBSET
    eval_batch_size: int = EVAL_BATCH_SIZE
    eval_every: int = EVAL_EVERY


MODEL_LABELS = {
    "fc": "Strong FC",
    "trainable_conv": "Trainable conv",
    "dendritic_fixed": "Dendritic fixed [0,4]",
}


def make_experiment_config() -> Config:
    return Config(
        data_dir=DATA_DIR,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=TRAIN_LR,
        weight_decay=TRAIN_WEIGHT_DECAY,
        seed=SEED,
        device=DEVICE,
        kernel_variant=KERNEL_VARIANT,
        num_workers=NUM_WORKERS,
        conv_out_channels=CONV_OUT_CHANNELS,
        classifier_hidden_dims=CLASSIFIER_HIDDEN_DIMS,
        classifier_dropout=CLASSIFIER_DROPOUT,
        num_classes=NUM_CLASSES,
        dendritic_current_gain=DENDRITIC_CURRENT_GAIN,
        dendritic_row_bias_init=DENDRITIC_ROW_BIAS_INIT,
        dendritic_sigmoid_start=DENDRITIC_SIGMOID_START,
        dendritic_sigmoid_saturation=DENDRITIC_SIGMOID_SATURATION,
        dendritic_steepness=DENDRITIC_STEEPNESS,
        dendritic_output_min=DENDRITIC_OUTPUT_MIN,
        dendritic_output_max=DENDRITIC_OUTPUT_MAX,
        dendritic_center_row_mix=DENDRITIC_CENTER_ROW_MIX,
        dendritic_seed=DENDRITIC_SEED,
    )


def make_noise_loaders(
    cfg: Config,
    batch_cfg: BatchEvalConfig,
) -> Dict[Tuple[str, float], DataLoader]:
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, transform=transforms.ToTensor(), download=True)
    subset = Subset(test_ds, list(range(min(batch_cfg.eval_subset, len(test_ds)))))
    images = torch.stack([subset[idx][0] for idx in range(len(subset))])
    labels = torch.tensor([subset[idx][1] for idx in range(len(subset))], dtype=torch.long)

    loaders: Dict[Tuple[str, float], DataLoader] = {}
    for level in PER_BATCH_GAUSSIAN_LEVELS:
        if level == 0.0:
            noisy = images
        else:
            generator = torch.Generator().manual_seed(cfg.seed + 1000 + int(round(level * 1000)))
            noisy = (images + torch.randn(images.shape, generator=generator) * level).clamp(0.0, 1.0)
        loaders[("gaussian", level)] = DataLoader(
            TensorDataset(noisy, labels),
            batch_size=batch_cfg.eval_batch_size,
            shuffle=False,
        )

    for level in PER_BATCH_SALT_PEPPER_LEVELS:
        if level == 0.0:
            noisy = images
        else:
            generator = torch.Generator().manual_seed(cfg.seed + 2000 + int(round(level * 1000)))
            mask = torch.rand(images.shape, generator=generator)
            noisy = images.clone()
            noisy[mask < (level / 2.0)] = 1.0
            noisy[(mask >= (level / 2.0)) & (mask < level)] = 0.0
        loaders[("salt_pepper", level)] = DataLoader(
            TensorDataset(noisy, labels),
            batch_size=batch_cfg.eval_batch_size,
            shuffle=False,
        )
    return loaders


def make_gaussian_sweep_loaders(
    cfg: Config,
    batch_cfg: BatchEvalConfig,
) -> Dict[float, DataLoader]:
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, transform=transforms.ToTensor(), download=True)
    subset = Subset(test_ds, list(range(min(batch_cfg.eval_subset, len(test_ds)))))
    images = torch.stack([subset[idx][0] for idx in range(len(subset))])
    labels = torch.tensor([subset[idx][1] for idx in range(len(subset))], dtype=torch.long)
    loaders: Dict[float, DataLoader] = {}
    for level in SWEEP_GAUSSIAN_LEVELS:
        if level == 0.0:
            noisy = images
        else:
            generator = torch.Generator().manual_seed(cfg.seed + 3000 + int(round(level * 1000)))
            noisy = (images + torch.randn(images.shape, generator=generator) * level).clamp(0.0, 1.0)
        loaders[level] = DataLoader(
            TensorDataset(noisy, labels),
            batch_size=batch_cfg.eval_batch_size,
            shuffle=False,
        )
    return loaders


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if was_training:
        model.train()
    return correct / total


def evaluate_all(
    model_name: str,
    model: nn.Module,
    batch_index: int,
    noise_loaders: Dict[Tuple[str, float], DataLoader],
    device: torch.device,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for (noise_type, level), loader in noise_loaders.items():
        acc = evaluate(model, loader, device)
        rows.append(
            {
                "model": model_name,
                "batch": batch_index,
                "noise_type": noise_type,
                "level": level,
                "accuracy": acc,
            }
        )
    return rows


def train_epoch1_per_batch(
    model_name: str,
    cfg: Config,
    batch_cfg: BatchEvalConfig,
    noise_loaders: Dict[Tuple[str, float], DataLoader],
) -> Tuple[List[Dict[str, float | int | str]], nn.Module]:
    device = torch.device(cfg.device)
    model = build_model(model_name, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    train_loader = get_train_loader(cfg)

    rows = evaluate_all(model_name, model, 0, noise_loaders, device)
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % batch_cfg.eval_every == 0:
            rows.extend(evaluate_all(model_name, model, batch_idx, noise_loaders, device))

        if PRINT_EVERY > 0 and batch_idx % PRINT_EVERY == 0:
            g20 = next(
                float(row["accuracy"])
                for row in reversed(rows)
                if row["noise_type"] == "gaussian" and float(row["level"]) == 0.20
            )
            sp10 = next(
                float(row["accuracy"])
                for row in reversed(rows)
                if row["noise_type"] == "salt_pepper" and float(row["level"]) == 0.10
            )
            print(
                f"[{model_name}] batch {batch_idx:03d} "
                f"gaussian0.20={g20 * 100:.2f}% salt0.10={sp10 * 100:.2f}%"
            )
    return rows, model


def plot_noise_type(
    rows: List[Dict[str, float | int | str]],
    out_dir: Path,
    noise_type: str,
    levels: Iterable[float],
) -> Path:
    levels = list(levels)
    n_cols = len(levels)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.3 * n_cols, 4.6), sharey=True)
    if n_cols == 1:
        axes = [axes]
    for ax, level in zip(axes, levels):
        for model_name, label in MODEL_LABELS.items():
            points = [
                row
                for row in rows
                if row["model"] == model_name
                and row["noise_type"] == noise_type
                and float(row["level"]) == level
            ]
            points.sort(key=lambda row: int(row["batch"]))
            xs = [int(row["batch"]) for row in points]
            ys = [float(row["accuracy"]) * 100 for row in points]
            ax.plot(xs, ys, linewidth=1.4, label=label)
            ax.scatter(xs, ys, s=8)
        ax.set_title(f"{noise_type}={level:.2f}")
        ax.set_xlabel("Training batch")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel(f"Accuracy on fixed {EVAL_SUBSET}-image subset (%)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Epoch 1 per-batch accuracy: {noise_type.replace('_', '-')}")
    fig.tight_layout()
    path = out_dir / f"{noise_type}_per_batch_epoch1.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    return path


def plot_combined(rows: List[Dict[str, float | int | str]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(REPRESENTATIVE_PANELS), figsize=(21, 4.6), sharey=True)
    for ax, (noise_type, level, title) in zip(axes, REPRESENTATIVE_PANELS):
        for model_name, label in MODEL_LABELS.items():
            points = [
                row
                for row in rows
                if row["model"] == model_name
                and row["noise_type"] == noise_type
                and float(row["level"]) == level
            ]
            points.sort(key=lambda row: int(row["batch"]))
            xs = [int(row["batch"]) for row in points]
            ys = [float(row["accuracy"]) * 100 for row in points]
            ax.plot(xs, ys, linewidth=1.4, label=label)
            ax.scatter(xs, ys, s=8)
        ax.set_title(title)
        ax.set_xlabel("Training batch")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel(f"Accuracy on fixed {EVAL_SUBSET}-image subset (%)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Epoch 1 per-batch accuracy with representative noise levels")
    fig.tight_layout()
    path = out_dir / "epoch1_per_batch_representative_noise.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    return path


def plot_summary_curve(
    rows: List[Dict[str, float | int | str]],
    out_dir: Path,
    noise_type: str,
    level: float,
    title: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for model_name, label in MODEL_LABELS.items():
        points = [
            row
            for row in rows
            if row["model"] == model_name
            and row["noise_type"] == noise_type
            and float(row["level"]) == level
        ]
        points.sort(key=lambda row: int(row["batch"]))
        xs = [int(row["batch"]) for row in points]
        ys = [float(row["accuracy"]) * 100 for row in points]
        ax.plot(xs, ys, linewidth=2.0, marker="o", markersize=3.5, label=label)
    ax.set_xlabel("训练 batch")
    ax.set_ylabel(f"固定 {EVAL_SUBSET} 张测试子集准确率 (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    return path


def plot_epoch1_noise_sweep(
    rows: List[Dict[str, float | int | str]],
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for model_name, label in MODEL_LABELS.items():
        points = [row for row in rows if row["model"] == model_name]
        points.sort(key=lambda row: float(row["noise_std"]))
        xs = [float(row["noise_std"]) for row in points]
        ys = [float(row["accuracy"]) * 100 for row in points]
        ax.plot(xs, ys, linewidth=2.0, marker="o", markersize=4.0, label=label)
    ax.set_xlabel("Gaussian 噪声标准差")
    ax.set_ylabel(f"固定 {EVAL_SUBSET} 张测试子集准确率 (%)")
    ax.set_title("第 1 个 epoch 结束后：准确率随 Gaussian 噪声变化")
    ax.set_xticks(SWEEP_GAUSSIAN_LEVELS)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "epoch1_noise_sweep.png"
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    return path


def main() -> None:
    batch_cfg = BatchEvalConfig()
    batch_cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_experiment_config()
    set_seed(cfg.seed)
    noise_loaders = make_noise_loaders(cfg, batch_cfg)
    gaussian_sweep_loaders = make_gaussian_sweep_loaders(cfg, batch_cfg)

    rows: List[Dict[str, float | int | str]] = []
    sweep_rows: List[Dict[str, float | int | str]] = []
    for model_name in MODEL_ORDER:
        set_seed(cfg.seed)
        model_rows, model = train_epoch1_per_batch(model_name, cfg, batch_cfg, noise_loaders)
        rows.extend(model_rows)
        device = torch.device(cfg.device)
        for level, loader in gaussian_sweep_loaders.items():
            acc = evaluate(model, loader, device)
            sweep_rows.append(
                {
                    "model": model_name,
                    "noise_std": level,
                    "accuracy": acc,
                }
            )

    csv_path = batch_cfg.out_dir / "epoch1_per_batch_noise_accuracy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "batch", "noise_type", "level", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    sweep_csv_path = batch_cfg.out_dir / "epoch1_noise_sweep.csv"
    with sweep_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "noise_std", "accuracy"])
        writer.writeheader()
        writer.writerows(sweep_rows)

    if GENERATE_PER_BATCH_PLOTS:
        gaussian_plot = plot_noise_type(rows, batch_cfg.out_dir, "gaussian", PER_BATCH_GAUSSIAN_LEVELS)
        salt_plot = plot_noise_type(rows, batch_cfg.out_dir, "salt_pepper", PER_BATCH_SALT_PEPPER_LEVELS)
        combined_plot = plot_combined(rows, batch_cfg.out_dir)
        print(f"Saved plot to {gaussian_plot}")
        print(f"Saved plot to {salt_plot}")
        print(f"Saved plot to {combined_plot}")

    if GENERATE_SUMMARY_PLOTS:
        clean_curve = plot_summary_curve(
            rows,
            batch_cfg.out_dir,
            "gaussian",
            0.00,
            "第 1 个 epoch：无噪声准确率随训练变化",
            "epoch1_clean_curve.png",
        )
        g20_curve = plot_summary_curve(
            rows,
            batch_cfg.out_dir,
            "gaussian",
            SUMMARY_GAUSSIAN_LEVEL,
            f"第 1 个 epoch：Gaussian {SUMMARY_GAUSSIAN_LEVEL:.2f} 准确率随训练变化",
            "epoch1_gaussian_0p20_curve.png",
        )
        sweep_plot = plot_epoch1_noise_sweep(sweep_rows, batch_cfg.out_dir)
        print(f"Saved plot to {clean_curve}")
        print(f"Saved plot to {g20_curve}")
        print(f"Saved plot to {sweep_plot}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved CSV to {sweep_csv_path}")


if __name__ == "__main__":
    main()
