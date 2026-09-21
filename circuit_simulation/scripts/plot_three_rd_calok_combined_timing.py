#!/usr/bin/env python3
"""Plot three RD cycles showing RD_END, delayed RD, CAL_OK, and dir_enable."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent.parent / "results"
HERE.mkdir(parents=True, exist_ok=True)
SOURCE_CIRCUIT = HERE / "initialized_full_chain_12345_combined_dir_enable.cir"
TIMING_CIRCUIT = HERE / "combined_enable_three_rd_timing.cir"
CSV = HERE / "combined_enable_three_rd_timing.csv"
PNG = HERE / "combined_enable_three_rd_timing_large_text.png"
SVG = HERE / "combined_enable_three_rd_timing_large_text.svg"

SIGNALS = ("time", "rd", "rd_end", "rd_neutral", "cal_ok", "dir_enable")
TRACKS = (
    ("RD", "rd", "#ff7f0e"),
    ("RD_END", "rd_end", "#9467bd"),
    ("Delayed RD", "rd_neutral", "#7f7f7f"),
    ("CAL_OK", "cal_ok", "#2ca02c"),
    ("Direction enable", "dir_enable", "#008c95"),
)


def build_and_run() -> None:
    text = SOURCE_CIRCUIT.read_text(encoding="utf-8")
    text = re.sub(r"^\.title .*?$", ".title combined_enable_three_rd_timing", text, flags=re.M)
    text = re.sub(r"^\.tran .*?$", ".tran 0.5n 16u", text, flags=re.M)
    text = re.sub(
        r"^wrdata .*?$",
        f"wrdata {CSV.name} time v(rd) v(rd_end) v(rd_neutral) v(cal_ok) v(dir_enable)",
        text,
        flags=re.M,
    )
    TIMING_CIRCUIT.write_text(text, encoding="utf-8")
    subprocess.run(
        ["ngspice", "-b", TIMING_CIRCUIT.name],
        cwd=HERE,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def load() -> dict[str, np.ndarray]:
    raw = np.loadtxt(CSV)
    if raw.ndim != 2 or raw.shape[1] != len(SIGNALS) + 1:
        raise ValueError(f"Unexpected timing CSV shape: {raw.shape}")
    return {name: raw[:, index + 1] for index, name in enumerate(SIGNALS)}


def plot(data: dict[str, np.ndarray]) -> None:
    time_us = data["time"] * 1e6
    fig, axes = plt.subplots(len(TRACKS), 1, figsize=(14.5, 9.2), sharex=True)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.97, bottom=0.13, hspace=0.16)

    for axis, (label, signal, color) in zip(axes, TRACKS):
        voltage = np.clip(data[signal], 0.0, 5.2)
        axis.plot(time_us, voltage, color=color, linewidth=3.5)
        axis.set_ylabel(
            f"{label} (V)", rotation=0, ha="right", va="center",
            labelpad=68, fontsize=20, fontweight="semibold",
        )
        axis.set_ylim(-0.2, 5.3)
        axis.set_yticks((0, 2, 4))
        axis.tick_params(axis="y", labelsize=16)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(1.2)

    axes[-1].set_xlim(5.6, 15.35)
    axes[-1].set_xticks(np.arange(6.0, 15.1, 1.0))
    axes[-1].tick_params(axis="x", labelsize=16)
    axes[-1].set_xlabel("Time (µs)", fontsize=20, fontweight="semibold")
    fig.savefig(PNG, dpi=220, facecolor="white")
    fig.savefig(SVG, facecolor="white")
    plt.close(fig)

    with Image.open(PNG) as image:
        image.convert("RGB").save(PNG)


def main() -> None:
    build_and_run()
    plot(load())
    print(PNG)
    print(SVG)


if __name__ == "__main__":
    main()
