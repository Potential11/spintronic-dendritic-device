#!/usr/bin/env python3
"""Generate initialized WR/WR' memory cases and RD-source waveforms."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent.parent / "results"
HERE.mkdir(parents=True, exist_ok=True)
MODEL_DIR = HERE.parent / "memory_model"
sys.path.insert(0, str(MODEL_DIR))
from five_port_position_io import simulate_positions  # noqa: E402

CASES = ("12345", "321", "345", "12333")
STEP_US = 4.0
WR_WIDTH_US = 1.0
WR_EDGE_US = 0.005
M_EDGE_US = 0.08
BIAS_V = 0.0
DB_V = 0.020
RED_OFFSET_V = 0.95


def stepped_pwl(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    times = [0.0]
    wave = [values[0]]
    for index, value in enumerate(values[1:], start=1):
        t = index * STEP_US
        times.extend((t, t + M_EDGE_US))
        wave.extend((values[index - 1], value))
    times.append(len(values) * STEP_US)
    wave.append(values[-1])
    return np.asarray(times), np.asarray(wave)


def pulse_pwl(active_positions: list[int], position: int) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = [(0.0, 5.0 if active_positions[0] == position else 0.0)]
    for index, active in enumerate(active_positions):
        if active != position:
            continue
        start = index * STEP_US
        stop = start + WR_WIDTH_US
        if start == 0:
            points.extend(((stop, 5.0), (stop + WR_EDGE_US, 0.0)))
        else:
            points.extend(((start, 0.0), (start + WR_EDGE_US, 5.0), (stop, 5.0), (stop + WR_EDGE_US, 0.0)))
    points.append((len(active_positions) * STEP_US, 0.0))
    return np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points])


def rd_pwl(beats: int) -> tuple[np.ndarray, np.ndarray]:
    points = [(0.0, 0.0)]
    # Beat 0 is initialization only: WR1 + WR'5, with no following RD.
    for index in range(1, beats):
        start = index * STEP_US + 2.0
        stop = index * STEP_US + 3.0
        points.extend(((start, 0.0), (start + 0.05, 5.0), (stop, 5.0), (stop + 0.05, 0.0)))
    points.append((beats * STEP_US, 0.0))
    return np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points])


def cal_ok_pwl(beats: int) -> tuple[np.ndarray, np.ndarray]:
    """CAL_OK rises after the falling edge of the first post-init RD pulse."""
    first_rd_stop = STEP_US + 3.0
    points = [
        (0.0, 0.0),
        (first_rd_stop, 0.0),
        (first_rd_stop + 0.05, 5.0),
        (beats * STEP_US, 5.0),
    ]
    return np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points])


def direction_pwl(directions: list[str], selected: str) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for index, direction in enumerate(directions, start=1):
        # The first RD after initialization calibrates the previous-value SHA.
        # CAL_OK is still low, so no direction output is allowed in this slot.
        if index == 1:
            continue
        if direction != selected:
            continue
        start = index * STEP_US + 2.0
        stop = index * STEP_US + 3.0
        points.extend(((start, 0.0), (start + 0.05, 5.0), (stop, 5.0), (stop + 0.05, 0.0)))
    points.append(((len(directions) + 1) * STEP_US, 0.0))
    return np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points])


def spice_pairs(times: np.ndarray, values: np.ndarray) -> str:
    return " ".join(f"{t:.6g}u {v:.10g}" for t, v in zip(times, values))


def generate_case(sequence: str) -> dict[str, list[float]]:
    positions = [int(value) for value in sequence]
    # Initialization is simultaneous WR1 and WR'5. Each branch then continues
    # from its own initialized memory state.
    wr_positions = [1] + positions
    wrp_positions = [5] + positions
    wr_blue = list(simulate_positions(wr_positions, use_memory=True, line="blue"))
    # The red model is natively plotted over -0.65..-0.30 V. Shift it into
    # the requested 0.30..0.65 V range before constructing every sum/PWL.
    wr_red_raw = list(simulate_positions(wr_positions, use_memory=True, line="red"))
    wr_red = [value + RED_OFFSET_V for value in wr_red_raw]
    wrp_blue = list(simulate_positions(wrp_positions, use_memory=True, line="blue"))
    wrp_red_raw = list(simulate_positions(wrp_positions, use_memory=True, line="red"))
    wrp_red = [value + RED_OFFSET_V for value in wrp_red_raw]
    m_wr = [blue + red for blue, red in zip(wr_blue, wr_red)]
    m_wrp = [blue + red for blue, red in zip(wrp_blue, wrp_red)]
    m_sum = [wr + wrp for wr, wrp in zip(m_wr, m_wrp)]
    beats = len(m_sum)
    directions = []
    for previous, current in zip(m_sum, m_sum[1:]):
        delta = current - previous
        directions.append("Right" if delta > DB_V else "Left" if delta < -DB_V else "Neutral")

    rows = []
    for index in range(beats):
        is_init = index == 0
        rows.append(
            {
                "beat": index,
                "phase": "INIT" if is_init else f"STEP{index}",
                "position": "" if is_init else positions[index - 1],
                "active_wr": "WR1" if is_init else f"WR{positions[index - 1]}",
                "active_wrp": "WRP5" if is_init else f"WRP{positions[index - 1]}",
                "WR_blue": wr_blue[index],
                "WR_red": wr_red[index],
                "M_wr": m_wr[index],
                "WRP_blue": wrp_blue[index],
                "WRP_red": wrp_red[index],
                "M_wrp": m_wrp[index],
                "M_sum": m_sum[index],
                "rd_high_v": 0.0 if is_init else 5.0,
                "rd_controlled_vout": 0.0 if is_init else m_sum[index],
                "direction": "INIT" if is_init else directions[index - 1],
            }
        )

    stem = f"initialized_wr1_wrp5_{sequence}"
    csv_path = HERE / f"{stem}.csv"
    inc_path = HERE / f"{stem}.inc"
    png_path = HERE / f"{stem}_waveforms_calok.png"
    svg_path = HERE / f"{stem}_waveforms_calok.svg"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    wr_blue_t, wr_blue_y = stepped_pwl(wr_blue)
    wr_red_t, wr_red_y = stepped_pwl(wr_red)
    wrp_blue_t, wrp_blue_y = stepped_pwl(wrp_blue)
    wrp_red_t, wrp_red_y = stepped_pwl(wrp_red)
    mwr_t, mwr_y = stepped_pwl(m_wr)
    mwrp_t, mwrp_y = stepped_pwl(m_wrp)
    sum_t, sum_y = stepped_pwl(m_sum)
    rd_t, rd_y = rd_pwl(beats)
    cal_t, cal_y = cal_ok_pwl(beats)
    source_lines = [
        f"VSOT_WR_BLUE sot_wr_blue 0 PWL({spice_pairs(wr_blue_t, wr_blue_y)})",
        f"VSOT_WR_RED sot_wr_red 0 PWL({spice_pairs(wr_red_t, wr_red_y)})",
        f"VSOT_WRP_BLUE sot_wrp_blue 0 PWL({spice_pairs(wrp_blue_t, wrp_blue_y)})",
        f"VSOT_WRP_RED sot_wrp_red 0 PWL({spice_pairs(wrp_red_t, wrp_red_y)})",
        f"VSOT_M_WR sot_m_wr 0 PWL({spice_pairs(mwr_t, mwr_y)})",
        f"VSOT_M_WRP sot_m_wrp 0 PWL({spice_pairs(mwrp_t, mwrp_y)})",
        f"VSOT_M_SUM sot_m_sum 0 PWL({spice_pairs(sum_t, sum_y)})",
        f"VSOT_RD rd 0 PWL({spice_pairs(rd_t, rd_y)})",
        f"VSOT_CAL_OK cal_ok 0 PWL({spice_pairs(cal_t, cal_y)})",
        "BSOT_RD_OUT sot_rd_out 0 V=V(sot_m_sum)*V(rd)/5",
    ]
    for direction, node in (("Right", "right"), ("Left", "left"), ("Neutral", "neutral")):
        t, y = direction_pwl(directions, direction)
        source_lines.append(f"VSOT_{direction.upper()} {node} 0 PWL({spice_pairs(t, y)})")
    for pos in range(1, 6):
        t, y = pulse_pwl(wr_positions, pos)
        source_lines.append(f"VSOT_WR{pos} wr{pos} 0 PWL({spice_pairs(t, y)})")
    for pos in range(1, 6):
        t, y = pulse_pwl(wrp_positions, pos)
        source_lines.append(f"VSOT_WR{pos}P wr{pos}p 0 PWL({spice_pairs(t, y)})")
    inc_path.write_text(
        f"* INIT: WR1 + WRP5; sequence={sequence}; red offset={RED_OFFSET_V} V\n"
        "* INIT has no RD pulse. sot_rd_out equals sot_m_sum during later RD pulses.\n"
        + "\n".join(source_lines)
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(11, 1, figsize=(13, 16.2), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.92, bottom=0.07, hspace=0.16)
    fig.suptitle(f"Initialized SOT response: WR1 + WR′5, then {sequence}", fontsize=14, fontweight="semibold")
    colors = ("#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f")
    for pos, color in zip(range(1, 6), colors):
        t, y = pulse_pwl(wr_positions, pos)
        axes[0].plot(t, y, color=color, linewidth=1.4)
        t, y = pulse_pwl(wrp_positions, pos)
        axes[1].plot(t, y, color=color, linewidth=1.4)
    axes[0].set_ylabel("WR1–WR5 (V)")
    axes[1].set_ylabel("WR1′–WR5′ (V)")
    axes[2].plot(wr_blue_t, wr_blue_y, color="#1f77b4", linewidth=1.5, label="WR blue")
    axes[2].plot(wr_red_t, wr_red_y, color="#d62728", linewidth=1.5, label="WR red")
    axes[2].plot(mwr_t, mwr_y, color="#222222", linewidth=1.8, label="sum")
    axes[2].set_ylabel("WR memory")
    axes[2].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    axes[3].plot(wrp_blue_t, wrp_blue_y, color="#1f77b4", linewidth=1.5, label="WR′ blue")
    axes[3].plot(wrp_red_t, wrp_red_y, color="#d62728", linewidth=1.5, label="WR′ red")
    axes[3].plot(mwrp_t, mwrp_y, color="#222222", linewidth=1.8, label="sum")
    axes[3].set_ylabel("WR′ memory")
    axes[3].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    axes[4].plot(sum_t, sum_y, color="#222222", linewidth=2.0)
    axes[4].set_ylabel("M_WR + M_WR′ (V)")
    axes[5].plot(rd_t, rd_y, color="#ff7f0e", linewidth=1.7)
    axes[5].set_ylabel("RD (V)")
    rd_out = np.where(rd_y > 0, np.interp(rd_t, sum_t, sum_y) * rd_y / 5.0, 0.0)
    axes[6].plot(rd_t, rd_out, color="#b5bd00", linewidth=1.8)
    axes[6].set_ylabel("RD output (V)")
    axes[7].plot(cal_t, cal_y, color="#2ca02c", linewidth=1.7)
    axes[7].set_ylabel("CAL_OK (V)")
    axes[7].set_ylim(-0.2, 5.3)
    direction_specs = (("Right", "#b2182b"), ("Left", "#1b7837"), ("Neutral", "#1f77b4"))
    for axis, (direction, color) in zip(axes[8:], direction_specs):
        t, y = direction_pwl(directions, direction)
        axis.plot(t, y, color=color, linewidth=1.7)
        axis.set_ylabel(f"{direction} (V)")
        axis.set_ylim(-0.2, 5.3)
    axes[-1].set_xlabel("Time (µs)")
    labels = ["INIT"] + list(sequence)
    ticks = np.arange(beats) * STEP_US + 0.5
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    for axis in axes:
        axis.grid(axis="y", color="#E6E8F0", linewidth=0.7)
        axis.set_xlim(0, beats * STEP_US)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(png_path, dpi=190, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return {
        "wr_blue": wr_blue,
        "wr_red": wr_red,
        "m_wr": m_wr,
        "wrp_blue": wrp_blue,
        "wrp_red": wrp_red,
        "m_wrp": m_wrp,
        "m_sum": m_sum,
    }


def main() -> None:
    for sequence in CASES:
        result = generate_case(sequence)
        print(sequence)
        print("  WR_blue =", [round(value, 9) for value in result["wr_blue"]])
        print("  WR_red  =", [round(value, 9) for value in result["wr_red"]])
        print("  M_wr  =", [round(value, 9) for value in result["m_wr"]])
        print("  WRP_blue=", [round(value, 9) for value in result["wrp_blue"]])
        print("  WRP_red =", [round(value, 9) for value in result["wrp_red"]])
        print("  M_wrp =", [round(value, 9) for value in result["m_wrp"]])
        print("  M_sum =", [round(value, 9) for value in result["m_sum"]])


if __name__ == "__main__":
    main()
