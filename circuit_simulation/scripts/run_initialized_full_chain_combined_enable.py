#!/usr/bin/env python3
"""Build cases with a shared delayed-RD-and-CAL_OK direction enable."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent.parent / "results"
HERE.mkdir(parents=True, exist_ok=True)
ROOT = HERE.parent
TEMPLATE = ROOT / "templates/full_chain_ad783_like_opamp_sha_rd_neutral_2inv_calok_behavioral_wr_logic.cir"
CASES = ("12345", "321", "345", "12333")
SEQUENCE = CASES[0]
POSITIONS = [1] + [int(value) for value in SEQUENCE]
STEP_US = 4.0
STOP_US = len(POSITIONS) * STEP_US
CIRCUIT = HERE / "unused.cir"
CSV = HERE / "unused.csv"
PNG = HERE / "unused.png"
SVG = HERE / "unused.svg"
FULLCHAIN_INC = HERE / "unused.inc"

COLS = [
    "time", "clk", "wr1", "wr2", "wr3", "wr4", "wr5",
    "wr1p", "wr2p", "wr3p", "wr4p", "wr5p",
    "m_wr", "m_wrp", "m_sum", "init", "rd_src", "rd", "cal_ok", "dir_enable", "vout",
    "v_cur", "v_prev", "vdiff", "right_valid",
    "left_valid", "neutral",
]


def pulse_pwl(windows: list[tuple[float, float]], *, invert: bool = False, edge: float = 0.005) -> str:
    low, high = (5.0, 0.0) if invert else (0.0, 5.0)
    starts_high = bool(windows and windows[0][0] == 0.0)
    points: list[tuple[float, float]] = [(0.0, high if starts_high else low)]
    for start, stop in windows:
        if start == 0.0:
            points.extend([(stop, high), (stop + edge, low)])
        else:
            points.extend([(start, low), (start + edge, high), (stop, high), (stop + edge, low)])
    points.append((STOP_US, low))
    return " ".join(f"{time:g}u {value:g}" for time, value in points)


def build_circuit() -> None:
    source_inc = HERE / f"initialized_wr1_wrp5_{SEQUENCE}.inc"
    source_lines = source_inc.read_text(encoding="utf-8").splitlines()
    excluded = ("VSOT_RD ", "VSOT_CAL_OK ", "VSOT_RIGHT ", "VSOT_LEFT ", "VSOT_NEUTRAL ", "BSOT_RD_OUT ")
    kept = [line for line in source_lines if not line.startswith(excluded)]
    kept.append("BSOT_RD_OUT sot_rd_out 0 V=V(sot_m_sum)*V(rd_src)/5")
    FULLCHAIN_INC.write_text("\n".join(kept) + "\n", encoding="utf-8")

    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.replace(
        ".title full_chain_ad783_like_opamp_sha_rd_neutral_2inv_calok_behavioral_wr_logic",
        f".title initialized_full_chain_{SEQUENCE}",
    )
    text = text.replace(".include AD783_like_SHA_opamp.lib", ".include ../models/AD783_like_SHA_opamp.lib")
    text = text.replace(".include TLV3501.lib", ".include ../models/TLV3501.lib")
    text = text.replace(".param DB=50m", ".param DB=20m")
    text = text.replace(
        ".include ../basic_models/OpAmps/generic_opamp.lib",
        ".include ../models/generic_opamp.lib\n"
        f".include {FULLCHAIN_INC.name}",
    )

    # Beat 0 initializes WR1 + WR'5 and intentionally has no RD.
    rd_windows = [(index * STEP_US + 2.0, index * STEP_US + 3.0) for index in range(1, len(POSITIONS))]
    wr_clock_windows = [(index * STEP_US, index * STEP_US + 1.0) for index in range(len(POSITIONS))]
    # One CLK-high pulse carries WR; the following CLK-high pulse is the RD
    # clock slot. Initialization also keeps both CLK pulses, but RD remains low
    # in its second slot as required.
    rd_clock_windows = [(index * STEP_US + 2.0, index * STEP_US + 3.0) for index in range(len(POSITIONS))]
    clock_windows = wr_clock_windows + rd_clock_windows
    clock_windows.sort()
    rd_source_windows = rd_windows
    rd_sha_windows = [(start + 0.06, stop - 0.06) for start, stop in rd_windows]
    text = re.sub(r"^VCLK .*?$", f"VCLK clk 0 PWL({pulse_pwl(clock_windows)})", text, flags=re.M)
    text = re.sub(r"^VWR[1-5]P? .*?\n", "", text, flags=re.M)
    text = re.sub(r"^VRD .*?$", f"VRD rd 0 PWL({pulse_pwl(rd_windows, edge=0.05)})", text, flags=re.M)
    text = re.sub(r"^VRDB .*?$", f"VRDB rd_b 0 PWL({pulse_pwl(rd_windows, invert=True, edge=0.05)})", text, flags=re.M)
    text = text.replace(
        f"VRD rd 0 PWL({pulse_pwl(rd_windows, edge=0.05)})",
        f"VRD_SRC rd_src 0 PWL({pulse_pwl(rd_source_windows, edge=0.05)})\n"
        f"VRD rd 0 PWL({pulse_pwl(rd_windows, edge=0.05)})\n"
        f"VRD_SHA rd_sha 0 PWL({pulse_pwl(rd_sha_windows, edge=0.005)})\n"
        f"VRD_SHA_B rd_sha_b 0 PWL({pulse_pwl(rd_sha_windows, invert=True, edge=0.005)})",
    )
    text = re.sub(r"^VINIT .*?$", f"VINIT init 0 PWL(0 5 0.5u 5 0.505u 0 {STOP_US:g}u 0)", text, flags=re.M)
    text = re.sub(r"^VLOGIC_[AB] .*?\n", "", text, flags=re.M)
    text = re.sub(r"^B_COEFF_TOTAL .*?\n", "", text, flags=re.M)
    text = text.replace("CRD_INV2 rd_neutral 0 47p", "CRD_INV2 rd_neutral 0 150p")
    text = text.replace(
        "CNEUTRAL neutral 0 22p",
        "CNEUTRAL neutral 0 22p\n"
        "* Fast RD-off clamp: prevent a Neutral pulse while delayed RD_neutral falls.\n"
        "S_NEUTRAL_RD_OFF neutral 0 rd_b 0 SW_NEUTRAL_RD_OFF\n"
        ".model SW_NEUTRAL_RD_OFF sw vt=0.01 vh=0.001 ron=1 roff=1e12",
    )
    # Build one shared qualification signal so every direction output has the
    # same delay and the plot needs only one combined control track.
    dir_enable_block = """* Shared direction enable: delayed RD AND CAL_OK.
MP_DIR_EN_N1 dir_enable_n rd_neutral vdd vdd PMOS_LOGIC W=20u L=1u
MP_DIR_EN_N2 dir_enable_n cal_ok     vdd vdd PMOS_LOGIC W=20u L=1u
MN_DIR_EN_N1 dir_enable_n rd_neutral dir_enable_mid 0 NMOS_LOGIC W=20u L=1u
MN_DIR_EN_N2 dir_enable_mid cal_ok 0 0 NMOS_LOGIC W=20u L=1u
MP_DIR_EN_INV dir_enable dir_enable_n vdd vdd PMOS_LOGIC W=20u L=1u
MN_DIR_EN_INV dir_enable dir_enable_n 0 0 NMOS_LOGIC W=20u L=1u
CDIR_ENABLE dir_enable 0 10p
DDIR_ENABLE_H dir_enable vdd DLOGIC_CLAMP
DDIR_ENABLE_L 0 dir_enable DLOGIC_CLAMP

"""
    text = text.replace("* CMOS NOR.\n", dir_enable_block + "* CMOS NOR.\n")

    # Replace each original 3-input qualification gate with a 2-input gate
    # driven by the shared dir_enable signal.
    text = text.replace("MP_NAND1 n_neutral_nand rd_neutral vdd vdd PMOS_LOGIC W=20u L=1u", "MP_NAND1 n_neutral_nand dir_enable vdd vdd PMOS_LOGIC W=20u L=1u")
    text = text.replace("MP_NAND3 n_neutral_nand cal_ok vdd vdd PMOS_LOGIC W=20u L=1u\n", "")
    text = text.replace("MN_NAND1 n_neutral_nand rd_neutral n_neutral_mid 0 NMOS_LOGIC W=20u L=1u", "MN_NAND1 n_neutral_nand dir_enable n_neutral_mid 0 NMOS_LOGIC W=20u L=1u")
    text = text.replace("MN_NAND2 n_neutral_mid  n_nor n_neutral_mid2 0 NMOS_LOGIC W=20u L=1u", "MN_NAND2 n_neutral_mid n_nor 0 0 NMOS_LOGIC W=20u L=1u")
    text = text.replace("MN_NAND3 n_neutral_mid2 cal_ok 0             0 NMOS_LOGIC W=20u L=1u\n", "")

    text = text.replace("MP_RIGHT_NAND2 n_right_valid_n rd_neutral vdd vdd PMOS_LOGIC W=20u L=1u", "MP_RIGHT_NAND2 n_right_valid_n dir_enable vdd vdd PMOS_LOGIC W=20u L=1u")
    text = text.replace("MP_RIGHT_NAND3 n_right_valid_n cal_ok vdd vdd PMOS_LOGIC W=20u L=1u\n", "")
    text = text.replace("MN_RIGHT_NAND2 n_right_mid rd_neutral n_right_mid2 0 NMOS_LOGIC W=20u L=1u", "MN_RIGHT_NAND2 n_right_mid dir_enable 0 0 NMOS_LOGIC W=20u L=1u")
    text = text.replace("MN_RIGHT_NAND3 n_right_mid2 cal_ok 0 0 NMOS_LOGIC W=20u L=1u\n", "")

    text = text.replace("MP_LEFT_NAND2 n_left_valid_n rd_neutral vdd vdd PMOS_LOGIC W=20u L=1u", "MP_LEFT_NAND2 n_left_valid_n dir_enable vdd vdd PMOS_LOGIC W=20u L=1u")
    text = text.replace("MP_LEFT_NAND3 n_left_valid_n cal_ok vdd vdd PMOS_LOGIC W=20u L=1u\n", "")
    text = text.replace("MN_LEFT_NAND2 n_left_mid rd_neutral n_left_mid2 0 NMOS_LOGIC W=20u L=1u", "MN_LEFT_NAND2 n_left_mid dir_enable 0 0 NMOS_LOGIC W=20u L=1u")
    text = text.replace("MN_LEFT_NAND3 n_left_mid2 cal_ok 0 0 NMOS_LOGIC W=20u L=1u\n", "")
    text = re.sub(r"^B_RD_VOUT .*?\n", "", text, flags=re.M)
    text = text.replace("CVOUT_LOAD vout 0 1p", "CVOUT_LOAD sot_rd_out 0 1p")
    text = text.replace("RINJ vout vout_sha 10", "RINJ sot_rd_out vout_sha 10")
    text = text.replace("XSH_CUR vout_sha rd 0 vee_sha v_cur vcc_sha", "XSH_CUR vout_sha rd_sha 0 vee_sha v_cur vcc_sha")
    text = text.replace("XSH_PREV v_cur rd_b 0 vee_sha v_prev vcc_sha", "XSH_PREV v_cur rd_sha_b 0 vee_sha v_prev vcc_sha")
    text = text.replace(
        "XSH_PREV v_cur rd_sha_b 0 vee_sha v_prev vcc_sha AD783_LIKE_SHA_OPAMP",
        "XSH_PREV v_cur rd_sha_b 0 vee_sha v_prev vcc_sha AD783_LIKE_SHA_OPAMP\n"
        "* Initialize both internal SHA hold capacitors to 1 V during INIT.\n"
        "VSHA_INIT_REF sha_init_ref 0 1\n"
        "SSHA_CUR_INIT xsh_cur.hold sha_init_ref init 0 SW_SHA_INIT\n"
        "SSHA_PREV_INIT xsh_prev.hold sha_init_ref init 0 SW_SHA_INIT\n"
        ".model SW_SHA_INIT sw vt=2.5 vh=0.1 ron=1 roff=1e12",
    )
    text = re.sub(r"^\.tran .*?$", f".tran 2n {STOP_US:g}u", text, flags=re.M)
    text = re.sub(
        r"^wrdata .*?$",
        f"wrdata {CSV.name} time v(clk) v(wr1) v(wr2) v(wr3) v(wr4) v(wr5) "
        "v(wr1p) v(wr2p) v(wr3p) v(wr4p) v(wr5p) "
        "v(sot_m_wr) v(sot_m_wrp) v(sot_m_sum) v(init) v(rd_src) v(rd) v(cal_ok) v(dir_enable) "
        "v(sot_rd_out) v(v_cur) v(v_prev) v(vdiff) "
        "v(right_valid) v(left_valid) v(neutral)",
        text,
        flags=re.M,
    )
    CIRCUIT.write_text(text, encoding="utf-8")


def load_data() -> dict[str, np.ndarray]:
    raw = np.loadtxt(CSV)
    if raw.ndim != 2 or raw.shape[1] != len(COLS) + 1:
        raise ValueError(f"unexpected ngspice output shape: {raw.shape}")
    return {name: raw[:, index + 1] for index, name in enumerate(COLS)}


def plot(data: dict[str, np.ndarray]) -> None:
    t_us = data["time"] * 1e6
    sha_ylim = (0.9, 2.72) if SEQUENCE == "12345" else (0, 2.72)
    tracks = [
        ("CLK (V)", ("clk",), ("#1f77b4",), (0, 5.3)),
        ("WR1–WR5 (V)", ("wr1", "wr2", "wr3", "wr4", "wr5"),
         ("#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f"), (0, 5.3)),
        ("WR1′–WR5′ (V)", ("wr1p", "wr2p", "wr3p", "wr4p", "wr5p"),
         ("#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f"), (0, 5.3)),
        ("M from WR (V)", ("m_wr",), ("#6a3d9a",), (0.5, 1.38)),
        ("M from WR′ (V)", ("m_wrp",), ("#e66101",), (0.5, 1.38)),
        ("M_WR + M_WR′ (V)", ("m_sum",), ("#222222",), (1.1, 2.72)),
        ("RD (V)", ("rd",), ("#ff7f0e",), (0, 5.3)),
        ("RD source enable (V)", ("rd_src",), ("#9467bd",), (0, 5.3)),
        ("Direction enable (V)", ("dir_enable",), ("#2ca02c",), (0, 5.3)),
        ("RD-controlled Vout (V)", ("vout",), ("#b5bd00",), (0, 2.72)),
        ("opamp-SHA current (V)", ("v_cur",), ("#17becf",), sha_ylim),
        ("opamp-SHA previous (V)", ("v_prev",), ("#888888",), sha_ylim),
        ("vdiff_cm (V)", ("vdiff",), ("#1f77b4",), (1.8, 3.55)),
        ("Right (V)", ("right_valid",), ("#b2182b",), (0, 5.3)),
        ("Left (V)", ("left_valid",), ("#1b7837",), (0, 5.3)),
        ("Neutral (V)", ("neutral",), ("#1f77b4",), (0, 5.3)),
    ]
    fig, axes = plt.subplots(len(tracks), 1, figsize=(16, 18), sharex=True)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.955, bottom=0.045, hspace=0.13)
    fig.suptitle(
        f"SOT full chain: shared delayed direction enable; {SEQUENCE}",
        fontsize=15,
        fontweight="semibold",
    )
    stride = 4
    for axis, (label, signals, colors, ylim) in zip(axes, tracks):
        track_linewidth = 2.3 if label.startswith("opamp-SHA") and SEQUENCE == "12345" else 1.5
        for signal, color in zip(signals, colors):
            axis.plot(t_us[::stride], data[signal][::stride], color=color, linewidth=track_linewidth)
        axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=58, fontsize=9, fontweight="semibold")
        axis.set_ylim(*ylim)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)
        axis.tick_params(labelsize=8, length=3)
    vdiff_axis = axes[next(i for i, track in enumerate(tracks) if track[0] == "vdiff_cm (V)")]
    vdiff_axis.axhline(2.52, color="#ff3b30", linestyle="--", linewidth=0.8)
    vdiff_axis.axhline(2.48, color="#2ca02c", linestyle="--", linewidth=0.8)
    axes[-1].set_xlim(0, STOP_US)
    axes[-1].set_xlabel("Time (µs)", fontweight="semibold")
    fig.savefig(PNG, dpi=180, facecolor="white")
    fig.savefig(SVG, facecolor="white")
    plt.close(fig)
    # Flatten the PNG to opaque RGB so preview clients cannot render its
    # background with a dark transparency fallback.
    with Image.open(PNG) as image:
        image.convert("RGB").save(PNG)


def main() -> None:
    global SEQUENCE, POSITIONS, STOP_US, CIRCUIT, CSV, PNG, SVG, FULLCHAIN_INC
    for case in CASES:
        SEQUENCE = case
        POSITIONS = [1] + [int(value) for value in case]
        STOP_US = len(POSITIONS) * STEP_US
        stem = f"initialized_full_chain_{case}_combined_dir_enable"
        CIRCUIT = HERE / f"{stem}.cir"
        CSV = HERE / f"{stem}.csv"
        PNG = HERE / f"{stem}_waveforms.png"
        SVG = HERE / f"{stem}_waveforms.svg"
        FULLCHAIN_INC = HERE / f"{stem}_sources.inc"
        build_circuit()
        subprocess.run(["ngspice", "-b", CIRCUIT.name], cwd=HERE, check=True)
        data = load_data()
        plot(data)
        print(CIRCUIT)
        print(CSV)
        print(PNG)


if __name__ == "__main__":
    main()
