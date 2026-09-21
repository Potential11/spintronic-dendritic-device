"""Five-port simulation using the requested memory SOT rule.

This replaces the notebook's original ``sot_update`` behavior with the rule:

- The current branch is evaluated against the full fixed branch range
  ``IC_POS -> IC_BAO`` or ``IC_NEG -> IC_NEG_MIN``.
- A turn point only changes the opposite branch target after the sweep reverses.
- Therefore a partial positive peak, such as +22 mA, does not saturate the
  positive branch early.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent / "outputs"

DEFAULT_PARAMS = {
    "k": 2.50,
    "IC_POS": 17.00,
    "IC_BAO": 29.00,
    "IC_NEG": -17.00,
    "IC_NEG_MIN": -29.00,
    "M_POS_SAT": 0.65,
    "M_NEG_SAT": 0.30,
    "INITIAL_M": 0.30,
    "INITIAL_X": 0.00,
    "CURVE_MODEL": "tanh",
    "CURVE_SHAPE": 1.00,
    "LEFT_RATE": 0.80,
    "RIGHT_RATE": 0.80,
}

DEFAULT_R_PORTS_OHM = [304.0, 500.0, 751.0, 844.0, 980.0]
DEFAULT_VOLTAGE_ABS_V = 20.0
CURRENT_SCALE_TO_MA = 1000.0
OUTPUT_CURRENT_SIGN = -1.0

FIVE_STEP_ORDER_FORWARD = ["R21", "R31", "R41", "R51", "R12"]
FIVE_STEP_ORDER_REVERSE = ["R12", "R51", "R41", "R31", "R21"]
THREE_STEP_ORDER_FORWARD = ["R21", "R31", "R41"]
THREE_STEP_ORDER_REVERSE = ["R41", "R31", "R21"]
PORT_BY_LABEL = {"R21": 1, "R31": 2, "R41": 3, "R51": 4, "R12": 5}


def normalized_kernel(t, k, curve_model="tanh", shape=1.0, left_rate=1.0, right_rate=1.0):
    t = np.asarray(t, dtype=float)
    left_rate = max(float(left_rate), 1e-6)
    right_rate = max(float(right_rate), 1e-6)
    shape = max(float(shape), 1e-6)
    out = np.empty_like(t, dtype=float)
    left_mask = t <= 0
    right_mask = ~left_mask

    if curve_model == "tanh":
        kl = max(float(k) * left_rate, 1e-6)
        kr = max(float(k) * right_rate, 1e-6)
        if np.any(left_mask):
            tl = t[left_mask]
            out[left_mask] = 0.5 * (np.tanh(kl * tl) / np.tanh(kl) + 1.0)
        if np.any(right_mask):
            tr = t[right_mask]
            out[right_mask] = 0.5 + 0.5 * np.tanh(kr * tr) / np.tanh(kr)
    elif curve_model == "sigmoid":
        kl = max(float(k) * left_rate, 1e-6)
        kr = max(float(k) * right_rate, 1e-6)

        def logistic(z):
            return 1.0 / (1.0 + np.exp(-z))

        if np.any(left_mask):
            tl = t[left_mask]
            lo = logistic(-2.0 * kl)
            out[left_mask] = 0.5 * (logistic(2.0 * kl * tl) - lo) / (0.5 - lo + 1e-12)
        if np.any(right_mask):
            tr = t[right_mask]
            hi = logistic(2.0 * kr)
            out[right_mask] = 0.5 + 0.5 * (logistic(2.0 * kr * tr) - 0.5) / (hi - 0.5 + 1e-12)
    elif curve_model == "gompertz":

        def gompertz_unit(u, rate):
            a = max(float(k) * float(rate), 1e-6)
            g0 = np.exp(-shape)
            g1 = np.exp(-shape * np.exp(-a))
            g = np.exp(-shape * np.exp(-a * u))
            return (g - g0) / (g1 - g0 + 1e-12)

        if np.any(left_mask):
            tl = t[left_mask]
            out[left_mask] = 0.5 * gompertz_unit(tl + 1.0, left_rate)
        if np.any(right_mask):
            tr = t[right_mask]
            out[right_mask] = 0.5 + 0.5 * gompertz_unit(tr, right_rate)
    else:
        raise ValueError(f"Unsupported curve_model: {curve_model}")

    return np.clip(out, 0.0, 1.0)


def s_curve_point(x1, x2, y1, y2, x, params):
    if x1 == x2:
        return float(y2)
    t = (x - x1) / (x2 - x1) * 2.0 - 1.0
    s = normalized_kernel(
        t,
        params["k"],
        params["CURVE_MODEL"],
        params["CURVE_SHAPE"],
        params["LEFT_RATE"],
        params["RIGHT_RATE"],
    )
    return float(y1 + (y2 - y1) * s)


class RequestedMemorySOT:
    def __init__(self, params=None):
        self.params = deepcopy(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.reset()

    def reset(self):
        self.m = float(self.params["INITIAL_M"])
        self.x_prev = float(self.params["INITIAL_X"])
        self.pos_peak = float(self.params["IC_POS"])
        self.neg_peak = float(self.params["IC_NEG"])
        self.pos_target = float(self.params["IC_BAO"])
        self.neg_target = float(self.params["IC_NEG_MIN"])
        self.pos_active = False
        self.neg_active = False
        self.pos_m0 = self.m
        self.neg_m0 = self.m

    def step(self, x_new):
        x = float(x_new)
        p = self.params
        before = self.m

        if x > 0:
            if self.x_prev <= 0:
                # Entering positive branch: keep the full positive target.
                self.pos_active = False
                self.neg_active = False
                self.pos_peak = float(p["IC_POS"])
            if x > self.x_prev and x > p["IC_POS"]:
                if not self.pos_active:
                    self.pos_active = True
                    self.pos_m0 = self.m
                self.m = s_curve_point(p["IC_POS"], p["IC_BAO"], self.pos_m0, p["M_POS_SAT"], x, p)
            self.pos_peak = max(self.pos_peak, x)
        else:
            if self.x_prev >= 0:
                # Entering negative branch: target is determined by positive turn point.
                self.neg_target = (
                    max(-self.pos_peak, p["IC_NEG_MIN"])
                    if self.pos_peak > p["IC_POS"]
                    else p["IC_NEG_MIN"]
                )
                self.pos_active = False
                self.neg_active = False
                self.neg_peak = float(p["IC_NEG"])
            if x < p["IC_NEG"]:
                if not self.neg_active:
                    self.neg_active = True
                    self.neg_m0 = self.m
                self.m = s_curve_point(p["IC_NEG"], self.neg_target, self.neg_m0, p["M_NEG_SAT"], x, p)
            self.neg_peak = min(self.neg_peak, x)

        self.x_prev = x
        return before, self.m, self.m - before


def make_resistances(r_ports=None):
    r = list(DEFAULT_R_PORTS_OHM if r_ports is None else r_ports)
    return {
        "R21_R12": float(r[1]),
        "R31": float(r[2]),
        "R41": float(r[3]),
        "R51": float(r[4]),
    }


def directed_voltage(label, voltage_abs_v):
    return -float(voltage_abs_v) if label == "R12" else float(voltage_abs_v)


def current_from_pair(label, voltage_abs_v, resistances):
    resistance = resistances["R21_R12"] if label in ("R21", "R12") else resistances[label]
    applied_v = directed_voltage(label, voltage_abs_v)
    return OUTPUT_CURRENT_SIGN * applied_v / resistance * CURRENT_SCALE_TO_MA


def simulate_sequence(order, voltage_abs_v=DEFAULT_VOLTAGE_ABS_V, r_ports=None, params=None):
    resistances = make_resistances(r_ports)
    model = RequestedMemorySOT(params)
    rows = []
    for step, label in enumerate(order, start=1):
        iout_ma = current_from_pair(label, voltage_abs_v, resistances)
        before, after, delta = model.step(iout_ma)
        resistance = resistances["R21_R12"] if label in ("R21", "R12") else resistances[label]
        rows.append(
            {
                "step": step,
                "port": PORT_BY_LABEL[label],
                "label": label,
                "resistance_ohm": resistance,
                "applied_v": directed_voltage(label, voltage_abs_v),
                "iout_ma": iout_ma,
                "M_before": before,
                "M_after": after,
                "delta_M": delta,
                "pos_peak": model.pos_peak,
                "neg_target": model.neg_target,
            }
        )
    return pd.DataFrame(rows)


def plot_pair(forward, reverse, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    axes[0].plot(forward["port"], forward["M_after"], "-o", color="#1f4e79", lw=2.2, ms=7, label="forward")
    axes[0].plot(reverse["port"], reverse["M_after"], "-s", color="#bf4d4d", lw=2.2, ms=6, label="reverse")
    axes[0].axhline(DEFAULT_PARAMS["INITIAL_M"], color="#888888", ls=":", lw=1.0)
    axes[0].set_xticks(sorted(set(forward["port"]).union(reverse["port"])))
    axes[0].set_xlabel("port")
    axes[0].set_ylabel("M")
    axes[0].set_title(title)
    axes[0].grid(True, color="#d9d9d9", lw=0.7, alpha=0.65)
    axes[0].legend(frameon=False, fontsize=9)

    width = 0.32
    axes[1].bar(forward["port"] - width / 2, forward["delta_M"], width=width, color="#d18b28", label="delta M forward")
    axes[1].bar(reverse["port"] + width / 2, reverse["delta_M"], width=width, color="#7aa6c2", label="delta M reverse")
    ax2 = axes[1].twinx()
    ax2.plot(forward["port"], forward["iout_ma"], "--o", color="#6b4c9a", lw=1.4, ms=4, label="Iout forward")
    ax2.plot(reverse["port"], reverse["iout_ma"], "--s", color="#3b7a57", lw=1.4, ms=4, label="Iout reverse")
    axes[1].axhline(0, color="#888888", ls=":", lw=1.0)
    axes[1].set_xticks(sorted(set(forward["port"]).union(reverse["port"])))
    axes[1].set_xlabel("port")
    axes[1].set_ylabel("delta M")
    ax2.set_ylabel("Iout (mA)")
    axes[1].set_title("Step change and current")
    axes[1].grid(True, color="#d9d9d9", lw=0.7, alpha=0.65)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    forward_1_to_5 = simulate_sequence(FIVE_STEP_ORDER_FORWARD)
    reverse_5_to_1 = simulate_sequence(FIVE_STEP_ORDER_REVERSE)
    forward_1_to_3 = simulate_sequence(THREE_STEP_ORDER_FORWARD)
    reverse_3_to_1 = simulate_sequence(THREE_STEP_ORDER_REVERSE)

    forward_1_to_5.to_csv(OUT_DIR / "requested_memory_1_to_5.csv", index=False)
    reverse_5_to_1.to_csv(OUT_DIR / "requested_memory_5_to_1.csv", index=False)
    forward_1_to_3.to_csv(OUT_DIR / "requested_memory_1_to_3.csv", index=False)
    reverse_3_to_1.to_csv(OUT_DIR / "requested_memory_3_to_1.csv", index=False)

    plot_pair(
        forward_1_to_5,
        reverse_5_to_1,
        "Requested-memory five-port: 1->5 and 5->1",
        OUT_DIR / "requested_memory_1_to_5_5_to_1.png",
    )
    plot_pair(
        forward_1_to_3,
        reverse_3_to_1,
        "Requested-memory ports: 1->3 and 3->1",
        OUT_DIR / "requested_memory_1_to_3_3_to_1.png",
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()

