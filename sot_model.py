"""Single-reversal SOT switching rules; current input in mA."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np


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
    "PRECONDITION_REPEATS": 20,
    "CURVE_MODEL": "tanh",
    "CURVE_SHAPE": 1.00,
    "LEFT_RATE": 0.80,
    "RIGHT_RATE": 0.80,
    "UPDATE_MODE": "branch_fixed",
}


def normalized_kernel(
    t: np.ndarray | float,
    k: float,
    curve_model: str = "tanh",
    shape: float = 1.0,
    left_rate: float = 1.0,
    right_rate: float = 1.0,
) -> np.ndarray:
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


def s_curve_point(x1: float, x2: float, y1: float, y2: float, x: float, params: dict) -> float:
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


@dataclass
class SOTSwitchingModel:
    """Stateful SOT switching model.

    The formulation considered here is restricted to a single-reversal protocol.
    A second sweep-direction reversal raises ValueError before changing state.

    Input mode:
        Call ``step(x_new)`` with an effective current in mA.

    Output mode:
        Returns the new magnetization ``M``. Use ``run(sequence)`` to get a
        table with input current, M, and delta_M.
    """

    params: dict = field(default_factory=lambda: deepcopy(DEFAULT_PARAMS))
    m: float = field(init=False)
    x: float = field(init=False)
    state: dict = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.m = float(self.params["INITIAL_M"])
        self.x = float(self.params["INITIAL_X"])
        self.state = {
            "sweep_direction": 0,
            "reversal_count": 0,
            "pos_peak": self.params["IC_POS"],
            "neg_peak": self.params["IC_NEG"],
            "pos_target": self.params["IC_BAO"],
            "neg_target": self.params["IC_NEG_MIN"],
            "pos_branch_active": False,
            "neg_branch_active": False,
        }

    def step(self, x_new: float) -> float:
        m_new, x_new, state = sot_update(float(x_new), self.m, self.x, self.state, self.params)
        self.m = float(m_new)
        self.x = float(x_new)
        self.state = state
        return self.m

    def run(self, x_sequence: Iterable[float]) -> np.ndarray:
        x_values = np.asarray(list(x_sequence), dtype=float)
        m_values = np.array([self.step(x) for x in x_values], dtype=float)
        delta_m = m_values - float(self.params["INITIAL_M"])
        return np.column_stack([x_values, m_values, delta_m])


def stateless_sot_response(x_new: float, params: dict | None = None) -> dict:
    """Notebook-style one-point SOT response from the initial state."""

    sot_params = deepcopy(DEFAULT_PARAMS)
    if params:
        sot_params.update(params)
    model = SOTSwitchingModel(sot_params)
    m = model.step(float(x_new))
    return {
        "input_current_mA": float(x_new),
        "M": m,
        "delta_M": m - float(sot_params["INITIAL_M"]),
    }


def _check_single_reversal(x_new: float, x_prev: float, state: dict) -> None:
    """Validate actual current direction, including the initial-current transition."""
    if not np.isfinite(x_new) or not np.isfinite(x_prev):
        raise ValueError("Current must be finite.")
    direction = 1 if x_new > x_prev else -1 if x_new < x_prev else 0
    previous = state.get("sweep_direction", 0)
    reversals = state.get("reversal_count", 0)
    if direction and previous and direction != previous:
        reversals += 1
    if reversals > 1:
        raise ValueError(
            "Single-reversal protocol violated: a second current sweep-direction "
            "reversal is not supported. Call reset() to start an independent protocol."
        )
    if direction:
        state["sweep_direction"] = direction
    state["reversal_count"] = reversals


def sot_update(x_new: float, m_prev: float, x_prev: float, state: dict, params: dict) -> tuple[float, float, dict]:
    _check_single_reversal(x_new, x_prev, state)
    ic_pos = params["IC_POS"]
    ic_bao = params["IC_BAO"]
    ic_neg = params["IC_NEG"]
    ic_neg_min = params["IC_NEG_MIN"]
    m_pos_sat = params["M_POS_SAT"]
    m_neg_sat = params["M_NEG_SAT"]
    update_mode = params["UPDATE_MODE"]
    pos_peak = float(state.get("pos_peak", ic_pos))
    pos_target = float(ic_bao)

    if x_new > 0:
        if x_prev <= 0:
            state["pos_target"] = pos_target
            state["neg_peak"] = ic_neg
            state["pos_peak"] = ic_pos
            state["neg_branch_active"] = False
        if x_new <= x_prev:
            return m_prev, x_new, state
        if x_new > ic_pos:
            if update_mode == "branch_fixed":
                if state.get("pos_branch_active") is not True:
                    state["pos_branch_active"] = True
                    state["pos_branch_m0"] = m_prev
                    state["pos_branch_x0"] = ic_pos
                m_start = float(state.get("pos_branch_m0", m_prev))
                x_start = float(state.get("pos_branch_x0", ic_pos))
                m_new = s_curve_point(x_start, float(state["pos_target"]), m_start, m_pos_sat, x_new, params)
            else:
                m_new = s_curve_point(ic_pos, float(state["pos_target"]), m_prev, m_pos_sat, x_new, params)
            state["pos_peak"] = max(pos_peak, x_new)
            return m_new, x_new, state
        state["pos_peak"] = max(pos_peak, x_new)
        return m_prev, x_new, state

    if x_prev >= 0:
        if pos_peak > ic_pos:
            state["neg_target"] = max(-pos_peak, ic_neg_min)
        state["pos_peak"] = ic_pos
        state["neg_peak"] = ic_neg
        state["pos_branch_active"] = False
    if x_new >= ic_neg:
        state["neg_peak"] = min(float(state.get("neg_peak", ic_neg)), x_new)
        return m_prev, x_new, state
    state["neg_peak"] = min(float(state.get("neg_peak", ic_neg)), x_new)
    neg_target = float(state.get("neg_target", ic_neg_min))
    if update_mode == "branch_fixed":
        if state.get("neg_branch_active") is not True:
            state["neg_branch_active"] = True
            state["neg_branch_m0"] = m_prev
            state["neg_branch_x0"] = ic_neg
        m_start = float(state.get("neg_branch_m0", m_prev))
        x_start = float(state.get("neg_branch_x0", ic_neg))
        m_new = s_curve_point(x_start, neg_target, m_start, m_neg_sat, x_new, params)
    else:
        m_new = s_curve_point(ic_neg, neg_target, m_prev, m_neg_sat, x_new, params)
    return m_new, x_new, state


def triangular_sweep(amplitude: float, points_per_leg: int = 401) -> np.ndarray:
    """Build a negative -> positive -> negative current sweep."""

    amplitude = abs(float(amplitude))
    up = np.linspace(-amplitude, amplitude, int(points_per_leg))
    down = np.linspace(amplitude, -amplitude, int(points_per_leg))[1:]
    return np.concatenate([up, down])
