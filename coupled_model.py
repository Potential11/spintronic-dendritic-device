"""Coupling and notebook-equivalent independent-condition scans."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
from sot_model import DEFAULT_PARAMS, SOTSwitchingModel
from resistor_network import DEFAULT_NETWORK, voltage_network_currents


DEFAULT_POS_SCAN_STOP_V = 16.4


INPUT_POINT_COUNT = 961


NEG_VALUES = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -6.5, -7.0, -8.0]


@dataclass
class NetworkSOTModel:
    """Notebook-equivalent voltage/resistor-network wrapper around SOT model.

    Input mode:
        ``step_voltage(bias_abs_v, input_a_v, input_b_v)``.

    Output mode:
        Returns network values plus the effective SOT current, M, and delta_M.
    """

    params: dict = field(default_factory=lambda: deepcopy(DEFAULT_PARAMS))
    network: dict = field(default_factory=lambda: deepcopy(DEFAULT_NETWORK))
    sot: SOTSwitchingModel = field(init=False)

    def __post_init__(self) -> None:
        self.sot = SOTSwitchingModel(deepcopy(self.params))

    def reset(self) -> None:
        self.sot = SOTSwitchingModel(deepcopy(self.params))

    def step_voltage(self, bias_abs_v: float, input_a_v: float, input_b_v: float) -> dict:
        net = voltage_network_currents(bias_abs_v, input_a_v, input_b_v, self.network)
        m = self.sot.step(net["sot_current_mA"])
        return {
            "bias_abs_v": float(bias_abs_v),
            "input_a_v": float(input_a_v),
            "input_b_v": float(input_b_v),
            "v_common": net["v_common"],
            "iout_a": net["iout_a"],
            "sot_current_mA": net["sot_current_mA"],
            "M": m,
            "delta_M": m - float(self.params["INITIAL_M"]),
        }


def run_plot2_condition(
    bias_abs_v: float,
    input_a_v: float,
    input_b_v: float,
    params: dict | None = None,
    network: dict | None = None,
) -> float:
    """Stateless notebook-equivalent voltage condition returning delta_M."""

    sot_params = deepcopy(DEFAULT_PARAMS)
    if params:
        sot_params.update(params)
    model = NetworkSOTModel(sot_params, deepcopy(DEFAULT_NETWORK) if network is None else network)
    return model.step_voltage(bias_abs_v, input_a_v, input_b_v)["delta_M"]


def simulate_one_bias(
    bias_abs_v: float = DEFAULT_NETWORK["BIAS_ABS_V"],
    params: dict | None = None,
    network: dict | None = None,
    pos_scan_stop_v: float = DEFAULT_POS_SCAN_STOP_V,
    point_count: int = INPUT_POINT_COUNT,
    neg_values: Iterable[float] = NEG_VALUES,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Replicate the notebook's voltage scan for one output bias."""

    input_values = np.linspace(0.0, float(pos_scan_stop_v), int(point_count))
    dm_a = np.array(
        [run_plot2_condition(bias_abs_v, vin, 0.0, params, network) for vin in input_values],
        dtype=float,
    )
    dm_b0 = run_plot2_condition(bias_abs_v, 0.0, 0.0, params, network)
    x_ref = dm_a + dm_b0
    curves = {0.0: x_ref.copy()}
    for neg_v in neg_values:
        if np.isclose(neg_v, 0.0):
            continue
        dm_ab = np.array(
            [run_plot2_condition(bias_abs_v, vin, neg_v, params, network) for vin in input_values],
            dtype=float,
        )
        curves[float(neg_v)] = dm_ab
    return x_ref, curves

