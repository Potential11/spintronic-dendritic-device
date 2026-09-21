"""Resistor network; voltages in V, resistances in ohm, branch currents in A."""
from __future__ import annotations
from copy import deepcopy


DEFAULT_NETWORK = {
    "BIAS_ABS_V": 8.0,
    "R_PORTS_OHM": [304.0, 500.0, 751.0, 844.0, 980.0],
    "OUTPUT_PORT_INDEX": 1,
    "INPUT_PORT_A": 2,
    "INPUT_PORT_B": 4,
    "CURRENT_SCALE_TO_MA": 1000.0,
    "OUTPUT_CURRENT_SIGN": -1.0,
    "COMMON_NODE_V": None,
}


def solve_common_voltage_with_mask(v_ports: list[float], r_ports: list[float], on_mask: list[bool]) -> float:
    num = 0.0
    den = 0.0
    for v, r, is_on in zip(v_ports, r_ports, on_mask):
        if not is_on:
            continue
        num += float(v) / float(r)
        den += 1.0 / float(r)
    if den == 0:
        raise ValueError("No conducting branch. Common node is undefined.")
    return num / den


def compute_currents_with_mask(
    v_ports: list[float],
    r_ports: list[float],
    v_common: float,
    on_mask: list[bool],
) -> list[float]:
    currents = []
    for v, r, is_on in zip(v_ports, r_ports, on_mask):
        currents.append(0.0 if not is_on else (float(v) - float(v_common)) / float(r))
    return currents


def voltage_network_currents(
    bias_abs_v: float,
    input_a_v: float,
    input_b_v: float,
    network: dict | None = None,
) -> dict:
    """Return node voltage, branch currents, and effective SOT current.

    This follows the notebook's ``run_plot2_condition`` network exactly.
    ``input_a_v`` is applied to INPUT_PORT_A and ``input_b_v`` to INPUT_PORT_B.
    The output port is held at ``-bias_abs_v``.
    """

    cfg = deepcopy(DEFAULT_NETWORK)
    if network:
        cfg.update(network)

    r_ports = list(cfg["R_PORTS_OHM"][:4])
    v_ports = [0.0] * len(r_ports)
    on_mask = [False] * len(r_ports)
    output_idx = int(cfg["OUTPUT_PORT_INDEX"]) - 1
    input_a_idx = int(cfg["INPUT_PORT_A"]) - 1
    input_b_idx = int(cfg["INPUT_PORT_B"]) - 1

    v_ports[output_idx] = -float(bias_abs_v)
    on_mask[output_idx] = True
    v_ports[input_a_idx] = float(input_a_v)
    v_ports[input_b_idx] = float(input_b_v)
    on_mask[input_a_idx] = True
    on_mask[input_b_idx] = True

    common_node = cfg["COMMON_NODE_V"]
    v_common = (
        solve_common_voltage_with_mask(v_ports, r_ports, on_mask)
        if common_node is None
        else float(common_node)
    )
    currents_a = compute_currents_with_mask(v_ports, r_ports, v_common, on_mask)
    iout_a = currents_a[output_idx]
    iout_ma = float(cfg["OUTPUT_CURRENT_SIGN"]) * iout_a * float(cfg["CURRENT_SCALE_TO_MA"])
    return {
        "v_ports": v_ports,
        "r_ports": r_ports,
        "on_mask": on_mask,
        "v_common": v_common,
        "currents_a": currents_a,
        "iout_a": iout_a,
        "sot_current_mA": iout_ma,
    }

