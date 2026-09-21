# Spintronic dendritic device: SOT and resistor-network core

A phenomenological spin–orbit torque (SOT) switching model coupled to a resistor network. The formulation considered here is restricted to a single-reversal protocol.

The network converts applied port voltages into an effective SOT current, which drives the magnetization response. The model includes a fixed positive switching target and a positive-peak-dependent negative switching target.

## Files

| File | Purpose |
|---|---|
| `sot_model.py` | SOT response and state updates |
| `resistor_network.py` | Common-node voltage and branch currents |
| `coupled_model.py` | Voltage-driven SOT response and independent-condition scans |
| `run_demo.py` | Examples producing simulated CSV results |
| `test_protocol.py` | Model protocol tests |
| [CORE_NOTES.md](CORE_NOTES.md) | Detailed model rules, parameters and scope (Chinese) |
| [VALIDATION.md](VALIDATION.md) | Validation details (Chinese) |
| `SOURCE_MANIFEST.json` | Source provenance and revision description |

## Getting started

Requires Python 3.10+ and NumPy.

```bash
python3 -m pip install -r requirements.txt
python3 run_demo.py
```

The demo creates `results/current_modes.csv` and `results/voltage_scan.csv` containing simulated responses.

### Current-driven example

```python
from sot_model import SOTSwitchingModel

model = SOTSwitchingModel()
response = model.run([0, 19, 22, 0, -19, -22])
print(response)
# Columns: current (mA), M, and M - INITIAL_M

model.reset()  # Restore the initial state for an independent run.
```

### Voltage-driven example

```python
from coupled_model import run_plot2_condition

# Voltages in V; each condition is evaluated from the initial state.
delta_m = run_plot2_condition(
    bias_abs_v=8.0,
    input_a_v=12.0,
    input_b_v=-4.0,
)
print(delta_m)
```

The stateful interfaces retain history between calls. Independent-condition interfaces initialize the model for each condition. The default network uses ports 1, 2 and 4; branch currents are in A and effective SOT current is in mA.

## Validation

```bash
python3 -m unittest -v test_protocol.py
```

See [model notes](CORE_NOTES.md) for the formulation and its scope, and [validation details](VALIDATION.md) for the checks performed. This repository contains the core behavioral model and examples; experimental data and peripheral circuit simulations are not included.
