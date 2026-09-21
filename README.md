# Spintronic dendritic device: SOT and resistor-network core

A phenomenological spin–orbit torque (SOT) switching model coupled to a resistor network.

**The formulation considered here is restricted to a single-reversal protocol.**

The positive switching target is fixed at +29 mA with the default parameters; it is not shortened by a previous negative-current peak. The negative target retains the positive-peak-dependent rule. A second change in current sweep direction raises `ValueError` before updating the rejected point.

## Protocol examples

- Allowed: `0 → +22 → −22 mA`.
- Allowed: `0 → −22 → +22 mA`.
- Rejected: `0 → +22 → −22 → 0 mA` (second reversal).
- Rejected: `0 → +25 → +20 → +22 mA` (second reversal).

Reversal means a change in the direction of current increments, not a zero crossing. Equal consecutive currents do not count. Direction is tracked from `INITIAL_X` (default 0 mA), across successive calls. Call `reset()` for an independent protocol; this also resets magnetization. In voltage-driven simulations the restriction applies to the effective SOT current.

## Files

| File | Purpose |
|---|---|
| `sot_model.py` | SOT response, state updates and protocol validation |
| `resistor_network.py` | Common-node voltage, branch currents and effective SOT current |
| `coupled_model.py` | Stateful voltage coupling and independent-condition scans |
| `run_demo.py` | Minimal CSV-producing examples |
| `test_protocol.py` | Protocol and state-behavior tests |
| `CORE_NOTES.md` | Detailed model rules, units and limitations (Chinese) |
| `VALIDATION.md` | Validation scope (Chinese) |
| `SOURCE_MANIFEST.json` | Local source provenance and revision description |

## Run

Python 3.10+ and NumPy are required. No experimental data or ngspice installation is needed for these examples.

```bash
python3 -m pip install -r requirements.txt
python3 run_demo.py
python3 -m unittest -v test_protocol.py
```

The demo creates `results/current_modes.csv` and `results/voltage_scan.csv`. These are simulated results, not experimental measurements.

```python
from sot_model import SOTSwitchingModel

model = SOTSwitchingModel()
print(model.run([0, 22, -22]))
# Columns: input_current_mA, M, M - INITIAL_M
# model.step(0) now raises ValueError: second sweep reversal.
model.reset()
```

`SOTSwitchingModel` and `NetworkSOTModel` preserve history. `stateless_sot_response` and `run_plot2_condition` reset the state for every condition and therefore do not enforce a protocol across separate calls. A failed batch keeps the previously accepted points; it does not roll back the entire batch.

## Scope and limitations

The default network wrapper uses the first four resistance entries and activates ports 1, 2 and 4; it is not a general five-port interface. Voltage is in V, resistance in ohm, branch current in A and effective SOT current in mA.

This is a behavior-level model, not a micromagnetic simulation. It has no explicit time or pulse-width dynamics. The original negative-branch interpolation is retained; the single-reversal check does not establish validity for arbitrary hysteresis histories. Full reproduction of every manuscript figure has not been verified. Experimental data, peripheral circuit simulations and vendor models are not included.
