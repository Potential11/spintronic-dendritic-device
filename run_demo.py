"""Run without experiment files or ngspice; write explicit CSV examples."""
from pathlib import Path
import csv
import numpy as np
from sot_model import DEFAULT_PARAMS, SOTSwitchingModel, stateless_sot_response, triangular_sweep
from resistor_network import voltage_network_currents
from coupled_model import run_plot2_condition


def main():
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    sweep = triangular_sweep(35, 201)
    # Start at the negative endpoint; the sweep reverses only at +35 mA.
    params = dict(DEFAULT_PARAMS, INITIAL_X=-35.0, INITIAL_M=0.30)
    model = SOTSwitchingModel(params)
    with (out / 'current_modes.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'current_mA', 'M_stateful', 'M_independent'])
        for step, current in enumerate(sweep):
            writer.writerow([step, current, model.step(current), stateless_sot_response(current)['M']])
    with (out / 'voltage_scan.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bias_abs_V', 'input_a_V', 'input_b_V', 'common_V', 'output_A', 'effective_mA', 'delta_M_independent'])
        for vb in [0., -4., -8.]:
            for va in np.linspace(0, 16.4, 101):
                net = voltage_network_currents(8., va, vb)
                writer.writerow([8., va, vb, net['v_common'], net['iout_a'], net['sot_current_mA'], run_plot2_condition(8., va, vb)])
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
