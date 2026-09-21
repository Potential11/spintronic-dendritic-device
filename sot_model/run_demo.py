"""Run without experiment files or ngspice; write explicit CSV examples."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sot_model import DEFAULT_PARAMS, SOTSwitchingModel, triangular_sweep
from resistor_network import voltage_network_currents
from coupled_model import run_plot2_condition


def main():
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    with (out / 'current_roundtrips.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['amplitude_mA', 'step', 'current_mA', 'M', 'delta_M'])
        for amplitude, color in [(35, '#246C9C'), (25, '#C47536')]:
            params = dict(DEFAULT_PARAMS, INITIAL_X=-float(amplitude), INITIAL_M=0.30)
            model = SOTSwitchingModel(params)  # Independent initial state for each round.
            data = model.run(triangular_sweep(amplitude, 401))
            for step, row in enumerate(data):
                writer.writerow([amplitude, step, *row])
            ax.plot(data[:401, 0], data[:401, 1], color=color, lw=2.3,
                    label=f'±{amplitude} mA: forward')
            ax.plot(data[400:, 0], data[400:, 1], color=color, lw=2.3, ls='--',
                    label=f'±{amplitude} mA: return')
    ax.set(xlabel='Effective SOT current (mA)', ylabel='Magnetization M',
           ylim=(0.275, 0.69), title='Independent ±25 and ±35 mA sweeps (simulation)')
    ax.grid(alpha=0.15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=2,
              frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'current_roundtrips.png', dpi=300)
    fig.savefig(out / 'current_roundtrips.pdf')
    fig.savefig(out / 'current_roundtrips.svg')
    plt.close(fig)
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
