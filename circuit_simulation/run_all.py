"""Run the three simulation stages in order."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
if not shutil.which("ngspice"):
    raise SystemExit("ngspice is required. Install it and add it to PATH.")
(root / "results").mkdir(exist_ok=True)
env = os.environ.copy()
env.setdefault("MPLBACKEND", "Agg")
for script in ["generate_initialized_sequence_cases.py", "run_initialized_full_chain_combined_enable.py", "plot_three_rd_calok_combined_timing.py"]:
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(root / "scripts" / script)], cwd=root / "results", env=env, check=True)
print("Completed. Results are in results/.")
