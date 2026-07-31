"""Local queue: dense (round 0) addition baselines for all 3 seeds x 4 configs.

Runs the 12 dense trainings sequentially on the local GPU while waiting for
Modal credits -- so the Modal/FarmShare chains can start directly at round 1.
Resumable; each run also saves the step-1000 rewind checkpoint IMP needs.
"""
import subprocess
import sys

SEEDS = (67, 68, 69)
CONFIGS = ("6x1", "3x2", "2x3", "1x6")

for seed in SEEDS:
    for cfg in CONFIGS:
        subprocess.run([sys.executable, "run_seed_add.py", "--seed", str(seed),
                        "--configs", cfg, "--rounds", "0"], check=True)
print("ALL DENSE ADDITION BASELINES DONE")
