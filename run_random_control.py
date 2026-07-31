"""Train the random-ticket control runs sequentially on the local GPU.

Each run is a full 200k-step retrain of a random mask whose per-layer density
matches the IMP mask at that (config, round), so accuracies are directly
comparable to checkpoints/seed_67/<config>/lth/round_<r>/results.json.

Resumable: skips runs that already have results.json.
"""
import os
import subprocess
import sys
import time

RUNS = [("6x1", 10), ("6x1", 13), ("1x6", 10), ("1x6", 13)]
SEED = 67
ROOT = "checkpoints_rand"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    t0 = time.time()
    for i, (cfg, r) in enumerate(RUNS, 1):
        out = os.path.join(ROOT, f"seed_{SEED}", cfg, "lth", f"round_{r}", "results.json")
        if os.path.exists(out):
            print(f"[{i}/{len(RUNS)}] {cfg} round {r}: already done, skipping")
            continue
        print(f"\n{'='*60}\n[{i}/{len(RUNS)}] RANDOM TICKET {cfg} round {r} "
              f"(density {0.8**r:.2%})  [{(time.time()-t0)/3600:.1f}h elapsed]\n{'='*60}")
        subprocess.run([sys.executable, "local_train.py", "--config", cfg,
                        "--round", str(r), "--seed", str(SEED),
                        "--ckpt-root", ROOT], check=True)
    print(f"\nrandom-ticket control complete in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
