"""Repeat the random-ticket control at the ambiguous cell with fresh mask seeds.

1x6 at 5.50% density is where random appeared to underperform IMP (0.483 vs
0.558), but that cell's cross-seed std is +/-0.167, so one draw cannot
distinguish a systematic deficit from bimodal noise. This runs additional
independent random masks at the same cell (and a 6x1 companion) so the
comparison has more than n=1.

Resumable: skips cells that already have results.json.
"""
import os
import subprocess
import sys
import time

# (config, round, mask_seed) -> each gets its own checkpoint root
RUNS = [
    ("1x6", 13, 2001),
    ("1x6", 13, 2002),
    ("1x6", 13, 2003),
    ("6x1", 13, 2001),
]
SEED = 67


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    t0 = time.time()
    for i, (cfg, r, ms) in enumerate(RUNS, 1):
        root = f"checkpoints_rand_m{ms}"
        out = os.path.join(root, f"seed_{SEED}", cfg, "lth", f"round_{r}", "results.json")
        if os.path.exists(out):
            print(f"[{i}/{len(RUNS)}] {cfg} r{r} mask-seed {ms}: done, skipping")
            continue
        print(f"\n{'='*60}\n[{i}/{len(RUNS)}] RANDOM TICKET {cfg} r{r} mask-seed {ms} "
              f"[{(time.time()-t0)/3600:.1f}h elapsed]\n{'='*60}")
        subprocess.run([sys.executable, "make_random_mask.py", "--seed", str(SEED),
                        "--config", cfg, "--round", str(r), "--mask-seed", str(ms),
                        "--out-root", root], check=True)
        subprocess.run([sys.executable, "local_train.py", "--config", cfg,
                        "--round", str(r), "--seed", str(SEED),
                        "--ckpt-root", root], check=True)
    print(f"\nrandom-ticket repeats complete in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
