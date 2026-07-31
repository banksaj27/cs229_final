"""Per-seed experiment harness for the LTH x looped-transformers main result.

For one seed, runs all 4 iso-depth configurations (6x1, 3x2, 2x3, 1x6) through
the dense baseline + 16 IMP rounds (17 density levels: 100% down to ~2.81%),
exactly as in the paper's Table 1, writing results under checkpoints/seed_S/.

Fully resumable: completed runs are skipped via results.json, and a run that
was killed mid-training continues from its resume.pt. Just re-run the same
command after any interruption.

Usage:
    python run_seed.py --seed 42
    python run_seed.py --seed 43
    python run_seed.py --seed 44
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

from model import Transformer
from phop import VOCAB_SIZE
from lth_util import create_mask_iterative, rewind_weights, load_lth_checkpoint, save_lth_checkpoint

D_MODEL = 128
N_HEADS = 8
D_FF = 512
MAX_SEQ_LEN = 258
REWIND_STEP = 1000

CONFIGS = ["6x1", "3x2", "2x3", "1x6"]


def strip_compiled_prefix(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def prune_round(config, r, base_dir, prune_frac):
    """CPU version of prune.py: build mask from previous round's final weights,
    rewind survivors to the step-1000 checkpoint, save init.pt + mask.pt."""
    K, L = map(int, config.split("x"))
    out_dir = os.path.join(base_dir, "lth", f"round_{r}")
    os.makedirs(out_dir, exist_ok=True)

    model = Transformer(K, L, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN)

    if r == 1:
        src = os.path.join(base_dir, "final.pt")
        prev_mask = None
    else:
        src = os.path.join(base_dir, "lth", f"round_{r - 1}", "final.pt")
        prev_ckpt = load_lth_checkpoint(os.path.join(base_dir, "lth", f"round_{r - 1}", "mask.pt"))
        prev_mask = prev_ckpt["mask"]

    sd = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(strip_compiled_prefix(sd))

    mask, sparsity = create_mask_iterative(prev_mask, model, prune_frac)
    print(f"  prune round {r}: sparsity={sparsity:.1%}")

    rewind_sd = torch.load(os.path.join(base_dir, f"lth_rewind_{REWIND_STEP}.pt"),
                           map_location="cpu", weights_only=True)
    rewind_weights(model, strip_compiled_prefix(rewind_sd), mask)

    torch.save(model.state_dict(), os.path.join(out_dir, "init.pt"))
    save_lth_checkpoint(os.path.join(out_dir, "mask.pt"), mask, sparsity, r, accuracy=None)


def train_run(config, r, args):
    cmd = [
        sys.executable, "local_train.py",
        "--config", config,
        "--round", str(r),
        "--seed", str(args.seed),
        "--ckpt-root", args.ckpt_root,
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]
    if args.no_compile:
        cmd.append("--no-compile")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--rounds", type=int, default=16, help="number of IMP rounds after dense")
    parser.add_argument("--prune-frac", type=float, default=0.2)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    configs = args.configs.split(",")
    t0 = time.time()
    total_runs = len(configs) * (args.rounds + 1)
    done_runs = 0
    run_times = []  # wall-hours of runs trained this session, for the sweep ETA

    def progress_line():
        remaining = total_runs - done_runs
        if run_times:
            # median, not mean: pauses (VALORANT, sleep) inflate individual runs
            typical = sorted(run_times)[len(run_times) // 2]
            eta = f", est. remaining {typical * remaining:.1f}h ({typical * remaining / 24:.1f} days)"
        else:
            eta = ""
        return (f"[seed {args.seed} progress] {done_runs}/{total_runs} runs done, "
                f"{(time.time() - t0) / 3600:.1f}h elapsed{eta}")

    for config in configs:
        base_dir = os.path.join(args.ckpt_root, f"seed_{args.seed}", config)
        os.makedirs(base_dir, exist_ok=True)

        for r in range(0, args.rounds + 1):
            run_dir = base_dir if r == 0 else os.path.join(base_dir, "lth", f"round_{r}")
            results_path = os.path.join(run_dir, "results.json")
            if os.path.exists(results_path):
                done_runs += 1
                print(f"[seed {args.seed}] {config} round {r}: already done, skipping "
                      f"({done_runs}/{total_runs})")
                continue

            print(f"\n{'='*60}")
            print(f"[seed {args.seed}] {config} round {r}/{args.rounds} "
                  f"(density {(1 - args.prune_frac) ** r:.2%})")
            print(progress_line())
            print(f"{'='*60}")

            if r > 0 and not (os.path.exists(os.path.join(run_dir, "init.pt"))
                              and os.path.exists(os.path.join(run_dir, "mask.pt"))):
                prune_round(config, r, base_dir, args.prune_frac)

            t_run = time.time()
            train_run(config, r, args)
            run_times.append((time.time() - t_run) / 3600)
            done_runs += 1

            with open(results_path) as f:
                accs = json.load(f)["accuracy"]
            acc_str = ", ".join(f"p{p}={a:.3f}" for p, a in sorted(accs.items(), key=lambda kv: int(kv[0])))
            print(f"[seed {args.seed}] {config} round {r} finished: {acc_str}")
            print(progress_line())

    print(f"\nSeed {args.seed} complete: {done_runs}/{total_runs} runs, "
          f"{(time.time() - t0) / 3600:.1f}h total.")
    print("Run `python make_table.py` to build the aggregate table.")


if __name__ == "__main__":
    main()
