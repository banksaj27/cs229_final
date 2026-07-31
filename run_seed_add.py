"""Per-seed IMP harness for the ADDITION task (2-digit n-ary, Saunshi format).

Mirrors run_seed.py: for each config, dense round 0 then 16 IMP rounds
(prune 20% -> rewind to step 1000 -> retrain). Tree layout matches phop so
make_table.py works via --ckpt-root checkpoints_addition --p 2,4,8,16,24,32.

Resumable: completed rounds are skipped via results.json; mid-round training
resumes from resume.pt. Also usable dense-only: --rounds 0.

Usage:
    python run_seed_add.py --seed 67                 # full chain, all configs
    python run_seed_add.py --seed 67 --rounds 0      # dense baselines only
    python run_seed_add.py --seed 67 --configs 2x3,1x6
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

from model import Transformer
import addition as task
from lth_util import create_mask_iterative, rewind_weights, load_lth_checkpoint, \
    save_lth_checkpoint

D_MODEL, N_HEADS, D_FF = 128, 8, 512
REWIND_STEP = 1000
OP_DIGITS = 2
CONFIGS = ["6x1", "3x2", "2x3", "1x6"]

task.set_op_digits(OP_DIGITS)
MAX_SEQ = task.seq_len_for(max(task.N_EVAL)) + 2


def strip_compiled_prefix(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def prune_round(config, r, base_dir, prune_frac=0.2):
    K, L = map(int, config.split("x"))
    out_dir = os.path.join(base_dir, "lth", f"round_{r}")
    os.makedirs(out_dir, exist_ok=True)

    model = Transformer(K, L, task.VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, MAX_SEQ)
    if r == 1:
        src, prev_mask = os.path.join(base_dir, "final.pt"), None
    else:
        src = os.path.join(base_dir, "lth", f"round_{r - 1}", "final.pt")
        prev_mask = load_lth_checkpoint(
            os.path.join(base_dir, "lth", f"round_{r - 1}", "mask.pt"))["mask"]

    model.load_state_dict(strip_compiled_prefix(
        torch.load(src, map_location="cpu", weights_only=True)))
    mask, sparsity = create_mask_iterative(prev_mask, model, prune_frac)
    print(f"  prune round {r}: sparsity={sparsity:.1%}")

    rewind_sd = torch.load(os.path.join(base_dir, f"lth_rewind_{REWIND_STEP}.pt"),
                           map_location="cpu", weights_only=True)
    rewind_weights(model, strip_compiled_prefix(rewind_sd), mask)
    torch.save(model.state_dict(), os.path.join(out_dir, "init.pt"))
    save_lth_checkpoint(os.path.join(out_dir, "mask.pt"), mask, sparsity, r,
                        accuracy=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--ckpt-root", default="checkpoints_addition")
    ap.add_argument("--eval-every", type=int, default=25000)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    configs = args.configs.split(",")
    t0 = time.time()
    total = len(configs) * (args.rounds + 1)
    done = 0
    for config in configs:
        base_dir = os.path.join(args.ckpt_root, f"seed_{args.seed}", config)
        os.makedirs(base_dir, exist_ok=True)
        for r in range(0, args.rounds + 1):
            run_dir = base_dir if r == 0 else os.path.join(base_dir, "lth", f"round_{r}")
            if os.path.exists(os.path.join(run_dir, "results.json")):
                done += 1
                print(f"[seed {args.seed}] ADD {config} round {r}: done, skipping "
                      f"({done}/{total})")
                continue
            print(f"\n{'='*60}\n[seed {args.seed}] ADDITION {config} round {r}"
                  f"/{args.rounds} (density {0.8**r:.2%}) "
                  f"[{(time.time()-t0)/3600:.1f}h elapsed]\n{'='*60}")
            if r > 0 and not (os.path.exists(os.path.join(run_dir, "init.pt"))
                              and os.path.exists(os.path.join(run_dir, "mask.pt"))):
                prune_round(config, r, base_dir)
            subprocess.run([sys.executable, "train_add.py", "--config", config,
                            "--round", str(r), "--seed", str(args.seed),
                            "--steps", str(args.steps), "--op-digits", str(OP_DIGITS),
                            "--eval-every", str(args.eval_every),
                            "--ckpt-root", args.ckpt_root], check=True)
            done += 1
    print(f"\n[seed {args.seed}] addition harness done: {done}/{total} runs, "
          f"{(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
