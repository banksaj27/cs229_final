"""Export one seed's results as paste-ready TSV blocks matching the
"CS 229 Results" Google Sheet layout: one block per p value, rows
6x1/3x2/2x3/1x6, columns Baseline, 80%, 64%, ..., 2.81% (17 levels).

Paste each block with the cursor on that block's (6x1, Baseline) cell.

Usage: python export_sheet.py --seed 67 [--out seed67_sheet.tsv]
"""
import argparse
import glob
import json
import os
import sys

CONFIGS = ["6x1", "3x2", "2x3", "1x6"]
P_VALUES = [1, 2, 4, 8, 16, 32]
N_ROUNDS = 17  # baseline + 16 IMP rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    acc = {}
    pattern = os.path.join(args.ckpt_root, f"seed_{args.seed}", "*", "**", "results.json")
    for path in glob.glob(pattern, recursive=True):
        r = json.load(open(path))
        acc[(r["config"], r["round"])] = r["accuracy"]

    n_done = len(acc)
    blocks = []
    for p in P_VALUES:
        lines = []
        for cfg in CONFIGS:
            cells = []
            for rnd in range(N_ROUNDS):
                a = acc.get((cfg, rnd))
                cells.append(f"{a[str(p)]:.4f}" if a and str(p) in a else "")
            lines.append("\t".join(cells))
        blocks.append((p, "\n".join(lines)))

    out_text = "\n\n".join(f"### p={p} (paste at that block's 6x1/Baseline cell)\n{b}"
                           for p, b in blocks)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
        print(f"Wrote {args.out} ({n_done} completed runs for seed {args.seed})")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
