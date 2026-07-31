"""Analysis B: mask structure across seeds.

Two parts:

1. Per-block density at the final IMP round -- replicates the paper's Table 2
   across all three seeds (does 6x1's depth gradient hold? does lm_head keep
   escaping pruning?).

2. Mask overlap (IoU) between seeds at the SAME (config, round). This speaks
   directly to the path-dependence confound: if a failed run and a successful
   run at identical density have nearly identical masks, then the mask is not
   what distinguishes them and the difference is optimization stochasticity.
   If their masks diverge sharply, mask history matters.

Usage: python analyze_masks.py [--round 16]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import torch

from lth_util import load_lth_checkpoint

SEEDS = (67, 68, 69)
CONFIGS = ("6x1", "3x2", "2x3", "1x6")
FAIL, OK = 0.35, 0.50


def block_of(name):
    m = re.match(r"blocks\.(\d+)\.", name)
    if m:
        return f"block {m.group(1)}"
    return "lm_head" if "lm_head" in name else name


def per_block_density(mask):
    alive, total = defaultdict(int), defaultdict(int)
    for name, m in mask.items():
        b = block_of(name)
        alive[b] += int(m.sum())
        total[b] += m.numel()
    return {b: alive[b] / total[b] for b in total}, sum(alive.values()) / sum(total.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=16)
    ap.add_argument("--out", default="results/mask_analysis.json")
    args = ap.parse_args()

    acc = {}
    for path in glob.glob("checkpoints/seed_*/*/**/results.json", recursive=True):
        r = json.load(open(path))
        acc[(r["seed"], r["config"], r["round"])] = r["accuracy"]

    report = {"per_block": {}, "iou": []}

    print(f"=== per-block weight retention at IMP round {args.round} "
          f"(paper Table 2 = {0.8**args.round:.2%} global) ===")
    hdr = f"{'seed':<6} {'cfg':<5} " + " ".join(f"{'blk'+str(i):>7}" for i in range(6)) + \
          f" {'lm_head':>8} {'total':>7}"
    print(hdr)
    for cfg in CONFIGS:
        for s in SEEDS:
            p = f"checkpoints/seed_{s}/{cfg}/lth/round_{args.round}/mask.pt"
            if not os.path.exists(p):
                continue
            dens, total = per_block_density(load_lth_checkpoint(p)["mask"])
            cells = []
            for i in range(6):
                v = dens.get(f"block {i}")
                cells.append(f"{v*100:6.2f}%" if v is not None else "      -")
            lm = dens.get("lm_head", float("nan"))
            print(f"{s:<6} {cfg:<5} " + " ".join(f"{c:>7}" for c in cells) +
                  f" {lm*100:7.2f}% {total*100:6.2f}%")
            report["per_block"][f"{s}_{cfg}"] = {"blocks": dens, "total": total}
        print()

    print("=== mask IoU between seeds at the same (config, round) ===")
    print("high IoU + different outcome => optimization noise, not mask structure\n")
    print(f"{'cfg':<5} {'round':>5} {'density':>8} {'seeds':>9} {'IoU':>7}  outcomes")
    for cfg in CONFIGS:
        for rnd in range(10, 17):
            masks = {}
            for s in SEEDS:
                p = f"checkpoints/seed_{s}/{cfg}/lth/round_{rnd}/mask.pt"
                if os.path.exists(p):
                    masks[s] = load_lth_checkpoint(p)["mask"]
            if len(masks) < 2:
                continue
            ss = sorted(masks)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    a, b = masks[ss[i]], masks[ss[j]]
                    inter = sum(int((a[k] & b[k]).sum()) for k in a)
                    union = sum(int((a[k] | b[k]).sum()) for k in a)
                    iou = inter / union if union else 0.0
                    va = acc.get((ss[i], cfg, rnd), {}).get("16", float("nan"))
                    vb = acc.get((ss[j], cfg, rnd), {}).get("16", float("nan"))
                    def tag(v):
                        return "FAIL" if v < FAIL else ("OK" if v > OK else "mid")
                    mixed = tag(va) != tag(vb)
                    print(f"{cfg:<5} {rnd:>5} {0.8**rnd*100:7.2f}% "
                          f"{ss[i]}/{ss[j]:>4} {iou:6.3f}  "
                          f"p16 {va:.3f}({tag(va)}) vs {vb:.3f}({tag(vb)})"
                          f"{'   <-- MIXED' if mixed else ''}")
                    report["iou"].append({"config": cfg, "round": rnd, "iou": iou,
                                          "seeds": [ss[i], ss[j]],
                                          "p16": [va, vb], "mixed": mixed})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

    mixed = [e["iou"] for e in report["iou"] if e["mixed"]]
    same = [e["iou"] for e in report["iou"] if not e["mixed"]]
    print("\n=== summary ===")
    if mixed:
        print(f"  mixed-outcome pairs : n={len(mixed):2d}  mean IoU={sum(mixed)/len(mixed):.3f}")
    if same:
        print(f"  same-outcome pairs  : n={len(same):2d}  mean IoU={sum(same)/len(same):.3f}")


if __name__ == "__main__":
    main()
