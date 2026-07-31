"""Aggregate multi-seed results into the paper's Table 1 format.

Scans checkpoints/seed_*/{config}/**/results.json and builds, per hop count p:
  - accuracy table: rows = configs, cols = density levels, cells = mean +/- std over seeds
  - ratio-to-dense-baseline table (each seed normalized by its own dense run)

Prints p=8 and p=16 (the paper's headline table) and writes all p to
results/table_p{p}.csv plus a combined results/table.md.

Usage:
    python make_table.py [--ckpt-root checkpoints] [--out results]
"""
import argparse
import glob
import json
import math
import os
import re

CONFIGS = ["6x1", "3x2", "2x3", "1x6"]
PRUNE_FRAC = 0.2


def density_label(r):
    d = round((1 - PRUNE_FRAC) ** r * 100, 2)
    return f"{d:.0f}%" if d == int(d) else f"{d:.2f}%"


def load_results(ckpt_root):
    """-> {(config, round, seed): {p(str): acc}}, sorted seed list, max round seen"""
    data, seeds, max_round = {}, set(), 0
    for path in glob.glob(os.path.join(ckpt_root, "seed_*", "*", "**", "results.json"),
                          recursive=True):
        with open(path) as f:
            res = json.load(f)
        key = (res["config"], res["round"], res["seed"])
        data[key] = res["accuracy"]
        seeds.add(res["seed"])
        max_round = max(max_round, res["round"])
    return data, sorted(seeds), max_round


def mean_std(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def build_table(data, seeds, rounds, p, ratio=False):
    """rows: config -> list of cell strings (one per round), '-' if missing"""
    table = {}
    for cfg in CONFIGS:
        row = []
        for r in rounds:
            vals = []
            for s in seeds:
                acc = data.get((cfg, r, s))
                if acc is None or str(p) not in acc:
                    continue
                v = acc[str(p)]
                if ratio:
                    dense = data.get((cfg, 0, s))
                    if dense is None or str(p) not in dense or dense[str(p)] == 0:
                        continue
                    v = v / dense[str(p)]
                vals.append(v)
            if not vals:
                row.append("-")
            else:
                m, sd = mean_std(vals)
                row.append(f"{m:.3f}" if sd is None else f"{m:.3f}±{sd:.3f}")
        table[cfg] = row
    return table


def format_table(table, rounds, title):
    cols = [density_label(r) for r in rounds]
    width = max(11, max(len(c) for c in cols) + 1)
    lines = [title]
    lines.append(f"{'Cfg':<5} " + " ".join(f"{c:>{width}}" for c in cols))
    for cfg, row in table.items():
        lines.append(f"{cfg:<5} " + " ".join(f"{c:>{width}}" for c in row))
    return "\n".join(lines)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--out", default="results")
    parser.add_argument("--p", default="1,2,4,8,16,32")
    args = parser.parse_args()

    data, seeds, max_round = load_results(args.ckpt_root)
    if not data:
        print(f"No results.json files found under {args.ckpt_root}/")
        return
    rounds = list(range(0, max_round + 1))
    p_values = [int(p) for p in args.p.split(",")]
    print(f"Found {len(data)} completed runs across seeds {seeds}\n")

    os.makedirs(args.out, exist_ok=True)
    md_lines = [f"# IMP results ({len(seeds)} seed(s): {seeds})\n"]

    for p in p_values:
        acc = build_table(data, seeds, rounds, p, ratio=False)
        rat = build_table(data, seeds, rounds, p, ratio=True)

        acc_str = format_table(acc, rounds, f"p = {p} (accuracy, mean±std over seeds)")
        rat_str = format_table(rat, rounds, f"p = {p} (ratio to each seed's dense baseline)")
        if p in (8, 16):
            print(acc_str + "\n")
            print(rat_str + "\n")

        # csv: one row per (config, metric), one col per density
        csv_path = os.path.join(args.out, f"table_p{p}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("config,metric," + ",".join(density_label(r) for r in rounds) + "\n")
            for cfg in CONFIGS:
                f.write(f"{cfg},accuracy," + ",".join(acc[cfg]) + "\n")
            for cfg in CONFIGS:
                f.write(f"{cfg},ratio," + ",".join(rat[cfg]) + "\n")

        md_lines.append(f"## p = {p}\n\n```\n{acc_str}\n\n{rat_str}\n```\n")

    md_path = os.path.join(args.out, "table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path} and per-p CSVs to {args.out}/")


if __name__ == "__main__":
    main()
