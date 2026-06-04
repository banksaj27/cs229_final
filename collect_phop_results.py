import argparse
import os
import torch

# =========================
# collect results
# =========================

def collect_config(base_dir, config):
    base = os.path.join(base_dir, config)
    rows = []
    dense = os.path.join(base, "dense_results.pt")
    if os.path.exists(dense):
        d = torch.load(dense, map_location="cpu", weights_only=False)
        rows.append((0, 0.0, d))

    lth_dir = os.path.join(base, "lth")
    if os.path.isdir(lth_dir):
        round_nums = []
        for name in os.listdir(lth_dir):
            if name.startswith("round_"):
                try:
                    round_nums.append(int(name.split("_")[1]))
                except ValueError:
                    pass
        for r in sorted(round_nums):
            mask_path = os.path.join(lth_dir, f"round_{r}", "mask.pt")
            if os.path.exists(mask_path):
                ckpt = torch.load(mask_path, map_location="cpu", weights_only=False)
                rows.append((r, ckpt.get("sparsity"), ckpt.get("accuracy")))
    return rows


def print_table(config, rows):
    print(f"\n=== ({config}) ===")
    if not rows:
        print("  no results found")
        return
    p_keys = None
    for _, _, acc in rows:
        if isinstance(acc, dict):
            p_keys = sorted(acc.keys())
            break
    if p_keys is None:
        p_keys = []

    hdr = f"{'round':>5} {'sparsity':>9} " + " ".join(f"{'p'+str(p):>7}" for p in p_keys)
    print(hdr)
    print("-" * len(hdr))
    for r, sp, acc in rows:
        sp_str = f"{sp:.1%}" if isinstance(sp, float) else str(sp)
        if isinstance(acc, dict):
            acc_str = " ".join(f"{acc.get(p, float('nan')):7.3f}" for p in p_keys)
        else:
            acc_str = str(acc)
        print(f"{r:>5} {sp_str:>9} {acc_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--dir", default="checkpoints")
    args = ap.parse_args()
    for config in args.config:
        rows = collect_config(args.dir, config)
        print_table(config, rows)


if __name__ == "__main__":
    main()
