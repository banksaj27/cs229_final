import modal
import os

# =========================
# modal setup
# =========================

app = modal.App("phop-lth")
vol = modal.Volume.from_name("phop-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy", "transformers")
    .add_local_dir(".", remote_path="/root/project")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 24,
    volumes={"/root/checkpoints": vol},
)
def run_lth_sweep(K: int, L: int, start_round: int = 1, n_rounds: int = 5,
                  prune_frac: float = 0.2, rewind_step: int = 1000):
    """Run pruning rounds [start_round, start_round+n_rounds) inside ONE container.
    Survives local disconnect when launched with --detach."""
    import subprocess
    import torch

    config = f"{K}x{L}"
    base = f"/root/checkpoints/{config}"

    end_round = start_round + n_rounds
    for r in range(start_round, end_round):
        sparsity = 1.0 - (1.0 - prune_frac) ** r
        print(f"\n{'='*60}")
        print(f"ROUND {r}  ({K}\u2297{L})  target sparsity {sparsity:.1%}")
        print(f"{'='*60}")

        # 1. prune + rewind
        print(f"=== PRUNE round {r} ===")
        subprocess.run([
            "python", "/root/project/prune.py",
            "--config", config,
            "--round", str(r),
            "--prune-frac", str(prune_frac),
            "--rewind-step", str(rewind_step),
        ], check=True)

        # 2. train with mask
        print(f"=== TRAIN round {r} ===")
        env = os.environ.copy()
        env["PHOP_K"] = str(K)
        env["PHOP_L"] = str(L)
        env["LTH_ROUND"] = str(r)
        subprocess.run(
            ["python", "/root/project/train_phop.py"],
            check=True, env=env,
        )
        vol.commit()

    # 3. collect results into one table (all rounds present on disk)
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE — RESULTS for ({K}\u2297{L})")
    print(f"{'='*60}")
    rows = []
    lth_dir = f"{base}/lth"
    present = []
    if os.path.isdir(lth_dir):
        for name in os.listdir(lth_dir):
            if name.startswith("round_"):
                try:
                    present.append(int(name.split("_")[1]))
                except ValueError:
                    pass
    for r in sorted(present):
        mask_path = f"{base}/lth/round_{r}/mask.pt"
        if not os.path.exists(mask_path):
            continue
        ckpt = torch.load(mask_path, map_location="cpu", weights_only=False)
        rows.append((r, ckpt.get("sparsity"), ckpt.get("accuracy")))

    if rows:
        any_acc = rows[0][2]
        p_keys = sorted(any_acc.keys()) if isinstance(any_acc, dict) else ["acc"]
        hdr = f"{'round':>5} {'sparsity':>9} " + " ".join(f"p{p:>5}" for p in p_keys)
        print(hdr)
        print("-" * len(hdr))
        for r, sp, acc in rows:
            sp_str = f"{sp:.1%}" if isinstance(sp, float) else str(sp)
            if isinstance(acc, dict):
                acc_str = " ".join(f"{acc[p]:6.3f}" for p in p_keys)
            else:
                acc_str = str(acc)
            print(f"{r:>5} {sp_str:>9} {acc_str}")
    print(f"{'='*60}")


@app.local_entrypoint()
def main(k: int = 6, l: int = 1, start_round: int = 1, rounds: int = 5,
         prune_frac: float = 0.2, rewind_step: int = 1000):
    end = start_round + rounds - 1
    final_sp = 1.0 - (1.0 - prune_frac) ** end
    print(f"LTH sweep ({k}\u2297{l}): rounds {start_round}..{end}")
    print(f"prune_frac={prune_frac}, rewind_step={rewind_step}")
    print(f"Sparsity after round {end}: {final_sp:.1%}")
    run_lth_sweep.spawn(k, l, start_round, rounds, prune_frac, rewind_step)
    print("Spawned. Safe to close laptop.")