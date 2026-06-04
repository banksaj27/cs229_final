import argparse
import os
import torch
from model import Transformer
from phop import VOCAB_SIZE
from lth_util import (
    create_mask_iterative, apply_mask, rewind_weights,
    get_prunable_params, get_sparsity, print_mask_summary,
    save_lth_checkpoint, load_lth_checkpoint,
)

# =========================
# setup + helper
# =========================

D_MODEL = 128
N_HEADS = 8
D_FF = 512
MAX_SEQ_LEN = 258

def strip_compiled_prefix(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd

# =========================
# prune
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. 8x1, 4x2")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--prune-frac", type=float, default=0.2)
    parser.add_argument("--rewind-step", type=int, default=1000)
    parser.add_argument("--ckpt-dir", default="/root/checkpoints")
    args = parser.parse_args()

    K, L = map(int, args.config.split("x"))
    base_dir = f"{args.ckpt_dir}/{args.config}"
    out_dir = f"{base_dir}/lth/round_{args.round}"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("-------------------------")
    print(f"IMP round {args.round} for ({K}⊗{L})")
    print(f"zeroing out {args.prune_frac:.0%} of remaining weights")
    print("-------------------------")

    # 1. load model
    model = Transformer(K, L, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN).to(device)

    if args.round == 1:
        src = f"{base_dir}/final.pt"
        prev_mask = None
    else:
        src = f"{base_dir}/lth/round_{args.round - 1}/final.pt"
        prev_ckpt = load_lth_checkpoint(
            f"{base_dir}/lth/round_{args.round - 1}/mask.pt", device
        )
        prev_mask = prev_ckpt["mask"]

    print(f"Loading weights from {src}")
    sd = torch.load(src, map_location=device, weights_only=True)
    model.load_state_dict(strip_compiled_prefix(sd))

    # 2. prune
    mask, sparsity = create_mask_iterative(prev_mask, model, args.prune_frac)
    print(f"sparsity={sparsity:.1%}")
    print_mask_summary(mask, model)

    # 3. rewind to previous initialization
    rewind_path = f"{base_dir}/lth_rewind_{args.rewind_step}.pt"
    print(f"Rewinding to step {args.rewind_step}")
    rewind_sd = torch.load(rewind_path, map_location=device, weights_only=True)
    rewind_sd = strip_compiled_prefix(rewind_sd)
    rewind_weights(model, rewind_sd, mask)

    # 4. save
    torch.save(model.state_dict(), f"{out_dir}/init.pt")
    save_lth_checkpoint(f"{out_dir}/mask.pt", mask, sparsity, args.round, accuracy=None)
    print(f"SAVE init.pt + mask.pt to {out_dir}/")

if __name__ == "__main__":
    main()