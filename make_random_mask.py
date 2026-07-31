"""Random-ticket control for the LTH claim.

For a given (seed, config, round), builds a RANDOM mask with the same
per-layer density as the IMP mask found at that round, applies it to the same
step-1000 rewind weights, and writes init.pt + mask.pt into a parallel
checkpoint tree so local_train.py can train it unchanged.

Per-layer density is matched (rather than only global density) so the
comparison isolates *which* weights IMP selected, not how many it kept per
layer -- the stronger form of the control.

Usage:
  python make_random_mask.py --seed 67 --config 1x6 --round 13
  python local_train.py --config 1x6 --round 13 --seed 67 --ckpt-root checkpoints_rand
"""
import argparse
import os

import torch

from model import Transformer
from phop import VOCAB_SIZE
from lth_util import load_lth_checkpoint, save_lth_checkpoint, apply_mask

D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN = 128, 8, 512, 258
REWIND_STEP = 1000


def strip_compiled_prefix(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--src-root", default="checkpoints")
    ap.add_argument("--out-root", default="checkpoints_rand")
    ap.add_argument("--mask-seed", type=int, default=None,
                    help="RNG seed for the random mask (default: 1000+round)")
    args = ap.parse_args()

    K, L = map(int, args.config.split("x"))
    src_base = os.path.join(args.src_root, f"seed_{args.seed}", args.config)
    src_round = os.path.join(src_base, "lth", f"round_{args.round}")
    out_base = os.path.join(args.out_root, f"seed_{args.seed}", args.config)
    out_round = os.path.join(out_base, "lth", f"round_{args.round}")
    os.makedirs(out_round, exist_ok=True)

    imp = load_lth_checkpoint(os.path.join(src_round, "mask.pt"))
    imp_mask, target_sparsity = imp["mask"], imp["sparsity"]

    g = torch.Generator().manual_seed(args.mask_seed if args.mask_seed is not None
                                      else 1000 + args.round)
    rand_mask = {}
    print(f"random ticket for {args.config} round {args.round} "
          f"(IMP sparsity {target_sparsity:.1%}) -- per-layer densities:")
    for name, m in imp_mask.items():
        keep = int(m.sum().item())
        flat = torch.zeros(m.numel(), dtype=torch.bool)
        flat[torch.randperm(m.numel(), generator=g)[:keep]] = True
        rand_mask[name] = flat.view_as(m)
        print(f"  {name:<40} keep {keep:>7}/{m.numel():<7} ({keep/m.numel():6.2%})")

    total = sum(m.numel() for m in rand_mask.values())
    alive = sum(int(m.sum()) for m in rand_mask.values())
    sparsity = 1.0 - alive / total
    assert abs(sparsity - target_sparsity) < 1e-6, (sparsity, target_sparsity)

    # same rewind weights the IMP ticket used
    model = Transformer(K, L, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN)
    rewind = torch.load(os.path.join(src_base, f"lth_rewind_{REWIND_STEP}.pt"),
                        map_location="cpu", weights_only=True)
    model.load_state_dict(strip_compiled_prefix(rewind))
    apply_mask(model, rand_mask)

    torch.save(model.state_dict(), os.path.join(out_round, "init.pt"))
    save_lth_checkpoint(os.path.join(out_round, "mask.pt"), rand_mask,
                        sparsity, args.round, accuracy=None)
    # local_train.py needs the rewind ckpt present in the parallel tree only if
    # it ever re-prunes; copy the pointer for completeness
    print(f"\nsparsity {sparsity:.4%} (matches IMP) -> {out_round}")


if __name__ == "__main__":
    main()
