import torch
import torch.nn as nn
from collections import OrderedDict

# =========================
# utils for LTH
# =========================

# return {name: param} for all prunable (i.e. Linear) parameters
def get_prunable_params(model):
    prunable = OrderedDict()
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() == 2 and "ln" not in name and "embedding" not in name:
            prunable[name] = param
    return prunable

# create a magnitude-based binary mask at the target sparsity (True = keep, False = pruned)
def create_mask(model, sparsity):
    prunable = get_prunable_params(model)
    all_magnitudes = torch.cat([p.data.abs().flatten() for p in prunable.values()])

    k = int(sparsity * all_magnitudes.numel())
    threshold = torch.kthvalue(all_magnitudes, k + 1).values.item()
    
    mask = {}
    for name, param in prunable.items():
        mask[name] = param.data.abs() >= threshold
    return mask

# prune `prune_fraction` of the remaining weights
def create_mask_iterative(prev_mask, model, prune_fraction):
    prunable = get_prunable_params(model)
    
    if prev_mask is None:
        prev_mask = {name: torch.ones_like(p, dtype=torch.bool) for name, p in prunable.items()}
    
    alive_magnitudes = []
    for name, param in prunable.items():
        m = prev_mask[name]
        alive_magnitudes.append(param.data.abs()[m])
    all_alive = torch.cat(alive_magnitudes)
    
    n_to_prune = int(prune_fraction * all_alive.numel())
    if n_to_prune == 0:
        return prev_mask, get_sparsity(prev_mask, prunable)
    
    threshold = torch.kthvalue(all_alive, n_to_prune).values.item()
    
    new_mask = {}
    for name, param in prunable.items():
        new_mask[name] = prev_mask[name] & (param.data.abs() >= threshold)
    
    return new_mask, get_sparsity(new_mask, prunable)

# zero out pruned weights in-place
def _mask_for_name(mask, name):
    if name in mask:
        return mask[name]
    prefix = "_orig_mod."
    if name.startswith(prefix) and name[len(prefix):] in mask:
        return mask[name[len(prefix):]]
    prefixed = prefix + name
    if prefixed in mask:
        return mask[prefixed]
    return None

def apply_mask(model, mask):
    for name, param in model.named_parameters():
        m = _mask_for_name(mask, name)
        if m is not None:
            param.data.mul_(m.to(device=param.device, dtype=param.dtype))

# zero out gradients for pruned weights
def apply_mask_to_grad(model, mask):
    for name, param in model.named_parameters():
        m = _mask_for_name(mask, name)
        if m is not None and param.grad is not None:
            param.grad.data.mul_(m.to(device=param.grad.device, dtype=param.grad.dtype))

# return fraction of prunable weights that have been zeroed out
def get_sparsity(mask, prunable=None):
    total = 0
    pruned = 0
    for _, m in mask.items():
        total += m.numel()
        pruned += (~m).sum().item()
    return pruned / total if total > 0 else 0.0

# rewind weights (non-prunable and alive) to previous initialization
def rewind_weights(model, rewind_state_dict, mask):
    current_sd = model.state_dict()
    new_sd = {}
    
    for name in current_sd:
        if name in rewind_state_dict:
            new_sd[name] = rewind_state_dict[name].clone()
        else:
            new_sd[name] = current_sd[name]
    
    model.load_state_dict(new_sd)
    apply_mask(model, mask)

# save mask and some metadata
def save_lth_checkpoint(path, mask, sparsity, round_num, accuracy):
    torch.save({
        "mask": {k: v.cpu() for k, v in mask.items()},
        "sparsity": sparsity,
        "round": round_num,
        "accuracy": accuracy,
    }, path)

# load mask and metadata
def load_lth_checkpoint(path, device="cpu"):
    """Load mask + metadata."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "mask" in ckpt:
        ckpt["mask"] = {k: v.to(device) for k, v in ckpt["mask"].items()}
    return ckpt

# print out the mask
def print_mask_summary(mask, model):
    prunable = get_prunable_params(model)
    total_params = 0
    total_pruned = 0
    
    print(f"{'Layer':<45} {'Total':>8} {'Alive':>8} {'Density':>8}")
    print("-" * 73)
    for name, m in mask.items():
        alive = m.sum().item()
        total = m.numel()
        density = alive / total
        total_params += total
        total_pruned += total - alive
        print(f"{name:<45} {total:>8} {int(alive):>8} {density:>7.1%}")
    
    print("-" * 73)
    print(f"{'TOTAL':<45} {total_params:>8} {total_params - total_pruned:>8} {(total_params - total_pruned)/total_params:>7.1%}")
    
    # also show non-prunable param count
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Non-prunable params: {all_params - total_params:,}")
    print(f"Total params: {all_params:,}")
    print(f"Effective params (alive + non-prunable): {all_params - total_pruned:,}")