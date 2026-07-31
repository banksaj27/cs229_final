"""Modal app: train ONE (config, round) of the LTH sweep on a cheap GPU.

Stateless by design: init/mask come in as bytes from the local orchestrator
(modal_run_seed.py), outputs (final.pt, results.json, rewind ckpt for round 0)
are written to the workspace's "phop-lth" volume, which the orchestrator
downloads from. This makes jobs trivially portable across Modal accounts.

GPU is chosen via the PHOP_GPU env var at `modal run` time (default L4).

Manual bench:  $env:PHOP_GPU='A10G'; modal run modal_phop.py::bench
Train round:   modal run --detach modal_phop.py --config 6x1 --round 0 --seed 68
"""
import os

import modal

GPU = os.environ.get("PHOP_GPU", "L4")

app = modal.App("phop-lth")
vol = modal.Volume.from_name("phop-lth", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "transformers")
    .env({"PHOP_GPU": GPU})  # forward GPU label into the container for logs/results
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("phop.py", "/root/phop.py")
    .add_local_file("lth_util.py", "/root/lth_util.py")
)

# paper hyperparameters
D_MODEL, N_HEADS, D_FF = 128, 8, 512
SEQ_LEN = 256
MAX_SEQ_LEN = SEQ_LEN + 2
P_TRAIN = (1, 2, 4, 8, 16, 32)
LR = 1e-3
WARMUP_STEPS = 2000
REWIND_STEP = 1000


def _build_training(K, L, seed, device):
    import random
    import torch
    from model import Transformer
    from phop import VOCAB_SIZE

    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    model = Transformer(K, L, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN).to(device)
    return model


def _make_loader(batch_size, num_workers):
    import random
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    from phop import generate_one_example, encode, PAD_ID

    class PhopDataset(IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            if info is not None:
                random.seed(info.seed % 2**32)
            while True:
                p = random.choice(P_TRAIN)
                seq_str, answer_str = generate_one_example(p, SEQ_LEN)
                prompt_ids = encode(seq_str + ">")
                answer_ids = encode(answer_str)
                all_ids = prompt_ids + answer_ids
                masked_ids = [-100] * (len(prompt_ids) - 1) + answer_ids
                yield (all_ids, masked_ids)

    def collate(batch):
        all_ids, masked_ids = zip(*batch)
        seq_len = max(len(x) for x in all_ids)
        all_ids = [x + [PAD_ID] * (seq_len - len(x)) for x in all_ids]
        masked_ids = [x + [-100] * (seq_len - len(x)) for x in masked_ids]
        return torch.tensor(all_ids), torch.tensor(masked_ids)

    return DataLoader(PhopDataset(), batch_size=batch_size, collate_fn=collate,
                      num_workers=num_workers, pin_memory=True, prefetch_factor=4,
                      persistent_workers=True)


def _evaluate(model, device, n_examples=5000):
    import torch
    from phop import generate_one_example, encode, PAD_ID, char_to_id

    model.eval()
    results = {}
    with torch.no_grad():
        for p in P_TRAIN:
            correct = 0
            for i in range(0, n_examples, 256):
                cb = min(256, n_examples - i)
                examples = [generate_one_example(p, SEQ_LEN) for _ in range(cb)]
                seqs, answers = zip(*examples)
                prompt_ids = [encode(s + ">") for s in seqs]
                answer_ids = [char_to_id[a] for a in answers]
                prompt_len = len(prompt_ids[0])
                padded = [pr + [PAD_ID] * (MAX_SEQ_LEN - len(pr)) for pr in prompt_ids]
                input_ids = torch.tensor(padded, device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                preds = logits[:, prompt_len - 1, :].argmax(dim=-1)
                correct += sum(int(preds[j].item() == answer_ids[j]) for j in range(cb))
            results[p] = correct / n_examples
    return results


@app.function(image=image, gpu=GPU, timeout=20 * 3600, cpu=4.0, memory=8192,
              volumes={"/vol": vol}, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train_round(config: str, round_num: int, seed: int, steps: int,
                init_bytes: bytes = None, mask_bytes: bytes = None,
                batch_size: int = 256):
    import io
    import json
    import math
    import sys
    import time

    import torch
    import torch.nn as nn
    from transformers import Adafactor
    from lth_util import apply_mask, apply_mask_to_grad, save_lth_checkpoint

    sys.stdout.reconfigure(line_buffering=True)
    device = "cuda"
    K, L = map(int, config.split("x"))
    out_dir = f"/vol/seed_{seed}/{config}/round_{round_num}"
    os.makedirs(out_dir, exist_ok=True)

    model = _build_training(K, L, seed, device)

    mask, sparsity = None, 0.0
    if round_num > 0:
        assert init_bytes is not None and mask_bytes is not None
        model.load_state_dict(torch.load(io.BytesIO(init_bytes), map_location=device,
                                         weights_only=True))
        mask_ckpt = torch.load(io.BytesIO(mask_bytes), map_location=device, weights_only=False)
        mask = {k: v.to(device) for k, v in mask_ckpt["mask"].items()}
        sparsity = mask_ckpt["sparsity"]
        print(f"[{config} r{round_num}] mask loaded, sparsity={sparsity:.1%}")

    base_model = model
    try:
        model = torch.compile(base_model)
        print("torch.compile enabled")
    except Exception as e:
        model = base_model
        print(f"torch.compile failed ({type(e).__name__}), eager")

    optimizer = Adafactor(base_model.parameters(), lr=LR, scale_parameter=False,
                          relative_step=False, warmup_init=False)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loader = _make_loader(batch_size, num_workers=3)
    data_iter = iter(loader)

    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
        return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

    print(f"[{config} r{round_num} seed {seed}] {steps} steps on {GPU}, "
          f"params={sum(p.numel() for p in base_model.parameters()):,}")
    t0 = time.time()
    last = time.time()

    for step in range(steps):
        all_ids, all_masked_ids = next(data_iter)
        all_ids, all_masked_ids = all_ids.to(device), all_masked_ids.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(all_ids)
            loss = loss_fn(logits.transpose(1, 2), all_masked_ids)
        loss.backward()
        if mask:
            apply_mask_to_grad(base_model, mask)
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step)
        optimizer.step()
        if mask:
            apply_mask(base_model, mask)

        if step % 500 == 0:
            sps = 500 / (time.time() - last) if step > 0 else 0
            last = time.time()
            print(f"step {step}: loss={loss.item():.4f}, {sps:.1f} steps/s")

        if step == REWIND_STEP and round_num == 0:
            torch.save(base_model.state_dict(), f"{out_dir}/lth_rewind_{REWIND_STEP}.pt")
            vol.commit()
            print(f"SAVE rewind checkpoint at step {step}")

    final_accs = _evaluate(base_model, device)
    acc_str = ", ".join(f"p{p}={a:.3f}" for p, a in final_accs.items())
    print(f"FINAL EVAL: {acc_str}")

    torch.save(base_model.state_dict(), f"{out_dir}/final.pt")
    results = {
        "config": config, "seed": seed, "round": round_num,
        "density": 0.8 ** round_num, "sparsity": sparsity,
        "accuracy": {str(p): a for p, a in final_accs.items()},
        "steps": steps, "wall_hours": (time.time() - t0) / 3600, "gpu": GPU,
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print("Done")
    return results


@app.function(image=image, gpu=GPU, timeout=1800, cpu=4.0, memory=8192)
def bench_fn(steps: int = 300, batch_size: int = 256):
    """Measure steps/s on this GPU type for cost planning."""
    import time

    import torch
    import torch.nn as nn
    from transformers import Adafactor

    device = "cuda"
    model = _build_training(6, 1, 0, device)
    base_model = model
    try:
        model = torch.compile(base_model)
    except Exception:
        pass
    optimizer = Adafactor(base_model.parameters(), lr=LR, scale_parameter=False,
                          relative_step=False, warmup_init=False)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loader = _make_loader(batch_size, num_workers=3)
    data_iter = iter(loader)

    t0 = None
    for step in range(steps):
        all_ids, all_masked_ids = next(data_iter)
        all_ids, all_masked_ids = all_ids.to(device), all_masked_ids.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(all_ids)
            loss = loss_fn(logits.transpose(1, 2), all_masked_ids)
        loss.backward()
        optimizer.step()
        if step == 49:  # skip compile/warmup
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    sps = (steps - 50) / (time.time() - t0)
    print(f"BENCH {GPU}: {sps:.1f} steps/s (batch {batch_size})")
    return {"gpu": GPU, "steps_per_s": sps}


@app.local_entrypoint()
def main(config: str = "6x1", round_num: int = 0, seed: int = 68,
         steps: int = 200000, init_path: str = "", mask_path: str = ""):
    init_bytes = open(init_path, "rb").read() if init_path else None
    mask_bytes = open(mask_path, "rb").read() if mask_path else None
    call = train_round.spawn(config, round_num, seed, steps, init_bytes, mask_bytes)
    print(f"SPAWNED {call.object_id} [{config} round {round_num} seed {seed} on {GPU}]")


@app.local_entrypoint()
def bench():
    print(bench_fn.remote())
