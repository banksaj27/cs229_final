"""Train the looped transformer on n-ary addition (Saunshi et al. format).

Task: addition.py -- uniform mixture over n ∈ {2,4,8,16,32} operands of 3
digits, evaluated per-n (including the unseen n=24) with greedy decode and
full-answer exact match, mirroring the per-p evaluation of the phop task.

Architecture and recipe stay identical to the phop runs (d_model=128, 8 heads,
d_ff=512, Adafactor lr=1e-3, cosine decay, bf16) so pruning results are
comparable across tasks; this deviates from Saunshi et al.'s larger setup
(d=256, ff=1024, batch 1024, lr 5e-3) intentionally.

Usage: python train_add.py --config 6x1 --seed 67 --steps 40000
"""
import argparse
import json
import math
import os
import sys
import time
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from model import Transformer
import addition as task
from local_train import PauseMonitor, atomic_torch_save, save_resume
from lth_util import (apply_mask, apply_mask_to_grad, load_lth_checkpoint,
                      save_lth_checkpoint)

REWIND_STEP = 1000

WARMUP_STEPS = 2000


class AddDataset(IterableDataset):
    def __init__(self, op_digits=3):
        self.op_digits = op_digits

    def __iter__(self):
        # spawned workers re-import the task module, so reapply the digit
        # setting inside the worker process
        task.set_op_digits(self.op_digits)
        info = torch.utils.data.get_worker_info()
        if info is not None:
            random.seed(info.seed % 2**32)
        while True:
            n = random.choice(task.N_TRAIN)
            prompt, answer = task.generate_example(n)
            all_ids = prompt + answer
            masked = [-100] * (len(prompt) - 1) + answer
            yield (all_ids, masked)


def collate(batch):
    # top-level (not a closure) so Windows spawn workers can pickle it
    all_ids, masked = zip(*batch)
    L = max(len(x) for x in all_ids)
    a = [x + [task.PAD_ID] * (L - len(x)) for x in all_ids]
    m = [x + [-100] * (L - len(x)) for x in masked]
    return torch.tensor(a), torch.tensor(m)


@torch.no_grad()
def evaluate_greedy(model, device, n_examples=2000, batch=256, n_values=task.N_EVAL):
    """per-n full-answer exact match under greedy autoregressive decode"""
    model.eval()
    results = {}
    for n in n_values:
        correct = 0
        for i in range(0, n_examples, batch):
            cb = min(batch, n_examples - i)
            ex = [task.generate_example(n) for _ in range(cb)]
            prompts = torch.tensor([p for p, _ in ex], device=device)
            answers = torch.tensor([a for _, a in ex], device=device)
            seq = prompts
            ok = torch.ones(cb, dtype=torch.bool, device=device)
            for t in range(task.ANS_LEN):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(seq)
                nxt = logits[:, -1, :].argmax(-1)
                ok &= (nxt == answers[:, t])
                seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            correct += int(ok.sum())
        results[n] = correct / n_examples
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, default=0,
                    help="0 = dense (saves rewind ckpt), >=1 = IMP round (loads init+mask)")
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--ckpt-root", default="checkpoints_add")
    ap.add_argument("--easy-bar", type=float, default=0.999,
                    help="stop early as TOO EASY if ALL n meet this twice running")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    ap.add_argument("--op-digits", type=int, default=3,
                    help="digits per operand (3 = paper; 2 = easier variant)")
    args = ap.parse_args()
    LR, D_MODEL, D_FF, N_HEADS = args.lr, args.d_model, args.d_ff, 8
    task.set_op_digits(args.op_digits)

    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    K, L = map(int, args.config.split("x"))
    base_dir = os.path.join(args.ckpt_root, f"seed_{args.seed}", args.config)
    run_dir = base_dir if args.round == 0 else os.path.join(base_dir, "lth",
                                                            f"round_{args.round}")
    os.makedirs(run_dir, exist_ok=True)
    device = "cuda"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    max_seq = task.seq_len_for(max(task.N_EVAL)) + 2
    base_model = Transformer(K, L, task.VOCAB_SIZE, D_MODEL, N_HEADS, D_FF,
                             max_seq).to(device)

    mask, sparsity = None, 0.0
    if args.round > 0:
        print(f"IMP round {args.round}: loading init + mask from {run_dir}")
        base_model.load_state_dict(torch.load(os.path.join(run_dir, "init.pt"),
                                              map_location=device, weights_only=True))
        mask_ckpt = load_lth_checkpoint(os.path.join(run_dir, "mask.pt"), device)
        mask = mask_ckpt["mask"]
        sparsity = mask_ckpt["sparsity"]
        print(f"mask sparsity: {sparsity:.1%}")

    model = base_model
    try:
        import triton  # noqa: F401
        compiled = torch.compile(base_model)
        dummy = torch.zeros(2, max_seq, dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            compiled(dummy)
        model = compiled
        print("torch.compile enabled")
    except Exception as e:
        print(f"eager mode ({type(e).__name__})")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=LR,
                                      weight_decay=0.01, betas=(0.9, 0.98))
    else:
        from transformers import Adafactor
        optimizer = Adafactor(base_model.parameters(), lr=LR, scale_parameter=False,
                              relative_step=False, warmup_init=False)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    resume_path = os.path.join(run_dir, "resume.pt")
    start_step = 0
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        print(f"RESUMED from step {ck['step']}")

    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / WARMUP_STEPS
        prog = (step - WARMUP_STEPS) / max(1, args.steps - WARMUP_STEPS)
        return LR * 0.5 * (1.0 + math.cos(math.pi * prog))

    loader = DataLoader(AddDataset(args.op_digits), batch_size=args.batch_size,
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=True, prefetch_factor=4,
                        persistent_workers=args.num_workers > 0)
    it = iter(loader)
    monitor = PauseMonitor()

    print(f"n-ary ADDITION (Saunshi format): mixture n={list(task.N_TRAIN)}, "
          f"config {K}x{L}, seed {args.seed}, {args.steps} steps, seq_len {max_seq}")
    t0, last = time.time(), time.time()
    history, easy_streak, verdict = [], 0, "trained"

    step = start_step
    for step in range(start_step, args.steps):
        if monitor.paused.is_set():
            save_resume(resume_path, max(step - 1, 0), base_model, optimizer)
            torch.cuda.empty_cache()
            print(f"PAUSED at {step} (VALORANT)")
            while monitor.paused.is_set():
                time.sleep(5)
            print("RESUMING")
            last = time.time()

        a, m = next(it)
        a, m = a.to(device), m.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(a).transpose(1, 2), m)
        loss.backward()
        if mask:
            apply_mask_to_grad(base_model, mask)
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step)
        optimizer.step()
        if mask:
            apply_mask(base_model, mask)

        if step == REWIND_STEP and args.round == 0:
            atomic_torch_save(base_model.state_dict(),
                              os.path.join(run_dir, f"lth_rewind_{REWIND_STEP}.pt"))
            print(f"SAVE rewind checkpoint at step {step}")

        if step % 500 == 0:
            sps = 500 / (time.time() - last) if step > start_step else 0
            last = time.time()
            print(f"step {step}: loss={loss.item():.4f}, {sps:.1f} steps/s")
        if step > 0 and step % args.save_every == 0:
            save_resume(resume_path, step, base_model, optimizer)
        if step > 0 and step % args.eval_every == 0:
            accs = evaluate_greedy(model, device)
            history.append({"step": step, "exact_match": {str(n): a for n, a in accs.items()}})
            print("EVAL step %d: " % step +
                  ", ".join(f"n{n}={a:.4f}" for n, a in accs.items()))
            easy_streak = easy_streak + 1 if all(a >= args.easy_bar for a in accs.values()) else 0
            if easy_streak >= 2:
                verdict = "too_easy"
                print("TOO EASY - stopping early")
                break

    final = evaluate_greedy(model, device, n_examples=5000)
    history.append({"step": step, "exact_match": {str(n): a for n, a in final.items()},
                    "final": True})
    print("FINAL exact-match: " + ", ".join(f"n{n}={a:.4f}" for n, a in final.items()))

    atomic_torch_save(base_model.state_dict(), os.path.join(run_dir, "final.pt"))
    if mask:
        save_lth_checkpoint(os.path.join(run_dir, "mask.pt"), mask=mask,
                            sparsity=sparsity, round_num=args.round,
                            accuracy={str(n): a for n, a in final.items()})
    res = {"task": "nary_addition", "n_train": list(task.N_TRAIN),
           "op_digits": args.op_digits, "round": args.round,
           "density": 0.8 ** args.round, "sparsity": sparsity,
           "lr": LR, "d_model": D_MODEL, "d_ff": D_FF,
           "config": args.config, "seed": args.seed,
           "steps_run": step + 1, "steps_target": args.steps,
           "accuracy": {str(n): a for n, a in final.items()},
           "verdict": verdict, "history": history,
           "wall_hours": (time.time() - t0) / 3600}
    tmp = os.path.join(run_dir, "results.json.tmp")
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, os.path.join(run_dir, "results.json"))
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print("Done")


if __name__ == "__main__":
    main()
