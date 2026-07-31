"""Local (Windows / RTX 5070 Ti) version of train_phop.py.

One invocation = one training run (dense round 0, or LTH round r under a mask).
Adds over the Modal version:
  - step-level resume via resume.pt (safe to kill / power off mid-run)
  - pauses training while VALORANT is running, resumes when it exits
  - graceful torch.compile fallback (triton is flaky on Windows)
  - writes results.json at the end so the harness can skip completed runs

Called by run_seed.py; can also be run standalone.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from model import Transformer
from phop import generate_one_example, encode, PAD_ID, VOCAB_SIZE, char_to_id, ALPHABET
from lth_util import apply_mask, apply_mask_to_grad, load_lth_checkpoint, save_lth_checkpoint

# same hyperparameters as the paper (section 4.3)
D_MODEL = 128
N_HEADS = 8
D_FF = 512
SEQ_LEN = 256
MAX_SEQ_LEN = SEQ_LEN + 2
P_TRAIN = (1, 2, 4, 8, 16, 32)
P_EVAL = (1, 2, 4, 8, 16, 32)
LR = 1e-3
WARMUP_STEPS = 2000
REWIND_STEP = 1000

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

# =========================
# VALORANT pause monitor
# =========================

VALORANT_NAMES = ("valorant.exe", "valorant-win64-shipping.exe")

def _valorant_running():
    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            name = (proc.info["name"] or "").lower()
            if name in VALORANT_NAMES:
                return True
        return False
    except ImportError:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq VALORANT-Win64-Shipping.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True,
        ).stdout
        return "valorant" in out.lower()

class PauseMonitor:
    """Background thread that sets `paused` while VALORANT is running."""
    def __init__(self, poll_seconds=10):
        self.paused = threading.Event()
        self._poll = poll_seconds
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while True:
            try:
                if _valorant_running():
                    self.paused.set()
                else:
                    self.paused.clear()
            except Exception:
                self.paused.clear()
            time.sleep(self._poll)

# =========================
# data (same as train_phop.py)
# =========================

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

# =========================
# eval (same as train_phop.py)
# =========================

def evaluate(model, device, n_examples=2000, n_show=2, p_values=P_EVAL):
    model.eval()
    results = {}
    with torch.no_grad():
        for p in p_values:
            correct, shown = 0, 0
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
                for j in range(cb):
                    ok = preds[j].item() == answer_ids[j]
                    if ok:
                        correct += 1
                    if shown < n_show:
                        pv = preds[j].item()
                        pc = ALPHABET[pv] if pv < len(ALPHABET) else "?"
                        st = f"{GREEN}ok{RESET}" if ok else f"{RED}X{RESET}"
                        print(f"  [p={p}] {st} ...{seqs[j][-16:]}> -> {pc} (correct: {answers[j]})")
                        shown += 1
            results[p] = correct / n_examples
    return results

# =========================
# checkpoint helpers
# =========================

def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_resume(path, step, base_model, optimizer):
    atomic_torch_save({
        "step": step,
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "py_random": random.getstate(),
    }, path)

# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="KxL, e.g. 6x1")
    parser.add_argument("--round", type=int, default=0, help="0 = dense baseline, >=1 = LTH round")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-mode", default="default",
                        help="torch.compile mode: default | reduce-overhead | max-autotune")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    K, L = map(int, args.config.split("x"))
    base_dir = os.path.join(args.ckpt_root, f"seed_{args.seed}", args.config)
    run_dir = base_dir if args.round == 0 else os.path.join(base_dir, "lth", f"round_{args.round}")
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This harness expects the 5070 Ti to be available."

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    base_model = Transformer(K, L, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, MAX_SEQ_LEN).to(device)

    mask, sparsity = None, 0.0
    if args.round > 0:
        init_path = os.path.join(run_dir, "init.pt")
        mask_path = os.path.join(run_dir, "mask.pt")
        print(f"LTH round {args.round}: loading init from {init_path}")
        base_model.load_state_dict(torch.load(init_path, map_location=device, weights_only=True))
        mask_ckpt = load_lth_checkpoint(mask_path, device)
        mask = mask_ckpt["mask"]
        sparsity = mask_ckpt["sparsity"]
        print(f"Mask loaded: {sparsity:.1%} sparsity")

    model = base_model
    if not args.no_compile:
        # torch.compile fails lazily (at first forward) if triton is broken on
        # Windows, so probe with a dummy forward before committing to it
        try:
            import triton  # noqa: F401
            compiled = torch.compile(base_model, mode=args.compile_mode)
            print(f"compiling model (mode={args.compile_mode}, first forward takes a minute)...")
            dummy = torch.zeros(2, MAX_SEQ_LEN, dtype=torch.long, device=device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                compiled(dummy)
            model = compiled
            print("torch.compile enabled")
        except ImportError:
            print("triton not available, running eager (pip install triton-windows to enable compile)")
        except Exception as e:
            torch._dynamo.reset()
            print(f"torch.compile failed ({type(e).__name__}), running eager")

    from transformers import Adafactor
    optimizer = Adafactor(base_model.parameters(), lr=LR,
                          scale_parameter=False, relative_step=False, warmup_init=False)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # ---- resume ----
    resume_path = os.path.join(run_dir, "resume.pt")
    start_step = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        random.setstate(ckpt["py_random"])
        start_step = ckpt["step"] + 1
        print(f"RESUMED from step {ckpt['step']} ({resume_path})")
        if mask is not None:
            apply_mask(base_model, mask)

    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, args.steps - WARMUP_STEPS)
        return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

    loader = DataLoader(
        PhopDataset(), batch_size=args.batch_size, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = iter(loader)

    monitor = PauseMonitor()

    print("-------------------------")
    print(f"model: ({K}x{L})  seed={args.seed}  round={args.round}")
    if args.round > 0:
        print(f"LTH round {args.round}, sparsity={sparsity:.1%}")
    print(f"task: p-hop induction (p_train={list(P_TRAIN)}, n={SEQ_LEN})")
    print(f"device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"batch_size={args.batch_size}, n_steps={args.steps}, lr={LR}, warmup={WARMUP_STEPS}")
    print(f"parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"starting at step {start_step}")
    print("-------------------------")

    last_log_time = time.time()
    t_start = time.time()

    try:
        for step in range(start_step, args.steps):
            # pause while VALORANT is running
            if monitor.paused.is_set():
                save_resume(resume_path, step - 1 if step > 0 else 0, base_model, optimizer)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"PAUSED at step {step}: VALORANT detected. Checkpoint saved; waiting...")
                while monitor.paused.is_set():
                    time.sleep(5)
                print("RESUMING: VALORANT exited.")
                last_log_time = time.time()

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
            current_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            optimizer.step()
            if mask:
                apply_mask(base_model, mask)

            if step % 100 == 0:
                sps = 100 / (time.time() - last_log_time) if step > start_step else 0
                last_log_time = time.time()
                if sps > 0:
                    secs = (args.steps - step) / sps
                    eta = f"{int(secs // 3600)}h{int(secs % 3600 // 60):02d}m"
                else:
                    eta = "???"
                print(f"step {step}: loss={loss.item():.4f}, lr={current_lr:.6f}, {sps:.1f} steps/s, eta={eta}")

            if step == REWIND_STEP and args.round == 0:
                atomic_torch_save(base_model.state_dict(), os.path.join(run_dir, f"lth_rewind_{REWIND_STEP}.pt"))
                print(f"SAVE rewind checkpoint at step {step}")

            if step > 0 and step % args.save_every == 0:
                save_resume(resume_path, step, base_model, optimizer)

            if step > 0 and step % args.eval_every == 0:
                accs = evaluate(model, device)
                acc_str = ", ".join(f"p{p}={a:.3f}" for p, a in accs.items())
                print(f"EVAL on step {step}: {acc_str}")
    except KeyboardInterrupt:
        save_resume(resume_path, step - 1 if step > start_step else start_step, base_model, optimizer)
        print(f"\nINTERRUPTED at step {step}. Resume checkpoint saved to {resume_path}")
        sys.exit(130)

    print("Final eval...")
    final_accs = evaluate(model, device, 5000, n_show=5)
    acc_str = ", ".join(f"p{p}={a:.3f}" for p, a in final_accs.items())
    print(f"FINAL EVAL: {acc_str}")

    atomic_torch_save(base_model.state_dict(), os.path.join(run_dir, "final.pt"))
    if mask:
        save_lth_checkpoint(os.path.join(run_dir, "mask.pt"), mask=mask,
                            sparsity=sparsity, round_num=args.round, accuracy=final_accs)

    results = {
        "config": args.config,
        "seed": args.seed,
        "round": args.round,
        "density": (1.0 - 0.2) ** args.round,
        "sparsity": sparsity,
        "accuracy": {str(p): a for p, a in final_accs.items()},
        "steps": args.steps,
        "wall_hours": (time.time() - t_start) / 3600,
    }
    tmp = os.path.join(run_dir, "results.json.tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, "results.json"))

    if os.path.exists(resume_path):
        os.remove(resume_path)
    print("Done")

if __name__ == "__main__":
    main()
