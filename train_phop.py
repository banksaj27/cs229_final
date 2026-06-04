import random
import os
import torch
import torch.nn as nn
from phop import generate_one_example, encode, decode, PAD_ID, VOCAB_SIZE, char_to_id, ALPHABET
from torch.utils.data import IterableDataset, DataLoader
import sys, time, math
from model import Transformer
from transformers import Adafactor
from lth_util import apply_mask, apply_mask_to_grad, load_lth_checkpoint, save_lth_checkpoint

# =========================
# setup
# =========================

K = int(os.environ.get("PHOP_K", 6))
L = int(os.environ.get("PHOP_L", 1))
SEQ_LEN = int(os.environ.get("PHOP_N", 256))
P_TRAIN = (1, 2, 4, 8, 16, 32)
P_EVAL = (1, 2, 4, 8, 16, 32)
d_model = 128
n_heads = 8
d_ff = 512
max_seq_len = SEQ_LEN + 2
batch_size = 256
n_steps = 200000
warmup_steps = 2000
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = int(os.environ.get("PHOP_SEED", "42"))
torch.manual_seed(seed)
random.seed(seed)
print(f"seed: {seed}")

ckpt_dir = f"/root/checkpoints/{K}x{L}"
os.makedirs(ckpt_dir, exist_ok=True)

LTH_ROUND = int(os.environ.get("LTH_ROUND", 0))
mask = None
if LTH_ROUND > 0:
    lth_dir = f"{ckpt_dir}/lth/round_{LTH_ROUND}"
    os.makedirs(lth_dir, exist_ok=True)
    ckpt_dir = lth_dir

model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len).to(device)

if LTH_ROUND > 0:
    init_path = f"{ckpt_dir}/init.pt"
    mask_path = f"{ckpt_dir}/mask.pt"
    print(f"LTH round {LTH_ROUND}: loading init from {init_path}")
    model.load_state_dict(torch.load(init_path, map_location=device, weights_only=True))
    mask_ckpt = load_lth_checkpoint(mask_path, device)
    mask = mask_ckpt["mask"]
    sparsity = mask_ckpt["sparsity"]
    print(f"Mask loaded: {sparsity:.1%} sparsity")

model = torch.compile(model)

optimizer = Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[0m"
sys.stdout.reconfigure(line_buffering=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# =========================
# data
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
# eval (per-p)
# =========================

def evaluate(model, n_examples=2000, n_show=2, p_values=P_EVAL):
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
                padded = [pr + [PAD_ID] * (max_seq_len - len(pr)) for pr in prompt_ids]
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
                        st = f"{GREEN}\u2713{BLUE}" if ok else f"{RED}\u2717{BLUE}"
                        print(f"  [p={p}] {st} ...{seqs[j][-16:]}> \u2192 {pc} (correct: {answers[j]})")
                        shown += 1
            results[p] = correct / n_examples
    return results

def get_lr(step):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# =========================
# training
# =========================

loader = DataLoader(
    PhopDataset(), batch_size=batch_size, collate_fn=collate,
    num_workers=8, pin_memory=True, prefetch_factor=4,
)
data_iter = iter(loader)

print("-------------------------")
print(f"model: ({K}\u2297{L})")
if LTH_ROUND > 0:
    print(f"LTH round {LTH_ROUND}, sparsity={sparsity:.1%}")
print(f"task: p-hop induction (p_train={list(P_TRAIN)}, n={SEQ_LEN})")
print(f"device: {device}")
print(f"d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
print(f"batch_size={batch_size}, n_steps={n_steps}, lr={lr}, warmup={warmup_steps}")
print(f"vocab_size={VOCAB_SIZE}, max_seq_len={max_seq_len}")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-------------------------")

start_time = time.time()
last_log_time = time.time()

for step in range(n_steps):
    all_ids, all_masked_ids = next(data_iter)
    all_ids, all_masked_ids = all_ids.to(device), all_masked_ids.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(all_ids)
        loss = loss_fn(logits.transpose(1, 2), all_masked_ids)
    loss.backward()
    if mask:
        apply_mask_to_grad(model, mask)
    current_lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = current_lr
    optimizer.step()
    if mask:
        apply_mask(model, mask)

    if step % 100 == 0:
        sps = 100 / (time.time() - last_log_time) if step > 0 else 0
        last_log_time = time.time()
        if sps > 0:
            secs = (n_steps - step) / sps
            h, m = int(secs // 3600), int(secs % 3600 // 60)
            eta = f"{h}h{m:02d}m"
        else:
            eta = "???"
        print(f"step {step}: loss={loss.item():.4f}, lr={current_lr:.6f}, {sps:.1f} steps/s, eta={eta}")
    if step == 0:
        continue

    if step == 1000:
        torch.save(model.state_dict(), f"{ckpt_dir}/lth_rewind_{step}.pt")
        print(f"SAVE on step {step}")

    if step % 10000 == 0:
        accs = evaluate(model)
        acc_str = ", ".join(f"p{p}={a:.3f}" for p, a in accs.items())
        print(f"EVAL on step {step}: {acc_str}")

print("Final eval...")
final_accs = evaluate(model, 5000, n_show=5)
acc_str = ", ".join(f"p{p}={a:.3f}" for p, a in final_accs.items())
print(f"FINAL EVAL: {acc_str}")

torch.save(model.state_dict(), f"{ckpt_dir}/final.pt")
if mask:
    # store the full per-p accuracy dict as the "accuracy" payload
    save_lth_checkpoint(
        f"{ckpt_dir}/mask.pt", mask=mask,
        sparsity=sparsity, round_num=LTH_ROUND, accuracy=final_accs,
    )
print("Done")