"""Analysis A: why do matched runs at the same density land in different basins?

Finds (config, round) cells where seeds disagree at p=16 -- one seed near
chance, another well above -- then for each member of the pair replicates the
paper's per-loop diagnostics (sections 5.2 / 5.3):

  * early-exit accuracy after each loop iteration
  * output entropy after each loop
  * hidden-state cosine similarity between consecutive loops
  * hidden-state / logit cosine similarity to the final loop

If the failed runs simply never perform the entropy collapse the paper
identifies as "committing to an answer", the bimodality is the iterative
refinement mechanism failing to engage -- not generic noise.

Usage: python analyze_loops.py [--p 16] [--n-examples 512]
"""
import argparse
import glob
import json
import math
import os

import torch
import torch.nn.functional as F

from model import Transformer
from phop import generate_one_example, encode, PAD_ID, VOCAB_SIZE, char_to_id

D_MODEL, N_HEADS, D_FF, SEQ_LEN = 128, 8, 512, 256
MAX_SEQ_LEN = SEQ_LEN + 2
SEEDS = (67, 68, 69)
CONFIGS = ("6x1", "3x2", "2x3", "1x6")
FAIL, OK = 0.35, 0.50


def strip(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_results():
    acc = {}
    for path in glob.glob("checkpoints/seed_*/*/**/results.json", recursive=True):
        r = json.load(open(path))
        acc[(r["seed"], r["config"], r["round"])] = r["accuracy"]
    return acc


def find_pairs(acc, p):
    """cells where >=1 seed failed and >=1 seed succeeded"""
    pairs = []
    for cfg in CONFIGS:
        for rnd in range(17):
            vals = {s: acc[(s, cfg, rnd)][str(p)]
                    for s in SEEDS if (s, cfg, rnd) in acc}
            if len(vals) < 2:
                continue
            failed = [s for s, v in vals.items() if v < FAIL]
            good = [s for s, v in vals.items() if v > OK]
            if failed and good:
                pairs.append((cfg, rnd, failed, good, vals))
    return pairs


@torch.no_grad()
def per_loop_stats(model, K, L, ids, prompt_len, answers, device):
    """returns dict of per-loop-iteration diagnostics"""
    X = model.token_embeddings(ids)
    states = []
    for _ in range(L):
        for block in model.blocks:
            X = block(X)
        states.append(X.clone())

    q = prompt_len - 1  # position of the query token's prediction
    out = {"early_acc": [], "entropy": [], "sim_prev": [], "sim_final": [],
           "logit_sim_final": []}
    final_h = states[-1][:, q, :]
    final_logits = model.lm_head(model.ln_final(states[-1]))[:, q, :]
    for i, S in enumerate(states):
        h = S[:, q, :]
        logits = model.lm_head(model.ln_final(S))[:, q, :]
        probs = F.softmax(logits.float(), dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum(-1).mean().item()
        preds = logits.argmax(-1)
        acc = (preds == answers).float().mean().item()
        out["early_acc"].append(acc)
        out["entropy"].append(ent)
        out["sim_final"].append(F.cosine_similarity(h, final_h, dim=-1).mean().item())
        out["logit_sim_final"].append(
            F.cosine_similarity(logits.float(), final_logits.float(), dim=-1).mean().item())
        out["sim_prev"].append(
            F.cosine_similarity(h, states[i - 1][:, q, :], dim=-1).mean().item()
            if i > 0 else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--n-examples", type=int, default=512)
    ap.add_argument("--out", default="results/loop_analysis.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc = load_results()
    pairs = find_pairs(acc, args.p)
    print(f"found {len(pairs)} matched failed/succeeded cells at p={args.p}\n")

    # one fixed eval batch so all models see identical inputs
    torch.manual_seed(0)
    import random as _r
    _r.seed(0)
    ex = [generate_one_example(args.p, SEQ_LEN) for _ in range(args.n_examples)]
    seqs, ans = zip(*ex)
    prompt_ids = [encode(s + ">") for s in seqs]
    prompt_len = len(prompt_ids[0])
    padded = [pr + [PAD_ID] * (MAX_SEQ_LEN - len(pr)) for pr in prompt_ids]
    ids = torch.tensor(padded, device=device)
    answers = torch.tensor([char_to_id[a] for a in ans], device=device)

    report = []
    for cfg, rnd, failed, good, vals in pairs:
        K, L = map(int, cfg.split("x"))
        density = 0.8 ** rnd
        print(f"=== {cfg} round {rnd} (density {density:.2%}) "
              f"p{args.p}: " + ", ".join(f"s{s}={v:.3f}" for s, v in sorted(vals.items())) + " ===")
        entry = {"config": cfg, "round": rnd, "density": density,
                 "accs": {str(s): v for s, v in vals.items()}, "runs": {}}
        for s in sorted(vals):
            d = (f"checkpoints/seed_{s}/{cfg}" if rnd == 0
                 else f"checkpoints/seed_{s}/{cfg}/lth/round_{rnd}")
            model = Transformer(K, L, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF,
                                MAX_SEQ_LEN).to(device).eval()
            model.load_state_dict(strip(torch.load(f"{d}/final.pt",
                                                   map_location=device, weights_only=True)))
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                st = per_loop_stats(model, K, L, ids, prompt_len, answers, device)
            tag = "FAIL" if vals[s] < FAIL else ("OK  " if vals[s] > OK else "mid ")
            print(f"  seed {s} [{tag}] acc={vals[s]:.3f}")
            print(f"    early-exit acc : " + " ".join(f"{v:.3f}" for v in st["early_acc"]))
            print(f"    entropy        : " + " ".join(f"{v:.3f}" for v in st["entropy"]))
            print(f"    sim to final   : " + " ".join(f"{v:.3f}" for v in st["sim_final"]))
            entry["runs"][str(s)] = {"tag": tag.strip(), "acc": vals[s], **st}
            del model
            torch.cuda.empty_cache()
        report.append(entry)
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")

    # headline summary: does entropy collapse in the last loops?
    print("\n=== entropy drop (loop 1 -> final), by outcome ===")
    drops = {"FAIL": [], "OK": []}
    for e in report:
        for s, r in e["runs"].items():
            if r["tag"] in drops and len(r["entropy"]) > 1:
                drops[r["tag"]].append(r["entropy"][0] - r["entropy"][-1])
    for tag, ds in drops.items():
        if ds:
            print(f"  {tag}: n={len(ds)}  mean drop={sum(ds)/len(ds):+.3f}  "
                  f"(max uniform entropy = {math.log(VOCAB_SIZE):.3f})")


if __name__ == "__main__":
    main()
