# Post-replication analyses (after the 3-seed sweep)

Three follow-up studies run after the 204-run replication completed.
All use the seed-67/68/69 checkpoints; the random-ticket control adds 8 new
200k-step training runs on the local RTX 5070 Ti.

- Analysis A — per-loop mechanism on matched failed/succeeded pairs
  (`analyze_loops.py`, raw traces in `results/loop_analysis.json`)
- Analysis B — mask structure across seeds: per-block density + inter-seed IoU
  (`analyze_masks.py`, raw data in `results/mask_analysis.json`)
- Random-ticket control — random masks with per-layer density matched to IMP
  (`make_random_mask.py`, `run_random_control.py`, `run_random_repeats.py`;
  results under `checkpoints_rand*/`)

Notation: FAIL = p16 accuracy < 0.35 (chance ≈ 0.25); OK = > 0.50.

---

## Analysis A — why matched runs land in different basins

Across the 8 (config, density) cells where seeds disagree at p=16
(23 runs: 10 FAIL, 13 OK), per-loop diagnostics on a fixed 512-example batch:

| outcome | n | output entropy, loop 1 → final |
|---|---|---|
| OK   | 13 | **falls by 0.307** (commits to an answer) |
| FAIL | 10 | **rises by 0.226** (never commits) |

(max/uniform entropy = ln 6 ≈ 1.792)

Successful runs reproduce the paper's §5.3 signature: logit direction set
early, entropy collapses in the final loops, early-exit accuracy jumps from
chance to ≥0.6 in the last loop. Failed runs never leave chance at ANY loop
and their entropy stays pinned near uniform (often slightly rising).

Example (3⊗2, 5.50% density, p16):

| | early-exit acc per loop | entropy per loop |
|---|---|---|
| seed 67 OK (0.628)  | 0.256 → 0.604 | 1.355 → 0.786 |
| seed 68 FAIL (0.250) | 0.227 → 0.252 | 1.342 → 1.380 |
| seed 69 OK (0.661)  | 0.264 → 0.660 | 1.372 → 0.746 |

**Conclusion:** the low-density failure is not degraded computation but the
iterative-refinement mechanism failing to engage at all. The bimodal outcome
distribution at p=16 is a bimodal "does refinement turn on" event.

---

## Analysis B — mask structure across seeds

### B1. Per-block weight retention at round 16 (2.81% global) — paper Table 2

| seed | cfg | blk0 | blk1 | blk2 | blk3 | blk4 | blk5 | lm_head | total |
|---|---|---|---|---|---|---|---|---|---|
| 67 | 6⊗1 | 0.09% | 0.41% | 1.25% | 3.08% | 4.91% | 7.12% | 13.80% | 2.82% |
| 68 | 6⊗1 | 0.02% | 0.43% | 1.38% | 3.35% | 4.96% | 6.69% | 15.62% | 2.82% |
| 69 | 6⊗1 | 0.12% | 0.49% | 2.10% | 3.57% | 4.74% | 5.84% | 10.81% | 2.82% |
| 67 | 3⊗2 | 2.12% | 3.06% | 3.24% | – | – | – | 10.03% | 2.82% |
| 68 | 3⊗2 | 1.98% | 3.08% | 3.36% | – | – | – | 11.20% | 2.82% |
| 69 | 3⊗2 | 2.19% | 3.01% | 3.22% | – | – | – |  9.77% | 2.82% |
| 67 | 2⊗3 | 2.53% | 3.08% | – | – | – | – |  8.85% | 2.82% |
| 68 | 2⊗3 | 2.64% | 2.97% | – | – | – | – |  9.11% | 2.82% |
| 69 | 2⊗3 | 2.59% | 3.02% | – | – | – | – |  7.29% | 2.82% |
| 67 | 1⊗6 | 2.80% | – | – | – | – | – |  8.20% | 2.82% |
| 68 | 1⊗6 | 2.80% | – | – | – | – | – |  7.29% | 2.82% |
| 69 | 1⊗6 | 2.80% | – | – | – | – | – |  7.68% | 2.82% |

Paper's Table 2 **replicates across all three seeds**: sharp depth gradient in
6⊗1 (early blocks pruned to near zero, late blocks hoard mass), near-uniform
allocation in looped configs, and lm_head escaping pruning with the same
architecture ordering as the paper (6⊗1 ≈ 14% vs 1⊗6 ≈ 7–8%).

### B2. Mask overlap (IoU) between seeds at the same (config, round)

Chance-level IoU for two independent masks at density d is d/(2−d):
0.057 at 10.74%, 0.028 at 5.50%, 0.014 at 2.81%.

Observed inter-seed IoU (rounds 10–16, all configs): 0.016–0.086 —
roughly 1.1–1.5× chance. Different seeds find almost entirely different
subnetworks.

Within a density level, IoU is the same whether the pair's outcomes agree or
disagree, e.g. 1⊗6 round 13: OK/FAIL pair IoU 0.031 vs OK/OK pair IoU 0.031.
Mask overlap does not predict which runs fail. (Aggregate means — mixed 0.030
vs same-outcome 0.042 — are confounded by density: failures concentrate at
low density where IoU is mechanically lower. Use the within-density
comparison.)

**Conclusion:** no evidence that "bad masks" cause the failures; points to
optimization stochasticity under the sparsity constraint rather than mask
path-dependence.

---

## Random-ticket control

Design: random masks with **per-layer density matched exactly** to the IMP
mask at that (config, round), applied to the same step-1000 rewind weights,
trained with the identical recipe (200k steps, seed 67 data order). This
isolates IMP's *weight selection* from its *per-layer allocation* — stronger
than the usual global-density random control.

### All runs (p8 / p16)

| cfg | density | ticket | p8 | p16 |
|---|---|---|---|---|
| 6⊗1 | 10.74% | IMP (s67) | 0.886 | 0.735 |
| 6⊗1 | 10.74% | random #1 | 0.886 | **0.742** |
| 6⊗1 | 5.50% | IMP (s67/68/69) | 0.848 / 0.851 / 0.887 | 0.629 / 0.691 / 0.742 |
| 6⊗1 | 5.50% | random #1 | 0.840 | 0.631 |
| 6⊗1 | 5.50% | random #2 | 0.849 | **0.252 (FAIL)** |
| 1⊗6 | 10.74% | IMP (s67) | 0.849 | 0.670 |
| 1⊗6 | 10.74% | random #1 | 0.835 | 0.651 |
| 1⊗6 | 5.50% | IMP (s67/68/69) | 0.806 / 0.863 / 0.774 | 0.558 / 0.253 / 0.522 |
| 1⊗6 | 5.50% | random #1 | 0.777 | 0.483 |
| 1⊗6 | 5.50% | random #2 | 0.774 | 0.514 |
| 1⊗6 | 5.50% | random #3 | 0.749 | 0.471 |
| 1⊗6 | 5.50% | random #4 | 0.752 | 0.281 (FAIL) |

### Findings

1. **p≤8: random ≈ IMP everywhere** (within 0.03, both architectures, both
   densities). At matched per-layer density, IMP's specific weight choices add
   nothing to easy-task accuracy at this scale.

2. **p16 at 5.50%: both ticket types are bimodal, means indistinguishable.**
   1⊗6 IMP {0.558, 0.253, 0.522} vs random {0.483, 0.514, 0.471, 0.281}.
   The apparent IMP advantage in the first random draw (+0.075) did not
   survive repeats.

3. **The failure basin is reachable with training seed and data order held
   fixed.** Random repeats #1–#4 differ only in which weights the mask keeps;
   outcomes still swing 0.281–0.514. Failure is a chaotic function of the
   mask/init interaction, not "unlucky training".

4. **First-ever 6⊗1 collapse came from a random mask.** IMP-masked 6⊗1 never
   failed in 3 seeds × 17 densities; a random mask at identical per-layer
   density failed on its 2nd draw (p8 unharmed: 0.849). Hypothesis: IMP
   selection buys collapse *robustness* for multi-block models rather than
   accuracy. Underpowered (0/3 vs 1/2 failures, Fisher p ≈ 0.4) — needs ~4
   more 6⊗1 random draws to test.

### Reframing for the paper

The "winning ticket" at this scale is the **per-layer density allocation**,
not the individual surviving weights. IMP's real discovery in 6⊗1 is the
depth gradient (block 0 → ~0.1%, block 5 → ~7%); give that allocation to a
random mask and it trains just as well on p≤8. In 1⊗6 there is only one block
— no allocation to discover — and correspondingly random matches IMP there
too. Combined with Analyses A and B: near the capacity floor, *which* weights
survive matters far less than *how many per layer*, and success at p=16
becomes a stochastic event that no ticket type controls.

---

## Compute

8 extra training runs × 200k steps on the local RTX 5070 Ti ≈ 21.8 GPU-hours,
$0. Analyses A and B: minutes, checkpoint-only.
