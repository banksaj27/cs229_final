# Running the addition IMP sweep on FarmShare

FarmShare fits this project's terms of use: coursework / unsponsored research,
low-risk data. 24 L40S GPUs behind Slurm.

## One-time setup (from any terminal)

```bash
ssh <sunetid>@rice.stanford.edu          # FarmShare login node

# get the code up (from your LOCAL machine, in cs229_final/):
#   scp -r *.py farmshare <sunetid>@rice.stanford.edu:~/cs229_final/

cd ~/cs229_final
python3 -m venv .venv-farmshare
source .venv-farmshare/bin/activate
pip install torch transformers psutil numpy

mkdir -p farmshare/logs
```

## Submit the whole 3-seed sweep (12 chains, 1 GPU each)

```bash
cd ~/cs229_final/farmshare
sbatch --array=0-11 chain_add.sbatch
squeue -u $USER            # watch
```

Each chain runs dense + 16 IMP rounds sequentially (~12-14h on an L40S).
If a job hits its 24h limit or dies, resubmit the same index — completed
rounds are skipped and mid-round training resumes from checkpoints:

```bash
sbatch --array=3 chain_add.sbatch      # e.g. retry chain 3 only
```

Check `sinfo` for the real GPU partition name and edit `#SBATCH --partition`
if `gpu` is wrong (docs call the GPU nodes "oat" servers).

## Pull results back (from your LOCAL machine)

```bash
scp -r <sunetid>@rice.stanford.edu:~/cs229_final/checkpoints_addition ./
python make_table.py --ckpt-root checkpoints_addition --p 2,4,8,16,24,32
```

Chain-to-index map: 0-3 = seed 67 (6x1, 3x2, 2x3, 1x6), 4-7 = seed 68,
8-11 = seed 69.
