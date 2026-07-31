#!/bin/bash
# One IMP chain (seed x config) for the addition task, rounds 0-16.
# Submit as an array over all 12 chains:
#   sbatch --array=0-11 chain_add.sbatch
# Fully resumable: if the job hits its time limit, just resubmit the same
# array index -- completed rounds are skipped, mid-round training resumes.
#SBATCH --job-name=lth-add
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/chain_%a.out

# NOTE: check the actual GPU partition name with `sinfo` and adjust
# --partition if needed (docs call the GPU nodes "oat" servers).

SEEDS=(67 67 67 67 68 68 68 68 69 69 69 69)
CONFIGS=(6x1 3x2 2x3 1x6 6x1 3x2 2x3 1x6 6x1 3x2 2x3 1x6)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR/.."
source .venv-farmshare/bin/activate

echo "chain: seed $SEED config $CFG on $(hostname), gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python run_seed_add.py --seed "$SEED" --configs "$CFG" --rounds 16
