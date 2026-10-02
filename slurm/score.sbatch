#!/bin/bash
# Score one model on one evalset as an array job: each task scores one stable shard and
# resumes where it stopped, so a task that hits the time limit can simply be resubmitted.
#
#   sbatch --array=0-7 slurm/score.sbatch aasist urdu_csalt
#SBATCH --job-name=dtb-score
#SBATCH --account=REPLACE_WITH_PROJECT
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
model=$1
evalset=$2
# Preprocessed inputs go to the node-local NVMe; DTB_CACHE defaults to $TMPDIR/dtb-cache.
unset DTB_CACHE
export UV_LINK_MODE=copy HF_HUB_OFFLINE=1
uv run --frozen scripts/score.py --model "$model" --evalset "$evalset" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" --device cuda
