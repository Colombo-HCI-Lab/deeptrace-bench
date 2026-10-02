#!/bin/bash
# Compute metrics once every shard of a run has finished.
#
#   sbatch --dependency=afterok:<score_job_id> slurm/evaluate.sbatch aasist urdu_csalt
#SBATCH --job-name=dtb-evaluate
#SBATCH --account=REPLACE_WITH_PROJECT
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy HF_HUB_OFFLINE=1
uv run --frozen scripts/evaluate.py --model "$1" --evalset "$2"
