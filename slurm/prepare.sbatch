#!/bin/bash
# Unpack downloads and build manifests on a CPU node. Download first on the login node
# (compute nodes may have no internet): uv run scripts/setup_datasets.py <ids>
#
#   sbatch slurm/prepare.sbatch urdu_csalt banglafake
#SBATCH --job-name=dtb-prepare
#SBATCH --account=REPLACE_WITH_PROJECT
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy HF_HUB_OFFLINE=1
uv run --frozen scripts/setup_datasets.py "$@" --skip-download --manifest
