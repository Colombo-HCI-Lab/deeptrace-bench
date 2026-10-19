---
title: "Running on Curnagl or any GPU machine"
description: "Running the harness on UNIL's Curnagl cluster or any Linux GPU machine: setup, storage, jobs"
group: working
order: 5
---

Where the harness runs, as of 2026-10-16:

| Where | Status | How |
| --- | --- | --- |
| The DESI lab's SPARK cluster | works; the first scoring run reproduced a published number | [spark.md](spark.md) |
| UNIL's Curnagl cluster | set up, never run: the team has no compute project yet | below |
| Any Linux machine with an NVIDIA GPU and CUDA 12 drivers | works | [below](#any-linux-gpu-machine) |
| A laptop (Apple Silicon) | tests and the smoke test only | [smoke_test.md](smoke_test.md) |

## Any Linux GPU machine

```bash
git clone https://github.com/Colombo-HCI-Lab/deeptrace-bench.git && cd deeptrace-bench
git config core.hooksPath scripts/hooks      # refuses commits containing data
cp .env.example .env                         # set DTB_ROOT (and DTB_CACHE if /tmp is small)
uv sync --frozen                             # CUDA 12.8 wheels on Linux
uv run pytest
uv run scripts/score.py --model aasist --evalset urdu_csalt --device cuda
```

The scripts are the same everywhere; only storage and job submission differ.

## Curnagl

**Access.** A DCSR compute project must be requested at <https://requests.dcsr.unil.ch>, and each member needs a UNIL account added to it. Then connect through the UNIL VPN, `ssh <unil-user>@curnagl.dcsr.unil.ch`, and put the project's account name (from `Sproject`) in the `--account` line of the files in [`slurm/`](/slurm).

**Partitions.** Jobs run at most 3 days. Ask for one GPU and a realistic `--time`: small, short jobs start sooner.

| Partition | GPUs | Use for |
| --- | --- | --- |
| `gpu-l40` | 8 x L40S 48 GB | inference; the default |
| `gpu` | 14 x A100 40 GB | inference, fine-tuning |
| `gpu-h100` | 8 x H100 94 GB | anything bigger |
| `interactive` | 20 GB A100 slices, 8 h | setup and debugging only |
| `gpu-gh` | GH200 (ARM) | not supported: `uv.lock` has no ARM wheels |

**Storage.** Put `DTB_ROOT` at `/work/<project>/deeptrace-bench/` (no backup) and use `$TMPDIR/dtb-cache/` (fast node-local disk, wiped after each job) as `DTB_CACHE`. The tree is in [store_layout.md](store_layout.md).

- `/users` holds only 50 GB: the repo clone fits there, the venv and uv cache don't. Set `UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` to folders under `/work`, and `UV_LINK_MODE=copy`.
- `/scratch` is purged when it fills, so nothing that must survive goes there.
- Don't write millions of small files (frames, crops) to `/work`; keep them in `$TMPDIR` during a job.

**Setup.** As above, plus `mkdir -p logs` for SLURM's logs. Downloads need internet, which compute nodes may not have, so run `setup_models.py` and `setup_datasets.py` on the login node; jobs set `HF_HUB_OFFLINE=1`.

**Jobs.** Scoring is sharded and resumable: resubmit a job that times out.

```bash
sbatch slurm/prepare.sbatch urdu_csalt banglafake           # build manifests (CPU)
sbatch --array=0-7 slurm/score.sbatch aasist urdu_csalt     # 8 shards on gpu-l40
sbatch --dependency=afterok:<jobid> slurm/evaluate.sbatch aasist urdu_csalt
squeue -u $USER                                             # your jobs
squeue --start -j <jobid>                                   # a job's estimated start
```
