---
title: "Running on Curnagl, SPARK or any Linux GPU machine"
description: "Running on UNIL's Curnagl cluster, the DESI lab's SPARK cluster or any Linux GPU machine: storage, environment, jobs"
group: working
order: 5
---

The harness runs on the DESI lab's SPARK cluster through a separate repo, since 2026-10-16 ([SPARK](#spark-desi-lab-cluster-ci)). It is set up for UNIL's Curnagl cluster, where the team has no project yet, and it runs on any Linux machine with an NVIDIA GPU and CUDA 12 drivers. Laptops (Apple Silicon) run the tests and the smoke test ([smoke_test.md](smoke_test.md)).

## Getting on Curnagl

As of 2026-10-16 the team's DCSR project is still unconfirmed, so nobody has run a job here yet: a compute project must be requested at <https://requests.dcsr.unil.ch>, and each member needs a UNIL account added to it. Connect through the UNIL VPN, then `ssh <unil-user>@curnagl.dcsr.unil.ch`. Get the project's account name from `Sproject` and put it in the `--account` line of the files in `slurm/`.

## Partitions

| Partition     | GPUs                   | Use for                                        |
| ------------- | ---------------------- | ---------------------------------------------- |
| `gpu-l40`     | 8 x L40S 48 GB         | inference (DCSR's recommendation); the default |
| `gpu`         | 14 x A100 40 GB        | inference, fine-tuning                         |
| `gpu-h100`    | 8 x H100 94 GB         | anything bigger                                |
| `interactive` | 20 GB A100 slices, 8 h | setup and debugging only                       |
| `gpu-gh`      | GH200 (ARM)            | not supported: the lock has no ARM wheels      |

Jobs run at most 3 days. Ask for one GPU and a realistic `--time`; small, short jobs start sooner. Scoring is sharded and resumable, so resubmit a job that times out.

## Storage layout

Full tree, naming rules and what's safe to delete: [store_layout.md](store_layout.md). On Curnagl:

```
/work/<project>/deeptrace-bench/      DTB_ROOT (no backup; quota set by the project)
├── datasets/<id>/                    downloads, or symlinks to copies obtained by hand
├── weights/<model>/                  weight files, hashes pinned in configs/weights.lock.yaml
├── upstream/<repo>-<commit>.tar.gz   archive of every pinned upstream checkout
├── manifests/<id>.parquet
├── splits/
├── faces/<dataset>/<detector>__<hash>/  cached face detections
├── scores/<run_id>/                  run.json + score parts
├── results/<run_id>/                 full evaluation output
├── smoke/                            pipeline tests (same layout); safe to delete
└── hf/                               HF_HOME
$TMPDIR/dtb-cache/                    DTB_CACHE inside jobs: node-local NVMe, wiped after
```

- `/users` is only 50 GB: the repo clone can live there, the venv and uv cache not. Set `UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` to folders under `/work`, and `UV_LINK_MODE=copy`.
- `/scratch` is purged when it fills: nothing that must survive goes there. Copy anything precious (published results, weight hashes) into git or onto `/nas`.
- No millions of small files (frames, crops) on `/work`. Cache crops in `$TMPDIR` during a job, or pack them into shards.

## First-time setup

```bash
git clone https://github.com/Colombo-HCI-Lab/deeptrace-bench.git && cd deeptrace-bench
git config core.hooksPath scripts/hooks      # refuses commits containing data
cp .env.example .env                         # set DTB_ROOT, HF_HOME
uv sync --frozen                             # CUDA 12.8 wheels on Linux
mkdir -p logs                                # SLURM writes job logs here
uv run pytest
```

`uv sync` and all downloads need internet. Run `setup_models.py` and `setup_datasets.py` on the login node (moving files is allowed there, computing is not). Compute-node internet is unverified, so jobs set `HF_HUB_OFFLINE=1` and assume none.

## Jobs

```bash
sbatch slurm/prepare.sbatch urdu_csalt banglafake           # build manifests (CPU)
sbatch --array=0-7 slurm/score.sbatch aasist urdu_csalt     # 8 shards on gpu-l40
sbatch --dependency=afterok:<jobid> slurm/evaluate.sbatch aasist urdu_csalt
```

Queue: `squeue -u $USER`; pending jobs on a partition: `squeue -p gpu-l40 -t PD`; a job's estimated start: `squeue --start -j <jobid>`.

## SPARK (DESI lab, Cluster-CI)

The DESI lab at HEC Lausanne runs its own GPU cluster, SPARK, and gave the team access on 2026-10-14 through the [UNIL-DESI](https://github.com/UNIL-DESI) GitHub org. It has no SSH or SLURM. Instead, Cluster-CI runs a repo's `dvc.yaml` in a Docker container on a free worker, and it only runs repos inside that org.

The harness runs there from a separate private repo, [UNIL-DESI/deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments). It pins a commit of this repo in `params.yaml`, builds this repo's own `uv.lock` environment on the worker and runs the scripts as DVC stages. Only aggregate summaries come back to git; per-item scores, crops and the store stay on the worker. Code changes always land here first, then the pin moves.

Status on 2026-10-16: this repo's tests and smoke test (`pytest`, then `scripts/smoke.py` for GenD and AASIST) pass there, on the x86 worker, in this repo's own environment. The first scoring run, AASIST on Urdu CSALT, matched the 2026-10-12 round within 2e-4 on AUC and EER, with the same manifest and counts. Scoring took under two minutes there.

| Worker | GPU | For us |
| --- | --- | --- |
| `isipol09` (x86) | RTX 3090, 24 GB, driver for CUDA 12.9 at most | runs this repo: our lock's torch is built for CUDA 12.8 |
| `HEC45801`, `HEC45803` (ARM) | NVIDIA GB10, about 120 GB unified memory | not yet: our lock has no ARM Linux wheels |

- **The cluster's default image** (`nvcr.io/nvidia/pytorch:26.05-py3`) brings torch for CUDA 13.2, which sees no GPU on `isipol09` because the driver there is too old. deeptrace-experiments gives that worker NGC `25.06-py3`, built for its driver. Our stages use this repo's own environment either way.
- **Downloads need a Hugging Face token**: the cluster shares one public address, and Hugging Face rate-limits anonymous downloads from it after a few hundred files. The token is in deeptrace-experiments' GitHub secrets as `HF_TOKEN`. Even with it, Hugging Face allows 1,000 API requests per 5 minutes, about one per file, so a dataset of thousands of files downloads over several windows; the score stage waits them out and checks the manifest before scoring.
- **Limits**: every job is capped at 24 hours, so long scoring must be sharded or resumed (`score.py` already resumes). A worker runs one job at a time, so a job can wait behind someone else's.
- **More**: deeptrace-experiments' `CLAUDE.md` has how to run a stage, which worker each needs and the setup traps (a repo's `pyproject.toml` must list `dvc-viewer`, `REQUIRED_RAM` of at least 16 GB). Cluster-CI's own docs are at <https://unil-desi.github.io/cluster-ci/>.

## Elsewhere

On any other Linux GPU machine, skip SLURM: set `DTB_ROOT` (and `DTB_CACHE` if `$TMPDIR` is small) and run the scripts directly, e.g. `uv run scripts/score.py --model aasist --evalset urdu_csalt --device cuda`.
