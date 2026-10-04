# Running on Curnagl (or any Linux GPU machine)

The harness targets UNIL's Curnagl cluster but runs on any Linux machine with an NVIDIA GPU and CUDA 12 drivers. Laptops (Apple Silicon) run the tests and the smoke test ([smoke_test.md](smoke_test.md)), which scores a few items per dataset on the laptop's GPU or CPU.

## Getting on Curnagl

As of 2026-10-03 nobody on the team can run jobs yet: a compute project has to be requested at <https://requests.dcsr.unil.ch>, and every member needs a UNIL account added to it. Connect through the UNIL VPN, then `ssh <unil-user>@curnagl.dcsr.unil.ch`. Find the project's account name with `Sproject` and put it in the `--account` line of the files in `slurm/`.

## Partitions

| Partition     | GPUs                   | Use for                                        |
| ------------- | ---------------------- | ---------------------------------------------- |
| `gpu-l40`     | 8 x L40S 48 GB         | inference (DCSR's recommendation); the default |
| `gpu`         | 14 x A100 40 GB        | inference, fine-tuning                         |
| `gpu-h100`    | 8 x H100 94 GB         | anything bigger                                |
| `interactive` | 20 GB A100 slices, 8 h | setup and debugging only                       |
| `gpu-gh`      | GH200 (ARM)            | not supported: the lock has no ARM wheels      |

Jobs run at most 3 days. Ask for one GPU and a realistic `--time`; small, short jobs start sooner. Scoring is sharded and resumable, so a job that times out is simply resubmitted.

## Storage layout

The full tree, naming rules and what's safe to delete are in [store_layout.md](store_layout.md). On Curnagl:

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

- `/users` is only 50 GB, so the repo clone can live there but the venv and uv cache should not: set `UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` to folders under `/work`, and `UV_LINK_MODE=copy`.
- `/scratch` is purged when it fills, so nothing that must survive goes there. Copy anything precious (published results, weight hashes) into git or onto `/nas`.
- Don't spread millions of small files (frames, crops) on `/work`. Cache crops in `$TMPDIR` during a job, or pack them into shards.

## First-time setup

```bash
git clone https://github.com/Colombo-HCI-Lab/deeptrace-bench.git && cd deeptrace-bench
git config core.hooksPath scripts/hooks      # refuses commits containing data
cp .env.example .env                         # set DTB_ROOT, HF_HOME
uv sync --frozen                             # CUDA 12.8 wheels on Linux
mkdir -p logs                                # SLURM writes job logs here
uv run pytest
```

`uv sync` needs internet, as do all downloads. Run `setup_models.py` and `setup_datasets.py` on the login node (moving files is allowed there; computing is not). Whether compute nodes have internet is unverified, so jobs set `HF_HUB_OFFLINE=1` and assume they don't.

## Jobs

```bash
sbatch slurm/prepare.sbatch urdu_csalt banglafake           # build manifests (CPU)
sbatch --array=0-7 slurm/score.sbatch aasist urdu_csalt     # 8 shards on gpu-l40
sbatch --dependency=afterok:<jobid> slurm/evaluate.sbatch aasist urdu_csalt
```

Check the queue with `squeue -u $USER`, pending jobs on a partition with `squeue -p gpu-l40 -t PD`, and a job's estimated start with `squeue --start -j <jobid>`.

## Elsewhere

On any other Linux GPU machine, the same commands work without SLURM: set `DTB_ROOT` (and `DTB_CACHE` if `$TMPDIR` is small), then run the scripts directly, e.g. `uv run scripts/score.py --model aasist --evalset urdu_csalt --device cuda`.
