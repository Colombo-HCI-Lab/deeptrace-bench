---
title: "Getting started on SPARK"
description: "From GitHub access to a scoring run on the DESI lab's GPU cluster: setup, your first run, scoring, moving the pin, and what to do when a run fails"
group: working
order: 6
---

SPARK is the DESI lab's GPU cluster at HEC Lausanne. You don't log in to it, and there is no queue to submit to by hand. You describe a run in a git repo and push a tag; a worker runs it in a Docker container, then commits the results back to the repo. This page takes you from nothing to a scoring run. The other places the harness runs (Curnagl, any Linux GPU machine) are in [cluster.md](cluster.md).

Everything here was checked on 2026-10-16, the day the first runs passed.

## How it fits together

Two repos are involved:

- **deeptrace-bench** (this repo, public) holds all the benchmark code. A change to how anything is scored lands here first.
- **[UNIL-DESI/deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments)** (private) holds what SPARK runs. Cluster-CI, the system behind SPARK, only runs repos inside the [UNIL-DESI](https://github.com/UNIL-DESI) GitHub org, so the runs live there. That repo holds no benchmark code. It pins a deeptrace-bench commit and says which stage to run.

```
you        edit params.yaml and .cluster-ci in deeptrace-experiments, commit, push the cluster-run tag
GitHub     the tag starts Cluster-CI's workflow, which hands the job to a free worker
worker     starts a container, checks out deeptrace-bench at bench.commit, builds its uv.lock
           environment, then runs the stage (fetch weights and data, score, evaluate)
bot        cluster-ci-bot commits reports/ (aggregates only) back to main
```

Datasets, weights and per-item scores stay on the worker, in a disk called the job volume, which survives between runs. A second run on a dataset already fetched skips the download.

## Before your first run

| You need | How to check | If it's missing |
| --- | --- | --- |
| Membership of the UNIL-DESI org | <https://github.com/UNIL-DESI> lists you under People, or an invitation email is waiting | ask Oshan, who asks the DESI lab |
| Write access to deeptrace-experiments | `gh repo view UNIL-DESI/deeptrace-experiments` works | ask Oshan |
| git, and the GitHub CLI signed in | `gh auth status` | install [gh](https://cli.github.com/), then `gh auth login` |

You don't need SSH, a UNIL account, a GPU or a Hugging Face token. Downloads use the repo's `HF_TOKEN` secret, a read-only token already set. The UNIL VPN only matters for Cluster-CI's dashboard (`http://130.223.73.209:5000/`). Logs reach you through GitHub without it.

Cluster-CI's own guide asks you to run its `install.sh` and use its `cluster-run` command. Skip both for this repo. The installer rewrites files deeptrace-experiments has already fixed (its workflow, `.cluster-ci`), and a plain git tag starts the same run.

## The repo in two minutes

```bash
gh repo clone UNIL-DESI/deeptrace-experiments
cd deeptrace-experiments
```

| File | What it's for | You edit it |
| --- | --- | --- |
| `.cluster-ci` | the job: RAM, GPU memory, disk, hours, `ALLOWED_WORKERS` (where it runs), `STAGES` (what it runs) | every run |
| `params.yaml` | `bench.commit` (the deeptrace-bench pin) and each stage's settings | when you score something new |
| `dvc.yaml` | the stages and the command each runs | rarely; the bot rewrites it, so pull first |
| `stages/` | small Python drivers that call deeptrace-bench's scripts | only to add a stage |
| `reports/` | what the bot commits back after a run | never; read it |

Never edit `.github/workflows/cluster-ci.yml`, `.cluster-ci-tools/`, `.dvc-viewer/` or the managed block in `AGENTS.md`: Cluster-CI owns them.

The four stages:

| Stage | Worker | What it does |
| --- | --- | --- |
| `hello` | any | prints the worker and Python version; the smallest possible run |
| `gpu_check` | `HEC45801`, `HEC45803` | GPU name, driver and matmul speed; reruns every time |
| `bench_smoke` | `isipol09` | this repo's `pytest` and smoke test at the pinned commit |
| `score` | `isipol09` | scores one model on one evalset and compares it with a published run |

## Your first run

`gpu_check` makes the best first run, because it always reruns (see [the troubleshooting table](#when-a-run-fails)). It shows the whole loop in a few minutes when a worker is free.

1. Pull first. The bot commits to `main` after every run.

   ```bash
   git pull --rebase
   ```

2. In `.cluster-ci`, point the job at the ARM workers and the stage:

   ```ini
   ALLOWED_WORKERS=HEC45801,HEC45803
   STAGES=gpu_check
   ```

3. Commit, push, then move the `cluster-run` tag to your commit. Pushing the tag is what starts the run:

   ```bash
   git commit -am "check the gpu on the arm workers"
   git push
   git tag -f cluster-run && git push -f origin cluster-run
   ```

4. Follow it:

   ```bash
   gh run list -R UNIL-DESI/deeptrace-experiments --limit 3
   gh run watch <run id> -R UNIL-DESI/deeptrace-experiments
   ```

5. Pull the bot's commit and read the report:

   ```bash
   git pull --rebase
   cat reports/gpu_check.json
   ```

   It names the GPU (an NVIDIA GB10) and its measured speed (about 90 TFLOP/s in bf16).

`.cluster-ci` on `main` always holds the settings of the last run, whoever started it, so read it before each run.

## Scoring a model on an evalset

The `score` stage runs the same scripts as a run on any other machine (`setup_models.py`, `setup_datasets.py --manifest`, `score.py --device cuda`, `evaluate.py`), in this repo's own locked environment, then checks the result against a published run.

1. In `params.yaml`, set the pair and the published run to compare with. Copy the reference numbers from the run's summary or [results.md](results.md):

   ```yaml
   score:
     model: aasist
     evalset: urdu_csalt
     reference:
       auc: 0.8632242084258313
       eer: 0.22218952324896998
       n: 6794
       n_failed: 19
     tolerance: 0.001
   ```

2. In `.cluster-ci`, set `ALLOWED_WORKERS=isipol09` and `STAGES=score`. Give it enough hours with `MAX_RUNTIME_HOURS`; 24 is the cap.
3. Commit, push and tag as in [your first run](#your-first-run), then pull once it ends.
4. Read `reports/score.json`:

   | Field | Means |
   | --- | --- |
   | `passed` | every step ran and the evaluation produced a summary |
   | `matches_reference` | same item and failure counts, and AUC and EER within `tolerance` of the reference |
   | `overall`, `diff` | the run's metrics, and how far they are from the reference |
   | `manifest_matches_repo` | the downloaded dataset is the same set of files as the one committed here |
   | `hf_token` | whether the Hugging Face secret reached the job (never its value) |
   | `steps` | each step's time; the dataset step repeats while downloads are rate-limited |

   `reports/score_by_method.csv` has the per-method table.

**What it can run today.** The stage re-checks a published pair, so it needs a reference. It also fetches the dataset named like the evalset and checks it against `results/manifests/<evalset>.json` in this repo. That covers the single-dataset evalsets with a committed manifest: `urdu_csalt`, `banglafake`, `bangla_voices`, `in_the_wild` and `indictts_challenge`. A pair never scored before, or an evalset built from another dataset (such as `indicsynth_bn`), needs the stage extended first. Ask Oshan before trying.

**How long it takes.** A dataset's first download is slow. Hugging Face lets the cluster fetch about 1,000 files per 5 minutes, so the stage downloads in rounds and waits out each window. Urdu CSALT's 6,800 clips took about 20 minutes. Scoring them took 88 seconds on the 3090. Later runs on the same worker skip the download.

## Moving the deeptrace-bench pin

When you change code here, land it here first: tests, the smoke test ([smoke_test.md](smoke_test.md)), push to `main`. Then point deeptrace-experiments at it:

```bash
git -C path/to/deeptrace-bench rev-parse HEAD     # the full commit hash
```

Set `bench.commit` in `params.yaml` to that hash, then commit and run. The worker checks the commit out and builds its environment from `uv.lock`, which takes about a minute when the lock changed and seconds when it didn't. A new pin reruns every stage that depends on it, so `bench_smoke` on `isipol09` is a good first check.

## Rules

- **Aggregates only.** `reports/` gets pass or fail, counts, timings and metric tables, never per-item scores, crops or anything from a licensed dataset ([data_policy.md](data_policy.md)). Don't declare those as DVC `outs` either: Cluster-CI backs outputs up to Google Drive.
- **Pull with `--rebase` before every edit.** The bot commits after each run and rewrites `dvc.yaml`.
- **Tag your own commit.** The bot's commits carry `[skip ci]`, so a tag on one starts nothing.
- **One run at a time.** Runs from this repo share `main`, and a worker runs one job at a time. Tell the team before starting a long one.
- **This repo's rules still hold**: score through evalsets only, keep the contamination guard on, and don't publish anything from the smoke namespace.
- **Secrets stay secrets.** Credentials go in the repo's GitHub secrets, which reach stages as environment variables. Never write one to a file or print it in a stage.

## When a run fails

Start with the full log, where every line a stage printed ends up:

```bash
gh run view <run id> --log -R UNIL-DESI/deeptrace-experiments > run.log
```

| What you see | Why | What to do |
| --- | --- | --- |
| The run passes in seconds and `reports/` doesn't change | DVC skips a stage whose inputs haven't changed since the last run | change what you meant to change (a param, the pin); rerunning the same settings does nothing |
| You pushed the tag and no run starts | the tagged commit is one of the bot's (`[skip ci]`) | make a commit of your own, tag it again |
| The run sits queued for hours | the worker is busy with someone else's job, which may run up to 24 hours | wait, or cancel with `gh run cancel <run id> -R UNIL-DESI/deeptrace-experiments` |
| It aborts in the "Workspace Sanitizer" with a message about `dvc.lock` hashes | a failed stage's reports were committed without a lock update | restore those reports from the last commit that matches the lock (`git checkout <commit> -- reports/<file>`), commit, run again |
| `dvc-viewer: command not found` before any stage | the worker's job volume lost a launcher (a Cluster-CI bug) | ask Oshan; the fix is one run on another image |
| "The NVIDIA driver on your system is too old" on `isipol09` | the image changed; only NGC `25.06-py3` matches that worker's driver | keep `DOCKER_IMAGE_AMD64=nvcr.io/nvidia/pytorch:25.06-py3` |
| `manifest_matches_repo: false` in `score.json` | the download didn't finish within its 12 rounds, or `hf_token` is false | check `hf_token`, then run again; the download resumes where it stopped |
| The job is killed by the memory guard | `REQUIRED_RAM` also has to cover Cluster-CI's own install | keep it at 16GB or more |
| The job stops at its time limit | it reached `MAX_RUNTIME_HOURS` | raise it (24 at most) and run again; scoring resumes from the parts already saved |

## Workers and limits

| Worker | GPU | Runs this repo |
| --- | --- | --- |
| `isipol09` (x86) | RTX 3090, 24 GB; driver for CUDA 12.9 at most | yes, with NGC `25.06-py3`; our torch is built for CUDA 12.8 |
| `HEC45801`, `HEC45803` (ARM) | NVIDIA GB10, about 120 GB shared with the CPU | not yet: our lock has no ARM Linux wheels |

- **The GB10s have far more memory, not more speed.** Their memory is slower than a 3090's, so a model that fits in 24 GB won't score faster there. They matter for work that doesn't fit in 24 GB, once the lock supports ARM.
- **Every job stops at 24 hours.** Long scoring has to be sharded or resumed, and `score.py` already resumes.
- **The cluster shares one public address.** Without a token, Hugging Face blocks downloads from it after a few hundred files. Even with the token, downloads come in rounds.

## Further reading

- deeptrace-experiments' `CLAUDE.md`: every stage in detail, how a Cluster-CI run works inside, and the full list of traps from the first runs.
- Cluster-CI's documentation: <https://unil-desi.github.io/cluster-ci/>. Its configuration reference lists every `.cluster-ci` key.
