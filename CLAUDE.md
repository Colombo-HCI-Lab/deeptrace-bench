# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this repo is

The benchmark harness for the Colombo HCI Lab's South Asia deepfake detection project: open video and audio detectors run on South Asian deepfake datasets under one evaluation setup, with results reported per group. It starts with existing datasets and models (reproduce published numbers, zero-shot runs, fine-tuned baselines) and later evaluates the data collected through the sibling repo `deeptrace`. `README.md` has the workflow; `docs/` has models, datasets, evaluation, cluster and data policy.

Scaffolded 2026-10-03: configs, setup scripts, contamination guard, metrics, resumable scoring and tests work; model adapters (`models/*.py` other than `dummy.py`), dataset builders (`datasets/*.py`) and face preprocessing are documented stubs.

## Commands

```bash
uv sync                                    # Python 3.12, torch 2.8 (cu128 on Linux)
uv run pytest                              # synthetic fixtures only
uv run ruff check . && uv run ruff format .
uv run scripts/setup_models.py --list | <ids> [--record] | --wave N
uv run scripts/setup_datasets.py --list | <ids> [--languages ..] | <id> --from PATH | --manifest
uv run scripts/score.py --model <id> --evalset <id> [--shard i/n]
uv run scripts/evaluate.py --model <id> --evalset <id> [--publish]
```

Paths come from `DTB_ROOT` and `DTB_CACHE` (`.env`, see `.env.example`).

## Rules

- **No data in git.** Datasets, weights, manifests, per-item scores, split assignments and per-subject tables live under `DTB_ROOT`. Only code, configs, docs and aggregate tables are committed. `scripts/hooks/pre-commit` enforces it; keep it working. Test fixtures are synthetic.
- **One environment.** No dlib, fairseq or mamba-ssm. Upstream code is cloned at a pinned commit into `third_party/` (git-ignored) and loaded by adapters; never copy code from Effort (no licence), SBI (research only) or DeepfakeBench (CC BY-NC) into the repo. Compatibility fixes go in `patches/<repo>/`.
- **Parity before results.** An adapter counts only after its scores match the upstream code in its original environment on about 200 items (within about 1e-3) and its score direction checks out on the reproduction evalset. See `docs/models.md`.
- **Evalsets, not datasets.** Score only through `configs/evalsets/`. Cross-corpus evalsets never feed headline numbers.
- **The contamination guard is only as good as `training_data`.** List every dataset a model's released weights saw, including pretraining, and give each dataset its `derived_from`. Don't bypass the guard with `--force` for anything reported.
- **One evaluation setup.** `configs/eval/default.yaml` is the only place settings live; it is `proposed` until the team settles decision 5.
- **Weight hashes** are pinned in `configs/weights.lock.yaml` via `--record` on first fetch, never edited by hand.
- **Failures are data.** Preprocessing failures get a status and are counted per group, never dropped.

## Conventions

- `uv` for everything. Logging, not print, in library code; scripts print only tables. Docstrings on public functions; comments say why.
- Config ids match file names. Dataset and model facts in configs and docs carry the date they were checked.
- Commit titles start lower case, no prefix, one short line.
- Docs are read by the whole team: no personal machines, no people outside the team's working group.
