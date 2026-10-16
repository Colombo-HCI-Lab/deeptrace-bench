# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this repo is

The benchmark harness for the Colombo HCI Lab's South Asia deepfake detection project: open video and audio detectors run on South Asian deepfake datasets under one evaluation setup, with results reported per group. It starts with existing datasets and models (reproduce published numbers, zero-shot runs, fine-tuned baselines) and later evaluates the data collected through the sibling repo `deeptrace`. `README.md` has the workflow. `docs/` is the public site (https://colombo-hci-lab.github.io/deeptrace-bench/, Astro): five guided pages (overview, Results, Progress, How it works, Models & data) plus every doc under `/docs/`, written in `docs/src/content/docs/` (models, datasets, evaluation, cluster, data policy and the rest).

Scaffolded 2026-10-03; runs end to end since 2026-10-04. As of 2026-10-05 every configured model has an adapter except AltFreezing (weights unreachable): GenD (CLIP and PE backbones), SBI, LipForensics (FAN landmarks, upstream's mouth crops), nine DeepfakeBench detectors and Effort through one adapter (Effort's weights wait on Google Drive's quota), AASIST, AASIST-L, AASIST3, the XLS-R family (AASIST, SLS and Mamba back ends, run without fairseq or mamba-ssm through `models/_fairseq.py` and `models/_mamba.py`), AntiDeepfake MMS-300M and wav2vec 2.0 small, DF Arena 500M and 1B, and the audio-visual HAVIC. Seventeen pass parity (`scripts/parity.py`, `results/parity/`); the rest, GenD among them, pass strict loads and the pipeline check only. The first zero-shot round ran on 2026-10-12: 48 model and evalset pairs scored in all, findings in `docs/src/content/docs/results.md`, the round-by-round record in `progress.md` beside it. Builders are written for every open or click-through dataset, including parquet-row datasets (IndicSynth, IndicTTS challenge), zips inside zips (BD-GRF6) and tar-only IndicSUPERB. Still stubs: the request-only and gated builders whose files nobody has seen (FakeAVCeleb, DeePhy, DF-Platter, InDeepFake, IndieFake, Deepfake-Eval-2024, Casual Conversations v2, SEA-Spoof, Svarah), native preprocessing, the shortcut probe.

## Commands

```bash
uv sync                                    # Python 3.12, torch 2.8 (cu128 on Linux)
uv run pytest                              # synthetic fixtures only
uv run ruff check . && uv run ruff format .
uv run scripts/setup_models.py --list | <ids> [--record] | --wave N
uv run scripts/setup_datasets.py --list | <ids> [--languages ..] | <id> --from PATH | --manifest
uv run scripts/setup_datasets.py <ids> --sample N [--seed S] --manifest   # smoke namespace
uv run scripts/score.py --model <id> --evalset <id> [--shard i/n] [--smoke] [--save-crops K]
uv run scripts/evaluate.py --model <id> --evalset <id> [--publish] [--smoke]
uv run scripts/smoke.py [--model gend] [--sample 2] [--device auto|cpu|mps|cuda] [--keep]
uv run scripts/parity.py export --model <id> --evalset <id> --smoke   # then the reference and compare it prints
uv run scripts/profile_models.py --device cuda   # params, FLOPs, GPU time per model -> results/profile/
uv run scripts/build_site.py [--citations] # rewrite the site's data (status.json, catalog.md) from the store
cd docs && npm ci && npm run dev           # the site locally; npm run build fails on a broken doc link
```

GPU runs on the DESI lab's SPARK cluster go through a separate private repo, [UNIL-DESI/deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (cloned at `~/workspace/UNIL-DESI/deeptrace-experiments/`, its own `CLAUDE.md`), because Cluster-CI only runs repos in that org. It pins a commit of this repo in `params.yaml` and runs these scripts as DVC stages in this repo's own locked venv, so code changes land here first and then the pin moves. Since 2026-10-16 the tests and smoke test pass there on the x86 worker (an RTX 3090), and a scoring run (AASIST on Urdu CSALT) reproduced the 2026-10-12 numbers within 2e-4. `docs/src/content/docs/cluster.md` has the team-facing summary.

Paths come from `DTB_ROOT` and `DTB_CACHE` (`.env`, see `.env.example`); `docs/src/content/docs/store_layout.md` maps the tree. `DTB_NAMESPACE=smoke` (set by `--smoke`, `--sample` and `smoke.py`, never in `.env`) moves everything item-level under `DTB_ROOT/smoke/`.

## Rules

- **No data in git.** Datasets, weights, manifests, per-item scores, split assignments and per-subject tables live under `DTB_ROOT`. Only code, configs, docs and aggregate tables are committed. `scripts/hooks/pre-commit` enforces it; keep it working. Test fixtures are synthetic.
- **One environment.** No dlib, fairseq or mamba-ssm. Upstream code is cloned at a pinned commit into `third_party/` (git-ignored) and loaded by adapters; never copy code from Effort (no licence), SBI (research only) or DeepfakeBench (CC BY-NC) into the repo. Compatibility fixes go in `patches/<repo>/`.
- **Parity before results.** An adapter counts only after its scores match the upstream code in its original environment on about 200 items (within about 1e-3) and its score direction checks out on the reproduction evalset. See `docs/src/content/docs/models.md`.
- **Evalsets, not datasets.** Score only through `configs/evalsets/`. Cross-corpus evalsets never feed headline numbers.
- **The contamination guard is only as good as `training_data`.** List every dataset a model's released weights saw, including pretraining, and give each dataset its `derived_from`. Don't bypass the guard with `--force` for anything reported.
- **One evaluation setup.** `configs/eval/default.yaml` is the only place settings live; it is `proposed` until the team settles decision 5.
- **Weight hashes** are pinned in `configs/weights.lock.yaml` via `--record` on first fetch, never edited by hand.
- **Failures are data.** Preprocessing failures get a status and are counted per group, never dropped.
- **Pipeline tests are never results.** Datasets and evalsets with `role: pipeline_test` only prove the code works; the registry keeps them out of real evalsets and `--publish` refuses them and anything in the smoke namespace. Run `scripts/smoke.py` after any change to the pipeline.
- **Builders are samplable.** A dataset builder defines `build_manifest` and a way to label items before download: `label_from_path`, or `METADATA_FILES` with `labels_from_metadata` where labels live in a file, or `ROW_COLUMNS` with `label_from_row` for audio embedded in parquet rows (`datasets/_parquet.py`). It works on a partial tree; test fixtures use invented file names.
- **Conversions never overwrite downloads.** A weight whose Hub file is called `model.safetensors` gets its own local `name`, since the converted file has that name; `save_converted` refuses the clash.
- **Upstream code is imported, never copied.** Through `models/_upstream.py`; code inside a Hugging Face repo is pinned as `upstream.host: hf` and never loaded with `trust_remote_code`. Downloads are converted once to `model.safetensors` and loaded with every key matching.

## Conventions

- `uv` for everything. Logging, not print, in library code; scripts print only tables. Docstrings on public functions; comments say why.
- Config ids match file names. Dataset and model facts in configs and docs carry the date they were checked.
- Commit titles start lower case, no prefix, one short line.
- Docs are read by the whole team: no personal machines, no people outside the team's working group.
- **Docs are site pages.** Each is a Markdown file in `docs/src/content/docs/` with `title`, `description`, `group` (`start`, `findings`, `benchmark` for the reference, or `working`) and `order` in its front matter, and no H1 (the layout prints the title). Link another doc as `name.md#section` and a repo file as `/path/from/repo/root`; both also work on GitHub, and the build fails on a broken one. `docs/src/reference/status.json` and `catalog.md` are written by `build_site.py`, never by hand; the descriptions behind them live in `docs/src/reference/catalog.yaml`. A push to `main` touching `docs/` redeploys the site (`.github/workflows/pages.yml`). The design comes from the Claude Design project for the docs: tokens and shared styles in `docs/src/styles/site.css`, light only, one diverging AUC scale (`--good`, `--bad`) as the only colour. The Results and Progress pages are built from `results.md` and `progress.md` by heading (`docs/src/lib/guide.ts`), so keep their H2 and H3 titles and the `- **Label.** body` form of their lists; links to those two docs land on the guided pages, the full write-ups stay at `/docs/results/` and `/docs/progress/`.
- **After each round of runs**, add a dated entry at the top of the timeline in `progress.md` (what ran, commit range), update its "Where things stand" and "Next", and refresh `results.md` and `experiments.md` if new numbers or experiments came in. Numbers there are dated snapshots copied from the store's aggregates.
