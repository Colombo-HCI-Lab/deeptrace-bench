# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this repo is

The benchmark harness for the Colombo HCI Lab's South Asia deepfake detection project: open video and audio detectors run on South Asian deepfake datasets under one evaluation setup, with results reported per group. It starts with existing datasets and models (reproduce published numbers, zero-shot runs, then fine-tuned baselines) and later evaluates the data collected through the sibling repo `deeptrace`, the crowdsourced collection platform. See `README.md` for scope.

Created 2026-10-03. No code yet.

## Conventions

- A `uv` project on Python 3.12. Add dependencies with `uv add`, run with `uv run`.
- Never commit datasets, model weights, checkpoints, run outputs or participant data; `.gitignore` covers the usual paths. Most dataset licences are non-commercial research only and forbid passing the data on.
- One evaluation setup for every model and dataset (face crop, frames per clip, audio segment length, score aggregation, metrics). Once it is fixed, record the settings in `README.md` and keep every run on them, or results stop being comparable.
- Upstream detector code is wrapped, not edited in place, so each model's published numbers stay reproducible.
- Planning and findings live outside the repo; this repo holds code, configs, job scripts and result tables.
