---
title: "How it works"
description: "From a published detector to one trusted number: the pipeline, the checks along it, the terms every doc uses, and the repo's layout"
---

deeptrace-bench runs published, open deepfake detectors on South Asian audio and video under one evaluation setup, and reports each result overall and per group (language, generator, gender, race). Each step below is one script; three checks decide whether a number counts.

## In plain words

Many groups have published deepfake detectors: programs that take a clip and say how likely it is to be fake. Nearly all were trained and tested on Western, mostly English data. This benchmark asks whether they still work on South Asian faces, voices and languages, and whether they work equally well for everyone. That is only a fair question if every detector is tested the same way, so the harness works like a conveyor belt: each detector and each test set is described once in a config, and every run passes through the same stations below, from download to a published summary.

A result counts only once it gets past three guards. Parity asks whether our wrapper scores exactly like the authors' own code, so a weak number means a weak model and not a bug of ours. The contamination guard stops a model being tested on data it saw in training. The publish guard keeps test runs and per-clip data out of the repo. Everything large (datasets, weights, per-clip scores) lives in a store outside git ([Data policy](data_policy.md)). The two numbers every run reports, AUC and EER, are explained under [Metrics](evaluation.md#metrics).

## The pipeline

| Step | Script | What it does | Writes, under `DTB_ROOT` |
| --- | --- | --- | --- |
| Configure | `configs/` | One YAML per model (upstream commit, weights, training data), dataset (access, licence, provenance) and evalset, plus the face detector and the one evaluation setup | nothing; configs live in git |
| Set up | `setup_models.py`, `setup_datasets.py` | Fetches pinned upstream code and hash-checked weights. Fetches a dataset, links a copy obtained by request (`--from`) or samples a few items (`--sample N`), then writes its manifest | `weights/`, `upstream/`, `datasets/`, `manifests/` |
| Score | `score.py` | Runs the contamination guard, then shared preprocessing (4 s audio windows; detected, aligned face crops, cached) and one P(fake) per item. Sharded; resumes after a kill | `faces/`, `scores/` |
| Evaluate | `evaluate.py` | AUC and EER with 95% intervals, overall and per group; failures counted, never dropped | `results/` |
| Publish | `evaluate.py --publish` | Copies the aggregate tables into the repo | the repo's `results/` |

| Other script | Use |
| --- | --- |
| `smoke.py` | the whole pipeline on a few items, with a readable report ([Getting started](getting_started.md)) |
| `parity.py` | checks an adapter against upstream code ([Adding a model](adding_a_model.md#5-before-any-result-counts)) |
| `profile_models.py` | parameters, FLOPs, GPU time and memory per model, into `results/profile/` |
| `build_site.py` | rewrites the site's data from the store; commit what it writes, and a push to `main` redeploys |
| `archive.py` | copies per-item results and run records into the project's private archive ([The store](store_layout.md#archive)) |

The store's full tree is in [The store](store_layout.md).

## The checks

| Check | What it stops |
| --- | --- |
| Parity | an adapter that scores differently from the authors' code: it must agree within 1e-3 on about 200 items |
| Contamination guard | a model scored on a dataset it trained on |
| Publish guard | pipeline tests, smoke runs and per-item data reaching the repo |

## Key terms

### What gets scored

| Term | Meaning |
| --- | --- |
| Dataset | a published corpus, configured once in `configs/datasets/` |
| Evalset | the exact slice that gets scored: which datasets, which items, which labels, how real and fake pair up. Everything is scored through evalsets |
| Manifest | one table per dataset, one row per item: label, generator, speaker or face, group attributes |
| Same corpus | real and fake items share recordings or speakers. Only these give headline numbers |
| Cross corpus | real and fake come from different corpora, so recording differences can give the label away. Never in headline numbers |
| Zero-shot | released weights scored on data outside their training, with no fine-tuning |
| P(fake) | a detector's output read as the probability that an item is fake; an item scores the mean over its windows or frames |
| False-positive rate | the share of real items flagged fake at the model's global threshold (its overall EER point), per group |
| Pipeline test | a dataset or evalset that only proves the code runs; never reported |
| Smoke namespace | `DTB_ROOT/smoke/`, where samples and smoke runs live apart from real ones |

### Why a number counts

| Term | Meaning |
| --- | --- |
| Adapter | the wrapper that runs a detector's upstream code in this repo's one environment |
| Parity | the adapter and the authors' code score the same items within 1e-3 |
| Reproduction | a detector scored on its own paper's test set and compared with the published number |
| Score direction | settled once a reproduction lands above chance, so a high score really means fake |
| Contamination guard | refuses a model on data it trained on. It compares datasets, not the generators behind them ([verdicts](evaluation.md#contamination-verdicts)) |
| Provisional | a run whose model lacks a parity check or a settled score direction |
| Publish guard | `evaluate.py --publish` refuses pipeline tests, the smoke namespace and per-item tables |
| Wave | the batch a model is scored in; wave 1 went first |
| Decision 4, decision 5 | choices the team hasn't settled yet ([Progress](progress.md#open-decisions-that-shape-the-results)) |

## The repo

| Folder | Holds |
| --- | --- |
| `configs/` | one YAML per model, dataset, evalset and tool; `eval/default.yaml`, the one evaluation setup; `corpora.yaml`, corpora seen only in training; `weights.lock.yaml`, every weight's hash |
| `src/deeptrace_bench/` | the harness: config registry, dataset builders, preprocessing, adapters, scoring, metrics and the guards |
| `scripts/` | the commands above, the pre-commit hook and the upstream-side parity scorers |
| `results/` | published aggregates, parity summaries, compute profiles. No per-item data |
| `patches/` | fixes for pinned upstream code, which is cloned into the git-ignored `third_party/` |
| `slurm/` | the same steps as job templates for Curnagl |
| `docs/` | this site |
| `tests/` | unit tests on synthetic fixtures |
| `DTB_ROOT`, outside git | datasets, weights, manifests, per-item scores ([Data policy](data_policy.md)) |
