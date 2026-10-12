---
title: "Overview"
description: "The docs at a glance, and how configs, setup, scoring and evaluation fit together"
group: start
order: 1
---

deeptrace-bench runs open deepfake detectors on South Asian audio and video under one evaluation setup and reports results overall and per group. [Models and data](https://colombo-hci-lab.github.io/deeptrace-bench/inventory/) lists every model, dataset and run; [Results](results.md) says what the runs show.

| Doc | What it covers |
| --- | --- |
| **Start here** | |
| [Progress](progress.md) | Where the benchmark stands, what each round of work added, and what comes next |
| **Experiments and results** | |
| [Experiments](experiments.md) | What has been run, the question each run answers, and how to run it again |
| [Results](results.md) | What the runs so far show, with numbers and caveats |
| **Reference** | |
| [Models](models.md) | Every detector: status, wave, where code and weights come from, training data, porting notes, parity check |
| [Datasets](datasets.md) | Every dataset: how to get it, who requests or signs, licence terms, what's in it |
| [Models and datasets](catalog.md) | Every detector and dataset in plain language: paper, size, cost, what each records (generated) |
| [Evaluation](evaluation.md) | Evalsets, preprocessing, metrics, failure accounting, contamination verdicts, the shortcut probe |
| [Data policy](data_policy.md) | What may enter git, licence clauses that bind us, sharing data across institutions |
| **Working on it** | |
| [Smoke test](smoke_test.md) | One command that runs the whole pipeline on a few items per dataset, and how to read its report |
| [The store](store_layout.md) | The `DTB_ROOT` tree: what each folder holds, naming rules, the smoke namespace, what's safe to delete |
| [Adding a model](adding_a_model.md) | Recipe: config, pinned weights, adapter, smoke test, parity (GenD as the worked example) |
| [Adding a dataset](adding_a_dataset.md) | Recipe: config, builder, sampling, evalset (the two pipeline-test datasets as worked examples) |
| [Running on Curnagl](cluster.md) | Running on UNIL's Curnagl cluster (or any Linux GPU machine): storage, environment, jobs |

## How the pieces fit

```
configs/datasets/*.yaml ──setup_datasets.py──▶ datasets/<id>/       (download, --from a copy, or --sample N)
                                     └─────▶ manifests/<id>.parquet  (one row per item)
configs/models/*.yaml ───setup_models.py───▶ third_party/<repo>-<commit>/  +  weights/<id>/
configs/tools/*.yaml ────(video models)────▶ weights/<tool>/        (the shared face detector)
configs/evalsets/*.yaml ─┐
configs/eval/default.yaml┼──score.py──────▶ faces/<dataset>/<detector>__<hash>/  (video and image: cached detections)
contamination guard ─────┘            └───▶ scores/<run_id>/        (run.json, score parts, optional crops)
                           evaluate.py ───▶ results/<run_id>/  ──--publish──▶ results/<run_id>/ (git)
                           smoke.py ──────▶ all of the above, in DTB_ROOT/smoke/, plus reports/<time>__<model>.md
```

Paths on the right are under `DTB_ROOT`; [store_layout.md](store_layout.md) has the full tree.

1. **Configs** declare models, datasets, evalsets, tools (the face detector) and the one evaluation setup.
2. **Setup** fetches code, weights and data and writes each dataset's manifest. `--sample N` fetches only N real and N fake items, for pipeline tests.
3. **Scoring** runs one model over one evalset, resumably, once the model's modality matches the evalset and the contamination guard has checked the pair. Video and image items go through the shared face pipeline; audio is cut into windows.
4. **Evaluation** computes overall and per-group metrics with intervals and publishes the aggregates (never for pipeline tests).
5. **Smoke test** runs 2 to 4 on a few items per dataset and writes a report.
