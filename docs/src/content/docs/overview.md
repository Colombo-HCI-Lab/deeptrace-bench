---
title: "Overview"
description: "The docs at a glance, and how configs, setup, scoring and evaluation fit together"
group: start
order: 1
---

deeptrace-bench runs open deepfake detectors on South Asian audio and video under one evaluation setup and reports results overall and per group. New here? Read [Progress](progress.md) for where things stand, then [Results](results.md) for what the runs show. [How it works](https://colombo-hci-lab.github.io/deeptrace-bench/method/) explains the key terms.

| I want to | Read |
| --- | --- |
| know where the benchmark stands and what's next | [Progress](progress.md) |
| see what the runs show | [Results](results.md), then [Experiments](experiments.md) for what each run asked |
| look up a model or a dataset | [Models](models.md), [Datasets](datasets.md), or the plain-language [Models and datasets](catalog.md) |
| understand how a number is made | [Evaluation](evaluation.md) |
| know what may go into git or be shared | [Data policy](data_policy.md) |
| check the pipeline works after a change | [Smoke test](smoke_test.md) |
| find a file the scripts wrote | [The store](store_layout.md) |
| add a model or a dataset | [Adding a model](adding_a_model.md), [Adding a dataset](adding_a_dataset.md) |
| run on the GPU cluster | [Running on SPARK](spark.md) |
| run on Curnagl or another GPU machine | [Curnagl or any GPU machine](cluster.md) |

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

1. **Configs** declare everything: models, datasets, evalsets (the exact slice of data a model is scored on), the face detector, and the one evaluation setup.
2. **Setup** fetches code, weights and data, and writes a manifest per dataset (one row per item). `--sample N` fetches only N real and N fake items, for pipeline tests.
3. **Scoring** runs one model over one evalset. It first checks the model reads that modality and wasn't trained on that data (the contamination guard). Faces are detected once and cached; audio is cut into windows. Scoring resumes after a kill.
4. **Evaluation** computes overall and per-group metrics with intervals, and publishes only the aggregates.
5. **Smoke test** runs steps 2 to 4 on a few items per dataset and writes a report.
