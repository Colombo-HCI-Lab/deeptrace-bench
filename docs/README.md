# Docs

| Doc                              | What it covers                                                                                             |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [models.md](models.md)           | Every detector: status, wave, where code and weights come from, training data, porting notes, parity check |
| [datasets.md](datasets.md)       | Every dataset: how to get it, who requests or signs, licence terms, what's in it                           |
| [evaluation.md](evaluation.md)   | Evalsets, preprocessing, metrics, failure accounting, contamination verdicts, the shortcut probe           |
| [cluster.md](cluster.md)         | Running on UNIL's Curnagl cluster (or any Linux GPU machine): storage, environment, jobs                   |
| [data_policy.md](data_policy.md) | What may enter git, licence clauses that bind us, sharing data across institutions                         |

## How the pieces fit

```
configs/datasets/*.yaml ──setup_datasets.py──▶ DTB_ROOT/datasets/<id>/   (download, or --from a copy)
                                     └─────▶ DTB_ROOT/manifests/<id>.parquet  (one row per item)
configs/models/*.yaml ───setup_models.py───▶ third_party/<repo>-<commit>/  +  DTB_ROOT/weights/<id>/
configs/evalsets/*.yaml ─┐
configs/eval/default.yaml┼──score.py──────▶ DTB_ROOT/scores/<run_id>/   (run.json + score parts)
contamination guard ─────┘
                           evaluate.py ───▶ DTB_ROOT/results/<run_id>/  ──--publish──▶ results/<run_id>/ (git)
```

1. **Configs** say what exists: models, datasets, evalsets and the one evaluation setup.
2. **Setup** gets code, weights and data into place, and turns each dataset into a manifest.
3. **Scoring** runs one model over one evalset, resumably, after the contamination guard has checked the pair.
4. **Evaluation** computes overall and per-group metrics with intervals, and publishes the aggregates.

Status as of 2026-10-03: the harness, configs, setup scripts, metrics and guard are in place and tested. Model adapters, dataset builders and face preprocessing are documented stubs, to be written as each model and dataset arrives.
