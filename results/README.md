# Published results

Aggregate tables only: run summaries (`<run_id>/summary.json`), per-group metrics (`<run_id>/by_<attribute>.csv`) and manifest summaries (`manifests/<dataset>.json`, counts and a content hash). Written by `scripts/evaluate.py --publish` and `scripts/setup_datasets.py --manifest`.

Nothing here names an item. Per-item scores, manifests and per-subject tables stay in `DTB_ROOT`, and the pre-commit hook refuses tables with an `item_id` or `rel_path` column.
