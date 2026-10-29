---
title: "The store (`DTB_ROOT`)"
description: "The DTB_ROOT tree: what each folder holds, naming rules, the smoke namespace, the archive, what's safe to delete"
---

Everything large or item-level lives outside git, in the folder `DTB_ROOT` names in `.env`. Which script writes which folder: [the pipeline](method.md#the-pipeline).

## The tree

```
$DTB_ROOT/
├── weights/                                 shared by every namespace
│   ├── gend/
│   │   ├── model.safetensors, config.json
│   │   └── clip-vit-large-patch14/          pinned backbone snapshot (8 files)
│   ├── aasist/AASIST.pth, model.safetensors  a download and its one-time conversion
│   └── scrfd_10g/det_10g.onnx               the shared face detector (a tool)
├── upstream/GenD-387a42266dd3.tar.gz        archive of each pinned upstream checkout (shared)
├── hf/                                      HF_HOME, if .env sets it there (shared)
├── parity/envs/dfb/                         old-torch virtualenv for DeepfakeBench references
│
├── datasets/<dataset>/                      downloads, or a symlink to a copy obtained by hand
├── manifests/<dataset>.parquet              one row per item
├── faces/<dataset>/<detector>__<8hex>/      cached face detections
│   ├── spec.json                            the settings the hex stands for
│   └── part-000of001-00000.parquet
├── scores/<model>__<evalset>__<8hex>/       one run
│   ├── run.json                             provenance: inputs, sessions (commit, device, host)
│   ├── part-000of001-00000.parquet          item_id, score, n_windows, status, load_s, infer_s
│   ├── telemetry/<part prefix>-<UTC>-<pid>.json   one per session: phases, GPU load, memory, energy
│   └── crops/<item_id>/frame_0007.png       only with score.py --save-crops
├── results/<model>__<evalset>__<8hex>/      summary.json (with compute), by_<group>.csv, items.parquet
├── splits/                                  identity-safe fine-tuning splits
├── parity/<model>/<UTC time>/               a parity run: inputs.npz, parity.json, adapter.npy,
│                                            reference.npy, reference.json
│
└── smoke/                                   the smoke namespace: same layout as above
    ├── README.txt                           "pipeline tests; never reported; safe to delete"
    ├── datasets/ manifests/ faces/ scores/ results/
    └── reports/<UTC time>__<model>.md       smoke reports (+ .log), and latest.md
```

## Naming rules

- **Ids** are config file names (`configs/models/gend.yaml` is model `gend`); folders reuse them (`datasets/unidatapro_videos/`).
- **Items** keep their dataset's relative paths under `datasets/<id>/`; their id is `<dataset>/<local path without extension>` (`unidatapro_videos/deepfake/3`). Saved crops follow the item id, so a crop traces back by eye.
- **Runs** are `<model>__<evalset>__<8 hex>`. The hex covers the upstream commit, every weight hash (including the face detector's for video and image runs), the model, evalset and eval configs, and the preprocessing mode. Re-running the same thing resumes the run; changing any input starts a new folder, so settings never mix. Repo commit and device are recorded per session in `run.json`, not in the id.
- **Face caches** are `<detector>__<8 hex>`, the hex over detector weights, thresholds, input size, face choice and frame sampling (spelled out in `spec.json`), so a changed setting never reuses stale detections.
- **Parts** are `part-<shard>of<shards>-<n>.parquet`, so array jobs starting together never write the same file.

## Namespaces

`DTB_NAMESPACE=smoke` moves everything item-level (datasets, manifests, faces, scores, results, reports) under `$DTB_ROOT/smoke/`. Weights and upstream archives stay at the root, shared, so a smoke run never downloads a model twice. Scripts set it (`smoke.py`, `--sample`, `--smoke`); never set it in `.env`.

`setup_datasets.py --sample` always switches to the smoke namespace, so a four-item sample can't be mistaken for the real dataset.

## Archive

A store is the only copy of its per-item scores, and each machine that runs waves has its own store. After each round, `scripts/archive.py` copies every evaluated run into one archive in the project's private Google Drive folder (phase 2's `benchmark results/`), so the scores outlive any one machine and the team can reanalyse them without a GPU:

```bash
uv run scripts/archive.py --to <archive> [--from <copy of another machine's store>]
```

| In the archive               | From the store                                           |
| ---------------------------- | -------------------------------------------------------- |
| `runs/<run_id>/`             | `results/<run_id>/` (`items.parquet`, `summary.json`, `by_*.csv`) plus `scores/<run_id>/run.json` and `telemetry/` |
| `waves/<wave>/`              | `matrix/<wave>/`, each wave's ledger                     |
| `index.csv`                  | one row per run: pair, verdict, parity, AUC, EER, TPR at 5% FPR, whether git has it |

Score parts stay behind (`items.parquet` holds the same scores), and so do datasets, weights and face caches, which can be fetched or rebuilt. Copying keeps the newest version of each file and never deletes, so re-running it after every round is safe. Runs on datasets whose terms forbid copies are left out ([Data policy](data_policy.md#what-may-enter-git)). Runs scored on SPARK keep their per-item scores on the cluster, since only aggregates come back from it.

## What is safe to delete

| Folder                  | Safe? | What it costs                                                            |
| ----------------------- | ----- | ------------------------------------------------------------------------ |
| `smoke/`                | yes   | the next `smoke.py` re-samples (a few MB)                                |
| `faces/`                | yes   | face detection runs again (the slow step for video)                      |
| `scores/<run>/crops/`   | yes   | only the PNGs kept for people to look at                                 |
| `results/`              | yes   | `evaluate.py` rewrites it from the scores                                |
| `scores/`               | care  | resuming needs the parts; [archived](#archive) runs keep their final scores |
| `manifests/`            | care  | rebuild with `setup_datasets.py <id> --manifest`                         |
| `datasets/`, `weights/` | care  | re-download; for request-only datasets, someone has to supply them again |
| `upstream/`             | no    | the archive is the copy that survives a deleted or rewritten upstream    |

Restricted datasets on a shared machine need their own permissions ([Data policy](data_policy.md#sharing-across-institutions)).
