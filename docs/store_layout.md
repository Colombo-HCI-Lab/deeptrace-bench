# The store (`DTB_ROOT`)

Everything large or item-level lives outside git, in one folder named by `DTB_ROOT` in `.env`. This page is the map: what each folder holds, how things are named, and what is safe to delete.

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
│   ├── part-000of001-00000.parquet          item_id, score, n_windows, status
│   └── crops/<item_id>/frame_0007.png       only with score.py --save-crops
├── results/<model>__<evalset>__<8hex>/      summary.json, by_<group>.csv, items.parquet
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

- **Ids** are config file names: `configs/models/gend.yaml` is model `gend`. Folders reuse them, so `datasets/unidatapro_videos/` is that dataset.
- **Items** keep their dataset's own relative paths under `datasets/<id>/`, and their id is `<dataset>/<local path without extension>` (`unidatapro_videos/deepfake/3`). Saved crops follow the item id, so a crop can be traced back by eye.
- **Runs** are `<model>__<evalset>__<8 hex>`. The hex covers the upstream commit, every weight hash (including the face detector's for video and image runs), the model, evalset and eval configs and the preprocessing mode. Re-running the same thing resumes the same run; changing any input starts a new folder, so results from different settings never mix. The repo commit and device are recorded per session in `run.json`, not in the id.
- **Face caches** are `<detector>__<8 hex>`, the hex over the detector's weights, thresholds, input size, face choice and frame sampling (`spec.json` spells them out). A changed setting never reuses stale detections.
- **Parts** are `part-<shard>of<shards>-<n>.parquet`, so array jobs that start together never write the same file.

## Namespaces

`DTB_NAMESPACE=smoke` moves everything item-level (datasets, manifests, faces, scores, results, reports) under `$DTB_ROOT/smoke/`. Weights and upstream archives stay at the root and are shared, so a smoke run never downloads a model twice. Scripts set it for you (`smoke.py`, `--sample`, `--smoke`); don't set it in `.env`.

A sampled dataset can only ever land in the smoke namespace: `setup_datasets.py --sample` switches to it. That keeps a four-item "dataset" from ever being mistaken for the real one.

## What is safe to delete

| Folder                  | Safe? | What it costs                                                            |
| ----------------------- | ----- | ------------------------------------------------------------------------ |
| `smoke/`                | yes   | the next `smoke.py` re-samples (a few MB)                                |
| `faces/`                | yes   | face detection runs again (the slow step for video)                      |
| `scores/<run>/crops/`   | yes   | only the PNGs kept for people to look at                                 |
| `results/`              | yes   | `evaluate.py` rewrites it from the scores                                |
| `scores/`               | care  | re-scoring takes GPU time; published aggregates in git stay              |
| `manifests/`            | care  | rebuild with `setup_datasets.py <id> --manifest`                         |
| `datasets/`, `weights/` | care  | re-download; for request-only datasets, someone has to supply them again |
| `upstream/`             | no    | the archive is the copy that survives a deleted or rewritten upstream    |

On a cluster, give each restricted dataset its own Unix group on `datasets/<id>/` and the matching `scores/` and `results/` folders (see [data_policy.md](data_policy.md)).
