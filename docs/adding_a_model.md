# Adding a model

A model is a config, a pinned upstream checkout, pinned weights and an adapter. GenD is the worked example throughout; its files are `configs/models/gend.yaml` and `src/deeptrace_bench/models/gend.py`.

## 1. Config

Add `configs/models/<id>.yaml` (the id is the file name):

- `upstream`: the GitHub repo and the full commit to pin.
- `weights`: every file, with where it comes from. Kinds: `hf` (one file at a pinned `revision`), `hf_snapshot` (several files of a repo at a full 40-hex revision, into a folder, picked by `allow_patterns`), `github_release`, `github_raw`, `url` (optionally `member:` to take one file out of a zip by range requests), `gdrive`, `gdrive_folder`, `manual`. Pin whatever the model loads at runtime: GenD's own code calls `CLIPModel.from_pretrained("openai/clip-vit-large-patch14")` unpinned, so its config adds that backbone as an `hf_snapshot` and the adapter points GenD at the local copy.
- `training_data`: every dataset or corpus the released weights saw, including pretraining, with `split` where only one split was used. Be thorough; the contamination guard is only as good as this list.
- `input`: what the adapter needs. Face models give a `crop` (`size`, `margin`, `align`, `normalize`; `size: null` keeps the native size).
- `modality`: `video` models also score `image` and `audio_video` evalsets; `audio` models score `audio` and `audio_video`.

## 2. Fetch and pin

```bash
uv run scripts/setup_models.py <id> --record
git diff configs/weights.lock.yaml          # one hash per file, folders per file
```

This clones the upstream repo at its commit into `third_party/` (git-ignored), archives it to `$DTB_ROOT/upstream/`, downloads the weights to `$DTB_ROOT/weights/<id>/` and records each sha256 in the lock. A video model also gets the shared face detector (`configs/tools/scrfd_10g.yaml`). Never edit the lock by hand.

## 3. Adapter

Subclass `deeptrace_bench.models.base.Detector` in `src/deeptrace_bench/models/<module>.py` and set `adapter:` in the config:

- `load(device)`: build the network and load the pinned weights. Use `resolve_device` so `auto` works. Load upstream code from the checkout (`self.upstream_dir`) with `importlib`; register the module in `sys.modules` if transformers needs to find it (GenD does). Never copy code from repos without an MIT-style licence; compatibility fixes go in `patches/<repo>/`.
- `score(inputs)`: P(fake) per window or frame, higher is more fake. Face models get a list of RGB uint8 crops (sizes may differ); audio models get an array of windows. Batch inside the adapter; the harness averages the scores into one per item.
- Keep heavy imports inside `load` and `score`, so the registry test can import every adapter cheaply.

## 4. Prove it runs

```bash
uv run pytest                                          # registry cross-checks import the adapter
uv run scripts/smoke.py --model <id>                   # every pipeline-test evalset it accepts
```

Open the smoke report and its crops (see [smoke_test.md](smoke_test.md)). Then add a skip-unless-weights test like `tests/test_gend.py`.

## 5. Before any result counts

- **Parity**: run the upstream code in its original environment (an Apptainer image or a throwaway old-torch venv, on CPU) on about 200 items, run the adapter on the same items, and require scores to agree within about 1e-3. Record the comparison in the model's notes. This is what licenses every patch.
- **Score direction**: on the reproduction evalset (FF++ test, ASVspoof 2019 LA eval), AUC must come out above 0.5. An inverted logit flips every result silently.
- Then give the model a `wave:`.

Porting notes per model, and the breakages to expect on Python 3.12 and torch 2.x, are in [models.md](models.md).
