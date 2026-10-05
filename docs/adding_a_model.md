# Adding a model

A model is a config, a pinned upstream checkout, pinned weights and an adapter. GenD is the worked example throughout; its files are `configs/models/gend.yaml` and `src/deeptrace_bench/models/gend.py`. Three other adapters show the other shapes upstream code comes in: `aasist.py` (one plain PyTorch file), `df_arena.py` (code inside a Hugging Face model repo) and `deepfakebench.py` (one detector out of a whole training framework).

## 1. Config

Add `configs/models/<id>.yaml` (the id is the file name):

- `upstream`: the GitHub repo and the full commit to pin. Code that lives in a Hugging Face model repo (custom modelling files next to the weights) uses `host: hf`, with the Hub revision as `commit`; it is cloned the same way, without its LFS weights.
- `weights`: every file, with where it comes from. Kinds: `hf` (one file at a pinned `revision`), `hf_snapshot` (several files of a repo at a full 40-hex revision, into a folder, picked by `allow_patterns`), `github_release`, `github_raw`, `url` (optionally `member:` to take one file out of a zip by range requests), `gdrive`, `gdrive_folder`, `manual`. Pin whatever the model loads at runtime: GenD's own code calls `CLIPModel.from_pretrained("openai/clip-vit-large-patch14")` unpinned, so its config adds that backbone as an `hf_snapshot` and the adapter points GenD at the local copy.
- `training_data`: every dataset or corpus the released weights saw, including pretraining, with `split` where only one split was used. Be thorough; the contamination guard is only as good as this list.
- `input`: what the adapter needs. Face models give a `crop` (`size`, `margin`, `align`; `size: null` keeps the native size). Normalisation (mean and std) belongs in the adapter, which knows what upstream does.
- `modality`: `video` models also score `image` and `audio_video` evalsets; `audio` models score `audio` and `audio_video`.

## 2. Fetch and pin

```bash
uv run scripts/setup_models.py <id> --record
git diff configs/weights.lock.yaml          # one hash per file, folders per file
```

This clones the upstream repo at its commit into `third_party/` (git-ignored), archives it to `$DTB_ROOT/upstream/`, downloads the weights to `$DTB_ROOT/weights/<id>/` and records each sha256 in the lock. Once the adapter is written, it also converts the download to `model.safetensors` (step 3). A video model also gets the shared face detector (`configs/tools/scrfd_10g.yaml`). Never edit the lock by hand.

## 3. Adapter

Subclass `deeptrace_bench.models.base.Detector` in `src/deeptrace_bench/models/<module>.py` and set `adapter:` in the config:

- `convert_checkpoint()`: turn a pickled download into `model.safetensors` once, with `self.save_converted(state, source)` (it unwraps `{"state_dict": ...}`, strips `module.` and records the source file's hash). Return early when `self.converted_is_current(source)`. Models whose weights are already safetensors (GenD) skip this.
- `load(device)`: build the network and load the weights; `self.load_converted(network)` loads with every key matching and refuses a conversion made from another download. Use `resolve_device` so `auto` works.
- Upstream code comes from the checkout (`self.upstream_dir`), through `models/_upstream.py`: `load_module` for one file (registered in `sys.modules` under a fixed name, as transformers needs for GenD), `load_package` for a folder of files that import each other relatively (DF Arena), and `scoped_modules` for stand-ins while importing a file whose imports drag in a whole framework (DeepfakeBench: its training losses, metrics and tensorboard). Never copy code from repos without an MIT-style licence; compatibility fixes go in `patches/<repo>/`.
- `score(inputs)`: P(fake) per window or frame, higher is more fake. Face models get a list of RGB uint8 crops (sizes may differ); audio models get an array of windows. Batch inside the adapter; the harness averages the scores into one per item.
- Keep heavy imports inside `load` and `score`, so the registry test can import every adapter cheaply.

## 4. Prove it runs

```bash
uv run pytest                                          # registry cross-checks import the adapter
uv run scripts/smoke.py --model <id>                   # every pipeline-test evalset it accepts
```

Audio models have no pipeline-test set, so their smoke test runs on a sample of `urdu_csalt` (still in the smoke namespace, never reported). Open the smoke report and its crops (see [smoke_test.md](smoke_test.md)). Then add a skip-unless-weights test like `tests/test_gend.py` or `tests/test_aasist.py`.

## 5. Before any result counts

- **Parity**: the adapter and the upstream code score the same ~200 items within 1e-3, the upstream side in its original environment where it differs from ours. This is what licenses every patch.

  ```bash
  uv run scripts/setup_datasets.py <evalset dataset> --sample 100 --manifest
  uv run scripts/parity.py export --model <id> --evalset <evalset> --smoke   # prints the next two commands
  <python> scripts/parity_reference/<kind>.py <run dir>                     # the upstream side
  uv run scripts/parity.py compare --model <id> --run <run dir>             # writes results/parity/<id>.json
  ```

  `export` saves the exact inputs the adapter saw (windows or crops from the shared loaders) and the adapter's scores; a reference script in `scripts/parity_reference/` runs upstream's own code on them; `compare` passes when every score agrees within 1e-3 and writes a committable summary (no item ids). A new adapter needs a reference script of its kind. DeepfakeBench's runs in a throwaway old-torch virtualenv, made once:

  ```bash
  uv venv --python 3.9 $DTB_ROOT/parity/envs/dfb
  uv pip install --python $DTB_ROOT/parity/envs/dfb/bin/python torch==1.13.1 torchvision==0.14.1 \
    "numpy<1.24" opencv-python-headless==4.6.0.66 efficientnet-pytorch==0.7.1 pyyaml "scikit-learn<1.3"
  ```

- **Score direction**: on the reproduction evalset (FF++ test, ASVspoof 2019 LA eval, or In-the-Wild for models the guard refuses on ASVspoof 2019), AUC must come out above 0.5. An inverted logit flips every result silently. A seeded sample is enough for the direction (`setup_datasets.py <id> --sample 300 --manifest`, then `score.py --smoke` and `evaluate.py --smoke`); the full set reproduces the published number.
- Then give the model a `wave:`.

Porting notes per model, and the breakages to expect on Python 3.12 and torch 2.x, are in [models.md](models.md).
