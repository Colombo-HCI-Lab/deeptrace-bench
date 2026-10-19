---
title: "Adding a model"
description: "Recipe: config, pinned weights, adapter, smoke test, parity (GenD as the worked example)"
group: working
order: 3
---

A model is a config, a pinned upstream checkout, pinned weights and an adapter. Worked example: GenD (`configs/models/gend.yaml`, `src/deeptrace_bench/models/gend.py`). Other upstream shapes: `aasist.py` (one plain PyTorch file), `df_arena.py` (code inside a Hugging Face model repo), `deepfakebench.py` (one detector out of a training framework).

## 1. Config

Add `configs/models/<id>.yaml` (id = file name):

- `upstream`: GitHub repo and full commit. Code in a Hugging Face model repo (custom modelling files next to the weights) uses `host: hf` with the Hub revision as `commit`; it is cloned the same way, without LFS weights.
- `weights`: every file and its source. Kinds: `hf` (one file at a pinned `revision`), `hf_snapshot` (several files of a repo at a full 40-hex revision, into a folder, picked by `allow_patterns`), `github_release`, `github_raw`, `url` (optionally `member:` to take one file out of a zip by range requests), `gdrive`, `gdrive_folder`, `manual`. Pin everything loaded at runtime: GenD's code calls `CLIPModel.from_pretrained("openai/clip-vit-large-patch14")` unpinned, so its config adds that backbone as an `hf_snapshot` and the adapter points GenD at the local copy.
- `training_data`: every dataset or corpus the released weights saw, including pretraining, with `split` where only one split was used. The contamination guard is only as good as this list.
- `input`: what the adapter needs. Face models give a `crop` (`size`, `margin`, `align`; `size: null` keeps native size). Normalisation (mean, std) belongs in the adapter.
- `modality`: `video` models also score `image` and `audio_video` evalsets; `audio` models score `audio` and `audio_video`.

## 2. Fetch and pin

```bash
uv run scripts/setup_models.py <id> --record
git diff configs/weights.lock.yaml          # one hash per file, folders per file
```

Clones upstream at its commit into `third_party/` (git-ignored), archives it to `$DTB_ROOT/upstream/`, downloads weights to `$DTB_ROOT/weights/<id>/` and records each sha256 in the lock. Once the adapter exists, it also converts the download to `model.safetensors` (step 3). Video models also get the shared face detector (`configs/tools/scrfd_10g.yaml`). Never edit the lock by hand.

## 3. Adapter

Subclass `deeptrace_bench.models.base.Detector` in `src/deeptrace_bench/models/<module>.py` and set `adapter:` in the config:

- `convert_checkpoint()`: turn a pickled download into `model.safetensors` once with `self.save_converted(state, source)` (unwraps `{"state_dict": ...}`, strips `module.`, records the source hash). Return early when `self.converted_is_current(source)`. Safetensors weights (GenD) skip this. A Hub file called `model.safetensors` needs another local `name` in the config, or the conversion would overwrite it (`save_converted` refuses).
- `load(device)`: build the network and load weights; `self.load_converted(network)` requires every key to match and refuses a conversion made from another download. Use `resolve_device` so `auto` works.
- Import upstream code from the checkout (`self.upstream_dir`) through `models/_upstream.py`, never by copying it (compatibility fixes go in `patches/<repo>/`):
  - `load_module`: one file (GenD);
  - `load_package`: a folder of files that import each other (DF Arena);
  - `scoped_modules`: stand-ins while importing a file that drags in a whole framework (DeepfakeBench);
  - `models/_fairseq.py` and `models/_mamba.py`: pure-PyTorch stand-ins for fairseq (wav2vec 2.0, XLS-R) and mamba-ssm, which we don't install.
- `score(inputs)`: P(fake) per window or frame. Inputs: face models get a list of RGB uint8 crops (sizes may differ); audio models an array of windows; audio-visual models (`modality: audio_video`) a dict of face crops, audio and frame counts; lip models (`input.inputs: mouths`) an array of grayscale mouth crops. Raise `PreprocessError` (e.g. `too_short`) when inputs can't make one window. Batch inside the adapter; the harness averages scores into one per item.
- Keep heavy imports inside `load` and `score` so the registry test imports every adapter cheaply.

## 4. Prove it runs

```bash
uv run pytest                                          # registry cross-checks import the adapter
uv run scripts/smoke.py --model <id>                   # every pipeline-test evalset it accepts
```

Audio models have no pipeline-test set, so they smoke-test on a sample of `urdu_csalt` (still the smoke namespace, never reported). Open the report and its crops ([smoke_test.md](smoke_test.md)). Then add a skip-unless-weights test like `tests/test_gend.py` or `tests/test_aasist.py`.

## 5. Before any result counts

- **Parity**: adapter and upstream score the same ~200 items within 1e-3, upstream in its original environment where it differs from ours. This is what licenses every patch.

  ```bash
  uv run scripts/setup_datasets.py <evalset dataset> --sample 100 --manifest
  uv run scripts/parity.py export --model <id> --evalset <evalset> --smoke   # prints the next two commands
  <python> scripts/parity_reference/<kind>.py <run dir>                     # the upstream side
  uv run scripts/parity.py compare --model <id> --run <run dir>             # writes results/parity/<id>.json
  ```

  `export` saves the exact inputs the adapter saw (windows or crops from the shared loaders) and its scores; a reference script in `scripts/parity_reference/` runs upstream's code on them; `compare` passes when every score agrees within 1e-3 and writes a committable summary (no item ids). A new adapter kind needs its own reference script. DeepfakeBench's runs in a throwaway old-torch virtualenv, made once:

  ```bash
  uv venv --python 3.9 $DTB_ROOT/parity/envs/dfb
  uv pip install --python $DTB_ROOT/parity/envs/dfb/bin/python torch==1.13.1 torchvision==0.14.1 \
    "numpy<1.24" opencv-python-headless==4.6.0.66 efficientnet-pytorch==0.7.1 pyyaml "scikit-learn<1.3" \
    timm==0.6.13   # RECCE's encoder; install it in the same command, or uv upgrades torch
  ```

  SBI's reference (`sbi.py`) uses the same one. The fairseq models need two more, on a local disk (a virtualenv's `python` is a symlink, which a CIFS-mounted store can't hold). Each needs pip older than 24.1 (fairseq's pinned omegaconf has metadata newer pip rejects) and torchaudio pinned beside torch (or fairseq's install pulls the newest torch). AntiDeepfake's (`antideepfake.py`) uses fairseq 0.12.2:

  ```bash
  uv venv --seed --python 3.9 <envs>/fairseq && <envs>/fairseq/bin/python -m pip install "pip<24.1"
  <envs>/fairseq/bin/pip install torch==1.13.1 torchaudio==0.13.1 "numpy<1.24" cython soundfile \
    safetensors fairseq==0.12.2
  ```

  The XLS-R wrapper models (`ssl_xlsr.py`) need upstream's own stack, since fairseq 0.12.2 rejects the config inside `xlsr2_300m.pt`: torch 1.8.1 and the fairseq vendored in the SSL_Anti-spoofing checkout, installed from a copy outside the checkout. `SSLModel` loads `xlsr2_300m.pt` (3.8 GB, <https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt>) from the working directory, so keep it in the virtualenv's root, where the script runs:

  ```bash
  uv venv --seed --python 3.9 <envs>/fairseq-tak && <envs>/fairseq-tak/bin/python -m pip install "pip<24.1"
  <envs>/fairseq-tak/bin/pip install torch==1.8.1 torchaudio==0.8.1 "numpy<1.24" cython soundfile
  cp -r third_party/SSL_Anti-spoofing-*/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 /tmp/fairseq-tak
  <envs>/fairseq-tak/bin/pip install /tmp/fairseq-tak
  curl -L -o <envs>/fairseq-tak/xlsr2_300m.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
  ```

- **Score direction**: AUC above 0.5 on the reproduction evalset (FF++ test, ASVspoof 2019 LA eval, or In-the-Wild for models the guard refuses on ASVspoof 2019); an inverted logit flips every result silently. A seeded sample settles direction (`setup_datasets.py <id> --sample 300 --manifest`, then `score.py --smoke` and `evaluate.py --smoke`); the full set reproduces the published number.
- Then give the model a `wave:`.

Per-model porting notes and expected Python 3.12 / torch 2.x breakages: [models.md](models.md).
