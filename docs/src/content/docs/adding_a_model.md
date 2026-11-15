---
title: "Adding a model"
description: "Recipe: config, pinned weights, adapter, smoke test, parity (GenD as the worked example)"
---

Adding a model means teaching the harness to run someone else's detector exactly as its authors did; you never rewrite the model. A model is a config, a pinned upstream checkout, pinned weights and an adapter. Worked example: GenD (`configs/models/gend.yaml`, `src/deeptrace_bench/models/gend.py`). Other upstream shapes: `aasist.py` (one plain PyTorch file), `df_arena.py` (code inside a Hugging Face model repo), `deepfakebench.py` (one detector out of a training framework).

## How adapters work

One environment runs everything (Python 3.12, torch 2.8, transformers 4.56). Upstream stacks (DeepfakeBench: Python 3.7, torch 1.12; XLS-R + AASIST: torch 1.8, fairseq) don't run on H100s, so they serve only as parity references. An adapter imports the upstream code it needs from a pinned checkout (steps 2 and 3); only MIT code may be copied into the repo ([Data policy](data_policy.md#licence-clauses-that-bind-us)). Expected breakages on Python 3.12 and torch 2.x: `torch.load` defaults to `weights_only=True`; `np.float`, `np.int`, `distutils` and `imp` are gone.

## 1. Config


| Field | What goes in |
| --- | --- |
| `upstream` | GitHub repo and full commit. Code in a Hugging Face model repo (custom modelling files next to the weights) uses `host: hf` with the Hub revision as `commit`; it is cloned the same way, without LFS weights |
| `weights` | every file and its source, by kind (below). Pin everything loaded at runtime: GenD's code calls `CLIPModel.from_pretrained("openai/clip-vit-large-patch14")` unpinned, so its config adds that backbone as an `hf_snapshot` and the adapter points GenD at the local copy |
| `training_data` | every dataset or corpus the released weights saw, including pretraining, with `split` where only one split was used. The contamination guard is only as good as this list |
| `input` | what the adapter needs. Face models give a `crop` (`size`, `margin`, `align`; `size: null` keeps native size). Normalisation (mean, std) belongs in the adapter |
| `modality` | `video` models also score `image` and `audio_video` evalsets; `audio` models score `audio` and `audio_video` |

| Weight kind | Fetches |
| --- | --- |
| `hf` | one file at a pinned `revision` |
| `hf_snapshot` | several files of a repo at a full 40-hex revision, into a folder, picked by `allow_patterns` |
| `url` | a file at a URL; `member:` takes one file out of a zip by range requests |
| `github_release`, `github_raw`, `gdrive`, `gdrive_folder`, `manual` | as named |

## 2. Fetch and pin

```bash
uv run scripts/setup_models.py <id> --record
git diff configs/weights.lock.yaml          # one hash per file, folders per file
```

Clones upstream at its commit into `third_party/` (git-ignored), archives it to `$DTB_ROOT/upstream/`, downloads weights to `$DTB_ROOT/weights/<id>/` and records each sha256 in the lock. Once the adapter exists, it also converts the download to `model.safetensors` (step 3). Video models also get the shared face detector (`configs/tools/scrfd_10g.yaml`). Never edit the lock by hand.

## 3. Adapter

Subclass `deeptrace_bench.models.base.Detector` in `src/deeptrace_bench/models/<module>.py` and set `adapter:` in the config:

| Method | Does |
| --- | --- |
| `convert_checkpoint()` | turns a pickled download into `model.safetensors` once with `self.save_converted(state, source)` (unwraps `{"state_dict": ...}`, strips `module.`, records the source hash). Return early when `self.converted_is_current(source)`. Safetensors weights (GenD) skip this. A Hub file called `model.safetensors` needs another local `name` in the config, or the conversion would overwrite it (`save_converted` refuses) |
| `load(device)` | builds the network and loads weights; `self.load_converted(network)` requires every key to match and refuses a conversion made from another download. Use `resolve_device` so `auto` works |
| `score(inputs)` | returns P(fake) per window or frame. Raise `PreprocessError` (e.g. `too_short`) when inputs can't make one window. Batch inside the adapter; the harness averages scores into one per item |

| Model | `inputs` |
| --- | --- |
| face | a list of RGB uint8 crops (sizes may differ) |
| audio | an array of windows |
| audio-visual (`modality: audio_video`) | a dict of face crops, audio and frame counts |
| lip (`input.inputs: mouths`) | an array of grayscale mouth crops |

Import upstream code from the checkout (`self.upstream_dir`) through `models/_upstream.py`, never by copying it (compatibility fixes go in `patches/<repo>/`):

| Helper | For |
| --- | --- |
| `load_module` | one file (GenD) |
| `load_package` | a folder of files that import each other (DF Arena) |
| `scoped_modules` | stand-ins while importing a file that drags in a whole framework (DeepfakeBench) |
| `models/_fairseq.py`, `models/_mamba.py` | pure-PyTorch stand-ins for fairseq (wav2vec 2.0, XLS-R) and mamba-ssm, which we don't install |

Keep heavy imports inside `load` and `score` so the registry test imports every adapter cheaply.

## 4. Prove it runs

```bash
uv run pytest                                          # registry cross-checks import the adapter
uv run scripts/smoke.py --model <id>                   # every pipeline-test evalset it accepts
```

Open the report and its crops ([reading the report](getting_started.md#reading-the-report)). Then add a skip-unless-weights test like `tests/test_gend.py` or `tests/test_aasist.py`.

## 5. Before any result counts

- **Parity**: adapter and upstream score the same ~200 items within 1e-3, upstream in its original environment where it differs from ours. This is what licenses every patch.

  ```bash
  uv run scripts/setup_datasets.py <evalset dataset> --sample 100 --manifest
  uv run scripts/parity.py export --model <id> --evalset <evalset> --smoke   # prints the next two commands
  <python> scripts/parity_reference/<kind>.py <run dir>                     # the upstream side
  uv run scripts/parity.py compare --model <id> --run <run dir>             # writes results/parity/<id>.json
  ```

  `export` saves the exact inputs the adapter saw (windows or crops from the shared loaders) and its scores; a reference script in `scripts/parity_reference/` runs upstream's code on them; `compare` passes when every score agrees within 1e-3 and writes a committable summary (no item ids). A new adapter kind needs its own reference script, and usually upstream's old stack in a virtualenv of its own: the ones built so far are in [Models](models.md#reference-environments).

- **Score direction**: AUC above 0.5 on the reproduction evalset (FF++ test, ASVspoof 2019 LA eval, or In-the-Wild for models the guard refuses on ASVspoof 2019); an inverted logit flips every result silently. A seeded sample settles direction (`setup_datasets.py <id> --sample 300 --manifest`, then `score.py --smoke` and `evaluate.py --smoke`); the full set reproduces the published number.
- Then give the model a `wave:`.

Per-model porting notes: [Models](models.md#per-model).
