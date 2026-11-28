---
title: "Models"
description: "Every detector: whether its scores can be trusted yet, why any is missing, the parity checks and their environments, and porting notes per adapter"
---

The configs (`configs/models/<id>.yaml`) are the source of truth, and `uv run scripts/setup_models.py --list` prints the current state. Each model's architecture, training data and paper are in the [catalog](catalog.md#models); terms such as parity and score direction are defined in [How it works](method.md#key-terms).

## Status

Checked on 2026-11-28 against the configs and the published runs. All 35 models with an adapter pass [parity](#parity), except DeMamba and D3, added that day, whose parity job is queued on SPARK.

| Models | Modality | Status | Score direction |
| --- | --- | --- | --- |
| Xception, EfficientNet-B4, SBI, GenD (CLIP L/14) | video | ready, wave 1 | settled |
| GenD (PE L/14), UCF, F3Net, SPSL, RECCE, SRM, CORE, FFD, Effort, UnivFD, LAA-Net, DFDC winner (7 x B7), FTCN, FakeSTormer, RealForensics, LipForensics | video | workable | settled |
| GenConViT (ED) | video | workable | can't be settled: trained on every reproduction set |
| HAVIC, AVH-Align, LipFD, AVAD | audio-video | workable | not yet |
| AASIST, AASIST-L | audio | ready, wave 1 | settled |
| XLS-R + AASIST | audio | workable, wave 1 | settled |
| XLS-R + SLS, XLSR-Mamba, AASIST3, AntiDeepfake MMS-300M and w2v2 small, DF Arena 500M and 1B | audio | workable | settled |
| DeMamba (GenVideo-100K checkpoint), D3 | video | workable, parity pending | not yet: reproduction on GenVideo-Val |
| AltFreezing, TALL | video | blocked, no adapter | |

A score direction is settled once the model scores above chance on a reproduction evalset. The video models' were settled on 2026-11-27 on Celeb-DF v2's official test list, where the 20 scored there score 0.68 to 0.98 ([Results](results.md#reproductions)). GenConViT trained on Celeb-DF v2, FF++ and DFDC, so the guard refuses it on all of them. The four audio-visual models' wait on an audio-visual reproduction set: FakeAVCeleb is on request ([Datasets](datasets.md#requests-to-send)), and HAVIC, trained on part of it, is refused there.

Both blocked models wait on weights. AltFreezing's are only on RecDrive and Baidu (password `altf`), and issues #7 and #8 report the file gone, with no reply or other mirror (rechecked 2026-10-05); someone with a working session has to fetch them. TALL has released none.

## Why these models

| Question | Models |
| --- | --- |
| Does a failure come from the data, not from one kind of model? | classic detectors beside foundation-model ones (GenD; the XLS-R family, AntiDeepfake, DF Arena): a failure both kinds share points at the data |
| Which cue fails? | the DeepfakeBench detectors each read a different one: texture (Xception, EfficientNet-B4), content-free forgery features (UCF), frequency (F3Net), phase (SPSL), reconstruction (RECCE), noise residuals (SRM), consistent representations (CORE), manipulation attention (FFD) |
| Are lip-sync fakes, v1's focus, caught? | LipForensics, HAVIC |
| Do cues that ignore face texture drop less on South Asian faces? | FTCN (motion only), RealForensics (lip motion from real talking faces) |
| Do detectors that never saw a fake avoid training-face bias? | AVAD, AVH-Align (audio-visual alignment of real speech) |
| Do more varied training faces help? | GenConViT, DFDC winner (both trained on DFDC's paid actors, not only FF++) |
| Are fully generated (text-to-video) fakes caught by detectors built for them? | DeMamba (trained on GenVideo), D3 (training-free), whole frames rather than face crops |
| How do newer relatives of what holds up on MAVOS-DD do? | LAA-Net, FakeSTormer (SBI-style training), LipFD (lip-sync specific), UnivFD (generic AI-image floor), TALL (MAVOS-DD's own baseline) |

The last four rows joined on 2026-10-28, chosen for the South Asian question, each with weights confirmed reachable that day except TALL's. DeepfakeBench releases no checkpoint for any of its video detectors (TALL, FTCN, AltFreezing, STIL, VideoMAE, TimeSformer, X-CLIP), only the frame-level set already here (checked 2026-10-28), so these come from their authors' repos. Each model's own reason, and whether it is a cut candidate, is on its page in the [Model catalog](catalog.md#models).

## Parity

Passed, each against upstream run in its original environment (`scripts/parity.py`; summaries in `results/parity/<id>.json`). Audio uses 521 windows of 200 Urdu (CSALT) clips, video 199 face crops from the Mendeley pipeline-test set.

| Models (max difference)                                                                                                            | Checked    | Upstream side                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------- |
| AASIST 2.6e-6, AASIST-L 3.3e-6                                                                                                     | 2026-10-05 | upstream `Model` and `.pth`, built as `main.py` does (torch 2.8)    |
| DF Arena 500M 0, 1B 0                                                                                                              | 2026-10-05 | the model card's `trust_remote_code` pipeline (torch 2.8)           |
| Xception 2.1e-6, EfficientNet-B4 3.5e-6, UCF 4.9e-6, F3Net 3.4e-6, SPSL 2.0e-6, RECCE 5.9e-6, SRM 1.7e-6, CORE 3.4e-6, FFD 4.2e-6 | 2026-10-05 | upstream detector, release `.pth`, test transform (torch 1.13; timm 0.6.13 for RECCE) |
| SBI 3.6e-6                                                                                                                         | 2026-10-12 | upstream `Detector`, `FFc23.tar` (torch 1.13)                       |
| XLS-R + AASIST 1.1e-4                                                                                                              | 2026-10-12 | upstream `model.py`, its vendored fairseq (torch 1.8.1)             |
| AntiDeepfake MMS-300M 9.0e-5, w2v2 small 3.2e-5                                                                                    | 2026-10-12 | upstream `W2V.py`, the Hub checkpoint, fairseq 0.12.2 (torch 1.13)  |
| FTCN 4.6e-6 (160 clips of 50 HiDF videos)                                                                                          | 2026-10-28 | upstream `PluginLoader` classifier and `load()`, `ftcn_tt.pth`, fvcore (torch 1.4.0, CPU) |
| AVH-Align 3.9e-6 (20 MAVOS-DD audio-visual videos)                                                                                 | 2026-10-28 | fairseq AV-HuBERT (`facebookresearch/av_hubert` and its pinned fairseq fork) `extract_finetune`, upstream `FusionModel` (torch 1.10.0, CPU) |
| DFDC winner 9.0e-6 (128 crops of 4 HiDF videos, all seven models)                                                                  | 2026-10-28 | upstream `DeepFakeClassifier` x7, `predict_folder.py`'s strict load, fp32 (torch 1.10.0, timm 0.4.12, CPU) |
| RealForensics 6.4e-6 (63 clips of 20 HiDF videos)                                                                                  | 2026-10-28 | upstream `csn_temporal_no_head` and `MeanLinear`, `eval.py`'s load, the test transform (torch 1.9.0, torchvision 0.10, pytorchvideo 0.1.2, CPU) |
| GenConViT (ED) 6.4e-6 (640 crops of 20 HiDF videos)                                                                                | 2026-10-28 | upstream `GenConViT(net="ed")` and its own strict load, `pred_func` preprocessing (timm 0.6.5, torch 1.13.1) |
| UnivFD 7.6e-6 (200 items, 6,400 crops)                                                                                             | 2026-11-02 | upstream CLIP ViT-L/14 and linear head (torch 2.1.0, CPU) |
| FakeSTormer 3.0e-6 (100 items, 800 runs of 4 frames)                                                                               | 2026-11-02 | upstream network and C23 checkpoint (torch 1.13.1, CPU) |
| LipFD 2.0e-6 (500 composites of 50 MAVOS-DD audio-visual videos)                                                                                   | 2026-11-04 | upstream `AVLip.__getitem__` and `validate`'s two calls, `ckpt.pth` strict (torch 2.1.0, torchvision 0.16, CPU) |
| AVAD 4.4e-8 (30 MAVOS-DD audio-visual videos)                                                                                      | 2026-11-04 | upstream `detect.test2` at `lam` 0 (torch 1.9.0, CPU) |
| LAA-Net 3.6e-6 (1,596 crops of 50 MAVOS-DD videos)                                                                                        | 2026-11-04 | upstream `test.py` path for `efn4_fpn_sbi_adv.yaml`, float64, flip test (torch 1.13.1, CPU) |
| AASIST3 8.5e-7 (519 windows of 200 Urdu CSALT clips)                                                                              | 2026-11-13 | upstream `aasist3.from_pretrained` on the Hub files, torchaudio pre-emphasis (torch 2.6, CPU) |
| XLS-R + SLS 1.0e-4 (519 windows of 200 Urdu CSALT clips)                                                                          | 2026-11-13 | upstream `model.py` with fairseq's `xlsr2_300m.pt` and `MMpaper_model.pth` (torch 1.8.1, CPU) |
| GenD 2.4e-6, GenD PE-L 1.8e-7 (1,596 crops of 50 MAVOS-DD videos)                                                                 | 2026-11-13 | upstream `GenD.from_pretrained` on the weights folder, its own preprocess (torch 2.8) |
| XLSR-Mamba 4.2e-5 (519 windows of 200 Urdu CSALT clips)                                                                          | 2026-11-13 | upstream `model.py` with fairseq's `xlsr2_300m.pt`, mamba-ssm 1.1.4's CUDA kernels and the released safetensors, TF32 off (torch 2.2.1, RTX 3090) |
| LipForensics 0 (200 clips of 50 MAVOS-DD videos)                                                                                  | 2026-11-13 | upstream `get_model` and `lipforensics_ff.pth` strict, `evaluate.py`'s transform (torch 1.9.0, torchvision 0.10, CPU) |
| HAVIC 1.3e-8 (444 windows of 30 MAVOS-DD audio-visual videos)                                                                    | 2026-11-13 | upstream `HAVIC_FT` and `best_ft_model.pth` as `sliding_window_infer.py` loads them, its resize and audio windows (torch 2.3.1, timm 0.4.5, CPU) |
| Effort 2.0e-6 (1,596 crops of 50 MAVOS-DD videos)                                                                                 | 2026-11-13 | upstream detector, release `.pth`, DeepfakeBench test transform (torch 1.13.1, CPU); CLIP's fixed `position_ids` buffer, absent from the checkpoint, filled in |

DeepfakeBench's reference shares the adapter's import stand-ins (and RECCE's and FFD's build stand-ins), so a pass there covers weights, numerics and transform, not upstream's imports; the strict load covers the architecture.

LipForensics' and HAVIC's references start from the adapter's mouth and face crops, so they check the network, weights and transform, not upstream's landmark and RetinaFace pipelines.

## Reference environments

Upstream's side of a parity check runs in upstream's own stack. The models checked on SPARK from 2026-11-02 on have theirs built on the worker by deeptrace-experiments' `parity` stage ([Where to run](spark.md#other-stages)), with the versions in the table above. The rest were made by hand:

| For | Python | Pins | Notes |
| --- | --- | --- | --- |
| DeepfakeBench's nine, SBI (`sbi.py`) | 3.9 | recipe `dfb` below | at `$DTB_ROOT/parity/envs/dfb` |
| AntiDeepfake (`antideepfake.py`) | 3.9 | recipe `fairseq` below | on a local disk |
| The XLS-R wrapper models (`ssl_xlsr.py`) | 3.9 | recipe `fairseq-tak` below | on a local disk. They need upstream's own stack, since fairseq 0.12.2 rejects the config inside `xlsr2_300m.pt` |
| FTCN | 3.8 | `torch==1.4.0+cpu`, `numpy<1.20`, `opencv-python-headless<4.6`, `fvcore==0.1.1.post20200716`, pyyaml, einops, termcolor, simplejson, psutil, `matplotlib<3.6` | Linux x86 only (torch 1.4 has no macOS arm64 wheels); set `DTB_PATH_MAP` when the run directory is mounted at another path |
| GenConViT | 3.9 | `torch==1.13.1`, `torchvision==0.14.1`, `numpy<1.24`, `timm==0.6.5`, pyyaml | at `$DTB_ROOT/parity/envs/genconvit` |
| AVH-Align | 3.8 | `torch==1.10.0+cpu`, `numpy<1.24`, `setuptools<60`, `pip<24.1`, cython, `omegaconf==2.0.6`, `hydra-core==1.0.7`, sentencepiece, scipy, python_speech_features, scikit-image, opencv | AV-HuBERT's repo with `git submodule update --init` and its fairseq fork installed editable (`--no-build-isolation`, after `wheel`); run with `AVHUBERT_DIR` set to its `avhubert/` folder |
| DFDC winner | 3.8 | `torch==1.10.0+cpu`, `torchvision==0.11.1+cpu`, `timm==0.4.12`, `numpy<1.24`, `setuptools<70` | Linux x86, with patchelf `--clear-execstack` on torch's libraries |
| RealForensics | 3.8 | `torch==1.9.0+cpu`, `torchvision==0.10.0+cpu`, `pytorchvideo==0.1.2`, `numpy<1.21`, `pillow<10`, `setuptools<70` | Linux x86, with patchelf `--clear-execstack` on torch's libraries where glibc refuses them |

`dfb`, a throwaway old-torch virtualenv, made once:

```bash
uv venv --python 3.9 $DTB_ROOT/parity/envs/dfb
uv pip install --python $DTB_ROOT/parity/envs/dfb/bin/python torch==1.13.1 torchvision==0.14.1 \
  "numpy<1.24" opencv-python-headless==4.6.0.66 efficientnet-pytorch==0.7.1 pyyaml "scikit-learn<1.3" \
  timm==0.6.13   # RECCE's encoder; install it in the same command, or uv upgrades torch
```

The two fairseq environments go on a local disk (a virtualenv's `python` is a symlink, which a CIFS-mounted store can't hold). Each needs pip older than 24.1 (fairseq's pinned omegaconf has metadata newer pip rejects) and torchaudio pinned beside torch (or fairseq's install pulls the newest torch). `fairseq`, with fairseq 0.12.2:

```bash
uv venv --seed --python 3.9 <envs>/fairseq && <envs>/fairseq/bin/python -m pip install "pip<24.1"
<envs>/fairseq/bin/pip install torch==1.13.1 torchaudio==0.13.1 "numpy<1.24" cython soundfile \
  safetensors fairseq==0.12.2
```

`fairseq-tak`: torch 1.8.1 and the fairseq vendored in the SSL_Anti-spoofing checkout, installed from a copy outside the checkout. `SSLModel` loads `xlsr2_300m.pt` (3.8 GB) from the working directory, so keep it in the virtualenv's root, where the script runs:

```bash
uv venv --seed --python 3.9 <envs>/fairseq-tak && <envs>/fairseq-tak/bin/python -m pip install "pip<24.1"
<envs>/fairseq-tak/bin/pip install torch==1.8.1 torchaudio==0.8.1 "numpy<1.24" cython soundfile
cp -r third_party/SSL_Anti-spoofing-*/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 /tmp/fairseq-tak
<envs>/fairseq-tak/bin/pip install /tmp/fairseq-tak
curl -L -o <envs>/fairseq-tak/xlsr2_300m.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
```

## Per model

What a reader of each adapter's scores should know: where the weights come from, what the model sees, which output is P(fake), and where we differ from upstream. Reproduced numbers are in [Results](results.md#reproductions); the sets the guard refuses each model on are in [Evaluation](evaluation.md#contamination-verdicts); how adapters work in general is in [Adding a model](adding_a_model.md#how-adapters-work).

### AASIST, AASIST-L

| | |
| --- | --- |
| Weights | ship in the upstream repo |
| P(fake) | softmax index 0: upstream scores the bona fide logit, and reading it the other way explains the inverted outputs in #17 |
| Differs from upstream | upstream scores each utterance's first four seconds; we average every window and refuse the 1,059 eval clips under a second |
| Note | the reproductions used TF32 convolutions (since turned off by `compute.tf32`), which move single scores by up to 2.5e-2 but left DF Arena 500M's In-the-Wild EER unchanged |

### XLS-R + AASIST, XLS-R + SLS, XLSR-Mamba

| | |
| --- | --- |
| Training | all three fine-tune XLS-R 300M inside Tak et al.'s `SSLModel` on ASVspoof 2019 LA alone, so they are clean zero-shot probes on every South Asian set |
| Weights | `LA_model.pth` (byte-identical to the MIT mirror `SpeechAntiSpoofingBenchmarks/W2V2-AASIST`) and `MMpaper_model.pth` from the authors' Drive, and `AustinXiao/XLSR-Mamba-LA` |
| Stand-ins | they run without fairseq (transformers' `Wav2Vec2Model` stands in, with the per-layer outputs SLS pools), and XLSR-Mamba without mamba-ssm (pure-PyTorch kernels checked against transformers' Mamba mixer, and since 2026-11-13 against mamba-ssm's own CUDA kernels by parity) |
| P(fake) | index 0 |

### DF Arena 500M and 1B

| | |
| --- | --- |
| Weights | `Speech-Arena-2025/DF_Arena_*_V_1`, code and weights in one Hugging Face repo, non-commercial; the top open models on the Speech DF Arena leaderboard (5.9% average EER for 1B) |
| Code | pinned, never run with `trust_remote_code` |
| P(fake) | the `spoof` softmax |
| Reproduction set | In-the-Wild, since the guard refuses them on ASVspoof 2019 LA eval |
| Differs from upstream | upstream scores a clip's first four seconds where we average every window, and we refuse the 865 In-the-Wild clips under a second where the card tiles them. Parity compares windows, so it can't see either; decision 5 (the evaluation setup, still open) should settle both |

### AntiDeepfake MMS-300M and wav2vec 2.0 small

| | |
| --- | --- |
| Weights | NII Yamagishi Lab (`nii-yamagishilab/*-anti-deepfake`), post-trained on 74k hours of real and fake speech in over 100 languages. MMS-300M is pretrained on about 1,400 languages, the small one on English only, isolating multilingual pretraining |
| Training data | includes FLEURS (Hindi, Bengali, Urdu, Tamil, Nepali and more, not Sinhala), MLAAD and all of ASVspoof 2019 LA |
| Input | each window is normalised to zero mean and unit variance (upstream normalises the whole utterance) |
| P(fake) | index 0 |
| Reproduction gap | on In-the-Wild, small matches its card's EER and MMS-300M misses its card by 1.4 points ([Results](results.md#reproductions)). Both are scored the same way and parity passed window by window, so that gap lies in the protocol or the card, not the adapter: a concrete case for decision 5 |

### AASIST3

| | |
| --- | --- |
| Weights | `lab260/AASIST3`, a weak multilingual baseline: the released weights aren't the paper's, and the Speech DF Arena measures them at 28 to 32% EER on In-the-Wild and ASVspoof 2021 |
| Input | pre-emphasised (0.97) per window |
| P(fake) | index 0. The README reads index 0 as bona fide but the training code reads index 1; the sanity check sides with the code |

### DeepfakeBench detectors

| | |
| --- | --- |
| Models | Xception, EfficientNet-B4, UCF, F3Net, SPSL, RECCE, SRM, CORE and FFD share `models/deepfakebench.py` (`input.detector` picks the upstream yaml) |
| Weights | release v1.0.1 at commit `f188b1c`. All nine load strictly (the EfficientNet-B4 mismatch in #79 doesn't occur with `efficientnet-pytorch` 0.7.1). The release's Capsule, CNN-Aug, MesoNet and MesoInception weights would load too |
| Input | RGB 256 px crops, cubic resize, mean and std 0.5 (not ImageNet's) |
| P(fake) | label 1 is fake. UCF scores with its common-forgery head |
| Build stand-ins | RECCE's timm encoder is built with `pretrained=False` (the checkpoint replaces every weight); FFD's template images, unused by its map type, are blank 19 x 19 images |
| Known issues | published weights don't reproduce the README for some users (#109, #159); #151 asks whether every checkpoint is FF++ c23 (unanswered); the yamls name one manipulation for F3Net (NeuralTextures) and SPSL (FaceSwap), which may not match the checkpoints. The guard treats all nine as trained on FF++ train |

### Effort

| | |
| --- | --- |
| Adapter | the DeepfakeBench adapter, with `loralib` stubbed and CLIP built from a pinned config |
| Weights | its Google Drive folder has answered "Quota exceeded" since 2026-10-05, so since 2026-11-13 the checkpoint comes from a private Hugging Face mirror under Oshan's account (no upstream licence, so not public), which needs an `HF_TOKEN` from that account; anyone else downloads it from Drive in a browser |
| Load | strict, filling in only CLIP's `position_ids` buffer, which newer transformers no longer save |
| Known issues | upstream reports a reproduction gap (CDF-v2 AUC 0.885 against 0.956, #19) and missing checkpoint keys (#18). No licence file (#3) |

### SBI

| | |
| --- | --- |
| Weights | `FFc23.tar` from Google Drive |
| Input | as in `inference_video.py`: the face box widened by an eighth per side, stretched to 380 x 380, RGB / 255, no normalisation |
| P(fake) | softmax index 1 |
| Differs from upstream | we keep the largest face (upstream takes the max over every face at least half its area) and use SCRFD boxes, not RetinaFace, so parity compares the network on identical crops |

### GenD (CLIP and PE)

| | |
| --- | --- |
| Weights | `yermandy/GenD_CLIP_L_14` and `yermandy/GenD_PE_L`, trained on FF++ only (the paper's 14 benchmarks, FakeAVCeleb among them, are test sets) |
| Input | crops from the shared SCRFD pipeline with five-point alignment ([Evaluation](evaluation.md#preprocessing)); its `det_10g.onnx` is insightface's `buffalo_l` |
| P(fake) | logit index 1; a video scores the mean of its frames |
| Known issues | PE fails on Apple MPS for one video; use CPU or CUDA. The DINOv3 variant waits for someone to accept Meta's terms on its gated backbone |

### UnivFD

| | |
| --- | --- |
| Network | one linear layer on CLIP ViT-L/14 features, through transformers on the snapshot GenD pins |

### LAA-Net

| | |
| --- | --- |
| Weights | the SBI checkpoint, in one zip of a Dropbox folder, so a fetch downloads the whole zip |
| Input | 32 frames with flip testing, in float32 where upstream uses float64 |
| Differs from upstream | upstream averages a video's logits before the sigmoid; here each frame gets a sigmoid, which orders videos slightly differently |

### GenConViT

| | |
| --- | --- |
| Weights | `genconvit_ed_inference.pth` from `Deressa/GenConViT` (CC BY-NC 4.0 weights over MIT code) |
| Training | DFDC, FF++, DeepfakeTIMIT, TrustedMedia and a random split of Celeb-DF v2 (paper, Section IV); the config missed the last two until 2026-11-28, so its Celeb-DF v2 run was withdrawn and its score direction can't be settled here |
| Branch scored | only ED, upstream's `--net ed`: the VAE branch draws `torch.randn_like` noise in its encoder at test time, so its scores differ between runs |
| Load | upstream's `GenConViTED` is built with `pretrained=False`; the Swin embedder it attaches as the ConvNeXt's `patch_embed` is never called by ConvNeXt's forward, but its weights are loaded strictly. The checkpoint was saved with timm 0.6.5, so conversion runs timm's own Swin `checkpoint_filter_fn` on both copies of the embedder, then saves one copy of the shared tensors |
| P(fake) | class 0 is fake: the sigmoid of logit 0 |
| Differs from upstream | crops are the shared detector's box stretched to 224 (upstream: dlib's tight box via `face_recognition`, `INTER_AREA`), 32 frames per video (upstream 15) |

### DFDC winner

| | |
| --- | --- |
| Weights | the seven EfficientNet-B7 checkpoints of Selim Seferbekov's winning DFDC entry (MIT, GitHub release 0.0.1; the release serves each stream slowly, so fetch them in parallel) |
| Load | each is built by upstream's own `DeepFakeClassifier` with timm's constructors told not to fetch ImageNet weights, converted to one safetensors file per checkpoint (their pickles carry a numpy scalar, so the safe unpickler is told about numpy's scalar and dtype) and loaded strictly |
| Input | the shared `letterbox` crop: the box widened by a third each side, resized to 380 on its longer side and centred on black, with upstream's two helpers (MIT, copied into `preprocess/faces.py`) |
| P(fake) | per frame, the mean of the seven sigmoids |
| Differs from upstream | upstream runs fp16 and summarises each model's frames with its "confident strategy"; here fp32 and the shared mean |

### FTCN

| | |
| --- | --- |
| Weights | `ftcn_tt.pth` (FTCN+TT, FF++ c23) from the GitHub release |
| Code | upstream states no licence, so it is imported from the checkout: built by upstream's own `config` and `PluginLoader` with yacs in place of fvcore's `CfgNode` and training-only imports stubbed. Two patches (`patches/FTCN/`): `nn.Conv3d`'s signature gained `device` and `dtype` after torch 1.4, and `get_crop_box` used `np.int` |
| Input | `preprocess/clips.py`: the first 100 consecutive frames (the frames LipForensics and RealForensics read, so the three share one detection cache), each face box widened by half on each side, then 32-frame clips every 16 frames, each aligned by upstream's `FasterCropAlignXRay` with one transform fitted to the whole clip's five landmarks |
| P(fake) | a clip's sigmoid |
| Differs from upstream | upstream reads up to 768 frames and slides one frame at a time. Its padding of short tracks is kept, including its failure on tracks under 9 frames (`too_short`; the config asks for 16) |

### FakeSTormer

| | |
| --- | --- |
| Weights | the C23 checkpoint, in one zip of a Dropbox folder, so a fetch downloads the whole zip |
| Input | 4 frames per input, not 16: the 32 evenly spaced frames are cut into 8 runs that don't overlap |

### RealForensics

| | |
| --- | --- |
| Weights | `realforensics_ff.pth` (FF++ c23 after LRW self-supervision) from Google Drive |
| Network | upstream's CSN-101 (`csn_temporal_no_head`, built from pytorchvideo's blocks, so pytorchvideo is a dependency) and normalised-linear head, loaded as `stage2/eval.py` does |
| Input | `preprocess/talking_faces.py`: the same first 100 consecutive frames and FAN landmarks as LipForensics, then upstream's own `crop_patch` (landmarks smoothed over up to 6 frames each side, warped to the LRW mean face on 8 stable points, a 150 x 150 colour patch around landmarks 15 to 67). Clips of 25 frames back to back, the last moved back to the video's end (pytorchvideo's `backpad_last`), centre-cropped to 140 and resized to 112 without antialiasing, as torchvision 0.10 resized tensors |
| P(fake) | a clip's sigmoid |

### LipForensics

| | |
| --- | --- |
| Weights | `lipforensics_ff.pth` from Google Drive |
| Input | `preprocess/mouths.py`: mouth crops reuse upstream's warp and crop code on 68 FAN landmarks per frame, from the shared detector's boxes (upstream suggests RetinaFace): the first 100 consecutive frames, landmarks smoothed over 12 frames, 96 x 96 grayscale mouth patches, then non-overlapping 25-frame clips centre-cropped to 88 |
| P(fake) | a clip's sigmoid |
| Known issues | users report its FF++ numbers don't reproduce (#9) |

### HAVIC

| | |
| --- | --- |
| Weights | `JielunPeng/HAVIC`, fine-tuned on an unpublished 70% of FakeAVCeleb |
| Patches | two (`patches/HAVIC/`) fix timm 1.x and a `use_mask` bug in upstream's own inference |
| Input | `preprocess/av.py`: up to 50 frames at 5 per second, raw face boxes resized to 224, in 16-frame windows at stride 2, with the audio under each window as 128-bin Kaldi filterbanks |
| P(fake) | each window's sigmoid |
| Differs from upstream | a frame without a face drops out (upstream skips the video) |
| Limits | it needs at least 3.2 s with a face in 16 sampled frames, so HiDF's 3 s clips are too short; MAVOS-DD Hindi is its South Asian set |

### AVH-Align

| | |
| --- | --- |
| Weights | `AVH-Align_AV1M.pt` (CC BY-NC-SA 4.0), the fusion network, and AV-HuBERT large's `self_large_vox_433h.pt`, loaded through torch's safe unpickler with inert stand-ins for the lip-reading vocabulary it pickles |
| Licence | AV-HuBERT's bars "biometric processing" and reverse-engineering; Oshan accepted that risk for this research use on 2026-10-28 |
| Stand-ins | fairseq doesn't install here, so `models/avh_align.py` rebuilds AV-HuBERT's encoder from its architecture (3D stem and ResNet-18 trunk with PReLU, the audio projection, concat fusion, and the 24-layer pre-norm transformer as transformers' `Wav2Vec2EncoderStableLayerNorm`, fairseq names mapped by `_fairseq.hf_name`) |
| Input | LipForensics' mouth crops (the same mpc001 recipe AV-HuBERT's `align_mouth.py` uses, here on FAN landmarks, not dlib's) over the first 100 frames, and the audio as 26 log filterbanks stacked four at a time. Neither side resamples to 25 fps, as upstream doesn't; MAVOS-DD is 24 fps |
| P(fake) | the fusion network scores the agreement of AV-HuBERT's video-only and audio-only features, frame by frame; a video's score is `logsumexp(-agreement)` over frames, reported through a sigmoid (monotone, so AUC and EER are upstream's) |
| Limits | audio-visual evalsets only (HiDF has no audio) |

### LipFD

| | |
| --- | --- |
| Input | upstream's reader takes `img[:, 500:, i:i + 500] for i in range(5)`, five crops of the first frame one pixel apart, so only the CLIP branch sees the other frames. The adapter keeps that, since the checkpoint was evaluated that way. Input is BGR on the 0 to 255 scale, unnormalised, and resizing has no antialiasing, as in torchvision 0.16 |
| Face crops | it learned from LRS clips, which are face tracks, and upstream crops no faces, so `preprocess/tracks.py` cuts SyncNet's face crop from the shared detector's boxes. That step is a choice of ours, not upstream's |
| Licence | none stated upstream |

### AVAD

| | |
| --- | --- |
| P(fake) | the delay-distribution score only (`lam` 0, upstream's default), so `pca.pkl` and the feature model stay pinned but unused |
| Input | video is mapped to 25 fps, which upstream assumes, and the first 50 windows are read |
| Face crops | as for LipFD: SyncNet's face crop cut by `preprocess/tracks.py`, a choice of ours |

## Candidates, not yet configured

| Candidate | Checked | Why not yet |
| --- | --- | --- |
| Nes2Net-X | 2026-10-05 | another back end on the same XLS-R; licence undeclared |
| Forensics Adapter | 2026-10-05 | CVPR 2025; CC BY-NC 4.0; no inference script |
| AI-Face fairness checkpoints | 2026-10-05 | login-walled, and they saw Celeb-DF v2 |
| MAVOS-DD baselines | 2026-10-05 | fine-tuned on MAVOS-DD |
| Referee | 2026-10-28 | needs a reference clip of each speaker |
| NPR, CLIPping the Deception | 2026-10-28 | GAN-image detectors that add little beside UnivFD |
| ExposeAnyone | 2026-10-28 | no code or weights released |

Hugging Face community classifiers with undisclosed training data are left out.
