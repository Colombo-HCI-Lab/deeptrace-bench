---
title: "Models"
description: "Every detector: what it is, whether its scores can be trusted yet, why any is missing, and porting notes per adapter"
---

The configs (`configs/models/<id>.yaml`) are the source of truth, and `uv run scripts/setup_models.py --list` prints the current state. Each model's architecture, training data and paper are in the [catalog](catalog.md#models); terms such as parity and score direction are defined in [How it works](method.md#key-terms).

## Status

Checked on 2026-10-12.

| Model                                   | Modality    | Status           | Parity     | Direction |
| --------------------------------------- | ----------- | ---------------- | ---------- | --------- |
| Xception, EfficientNet-B4               | video       | ready, wave 1    | yes        | not yet   |
| SBI                                     | video       | ready, wave 1    | yes        | not yet   |
| GenD (CLIP L/14)                        | video       | ready, wave 1    | not yet    | not yet   |
| GenD (PE L/14)                          | video       | workable         | not yet    | not yet   |
| UCF, F3Net, SPSL, RECCE, SRM, CORE, FFD | video       | workable         | yes        | not yet   |
| Effort                                  | video       | workable         | not yet    | not yet   |
| LipForensics                            | video       | workable         | not yet    | not yet   |
| AltFreezing                             | video       | blocked          | no adapter |           |
| FTCN                                    | video       | workable         | yes        | not yet   |
| GenConViT, DFDC winner (B7), UnivFD     | video       | later            | no adapter |           |
| RealForensics, LAA-Net, FakeSTormer     | video       | later            | no adapter |           |
| TALL                                    | video       | blocked          | no weights |           |
| AVAD, AVH-Align, LipFD                  | audio-video | later            | no adapter |           |
| AASIST, AASIST-L                        | audio       | ready, wave 1    | yes        | settled   |
| XLS-R + AASIST                          | audio       | workable, wave 1 | yes        | settled   |
| XLS-R + SLS, XLSR-Mamba                 | audio       | workable         | not yet    | sanity    |
| AASIST3                                 | audio       | workable         | not yet    | sanity    |
| AntiDeepfake MMS-300M, w2v2 small       | audio       | workable         | yes        | settled   |
| DF Arena 500M, 1B                       | audio       | workable         | yes        | settled   |
| HAVIC                                   | audio-video | workable         | not yet    | not yet   |

"Sanity": bona fide ASVspoof 2019 LA eval clips score near 0 and spoofed ones near 1, short of the formal check.

| Missing | Why |
| --- | --- |
| Video score direction | waits on FF++ test or Celeb-DF v2, both on request ([Datasets](datasets.md#requests-to-send)) |
| AltFreezing (`blocked`) | weights only on RecDrive and Baidu (password `altf`), and issues #7 and #8 report the file gone, with no reply or other mirror (rechecked 2026-10-05). Someone with a working session has to fetch them |
| Effort's weights | Google Drive refuses scripted downloads over quota; fetched by hand through a signed-in browser on 2026-10-28 ([Effort](#effort)) |
| Parity for eight adapters | [Parity](#parity) says what each needs |

**Why these models.** Classic detectors sit beside foundation-model ones (GenD; the XLS-R family, AntiDeepfake, DF Arena), so a failure both kinds share points at the data. The DeepfakeBench detectors each read a different cue: texture (Xception, EfficientNet-B4), content-free forgery features (UCF), frequency (F3Net), phase (SPSL), reconstruction (RECCE), noise residuals (SRM), consistent representations (CORE), manipulation attention (FFD). LipForensics and HAVIC target lip-sync fakes, v1's focus. Each model's own reason, and whether it is a cut candidate, is on its page in the [Model catalog](catalog.md#models).

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

DeepfakeBench's reference shares the adapter's import stand-ins (and RECCE's and FFD's build stand-ins), so a pass there covers weights, numerics and transform, not upstream's imports; the strict load covers the architecture.

Not checked yet (strict loads, tests and the pipeline check only):

| Model | What parity needs |
| --- | --- |
| AASIST3, GenD PE-L | nothing special; upstream runs in our environment, so these are the cheapest next |
| GenD (CLIP) | `parity.py export` to handle its native-size crops, which differ in shape |
| XLS-R + SLS | the patched fairseq zip in its checkout; the vendored commit returns no per-layer outputs |
| XLSR-Mamba | mamba-ssm's CUDA kernels on a GPU |
| HAVIC, LipForensics | upstream's face and landmark pipelines |
| Effort | its weights |

## Per model

Adapters date from 2026-10-04 (GenD) and 2026-10-05 (the rest). Reproduced numbers are in [Results](results.md#reproductions); the protocol differences behind their gaps are here. How adapters work in general: [Adding a model](adding_a_model.md#how-adapters-work).

### DeepfakeBench detectors

Xception, EfficientNet-B4, UCF, F3Net, SPSL, RECCE, SRM, CORE and FFD share `models/deepfakebench.py` (`input.detector` picks the upstream yaml): commit `f188b1c`, release v1.0.1 weights. The release's Capsule, CNN-Aug, MesoNet and MesoInception weights would load too. All nine load strictly (the EfficientNet-B4 mismatch in #79 doesn't occur with `efficientnet-pytorch` 0.7.1).

- **Input:** RGB 256 px crops, cubic resize, mean and std 0.5 (not ImageNet's). Label 1 is fake. UCF scores with its common-forgery head.
- **Build stand-ins:** RECCE's timm encoder is built with `pretrained=False` (the checkpoint replaces every weight); FFD's template images, unused by its map type, are blank 19 x 19 images.
- **Known issues:** published weights don't reproduce the README for some users (#109, #159); #151 asks whether every checkpoint is FF++ c23 (unanswered); the yamls name one manipulation for F3Net (NeuralTextures) and SPSL (FaceSwap), which may not match the checkpoints. The guard treats all nine as trained on FF++ train.

### SBI

`FFc23.tar` from Google Drive. Input as in `inference_video.py`: the face box widened by an eighth per side, stretched to 380 x 380, RGB / 255, no normalisation; P(fake) is softmax index 1. We keep the largest face (upstream takes the max over every face at least half its area) and use SCRFD boxes, not RetinaFace, so parity compares the network on identical crops.

### GenD (CLIP and PE)

`yermandy/GenD_CLIP_L_14` and `yermandy/GenD_PE_L`, trained on FF++ only (the paper's 14 benchmarks, FakeAVCeleb among them, are test sets). Crops come from the shared SCRFD pipeline with five-point alignment ([Evaluation](evaluation.md#preprocessing)); its `det_10g.onnx` is insightface's `buffalo_l`. Logit index 1 is fake; a video scores the mean of its frames. PE fails on Apple MPS for one video; use CPU or CUDA. The DINOv3 variant waits for someone to accept Meta's terms on its gated backbone.

### AASIST, AASIST-L

Weights ship in the upstream repo. Upstream scores the bona fide logit, so P(fake) is softmax index 0; reading it the other way explains the inverted outputs in #17. **Protocol:** upstream scores each utterance's first four seconds; we average every window and refuse the 1,059 eval clips under a second. The reproductions used TF32 convolutions (since turned off by `compute.tf32`), which move single scores by up to 2.5e-2 but left DF Arena 500M's In-the-Wild EER unchanged.

### XLS-R + AASIST, XLS-R + SLS, XLSR-Mamba

All three fine-tune XLS-R 300M inside Tak et al.'s `SSLModel` on ASVspoof 2019 LA alone, so they are clean zero-shot probes on every South Asian set. They run without fairseq (transformers' `Wav2Vec2Model` stands in, with the per-layer outputs SLS pools) and XLSR-Mamba without mamba-ssm (pure-PyTorch kernels checked against transformers' Mamba mixer). P(fake) is index 0. Weights: `LA_model.pth` (byte-identical to the MIT mirror `SpeechAntiSpoofingBenchmarks/W2V2-AASIST`) and `MMpaper_model.pth` from the authors' Drive, and `AustinXiao/XLSR-Mamba-LA`. XLS-R + AASIST's parity used the fairseq commit its checkout vendors, since fairseq 0.12.2 rejects the config inside `xlsr2_300m.pt`.

### DF Arena 500M and 1B

`Speech-Arena-2025/DF_Arena_*_V_1`, code and weights in one Hugging Face repo, non-commercial; the top open models on the Speech DF Arena leaderboard (5.9% average EER for 1B). XLS-R (300M or 1B) with layer-wise attention pooling and a conformer head; the code is pinned, never run with `trust_remote_code`. P(fake) is the `spoof` softmax. The guard refuses it on ASVspoof 2019 LA eval (training partitions not named) and `mlaad_*`, so In-the-Wild is its reproduction set. **Protocol:** upstream scores a clip's first four seconds where we average every window, and we refuse the 865 In-the-Wild clips under a second where the card tiles them. Parity compares windows, so it can't see either; decision 5 (the evaluation setup, still open) should settle both.

### AntiDeepfake MMS-300M and wav2vec 2.0 small

NII Yamagishi Lab (`nii-yamagishilab/*-anti-deepfake`), post-trained on 74k hours of real and fake speech in over 100 languages. MMS-300M is pretrained on about 1,400 languages, the small one on English only, isolating multilingual pretraining. Each window is normalised to zero mean and unit variance (upstream normalises the whole utterance); P(fake) is index 0. The post-training data includes FLEURS (Hindi, Bengali, Urdu, Tamil, Nepali and more, not Sinhala), MLAAD and all of ASVspoof 2019 LA. On In-the-Wild, small matches its card's EER and MMS-300M misses its card by 1.4 points ([Results](results.md#reproductions)). Both are scored the same way and parity passed window by window, so that gap lies in the protocol or the card, not the adapter: a concrete case for decision 5.

### AASIST3

`lab260/AASIST3`, a weak multilingual baseline: the released weights aren't the paper's, and the Speech DF Arena measures them at 28 to 32% EER on In-the-Wild and ASVspoof 2021. Input is pre-emphasised (0.97) per window. The README reads index 0 as bona fide but the training code reads index 1; the sanity check sides with the code, so P(fake) is index 0. Refused on `mlaad_*`.

### FTCN

`ftcn_tt.pth` (FTCN+TT, FF++ c23) from the GitHub release. Upstream states no licence, so it is imported from the checkout: built by upstream's own `config` and `PluginLoader` with yacs in place of fvcore's `CfgNode` and training-only imports stubbed. Two patches (`patches/FTCN/`): `nn.Conv3d`'s signature gained `device` and `dtype` after torch 1.4, and `get_crop_box` used `np.int`. Input (`preprocess/clips.py`): the first 96 consecutive frames, each face box widened by half on each side, then 32-frame clips every 16 frames, each aligned by upstream's `FasterCropAlignXRay` with one transform fitted to the whole clip's five landmarks; P(fake) is a clip's sigmoid. Upstream reads up to 768 frames and slides one frame at a time; its padding of short tracks is kept, including its failure on tracks under 9 frames (`too_short`; the config asks for 16). The reference needs torch 1.4, which has no macOS arm64 wheels: Python 3.8, `torch==1.4.0+cpu`, `numpy<1.20`, `opencv-python-headless<4.6`, `fvcore==0.1.1.post20200716`, pyyaml, einops, termcolor, simplejson, psutil and `matplotlib<3.6` on a Linux x86 machine, with `DTB_PATH_MAP` when the run directory is mounted at another path.

### Effort

Runs through the DeepfakeBench adapter, with `loralib` stubbed and CLIP built from a pinned config. Its Google Drive folder answered "Quota exceeded" on 2026-10-05; retry `setup_models.py effort`. Upstream reports a reproduction gap (CDF-v2 AUC 0.885 against 0.956, #19) and missing checkpoint keys (#18), which the strict load will settle. No licence file (#3).

### HAVIC

The only open audio-visual detector (`JielunPeng/HAVIC`). Two patches (`patches/HAVIC/`) fix timm 1.x and a `use_mask` bug in upstream's own inference. Input: up to 50 frames at 5 per second, raw face boxes resized to 224, in 16-frame windows at stride 2, with the audio under each window as 128-bin Kaldi filterbanks; P(fake) is each window's sigmoid. A frame without a face drops out (upstream skips the video). It needs at least 3.2 s with a face in 16 sampled frames, so HiDF's 3 s clips are too short; MAVOS-DD Hindi is its South Asian set once its terms are accepted. Fine-tuned on an unpublished 70% of FakeAVCeleb, so refused there.

### LipForensics

`lipforensics_ff.pth` from Google Drive. Mouth crops reuse upstream's warp and crop code on 68 FAN landmarks per frame, from the shared detector's boxes (upstream suggests RetinaFace): the first 100 consecutive frames, landmarks smoothed over 12 frames, 96 x 96 grayscale mouth patches, then non-overlapping 25-frame clips centre-cropped to 88. P(fake) is a clip's sigmoid. Users report its FF++ numbers don't reproduce (#9).

### Added 2026-10-28, not adapted yet

Eleven video and audio-visual models chosen for the South Asian question, each with weights confirmed reachable on 2026-10-28 except TALL's. They fall into four tests:

| Test | Models |
| --- | --- |
| Do cues that ignore face texture drop less on South Asian faces? | FTCN (motion only), RealForensics (lip motion from real talking faces) |
| Do detectors that never saw a fake avoid training-face bias? | AVAD, AVH-Align (audio-visual alignment of real speech) |
| Do more varied training faces help? | GenConViT, DFDC winner (both trained on DFDC's paid actors, not only FF++) |
| Newer relatives of what holds up on MAVOS-DD | LAA-Net, FakeSTormer (SBI-style training), LipFD (lip-sync specific), UnivFD (generic AI-image floor), TALL (MAVOS-DD's own baseline) |

DeepfakeBench releases no checkpoint for any of its video detectors (TALL, FTCN, AltFreezing, STIL, VideoMAE, TimeSformer, X-CLIP), only the frame-level set already here (checked 2026-10-28), so these come from their authors' repos. LAA-Net's and FakeSTormer's checkpoints are in Dropbox folders fetched by hand; AVH-Align needs AV-HuBERT without fairseq; FTCN and LipFD state no licence.

### Candidates, not yet configured

Checked 2026-10-05: **Nes2Net-X** (another back end on the same XLS-R; licence undeclared), **Forensics Adapter** (CVPR 2025; CC BY-NC 4.0; no inference script), the **AI-Face fairness checkpoints** (login-walled, and they saw Celeb-DF v2) and the **MAVOS-DD baselines** (fine-tuned on MAVOS-DD). Checked 2026-10-28: **Referee** (needs a reference clip of each speaker), **NPR** and **CLIPping the Deception** (GAN-image detectors that add little beside UnivFD), **ExposeAnyone** (no code or weights released). Hugging Face community classifiers with undisclosed training data are left out.
