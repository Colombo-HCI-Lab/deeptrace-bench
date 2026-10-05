# Models

Availability checked against the live repos on 2026-10-03, and for every video model again on 2026-10-04 (pinned commits still current, every weight file still public); models added on 2026-10-05 were checked that day. The source of truth is `configs/models/<id>.yaml`; `uv run scripts/setup_models.py --list` prints the current table, including which adapters are written. As of 2026-10-05 they are GenD, AASIST, AASIST-L, DF Arena 500M and 1B, and the five DeepfakeBench detectors (Xception, EfficientNet-B4, UCF, F3Net, SPSL); parity results are under [Parity](#parity).

| Model           | Modality    | Status   | Wave | Weights                      | Trained on (released weights) | Licence         |
| --------------- | ----------- | -------- | ---- | ---------------------------- | ----------------------------- | --------------- |
| Xception        | video       | ready    | 1    | DeepfakeBench release v1.0.1 | FF++ c23                      | CC BY-NC 4.0    |
| EfficientNet-B4 | video       | ready    | 1    | DeepfakeBench release v1.0.1 | FF++ c23                      | CC BY-NC 4.0    |
| SBI             | video       | ready    | 1    | Google Drive                 | FF++ real, self-blended       | research only   |
| GenD            | video       | ready    | 1    | Hugging Face                 | FF++                          | MIT             |
| UCF             | video       | workable |      | DeepfakeBench release v1.0.1 | FF++ c23, all four methods    | CC BY-NC 4.0    |
| F3Net           | video       | workable |      | DeepfakeBench release v1.0.1 | FF++ c23                      | CC BY-NC 4.0    |
| SPSL            | video       | workable |      | DeepfakeBench release v1.0.1 | FF++ c23                      | CC BY-NC 4.0    |
| AASIST          | audio       | ready    | 1    | in the upstream repo         | ASVspoof 2019 LA              | MIT             |
| AASIST-L        | audio       | ready    | 1    | in the upstream repo         | ASVspoof 2019 LA              | MIT             |
| XLS-R + AASIST  | audio       | workable | 1    | Google Drive + Hugging Face  | ASVspoof 2019 LA              | MIT             |
| DF Arena 500M   | audio       | workable |      | Hugging Face                 | ASVspoof 2019 + 5, MLAAD, six more | non-commercial |
| DF Arena 1B     | audio       | workable |      | Hugging Face                 | ASVspoof 2019 + 5, MLAAD, six more | non-commercial |
| Effort          | video       | workable |      | Google Drive                 | FF++ c23                      | none published  |
| HAVIC           | audio-video | later    |      | Hugging Face                 | LRS2, then 70% of FakeAVCeleb | MIT             |
| LipForensics    | video       | later    |      | Google Drive                 | FF++ c23 (LRW pretraining)    | MIT             |
| AltFreezing     | video       | blocked  |      | RecDrive / Baidu only        | FF++                          | MIT             |
| XLSR-Mamba      | audio       | dropped  |      | Hugging Face                 | ASVspoof 2019 LA              | MIT             |
| AASIST3         | audio       | dropped  |      | Hugging Face                 | ASVspoof 2019 + 5, MLAAD      | CC BY-NC-ND 4.0 |

Each modality pairs classic detectors with a foundation-model one (GenD for video; XLS-R + AASIST and DF Arena for audio). The DeepfakeBench detectors add different cues on the classic side: plain texture (Xception, EfficientNet-B4), forgery features separated from content (UCF), frequency (F3Net) and phase (SPSL). If both kinds fail on the same attribute, the cause is likely in the data rather than one model.

## One environment, ported adapters

Everything runs in this repo's single environment (Python 3.12, torch 2.8, transformers 4.56). Upstream repos pin much older stacks (DeepfakeBench: Python 3.7 and torch 1.12; XLS-R + AASIST: torch 1.8 and fairseq) that don't run on H100s, so we don't use them for results. Instead:

1. `setup_models.py` clones the upstream repo at its pinned commit into `third_party/` and archives it to `DTB_ROOT/upstream/`. Code from Effort (no licence), SBI (research only) and DeepfakeBench (CC BY-NC) is never copied into this repo; MIT code (AASIST, GenD, HAVIC) may be.
2. The adapter (`src/deeptrace_bench/models/<module>.py`) loads the one model file it needs from the checkout with `importlib`. Small compatibility fixes go in `patches/<repo>/` (see `patches/README.md`).
3. `convert_checkpoint` turns the download into `model.safetensors` with normalised keys, once. Adapters then never unpickle Google Drive files.
4. **Parity check** before any result counts: run the upstream code in its original environment (an Apptainer image or a throwaway old-torch venv, on CPU or an A100) on about 200 items, run the adapter on the same items, and require scores to agree within about 1e-3. `scripts/parity.py` does this with a reference script per upstream (see [adding_a_model.md](adding_a_model.md)) and writes `results/parity/<id>.json`. This is what licenses every patch.
5. **Score direction**: on the reproduction evalset (FF++ test, ASVspoof 2019 LA eval), AUC must come out above 0.5. An inverted logit otherwise flips every result silently.

Expect these breakages on Python 3.12 and torch 2.x: `torch.load` now defaults to `weights_only=True`; `np.float` and `np.int` are gone; `distutils` and `imp` are gone.

## Per model

**DeepfakeBench: Xception, EfficientNet-B4, UCF, F3Net, SPSL.** Commit `f188b1c` and release v1.0.1 weights, which ship 13 frame-level checkpoints (also Capsule, CNN-Aug, CORE, FFD, MesoNet, MesoInception and RECCE, which the same adapter could load). **Adapter written 2026-10-05** (`models/deepfakebench.py`, one class, `input.detector` picks the yaml): it imports the one detector file and its backbone from the checkout with stand-ins for DeepfakeBench's training losses, metrics and tensorboard, builds the network with upstream's own class and yaml, and loads the converted checkpoint with every key matching. All five load strictly; the EfficientNet-B4 mismatch reported in #79 doesn't occur with `efficientnet-pytorch` 0.7.1. Input as DeepfakeBench's test transform: RGB 256 px crops, cubic resize, mean and std 0.5 (not ImageNet's, as the configs said before); label 1 is fake for all five. UCF scores with its common-forgery head; its inference path also runs the unused content encoder. Known issues: published weights don't reproduce the README table for some users (#109, #159). Issue #151 asks whether every checkpoint is FF++ c23, unanswered, and the detector yamls name single FF++ manipulations for F3Net (NeuralTextures) and SPSL (FaceSwap), which may not describe the released checkpoints; the guard treats all five as trained on FF++ train either way.

**SBI.** `FFc23.tar` and `FFraw.tar` from Google Drive, about 135 MB each. A plain EfficientNet-B4 on RetinaFace crops at 380 px. Upstream's `src/inference/inference_video.py` scores any video (max over faces, mean over frames), which makes the parity check easy. Install the right `retinaface_pytorch` package for parity (#55, #29).

**GenD.** `yermandy/GenD_CLIP_L_14` on Hugging Face, pinned to revision `891ce01`. Already modern. Trained on FF++ only: the paper's 14 benchmarks, FakeAVCeleb among them, are test sets. **Adapter written 2026-10-04** (`models/gend.py`): it loads upstream's `src/hf/modeling_gend.py` from the pinned checkout and points it at a pinned local copy of its CLIP backbone (`openai/clip-vit-large-patch14` at `32bd642`), which upstream would otherwise fetch unpinned; all ten weight files are hash-pinned. Index 1 of the logits is fake; a video scores the mean of its frames' P(fake). Its face crops come from the shared pipeline, which ports `detector.py`'s SCRFD detector and five-point alignment (see [evaluation.md](evaluation.md)). The detector model, `det_10g.onnx`, is taken from insightface's official v0.7 `buffalo_l` pack (licensed for non-commercial research only) and hashes identical to the third-party mirror upstream downloads. Runs end to end in the smoke test on a laptop (MPS and CPU agree to 2e-6). **Parity with upstream is not checked yet.**

**AASIST, AASIST-L.** Weights committed in the upstream repo; `models/AASIST.py` needs no patch (only `main.py` imports `torchcontrib`). **Adapter written 2026-10-05** (`models/aasist.py`). Upstream labels bona fide 1 and scores the bona fide logit, so P(fake) is the softmax at index 0. On a seeded sample of ASVspoof 2019 LA eval (186 real, 187 fake) that gives AUC 0.999 and EER 1.6% (AASIST, paper 0.83%) and 1.1% (AASIST-L, paper 0.99%): the direction is settled, and the inverted outputs reported in #17 come from reading the bona fide logit as a fake score. Full-eval reproduction waits for the full download.

**XLS-R + AASIST.** Detector weights in the authors' Google Drive folder (`LA_model.pth`, `Best_LA_model_for_DF.pth`); the XLS-R 300M backbone from `facebook/wav2vec2-xls-r-300m`, pinned to revision `1a640f3`. The port drops fairseq: build the front end in transformers and remap the fairseq keys, checking each one (layer names differ). Load with `strict=False` and list missing keys (#1). About 1 to 2 days of work.

**DF Arena 500M and 1B.** `Speech-Arena-2025/DF_Arena_*_V_1` on Hugging Face, code and weights in one repo, non-commercial licence; the top open models on the Speech DF Arena leaderboard (5.9% average EER for 1B). An XLS-R (300M for the 500M model, 1B for the 1B) with attention pooling over every hidden layer and a conformer head. **Adapter written 2026-10-05** (`models/df_arena.py`): the code is pinned like any upstream (`host: hf`) and loaded as a package from the checkout, never with `trust_remote_code`; the XLS-R config that `backbone.py` fetches unpinned comes from a pinned snapshot instead, so loading works offline. Windows go through one at a time (upstream's `forward` unsqueezes a 1-D waveform); P(fake) is the `spoof` softmax. Trained on ASVspoof 2019 (partitions not named, so the guard refuses ASVspoof 2019 LA eval) and 2024, MLAAD (refused on `mlaad_*`), Codecfake, LibriSeVoc, DFADD, CtrSVDD, SpoofCeleb and EnvSDD; In-the-Wild is its reproduction set (card: EER 1.76% for 500M, 0.91% for 1B). On a four-clip Urdu smoke sample it scored both real clips as fake (0.9996 and 0.75): noise at that size, but the kind of gap this benchmark measures.

**Effort (second wave).** The FF++ checkpoint is on Google Drive. Hard-coded local CLIP path (#15), a reported reproduction gap (CDF-v2 AUC 0.885 against 0.956, #19), parameters reported missing from the FF++ checkpoint (#18).

**HAVIC (later).** The only open audio-visual detector. Fine-tuned on a random 70% of FakeAVCeleb with no published item list, so the guard refuses it on FakeAVCeleb. Useful once DeePhy, DF-Platter or InDeepFake arrive.

**LipForensics (later).** Needs a RetinaFace plus FAN landmark pipeline for mouth crops. Users can't reproduce its FF++ numbers (#9). Relevant for lip-sync fakes.

**AltFreezing (blocked).** Weights only on RecDrive and Baidu (password `altf`). Someone with a working session has to fetch them first.

**Dropped.** XLSR-Mamba needs fairseq plus compiled mamba-ssm kernels. AASIST3's released weights aren't the paper's, score 28 to 32% EER in independent tests, and were trained on MLAAD, one of our test sets.

**Candidates, not yet configured.** The AntiDeepfake family (NII Yamagishi Lab, e.g. `nii-yamagishilab/mms-300m-anti-deepfake`, CC BY-NC-SA 4.0) is post-trained on 74k hours across 29 corpora, including ASVspoof 2019, MLAAD and FLEURS, so it is unusable on MLAAD and on FLEURS reals. It isn't strictly fairseq-bound (checked 2026-10-05): its `model.safetensors` uses fairseq key names on a standard wav2vec2 network plus mean pooling and one linear layer (index 0 fake), so a key remap onto transformers' `Wav2Vec2Model` should load it; the parity check then needs fairseq 0.12.2 in an old environment. The same remap would serve XLS-R + AASIST and XLS-R + SLS (ACM MM 2024, the strongest open model trained on ASVspoof 2019 LA alone), so it is the next round's shared piece. On the video side, LipFD (NeurIPS 2024) is the lip-sync detector that fits v1, but needs combined audio-video input and states no licence.

## Parity

Checked 2026-10-05 with `scripts/parity.py`; summaries in `results/parity/<id>.json`. Every input score agrees with upstream's within the 1e-3 tolerance.

| Model           | Inputs                        | Max difference | Upstream side                                                       |
| --------------- | ----------------------------- | -------------: | ------------------------------------------------------------------- |
| AASIST          | 521 windows of 200 Urdu clips |         2.6e-6 | upstream `Model` and `.pth`, built as `main.py` does (torch 2.8)    |
| AASIST-L        | 521 windows of 200 Urdu clips |         3.3e-6 | same                                                                |
| DF Arena 500M   | 521 windows of 200 Urdu clips |              0 | the model card's `trust_remote_code` pipeline (torch 2.8)           |
| DF Arena 1B     | 521 windows of 200 Urdu clips |              0 | same                                                                |
| Xception        | 199 Mendeley face crops       |         2.1e-6 | upstream detector, release `.pth`, its test transform (torch 1.13) |
| EfficientNet-B4 | 199 Mendeley face crops       |         3.5e-6 | same                                                                |
| UCF             | 199 Mendeley face crops       |         4.9e-6 | same                                                                |
| F3Net           | 199 Mendeley face crops       |         3.4e-6 | same                                                                |
| SPSL            | 199 Mendeley face crops       |         2.0e-6 | same                                                                |

GenD's parity isn't checked yet: its native-size crops differ in shape, which `parity.py export` doesn't handle. DeepfakeBench's reference shares the adapter's import stand-ins, so parity there covers the weights, an old torch's numerics and the transform, not upstream's own package imports; the strict load covers the architecture.

## Adding a model

See [adding_a_model.md](adding_a_model.md): config, pinned weights, adapter, smoke test, then parity and score direction before any result counts.
