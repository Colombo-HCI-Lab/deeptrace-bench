# Models

Availability checked against the live repos on 2026-10-03. The source of truth is `configs/models/<id>.yaml`; `uv run scripts/setup_models.py --list` prints the current table.

| Model           | Modality    | Status   | Wave | Weights                      | Trained on (released weights) | Licence         |
| --------------- | ----------- | -------- | ---- | ---------------------------- | ----------------------------- | --------------- |
| Xception        | video       | ready    | 1    | DeepfakeBench release v1.0.1 | FF++ c23                      | CC BY-NC 4.0    |
| EfficientNet-B4 | video       | ready    | 1    | DeepfakeBench release v1.0.1 | FF++ c23                      | CC BY-NC 4.0    |
| SBI             | video       | ready    | 1    | Google Drive                 | FF++ real, self-blended       | research only   |
| GenD            | video       | ready    | 1    | Hugging Face                 | FF++                          | MIT             |
| AASIST          | audio       | ready    | 1    | in the upstream repo         | ASVspoof 2019 LA              | MIT             |
| AASIST-L        | audio       | ready    | 1    | in the upstream repo         | ASVspoof 2019 LA              | MIT             |
| XLS-R + AASIST  | audio       | workable | 1    | Google Drive + Hugging Face  | ASVspoof 2019 LA              | MIT             |
| Effort          | video       | workable |      | Google Drive                 | FF++ c23                      | none published  |
| HAVIC           | audio-video | later    |      | Hugging Face                 | LRS2, then 70% of FakeAVCeleb | MIT             |
| LipForensics    | video       | later    |      | Google Drive                 | FF++ c23 (LRW pretraining)    | MIT             |
| AltFreezing     | video       | blocked  |      | RecDrive / Baidu only        | FF++                          | MIT             |
| XLSR-Mamba      | audio       | dropped  |      | Hugging Face                 | ASVspoof 2019 LA              | MIT             |
| AASIST3         | audio       | dropped  |      | Hugging Face                 | ASVspoof 2019 + 5, MLAAD      | CC BY-NC-ND 4.0 |

Each modality pairs classic detectors with a foundation-model one (GenD for video, XLS-R + AASIST for audio). If both kinds fail on the same attribute, the cause is likely in the data rather than one model.

## One environment, ported adapters

Everything runs in this repo's single environment (Python 3.12, torch 2.8, transformers 4.56). Upstream repos pin much older stacks (DeepfakeBench: Python 3.7 and torch 1.12; XLS-R + AASIST: torch 1.8 and fairseq) that don't run on H100s, so we don't use them for results. Instead:

1. `setup_models.py` clones the upstream repo at its pinned commit into `third_party/` and archives it to `DTB_ROOT/upstream/`. Code from Effort (no licence), SBI (research only) and DeepfakeBench (CC BY-NC) is never copied into this repo; MIT code (AASIST, GenD, HAVIC) may be.
2. The adapter (`src/deeptrace_bench/models/<module>.py`) loads the one model file it needs from the checkout with `importlib`. Small compatibility fixes go in `patches/<repo>/` (see `patches/README.md`).
3. `convert_checkpoint` turns the download into `model.safetensors` with normalised keys, once. Adapters then never unpickle Google Drive files.
4. **Parity check** before any result counts: run the upstream code in its original environment (an Apptainer image or a throwaway old-torch venv, on CPU or an A100) on about 200 items, run the adapter on the same items, and require scores to agree within about 1e-3. Record the comparison in the model's notes. This is what licenses every patch.
5. **Score direction**: on the reproduction evalset (FF++ test, ASVspoof 2019 LA eval), AUC must come out above 0.5. An inverted logit otherwise flips every result silently.

Expect these breakages on Python 3.12 and torch 2.x: `torch.load` now defaults to `weights_only=True`; `np.float` and `np.int` are gone; `distutils` and `imp` are gone.

## Per model

**Xception, EfficientNet-B4.** DeepfakeBench commit `f188b1c` and release v1.0.1 weights. There's no single-video inference script, so the adapter does its own frame handling. Load only the detector and backbone files, not the `detectors` package, which imports every detector. Known issues: published weights don't reproduce the README table for some users (#109, #159), and there's an EfficientNet-B4 architecture mismatch report (#79). Issue #151 asks whether every checkpoint is FF++ c23, unanswered.

**SBI.** `FFc23.tar` and `FFraw.tar` from Google Drive, about 135 MB each. A plain EfficientNet-B4 on RetinaFace crops at 380 px. Upstream's `src/inference/inference_video.py` scores any video (max over faces, mean over frames), which makes the parity check easy. Install the right `retinaface_pytorch` package for parity (#55, #29).

**GenD.** `yermandy/GenD_CLIP_L_14` on Hugging Face, pinned to revision `891ce01`. Already modern. Needs face crops from upstream `detector.py` (RetinaFace ONNX, alignment at scale 1.3). Trained on FF++ only: the paper's 14 benchmarks, FakeAVCeleb among them, are test sets.

**AASIST, AASIST-L.** Weights committed in the upstream repo. Drop the unused `torchcontrib` import. A user reports inverted outputs with the shipped weights (#17, 2026-08-30), so settle the score direction on ASVspoof 2019 LA eval first.

**XLS-R + AASIST.** Detector weights in the authors' Google Drive folder (`LA_model.pth`, `Best_LA_model_for_DF.pth`); the XLS-R 300M backbone from `facebook/wav2vec2-xls-r-300m`, pinned to revision `1a640f3`. The port drops fairseq: build the front end in transformers and remap the fairseq keys, checking each one (layer names differ). Load with `strict=False` and list missing keys (#1). About 1 to 2 days of work.

**Effort (second wave).** The FF++ checkpoint is on Google Drive. Hard-coded local CLIP path (#15), a reported reproduction gap (CDF-v2 AUC 0.885 against 0.956, #19), parameters reported missing from the FF++ checkpoint (#18).

**HAVIC (later).** The only open audio-visual detector. Fine-tuned on a random 70% of FakeAVCeleb with no published item list, so the guard refuses it on FakeAVCeleb. Useful once DeePhy, DF-Platter or InDeepFake arrive.

**LipForensics (later).** Needs a RetinaFace plus FAN landmark pipeline for mouth crops. Users can't reproduce its FF++ numbers (#9). Relevant for lip-sync fakes.

**AltFreezing (blocked).** Weights only on RecDrive and Baidu (password `altf`). Someone with a working session has to fetch them first.

**Dropped.** XLSR-Mamba needs fairseq plus compiled mamba-ssm kernels. AASIST3's released weights aren't the paper's, score 28 to 32% EER in independent tests, and were trained on MLAAD, one of our test sets.

**Candidate, not yet configured.** The AntiDeepfake family (NII Yamagishi Lab, e.g. `nii-yamagishilab/mms-300m-anti-deepfake`) is post-trained on 74k hours across many corpora, including MLAAD and FLEURS. Strong and multilingual, but fairseq-bound and unusable on MLAAD.

## Adding a model

1. Add `configs/models/<id>.yaml`: pinned upstream commit, every weight with its source, and every dataset the released weights saw in `training_data` (be thorough; the guard is only as good as this list).
2. `uv run scripts/setup_models.py <id> --record` to fetch and pin hashes.
3. Write the adapter (subclass `models.base.Detector`), run the parity check and the score-direction check, then set `adapter:` and a `wave:`.
4. `uv run pytest` (the registry test checks the config cross-references).
