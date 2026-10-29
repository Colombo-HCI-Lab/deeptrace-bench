---
title: "Progress"
description: "Where the benchmark stands, what each round of work added, and what comes next"
---

## Where things stand

As of 2026-10-29:

- **Models.** 30 detectors have adapters and weights (FTCN, GenConViT, RealForensics, the DFDC winner and AVH-Align joined on 2026-10-28, and Effort's weights arrived). 22 pass parity against upstream code ([models.md](models.md#parity)). Seven more video and audio-visual models are in the registry without adapters ([Models](models.md#status)).
- **Data.** Builders cover every open South Asian set, the reproduction sets and HiDF. Request-only sets (FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, FF++, Celeb-DF v2 and others) wait on requests ([datasets.md](datasets.md#requests-to-send)).
- **Runs.** The first full benchmark round (2026-10-27 to 29) scored all 18 video models with weights on all of HiDF and MAVOS-DD Hindi and both audio-visual models on MAVOS-DD, and 40 same-corpus and 20 cross-corpus audio pairs before it was paused on 2026-10-29, with 84 audio pairs left ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)). Every audio model with a parity check has a settled score direction; no video model does yet (that needs FF++ or Celeb-DF v2).
- **Compute.** The DESI lab's SPARK cluster runs the harness and scores any pair or a whole wave of them, ordered with one command; the same wave stage runs on any other GPU too ([Where to run](spark.md)). The two 24 GB GPUs outside SPARK that ran most of the round are switched off while it is paused.
- **Rules.** The evaluation setup is still `proposed` (decision 5, below), so every number can still move.

## Timeline

### 2026-10-29: audio round paused

Commits after `622bff9`.

- IndicSynth Bengali and Urdu are scored for all ten audio models; the three models without parity split them, XLS-R + SLS best ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)).
- The first cross-corpus wave scored MLAAD Bengali, Hindi and Sinhala and most of BD-GRF6. The guard refused 15 MLAAD pairs for the five models trained on it. These numbers wait on the shortcut probe.
- The round was paused with 84 audio pairs left: the three models without parity on the six reproduction sets, 11 retries of earlier failures, the rest of BD-GRF6 and all 50 SpeechFake pairs. A rerun of each wave skips what's done and resumes the interrupted pair from its saved parts. The two GPUs outside SPARK were switched off until it resumes.
- [Coverage](https://colombo-hci-lab.github.io/deeptrace-bench/coverage/) is laid out like the round's dashboard: a column per model, a row per evalset, AUC in bands of 0.1. The site had been dropping 28 audio runs scored before the evalset config gained its `subset` field on 2026-10-27; unset, the field changes nothing a run measured, so they count again.

### 2026-10-29: five more video adapters scored

Commits `cefce3a` to `18a08d5`.

- FTCN, GenConViT (its ED branch), RealForensics, the DFDC winner's seven-model ensemble and AVH-Align have adapters, each matching upstream in upstream's own environment within 1e-5. AVH-Align runs on an AV-HuBERT rebuilt without fairseq.
- All five were scored on full HiDF and MAVOS-DD Hindi (AVH-Align on MAVOS-DD's audio and video). The temporal models hold up best on MAVOS-DD; the DFDC winner falls below chance there ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)).
- A power cut on 2026-10-28 stopped every machine mid-wave; runs resumed from their saved parts, so two RealForensics runs record only the resumed session's compute.
- `evaluate.py` now writes each result through a temporary file and a rename: over the workers' SMB mount, a shorter file copied over a longer one had kept the old last byte.

### 2026-10-28: the first full benchmark round

The waves and their ledgers are in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (`waves/`); the runs are in `results/` and `results/ledger.csv` here.

- A `matrix` stage scores a whole wave of pairs, measuring each one (time per phase and per item, GPU memory and utilisation, energy), and runs on SPARK or any other GPU. Summaries now carry accuracy, balanced accuracy, average precision and TPR at 1% and 5% FPR.
- All 15 video models with weights, Effort among them, were scored on full HiDF and full MAVOS-DD Hindi. MAVOS-DD is far harder than HiDF for every one of them.
- Re-runs on other GPUs matched the 2026-10-12 round (Urdu CSALT audio, HiDF video), and Effort agreed with itself across two GPUs within 1e-4.
- FTCN, GenConViT and RealForensics passed parity and are queued on both video sets. The rest of the audio grid follows: IndicSynth Bengali and Urdu, MAVOS-DD Hindi audio, and the cross-corpus sets.

### 2026-10-28: a wider video roster

Config and docs only; no new runs.

- Eleven video and audio-visual models join the registry with weights confirmed reachable, none adapted yet ([Models](models.md#added-2026-10-28-not-adapted-yet)): FTCN, RealForensics, AVAD, AVH-Align, GenConViT, the DFDC winner, LipFD, LAA-Net, FakeSTormer, UnivFD, and TALL (blocked, no weights).
- Ten datasets join it ([Datasets](datasets.md#at-a-glance)). Four are South Asian: HAV-DF (Hindi), DeepFakeX (Urdu), ILLUSION (Indic languages) and INDIFACE, all on request. Counterfactual-Deepfake-Bench is open and has an explicit Indian cell. TalkingHeadBench, KoDF, Celeb-DF++, AV-Deepfake1M and DFDC are anchors and contrasts.
- Video comes before audio in the work queue from here on.

### 2026-10-19: any pair on SPARK, one command to order it

The changes are in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments) (`abeac55` to `32d361c`); here, only [Where to run](spark.md) changed.

- SPARK's `score` stage now takes any pair. It reads an evalset's datasets from this repo's registry, and it still refuses to score before a download is proved complete. A reference run is optional.
- A helper, `scripts/exp.py`, places an order (it looks up the published reference itself), starts the run, waits for it and prints the report. It also lists every past run and repairs the reports a failed run leaves behind.
- Two test runs passed. AASIST-L on Urdu CSALT matched the 2026-10-12 round within 1e-3. XLS-R + SLS on Urdu CSALT ran with no reference; its number waits on parity.

### 2026-10-16: running on SPARK

Commits `6ae9dd1` to `5327e2f` here; the runs themselves are in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- Runs on the DESI lab's SPARK cluster go through `UNIL-DESI/deeptrace-experiments`, which pins a commit of this repo ([Where to run](spark.md)).
- On its x86 worker (an RTX 3090), the full test suite and the GenD and AASIST smoke tests passed in this repo's locked environment.
- The first scoring run there repeated AASIST on Urdu CSALT and matched the 2026-10-12 round: same manifest, same 6,794 items and 19 failures, AUC 0.8632 (+1.2e-5) and EER 0.2224 (+2.0e-4). The small EER shift fits the eval config's move from TF32 to fp32 on CUDA since then.
- Nine setup failures came before that, all fixed on our side; deeptrace-experiments' `CLAUDE.md` lists them.

### 2026-10-12: first zero-shot round

Commits `7928960` to `4ab78ad`, then the site (`429fb2d`, `7a79c22`).

- Four more adapters pass parity, making 17: XLS-R + AASIST (against upstream's torch 1.8.1 and vendored fairseq), both AntiDeepfake models (fairseq 0.12.2) and SBI.
- New reproductions of XLS-R + AASIST and both AntiDeepfake models, and the first zero-shot round: seven audio models on the five open South Asian sets, three video models on HiDF ([Results](results.md)).
- IndicSynth's evalsets switched to a seeded sample of 5,000 fakes per generator.
- The status page flags runs whose model lacks a parity or score-direction check. Docs and status page became one Astro site, published from `main` by GitHub Actions.

### 2026-10-05: every model in the kickoff Doc

Commits `bf2ae26` to `0424f48`, then the first status page (`d17c65c` to `4d77c1d`).

- 25 adapters: SBI, LipForensics, HAVIC, GenD's PE-L backbone, the XLS-R family (run without fairseq or mamba-ssm), AntiDeepfake, AASIST3 and four more DeepfakeBench detectors.
- New builders: IndicSynth with IndicSUPERB, the IndicTTS challenge set, BD-GRF6, Bangla voices, HiDF, SpeechFake.
- A pipeline check runs every working adapter on two real and two fake items of each open evalset: 195 pairs pass.
- RECCE, SRM, CORE and FFD pass parity.

### 2026-10-05: ten adapters and the first full runs

Commits `446714f` to `e524148`, then `d949b74` to `af30749`.

- AASIST, AASIST-L, DF Arena 500M and 1B, and five DeepfakeBench detectors, all passing parity.
- Full reproductions of AASIST and AASIST-L on ASVspoof 2019 LA eval and both DF Arena models on In-the-Wild ([Results](results.md#reproductions)).
- cuDNN's default TF32 convolutions broke GPU parity, so CUDA runs use full fp32 (`compute.tf32`).
- The first South Asian runs: AASIST and DF Arena on Urdu CSALT and BanglaFake.

### 2026-10-04: the pipeline runs end to end

Commits `1d0c0ab` to `30587a5`.

- The smoke namespace, dataset sampling, the shared SCRFD face pipeline, the GenD adapter and the smoke test ([Getting started](getting_started.md)).

### 2026-10-03: the scaffold

Commits `15a2e2c` and `d3a6d03`.

- Configs for 13 models and 16 datasets, checked live that day; setup scripts, the contamination guard, metrics and resumable scoring; a pre-commit hook that keeps data out of git.

## Next

In order:

1. **Settle the evaluation setup** (decision 5). The audio question now has a concrete case: average 4-second windows or score each utterance whole, which may explain MMS-300M's In-the-Wild gap ([Models](models.md#per-model)).
2. **The rest of the round**, paused on 2026-10-29 with its two GPUs outside SPARK switched off: 84 audio pairs (the reproduction sets for AASIST3, XLS-R + SLS and XLSR-Mamba, 11 retries, the rest of BD-GRF6 and SpeechFake).
3. **FF++ or Celeb-DF v2**, to settle every video model's score direction.
4. **A generator-level contamination check** of DF Arena and AntiDeepfake. The guard compares datasets, so a model trained on an evalset's TTS systems still counts as clean.
5. **The shortcut probe**, then the cross-corpus sets that wait on it (E5: MLAAD, SpeechFake, BD-GRF6, `mlaad_si`, the only Sinhala fakes).
6. **Parity still owed:** GenD (the letterbox crop added for the DFDC winner may unblock its export), XLS-R + SLS (needs the fairseq zip its checkout ships), XLSR-Mamba, AASIST3, GenD PE-L, HAVIC, LipForensics, Effort.
7. **Fine-tuned baselines** (E6) on [identity-safe splits](evaluation.md#fine-tuning-splits).
8. **deeptrace's own data** (E7), once collection starts.

## Open decisions that shape the results

- **Decision 4**, the rule for keeping an attribute. As proposed it uses AUC only. On HiDF one group's AUC barely moves while its false-positive rate doubles ([Results](results.md#video-hidf)), so a false-positive-rate rule at a fixed threshold may be needed too ([the rule](evaluation.md#metrics)).
- **Decision 5**, the evaluation setup in [`configs/eval/default.yaml`](/configs/eval/default.yaml): due 2026-10-09, still open. The settings in question are in [Evaluation](evaluation.md#settings-still-open).
- **`group_gap`**, the headline fairness number, is still undefined ([evaluation.md](evaluation.md#metrics)).
- **Cut candidates** (proposed 2026-10-26): AASIST-L, AASIST3, DF Arena 500M, EfficientNet-B4 and Deepfake-Eval-2024 add little the others don't show for South Asia. Each is marked in the [Model catalog](catalog.md#models) and [Dataset catalog](catalog.md#datasets) with its reason; the team decides whether to drop them.

## Adding an entry

After each round or milestone: a dated entry at the top of the timeline (3 to 5 bullets, commit range, no numbers that belong in Results), then update "Where things stand" and "Next", and [Results](results.md) if numbers or experiments changed. Once a round's waves are imported, refresh the [archive](store_layout.md#archive) of per-item results.
