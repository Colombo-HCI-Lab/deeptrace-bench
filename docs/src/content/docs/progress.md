---
title: "Progress"
description: "Where the benchmark stands, what each round of work added, and what comes next"
---

## Where things stand

As of 2026-10-30:

- **Models.** 30 detectors have adapters and weights (FTCN, GenConViT, RealForensics, the DFDC winner and AVH-Align joined on 2026-10-28, and Effort's weights arrived). 22 pass parity against upstream code ([models.md](models.md#parity)). Seven more video and audio-visual models are in the registry without adapters ([Models](models.md#status)).
- **Data.** Builders cover every open South Asian set, the reproduction sets and HiDF. Request-only sets (FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, FF++, Celeb-DF v2 and others) wait on requests ([datasets.md](datasets.md#requests-to-send)).
- **Runs.** The first full benchmark round (2026-10-27 to 29) scored all 18 video models with weights on all of HiDF and MAVOS-DD Hindi and both audio-visual models on MAVOS-DD, and 40 same-corpus and 20 cross-corpus audio pairs before it was paused on 2026-10-29. SPARK scored 81 more on 2026-10-30 and 31, so every audio model now has its reproductions, all of BD-GRF6 and SpeechFake and all but IndicSynth Hindi of the South Asian sets; the last three pairs (IndicSynth Hindi for the models without parity) are scoring there ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)). Every audio model with a parity check has a settled score direction; no video model does yet (that needs FF++ or Celeb-DF v2).
- **Compute.** The DESI lab's SPARK cluster runs the harness and scores any pair or a whole wave of them, ordered with one command; the same wave stage runs on any other GPU too ([Where to run](spark.md)). The two 24 GB GPUs outside SPARK that ran most of the round are switched off; SPARK scores the rest of it.
- **Rules.** The evaluation setup is still `proposed` (decision 5, below), so every number can still move.

## Timeline

### 2026-10-31: SpeechFake done

Commits after `ea65acb` here, and `52303c4` to `f39e6c8` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- `s3_speechfake` scored 49 of its 50 pairs in 20 hours, about 1.1 million clips and 4.6 kWh; the split-zip fetch held and deleted its parts. The post-trained models score 0.978 or more on all five languages and the other six run from 0.148 to 0.986 ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)). It is cross-corpus, so none of this counts until the shortcut probe.
- AASIST on Bengali lost the GPU at load in `s3_speechfake` and again in a wave of its own, both times as the first process of its job (`CUDA driver initialization failed`), and passed on the next try. The wave stage now retries once a score that dies within a minute.
- `s4_indicsynth_hi`, the last three pairs, started the same day with SpeechFake's data cleared off the disk.

### 2026-10-30: BD-GRF6 done, SpeechFake unblocked

Commits after `e6c05c0` here, and `db4650c` to `a97f3e6` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- `s2_cross_rest` scored the last five BD-GRF6 pairs, so all ten audio models have it ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)). AASIST3 reaches 0.952; every other model stays at 0.71 or below. It is cross-corpus, so it waits on the shortcut probe.
- SpeechFake's 50 pairs didn't run. Its 116 GB of multilingual fakes downloaded in 10 hours, but they are one zip split in three parts, which Python's `zipfile` can't open, and the fetcher had only ever read them by range requests for samples. A full fetch now opens the parts with the same split-zip reader, extracts only what the builder labels (the 8 GB of South Asian clips, 93,741 of them) and deletes the parts. The parts' checksums are pinned in the dataset config.
- `s3_speechfake` (the 50 pairs) started the same evening from the parts already on the worker. IndicSynth Hindi's three pairs follow in a wave of their own, since its 115 GB can't share the disk with SpeechFake.

### 2026-10-30: the first SPARK wave lands

Commits after `2282567` here, and `acbc9f9` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- `s1_audio_same_rest` scored 26 of its 29 pairs in 21 hours on SPARK's RTX 3090 (877K clips, 4.6 kWh). AASIST3, XLS-R + SLS and XLSR-Mamba now have their reproductions and five more South Asian sets each ([Results](results.md#audio-south-asian-speech)), and the 11 retries filled the gaps from earlier failures.
- XLS-R + SLS and XLSR-Mamba reach 6.7% and 6.3% EER on In-the-Wild, a little better than their papers, and AASIST3's 30.8% matches what the Speech DF Arena measured for its released weights ([Reproductions](results.md#reproductions)).
- The seven first-round models scored on IndicTTS again matched their 2026-10-12 runs within 0.001, a second cross-machine reproduction after Urdu CSALT.
- IndicSynth Hindi didn't fit: the worker's shared disk filled at 73 of its 115 GB, so its three pairs wait until `s2_cross_rest` frees the space. s2 (BD-GRF6 and SpeechFake, 55 pairs) started straight after.

### 2026-10-29: audio round paused, then moved to SPARK

Commits after `622bff9` here, and `c64c9f4` and `fb2e779` in [deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments).

- IndicSynth Bengali and Urdu are scored for all ten audio models; the three models without parity split them, XLS-R + SLS best ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)).
- The first cross-corpus wave scored MLAAD Bengali, Hindi and Sinhala and most of BD-GRF6. The guard refused 15 MLAAD pairs for the five models trained on it. These numbers wait on the shortcut probe.
- The round was paused with 84 audio pairs left: the three models without parity on the six reproduction sets, 11 retries of earlier failures, the rest of BD-GRF6 and all 50 SpeechFake pairs. A rerun of each wave skips what's done and resumes the interrupted pair from its saved parts. The two GPUs outside SPARK were switched off.
- The 84 pairs went to SPARK the same afternoon as two waves: `s1_audio_same_rest` (29 pairs: AASIST3, XLS-R + SLS and XLSR-Mamba on the six reproduction and South Asian sets, and the 11 retries) and `s2_cross_rest` (55: the rest of BD-GRF6, then all of SpeechFake). s1 started at once; s2 follows it.
- [Coverage](https://colombo-hci-lab.github.io/deeptrace-bench/coverage/) is now one grid: a column per model, a row per evalset, AUC in bands of 0.1. The site had been dropping 28 audio runs scored before the evalset config gained its `subset` field on 2026-10-27; unset, the field changes nothing a run measured, so they count again. Ten audio runs from 2026-10-12 (seven on IndicTTS, three reproductions) had dropped out the same way after a notes-only edit to their models' configs on 2026-10-19; a run now also counts while its model config differs only in descriptive fields, so they are back too, and every one of the 11 retries has a counted run.
- Every page of this site has one address. Results, the Timeline and How it works are now the only pages of their docs, and a section without a layout of its own shows there as prose, which put the 2026-10-28 round's write-up on [Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi). Every other doc moved from `/docs/<slug>/` to `/<slug>/`, and old addresses forward. The build now fails on a link to a missing page, to a forwarding address or to a missing anchor.

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
2. **The rest of the round on SPARK**: `s4_indicsynth_hi`, the three IndicSynth Hindi pairs, running since 2026-10-31 ([deeptrace-experiments](https://github.com/UNIL-DESI/deeptrace-experiments)). Import it when it lands.
3. **FF++ or Celeb-DF v2**, to settle every video model's score direction.
4. **A generator-level contamination check** of DF Arena and AntiDeepfake, then generators they haven't seen. The guard compares datasets, so a model trained on an evalset's TTS systems still counts as clean, and their near-ceiling scores on the open South Asian sets (0.978 and up for MMS-300M and both DF Arena models) may be familiarity rather than transfer.
5. **A language control for the video drop**: the same video models on a non-South-Asian set made with MAVOS-DD's generators (its other-language subsets, if they share Hindi's), to tell "Hindi" apart from "talking head" ([Results](results.md#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)).
6. **The shortcut probe**, then the cross-corpus sets that wait on it (E5: MLAAD, SpeechFake, BD-GRF6, `mlaad_si`, the only Sinhala fakes).
7. **Parity still owed:** GenD (the letterbox crop added for the DFDC winner may unblock its export), XLS-R + SLS (needs the fairseq zip its checkout ships), XLSR-Mamba, AASIST3, GenD PE-L, HAVIC, LipForensics, Effort.
8. **Fine-tuned baselines** (E6) on [identity-safe splits](evaluation.md#fine-tuning-splits).
9. **deeptrace's own data** (E7), once collection starts.

## Open decisions that shape the results

- **Decision 4**, the rule for keeping an attribute. As proposed it uses AUC only. On HiDF one group's AUC barely moves while its false-positive rate doubles ([Results](results.md#video-hidf)), so a false-positive-rate rule at a fixed threshold may be needed too ([the rule](evaluation.md#metrics)).
- **Decision 5**, the evaluation setup in [`configs/eval/default.yaml`](/configs/eval/default.yaml): due 2026-10-09, still open. The settings in question are in [Evaluation](evaluation.md#settings-still-open).
- **`group_gap`**, the headline fairness number, is still undefined ([evaluation.md](evaluation.md#metrics)).
- **Cut candidates** (proposed 2026-10-26): AASIST-L, AASIST3, DF Arena 500M, EfficientNet-B4 and Deepfake-Eval-2024 add little the others don't show for South Asia. Each is marked in the [Model catalog](catalog.md#models) and [Dataset catalog](catalog.md#datasets) with its reason; the team decides whether to drop them.

## Adding an entry

After each round or milestone: a dated entry at the top of the timeline (3 to 5 bullets, commit range, no numbers that belong in Results), then update "Where things stand" and "Next", and [Results](results.md) if numbers or experiments changed. Once a round's waves are imported, refresh the [archive](store_layout.md#archive) of per-item results.
