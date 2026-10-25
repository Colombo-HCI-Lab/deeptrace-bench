---
title: "Progress"
description: "Where the benchmark stands, what each round of work added, and what comes next"
---

## Where things stand

As of 2026-10-19:

- **Models.** 25 detectors have adapters; 24 have weights (Effort's wait on Google Drive's download quota; AltFreezing's are unreachable). 17 pass parity against upstream code ([models.md](models.md#parity)).
- **Data.** Builders cover every open South Asian set, the reproduction sets and HiDF. Request-only sets (FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, FF++, Celeb-DF v2 and others) wait on requests ([datasets.md](datasets.md#requests-to-send)).
- **Runs.** Ten models have full runs, 48 model and evalset pairs in all ([Results](results.md)). Every audio model with a parity check has a settled score direction; no video model does yet (that needs FF++ or Celeb-DF v2).
- **Compute.** The DESI lab's SPARK cluster runs the harness and scores any model and evalset pair, ordered with one command ([Where to run](spark.md)).
- **Rules.** The evaluation setup is still `proposed` (decision 5, below), so every number can still move.

## Timeline

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
2. **MAVOS-DD Hindi** (E4), the only open South Asian video: face swaps, audio-driven talking heads and voice conversion, so it tests v1's focus, lip-sync with cloned voices. Needs its Hugging Face terms accepted and a token.
3. **FF++ or Celeb-DF v2**, to settle every video model's score direction.
4. **A generator-level contamination check** of DF Arena and AntiDeepfake. The guard compares datasets, so a model trained on an evalset's TTS systems still counts as clean.
5. **The shortcut probe**, then the cross-corpus sets that wait on it (E5: MLAAD, SpeechFake, BD-GRF6, `mlaad_si`, the only Sinhala fakes).
6. **Parity still owed:** GenD (the export can't handle native-size crops yet), XLS-R + SLS (needs the fairseq zip its checkout ships), XLSR-Mamba, AASIST3, GenD PE-L, HAVIC, LipForensics.
7. **IndicSynth Bengali and Urdu** (72 GB); their evalsets have real clips but no fakes until then.
8. **Fine-tuned baselines** (E6) on [identity-safe splits](evaluation.md#fine-tuning-splits).
9. **deeptrace's own data** (E7), once collection starts.

## Open decisions that shape the results

- **Decision 4**, the rule for keeping an attribute. As proposed it uses AUC only. On HiDF one group's AUC barely moves while its false-positive rate doubles ([Results](results.md#video-hidf)), so a false-positive-rate rule at a fixed threshold may be needed too ([the rule](evaluation.md#metrics)).
- **Decision 5**, the evaluation setup in [`configs/eval/default.yaml`](/configs/eval/default.yaml): due 2026-10-09, still open. The settings in question are in [Evaluation](evaluation.md#settings-still-open).
- **`group_gap`**, the headline fairness number, is still undefined ([evaluation.md](evaluation.md#metrics)).

## Adding an entry

After each round or milestone: a dated entry at the top of the timeline (3 to 5 bullets, commit range, no numbers that belong in Results), then update "Where things stand" and "Next", and [Results](results.md) if numbers or experiments changed.
