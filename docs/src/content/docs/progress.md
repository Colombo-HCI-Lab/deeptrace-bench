---
title: "Progress"
description: "Where the benchmark stands, what each round of work added, and what comes next"
group: start
order: 2
---

The newest entry is at the top of the [timeline](#timeline). For what the numbers say, see [Results](results.md); for what was run and why, [Experiments](experiments.md).

## Where things stand

As of 2026-10-12:

- **Models.** 25 detectors have adapters, and 24 have their weights (Effort's wait on Google Drive's download quota; AltFreezing's weights can't be reached at all). 17 pass their parity check against upstream code ([models.md](models.md#parity)).
- **Data.** Builders cover every open South Asian set, the reproduction sets and HiDF. The request-only sets (FakeAVCeleb, InDeepFake, DeePhy, DF-Platter, FF++, Celeb-DF v2 and others) wait on their requests ([datasets.md](datasets.md#requests-to-send)).
- **Runs.** Ten models have full runs, 48 model and evalset pairs in all, shown in the [all-runs table](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs). Every audio model with a parity check has a settled score direction. No video model does yet, since that needs FF++ or Celeb-DF v2.
- **Rules.** The evaluation setup in [`configs/eval/default.yaml`](/configs/eval/default.yaml) is still `proposed`: decision 5 was due 2026-10-09 and isn't settled. Every number so far can still move.

## Timeline

### 2026-10-12: first zero-shot round

Commits `7928960` to `4ab78ad`, then the site (`429fb2d`, `7a79c22`).

- Four more adapters pass parity: XLS-R + AASIST (against upstream's own torch 1.8.1 and vendored fairseq), both AntiDeepfake models (fairseq 0.12.2) and SBI. That makes 17.
- New reproductions: XLS-R + AASIST at 0.30% EER on ASVspoof 2019 LA eval and 10.8% on In-the-Wild, AntiDeepfake w2v2 small at 4.23% on In-the-Wild (its card says 4.24%), and MMS-300M at 4.29% (its card says 2.90%).
- Zero-shot audio: seven models on Urdu CSALT, BanglaFake, Bangla voices, the IndicTTS challenge set (16 languages, Nepali among them) and IndicSynth Hindi. Zero-shot video: Xception, SBI and GenD on HiDF. See [Results](results.md).
- IndicSynth's evalsets now score a seeded sample of 5,000 fakes per generator. All of them would outnumber the real clips about 40 to 1.
- The status page flags runs whose model lacks a parity check or a score-direction check. The docs and the status page became one Astro site, published from `main` by a GitHub Actions workflow.

### 2026-10-05: every model in the kickoff Doc

Commits `bf2ae26` to `0424f48`, then the first status page (`d17c65c` to `4d77c1d`).

- 25 adapters: SBI, LipForensics, HAVIC, GenD's PE-L backbone, the XLS-R family (run without fairseq or mamba-ssm), AntiDeepfake, AASIST3 and four more DeepfakeBench detectors.
- New builders: IndicSynth with IndicSUPERB, the IndicTTS challenge set, BD-GRF6, Bangla voices, HiDF, SpeechFake.
- A pipeline check runs every working adapter on two real and two fake items of every open evalset: 195 pairs pass.
- RECCE, SRM, CORE and FFD pass parity.

### 2026-10-05: ten adapters and the first full runs

Commits `446714f` to `e524148`, then `d949b74` to `af30749`.

- AASIST, AASIST-L, DF Arena 500M and 1B, and five DeepfakeBench detectors, all passing parity.
- Full reproductions: AASIST 1.58% and AASIST-L 1.49% EER on ASVspoof 2019 LA eval; DF Arena 1B 0.67% and 500M 1.51% on In-the-Wild.
- cuDNN's default TF32 convolutions broke parity on GPU, so CUDA runs now use full fp32 (`compute.tf32`).
- The first South Asian runs: AASIST and DF Arena on Urdu CSALT and BanglaFake.

### 2026-10-04: the pipeline runs end to end

Commits `1d0c0ab` to `30587a5`.

- The smoke namespace, dataset sampling, the shared SCRFD face pipeline, the GenD adapter and `scripts/smoke.py`, which runs the whole pipeline on a few items and writes a readable report ([smoke_test.md](smoke_test.md)).

### 2026-10-03: the scaffold

Commits `15a2e2c` and `d3a6d03`.

- Configs for 13 models and 16 datasets, checked live that day; setup scripts, the contamination guard, metrics and resumable scoring; a pre-commit hook that keeps data out of git.

## Next

In order:

1. **Settle the evaluation setup** (decision 5). The audio protocol question has a concrete case now: whether to average 4-second windows or score each utterance whole, which may explain MMS-300M's In-the-Wild gap.
2. **MAVOS-DD Hindi**, the only open South Asian video, with the audio-driven talking heads v1 is about. It needs its Hugging Face terms accepted and a token.
3. **FF++ or Celeb-DF v2**, to settle the score direction of every video model.
4. **A generator-level contamination check** of DF Arena and AntiDeepfake. The guard compares datasets, so a model trained on the same TTS systems as an evalset still counts as clean.
5. **The shortcut probe**, then the cross-corpus sets that wait on it (MLAAD, SpeechFake, BD-GRF6, `mlaad_si`, the only Sinhala fakes).
6. **Parity still owed:** GenD (the export can't handle native-size crops yet), XLS-R + SLS (it needs the fairseq zip its checkout ships), XLSR-Mamba, AASIST3, GenD PE-L, HAVIC, LipForensics.
7. **IndicSynth Bengali and Urdu** (72 GB); their evalsets have real clips but no fakes until then.

## Open decisions that shape the results

- **Decision 4**, the rule for keeping an attribute. As proposed it looks at AUC only. HiDF shows a group whose AUC barely moves while its false-positive rate doubles ([Results](results.md#video-hidf)), so a rule on the false-positive rate at a fixed threshold may be needed too.
- **Decision 5**, the evaluation setup, above.
- **`group_gap`**, the headline fairness number, is still undefined ([evaluation.md](evaluation.md#metrics)).

## Adding an entry

After each round of runs or a milestone, add a dated entry at the top of the timeline: what was added or run (3 to 5 bullets) and its commit range. Update "Where things stand" and "Next", and refresh [Results](results.md) if new numbers came in.
