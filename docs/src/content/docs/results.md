---
title: "Results"
description: "What the runs so far show, with numbers and caveats"
group: findings
order: 2
---

## Read this first

- **A snapshot as of 2026-10-12.** The [all-runs table](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs) holds the latest numbers for every run. What was run and why: [Experiments](experiments.md).
- **Zero-shot.** No model has seen these evalsets or been fine-tuned.
- **Provisional rules.** The evaluation setup is still `proposed` (decision 5) and nothing has been published with `evaluate.py --publish`. Numbers can move.
- **The guard compares datasets, not generators.** `clean` means no known dataset overlap. A model may still have trained on the TTS systems that made an evalset's fakes.
- **Video is provisional.** No video model has a settled score direction, and GenD has no parity check.

## Reproductions

| Model                   | Evalset               | EER (95% interval)    | AUC    | Published               |
| ----------------------- | --------------------- | --------------------- | -----: | ----------------------- |
| AASIST                  | ASVspoof 2019 LA eval | 1.58% (1.41 to 1.75)  | 0.999  | 0.83% (paper)           |
| AASIST-L                | ASVspoof 2019 LA eval | 1.49% (1.32 to 1.66)  | 0.999  | 0.99% (paper)           |
| XLS-R + AASIST          | ASVspoof 2019 LA eval | 0.30% (0.26 to 0.34)  | 0.9999 | trained on its train split |
| XLS-R + AASIST          | In-the-Wild           | 10.8% (10.4 to 11.2)  | 0.961  | about 10.5% (later papers) |
| DF Arena 500M           | In-the-Wild           | 1.51% (1.35 to 1.66)  | 0.999  | 1.76% (model card)      |
| DF Arena 1B             | In-the-Wild           | 0.67% (0.58 to 0.77)  | 0.999  | 0.91% (model card)      |
| AntiDeepfake w2v2 small | In-the-Wild           | 4.23% (3.97 to 4.47)  | 0.993  | 4.24% (model card)      |
| AntiDeepfake MMS-300M   | In-the-Wild           | 4.29% (4.04 to 4.54)  | 0.991  | 2.90% (model card)      |

Every score direction here is settled. The AASIST and DF Arena gaps are protocol differences: upstream scores each clip's first 4 seconds, or the whole clip; this setup averages every 4-second window and refuses clips under a second ([models.md](models.md#per-model)).

MMS-300M is 1.4 points worse than its card, while w2v2 small, scored the same way, matches its card. Parity passed window by window, so the gap is in the protocol or the card, not the adapter: a concrete case for decision 5.

On In-the-Wild (real public figures' speech) the ASVspoof-only models collapse: AASIST 44.0% EER, AASIST-L 44.9%.

## Audio: South Asian speech

AUC (EER). * marks runs from 2026-10-05 under the previous setup version, which differs only in a video setting. "so" marks a `source_overlap` verdict.

| Model                   | Urdu CSALT    | BanglaFake       | Bangla voices | IndicTTS        | IndicSynth Hindi |
| ----------------------- | ------------- | ---------------- | ------------- | --------------- | ---------------- |
| AASIST                  | 0.863 (22%)*  | 0.365 (60%)*     | 0.048 (87%)   | 0.403 (58%)     | 0.393 (59%)      |
| AASIST-L                | 0.861 (23%)*  | 0.379 (58%)*     | 0.055 (87%)   | 0.377 (59%)     | 0.390 (59%)      |
| XLS-R + AASIST          | 0.802 (28%)   | 0.467 (55%) so   | 0.389 (58%)   | 0.581 (46%)     | 0.828 (25%)      |
| AntiDeepfake w2v2 small | 0.969 (9.3%)  | 0.955 (11%)      | 1.000 (0.2%)  | 0.895 (17%) so  | 0.997 (2.5%)     |
| AntiDeepfake MMS-300M   | 0.998 (2.2%)  | 0.999 (1.1%) so  | 1.000 (0.5%)  | 0.997 (2.3%) so | 1.000 (0.1%)     |
| DF Arena 500M           | 0.981 (7.8%)* | 0.997 (2.2%)* so | 0.998 (2.1%)  | 0.978 (6.9%)    | 0.997 (2.6%)     |
| DF Arena 1B             | 0.995 (4.0%)* | 0.999 (0.7%)* so | 0.993 (4.3%)  | 0.988 (5.1%)    | 0.999 (1.7%)     |

### Three tiers

1. **Trained only on ASVspoof 2019 LA (AASIST, AASIST-L): they fail.** Below chance on every Bengali and Indic set. On Bangla voices, AUC 0.05: confidently backwards. Likely cause: they learned studio-clean audio as a cue for real, and these crowd-sourced real recordings are noisier than the TTS fakes. Urdu is the exception (0.86).
2. **The same training data with a multilingual front end (XLS-R + AASIST): better, but uneven.** In-the-Wild drops from 44% to 10.8% EER and Urdu (0.80) and IndicSynth Hindi (0.83) hold up, but the Bengali sets (0.39 and 0.47) and most IndicTTS languages (0.49 to 0.70) stay near or below chance.
3. **Post-trained on large multilingual collections of real and fake speech (AntiDeepfake, DF Arena): near the ceiling everywhere.** AUC 0.98 or more on every set, mostly above 0.99.

The open South Asian TTS sets separate classic from modern detectors but no longer rank the modern ones. Part of the ceiling may be familiarity: these models trained on multilingual TTS collections that likely share generators (VITS, XTTS) with these sets, which the guard can't see.

### Multilingual pretraining helps

The two AntiDeepfake models share post-training data and differ only in pretraining.

| Evalset    | w2v2 small (English pretraining) | MMS-300M (about 1,400 languages) |
| ---------- | -------------------------------: | -------------------------------: |
| IndicTTS   | 0.895                            | 0.997                            |
| BanglaFake | 0.955                            | 0.999                            |
| Urdu CSALT | 0.969                            | 0.998                            |

Multilingual pretraining cuts 1 minus AUC by 15 to 45 times on these sets.

### Language matters for the classic models only

AUC by language on the IndicTTS challenge set:

| Language   | AASIST | XLS-R + AASIST | AntiDeepfake w2v2 small | AntiDeepfake MMS-300M | DF Arena 1B |
| ---------- | -----: | -------------: | ----------------------: | --------------------: | ----------: |
| Assamese   | 0.293  | 0.581          | 0.922                   | 0.996                 | 0.993       |
| Bengali    | 0.234  | 0.596          | 0.954                   | 0.999                 | 0.997       |
| Bodo       | 0.433  | 0.572          | 0.956                   | 1.000                 | 0.990       |
| Dogri      | 0.371  | 0.520          | 0.983                   | 1.000                 | 0.999       |
| English    | 0.453  | 0.861          | 0.960                   | 1.000                 | 0.999       |
| Gujarati   | 0.390  | 0.493          | 0.927                   | 1.000                 | 1.000       |
| Hindi      | 0.456  | 0.617          | 0.993                   | 1.000                 | 1.000       |
| Kannada    | 0.301  | 0.598          | 0.966                   | 1.000                 | 0.993       |
| Malayalam  | 0.444  | 0.541          | 0.962                   | 1.000                 | 0.996       |
| Manipuri   | 0.523  | 0.543          | 0.950                   | 0.999                 | 0.989       |
| Marathi    | 0.315  | 0.606          | 0.963                   | 1.000                 | 0.990       |
| Nepali     | 0.442  | 0.641          | 0.926                   | 0.999                 | 0.991       |
| Odia       | 0.452  | 0.699          | 0.881                   | 1.000                 | 0.999       |
| Sanskrit   | 0.246  | 0.684          | 0.955                   | 0.999                 | 0.994       |
| Tamil      | 0.297  | 0.686          | 0.928                   | 0.999                 | 0.996       |
| Telugu     | 0.388  | 0.680          | 0.935                   | 1.000                 | 0.999       |

XLS-R + AASIST ranges from 0.49 (Gujarati) to 0.86 (English, its training language); AASIST from 0.23 to 0.52. MMS-300M stays at 0.996 or above in all 16. Under decision 4's proposed rule (an attribute matters if both models in a modality shift by 5 or more AUC points), language qualifies for the classic pair, not the multilingual one.

### Voice conversion is the classic models' blind spot

On IndicSynth Hindi, at the global (overall EER) threshold, XLS-R + AASIST misses 46% of FreeVC24 voice conversions and 3% of XTTS-v2 clips; AASIST misses 92% and 27%. The multilingual models miss at most 5% of either.

### Gender

Mostly flat. Exception: XLS-R + AASIST on Bangla voices, AUC 0.29 for women, 0.48 for men.

### Ranking isn't calibration

DF Arena 1B ranks Bangla voices at AUC 0.993 and Urdu CSALT at 0.995, yet its EER there is 4.3% and 4.0%, against 0.7% on In-the-Wild. A fixed threshold needs per-group error rates, not just AUC.

## Video: HiDF

| Model    | AUC (95% interval)    | EER   |
| -------- | --------------------- | ----: |
| GenD     | 0.952 (0.948 to 0.956) | 11.5% |
| SBI      | 0.877 (0.870 to 0.884) | 20.0% |
| Xception | 0.804 (0.794 to 0.812) | 27.2% |

All three trained on FF++, so HiDF's commercial face swaps are a true cross-dataset test. The newest model generalises best, as the literature reports.

By race: AUC, and the false-positive rate on real faces at each model's global threshold.

| Group  | Real / fake | GenD AUC | GenD FPR | SBI AUC | SBI FPR | Xception AUC | Xception FPR |
| ------ | ----------: | -------: | -------: | ------: | ------: | -----------: | -----------: |
| Asian  | 215 / 206   | 0.963    | 13%      | 0.897   | 21%     | 0.844        | 31%          |
| Black  | 456 / 241   | 0.966    | 6%       | 0.891   | 22%     | 0.789        | 20%          |
| Indian | 90 / 65     | 0.940    | **28%**  | 0.865   | 21%     | 0.782        | 32%          |
| Latino | 194 / 365   | 0.954    | 11%      | 0.879   | 20%     | 0.851        | 21%          |
| White  | 3,406 / 3,484 | 0.951  | 12%      | 0.876   | 20%     | 0.797        | 28%          |

- **Ranking barely moves.** "Indian" has the lowest AUC for all three models, but its intervals are wide (GenD 0.901 to 0.972) and overlap the others. Spread across groups is 2.6 points for GenD and 3.2 for SBI, so race doesn't meet decision 4's AUC rule.
- **False alarms do.** At GenD's threshold, 28% of real Indian faces are flagged fake (25 of 90), against 6 to 13% for every other group; the white group's rate would predict about 10. This is the gap [evaluation.md](evaluation.md#metrics) warned AUC would miss. SBI is the most even, about 21% for every group.
- **Provisional.** The group is small, the direction check is pending, and HiDF isn't South Asian. This is a reason to measure false-positive rates on real South Asian faces once the request-only sets arrive, not a result in itself.

## What the results can't say yet

- Nothing about **lip-sync with cloned voices**, v1's focus: none scored yet. MAVOS-DD Hindi is next ([Progress](progress.md#next)).
- Nothing about **Sinhala**: its only fakes are in a cross-corpus set waiting on the shortcut probe.
- Nothing about **in-the-wild South Asian audio**: every South Asian audio set so far is read or studio speech.
- Little about **demographics in audio**: these sets carry gender at most.
