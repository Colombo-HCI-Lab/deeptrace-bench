---
title: "Results"
description: "What the zero-shot runs so far show, and how far to trust each number"
group: findings
order: 2
---

Read this page for a finding or a number from the runs so far. The caveats come first.

## Read this first

- **A snapshot as of 2026-10-12.** Latest numbers for every run: the [all-runs table](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs). What ran and why: [Experiments](experiments.md).
- **Zero-shot.** No model was trained or fine-tuned on these evalsets.
- **Provisional rules.** The evaluation setup is still `proposed` (decision 5) and nothing is published with `evaluate.py --publish`, so numbers can move.
- **The guard compares datasets, not generators.** The contamination guard refuses a model on data it trained on. `clean` means no known dataset overlap, not that the model never saw the TTS systems behind an evalset's fakes.
- **Video is provisional.** No video model has a settled score direction (shown on a reproduction set that higher means fake), and GenD has no parity check (shown to score like the upstream code, [models.md](models.md#parity)).

## The datasets

Every dataset a detector was scored on, plus FF++, the training set of every video detector.

| Dataset | Type | Role | Language | Real / fake | Fakes made by | Groups | Licence |
| ------- | ---- | ---- | -------- | ----------: | ------------- | ------ | ------- |
| ASVspoof 2019 LA (eval) | Audio · TTS and voice conversion | Reproduction | English | 7,355 / 63,882 | 13 attack systems (A07 to A19) | attack | open |
| In-the-Wild | Audio · real-world deepfakes | Reproduction | English | 19,963 / 11,816 | collected from the web | none | CC BY-SA 4.0 |
| Urdu CSALT | Audio · TTS | South Asian | Urdu | 3,398 / 3,396 | VITS, Tacotron | generator | CC BY-NC |
| BanglaFake | Audio · TTS | South Asian | Bengali | 12,796 / 12,796 | VITS | sub-corpus | none declared |
| Bangla voices | Audio · TTS, each fake paired with its real twin | South Asian | Bengali | 2,250 / 2,250 | not named | gender | CC BY 4.0 |
| IndicTTS challenge | Audio · TTS | South Asian | 16 Indian languages, Nepali among them | 15,558 / 15,544 | not named | language, gender | CC BY 4.0 |
| IndicSynth Hindi | Audio · TTS and voice conversion | South Asian | Hindi | 5,059 / 10,000 | XTTS-v2, FreeVC24 | generator, gender | CC BY-NC |
| HiDF | Video · face swap | Race-labelled, not South Asian | n/a | 4,361 / 4,361 | a commercial face-swap tool | race, gender, age | CC BY-NC 4.0 |
| FF++ | Video · face swap and reenactment | Training set only | n/a | 1,000 / 4,000 | Deepfakes, Face2Face, FaceSwap, NeuralTextures | none | on request |

IndicSynth Hindi's fakes are a seeded sample of 5,000 per generator from 205,938; its real side is IndicSUPERB, the corpus they were cloned from.

## The detectors

| Detector | Type | Backbone | Trained on |
| -------- | ---- | -------- | ---------- |
| `aasist` | Graph attention network, 0.3 M parameters | none, trained from scratch | ASVspoof 2019 LA |
| `aasist_l` | Graph attention network, 85 K parameters | none, trained from scratch | ASVspoof 2019 LA |
| `xlsr_aasist` | Speech foundation model with a graph attention head | XLS-R 300M, 128 languages | ASVspoof 2019 LA |
| `antideepfake_w2v_small` | Speech foundation model with a linear head | wav2vec 2.0 base, English only | 74k hours of real and fake speech, 100+ languages |
| `antideepfake_mms_300m` | Speech foundation model with a linear head | MMS 300M, about 1,400 languages | 74k hours of real and fake speech, 100+ languages |
| `df_arena_500m` | Speech foundation model with a conformer head | XLS-R 300M | ASVspoof 2019 and 5, MLAAD and six more |
| `df_arena_1b` | Speech foundation model with a conformer head | XLS-R 1B | ASVspoof 2019 and 5, MLAAD and six more |
| `xception` | CNN, 22 M parameters | ImageNet | FF++ |
| `sbi` | CNN (EfficientNet-B4), 18 M parameters | ImageNet | FF++ real videos, with fakes made by blending each face into itself |
| `gend` | Vision foundation model, only its LayerNorms tuned | CLIP ViT-L/14 | FF++ |

## Insights

1. **Doing well on its home set says nothing about South Asian speech.** AASIST scores 0.999 on ASVspoof, 0.578 on In-the-Wild and 0.048 on Bangla voices, where it is confidently backwards. It most likely learned studio-clean audio as a sign of real speech, which flips on crowd-sourced recordings.
2. **Training data matters more than size.** XLS-R + AASIST, a 300M multilingual backbone with AASIST's English-only training, fails on Bengali (0.467 BanglaFake, 0.389 Bangla voices). AntiDeepfake w2v2 small, a third the size but trained on 74k hours of multilingual real and fake speech, scores 0.955 and 1.000. AASIST-L, at 85K parameters, moves no score of AASIST's by more than 0.03.
3. **With training held fixed, multilingual pretraining still helps.** The two AntiDeepfake models differ only in pretraining, and going from English-only to about 1,400 languages lifts IndicTTS from 0.895 to 0.997 (full table below). This is the cleanest comparison in the grid.
4. **Past a point, bigger doesn't help.** DF Arena 1B beats 500M on IndicTTS (0.988 against 0.978) and loses on Bangla voices (0.993 against 0.998). Both sit at a ceiling these sets can't see past.
5. **The open South Asian sets no longer separate modern detectors.** MMS-300M and both DF Arena models score 0.978 or more on all five; w2v2 small 0.895 or more. Some runs carry a source-overlap flag, and these models trained on collections such as MLAAD that likely share generators (VITS, XTTS) with these sets, which the guard can't see. So the dataset track is moving to harder material: lip-sync with cloned voices, in-the-wild clips, Sinhala and Nepali.
6. **Urdu is the odd one out, a warning about dataset artefacts.** The classic detectors score 0.86 on Urdu, about 0.4 on the other Bengali and Indic sets and 0.05 on Bangla voices. Nothing in them favours Urdu, so the gap likely comes from how each corpus was recorded. The planned shortcut probe ([evaluation.md](evaluation.md)) will test this.
7. **Voice conversion is the classic detectors' blind spot.** On IndicSynth Hindi, at the overall EER threshold, XLS-R + AASIST misses 46% of FreeVC24 voice conversions but 3% of XTTS-v2 clips; AASIST misses 92% against 27%. The multilingual detectors miss at most 5% of either.
8. **In video, newer and foundation-based detectors generalise better from FF++.** On HiDF, GenD scores 0.952, SBI 0.877, Xception 0.804; SBI beats Xception without seeing a real deepfake in training. Below AUC, GenD flags 28% of real Indian faces as fake at its threshold, against 6 to 13% for the other groups. Provisional until FF++ or Celeb-DF confirms the score direction.

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

Every score direction here is settled. The AASIST and DF Arena gaps are protocol: upstream scores a clip's first 4 seconds or the whole clip, while we average every 4-second window and refuse clips under a second ([models.md](models.md#per-model)).

MMS-300M is 1.4 points worse than its card, yet w2v2 small, scored the same way, matches its own and parity passed window by window. So the gap is in the protocol or the card, not the adapter: a concrete case for decision 5.

On In-the-Wild (real public figures' speech) the ASVspoof-only models collapse: AASIST 44.0% EER, AASIST-L 44.9%.

## Audio: South Asian speech

Each cell is AUC (EER). * marks a run from 2026-10-05 under the previous setup version, which differs only in a video setting. "so" marks a `source_overlap` verdict: the model's training shares a source corpus, such as Common Voice, with the evalset.

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

Training data sorts the audio detectors into three tiers.

1. **ASVspoof 2019 LA only (AASIST, AASIST-L): they fail.** Below chance on every South Asian set except Urdu (0.86).
2. **Same training, multilingual front end (XLS-R + AASIST): uneven.** It holds up on Urdu (0.80) and IndicSynth Hindi (0.83) but stays near or below chance on the Bengali sets and most IndicTTS languages (0.49 to 0.70).
3. **Post-trained on large multilingual collections of real and fake speech (AntiDeepfake, DF Arena): near the ceiling.** 0.978 or more everywhere, except w2v2 small, which dips to 0.895 on IndicTTS.

These sets no longer rank the top tier, and part of its ceiling may be familiarity: its training collections likely share generators (VITS, XTTS) with these sets, which the guard can't see.

### Multilingual pretraining helps

The two AntiDeepfake models share post-training data and differ only in pretraining.

| Evalset    | w2v2 small (English pretraining) | MMS-300M (about 1,400 languages) |
| ---------- | -------------------------------: | -------------------------------: |
| IndicTTS   | 0.895                            | 0.997                            |
| BanglaFake | 0.955                            | 0.999                            |
| Urdu CSALT | 0.969                            | 0.998                            |

Multilingual pretraining cuts the remaining error (1 minus AUC) by 15 to 45 times on these sets.

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

XLS-R + AASIST runs from 0.49 (Gujarati) to 0.86 (English, its training language), AASIST from 0.23 to 0.52; MMS-300M stays at 0.996 or above. Under decision 4's proposed rule (an attribute matters if both models in a modality shift by 5 or more AUC points), language counts for the classic pair only.

### Gender

Mostly flat. The exception is XLS-R + AASIST on Bangla voices: AUC 0.29 for women, 0.48 for men.

### Ranking isn't calibration

DF Arena 1B reaches AUC 0.993 on Bangla voices and 0.995 on Urdu CSALT, yet its EER there is 4.3% and 4.0%, against 0.7% on In-the-Wild. A fixed threshold needs per-group error rates, not just AUC.

## Video: HiDF

All three detectors trained on FF++, so HiDF's commercial face swaps are a true cross-dataset test. The newest, GenD, generalises best, as the literature reports.

| Model    | AUC (95% interval)    | EER   |
| -------- | --------------------- | ----: |
| GenD     | 0.952 (0.948 to 0.956) | 11.5% |
| SBI      | 0.877 (0.870 to 0.884) | 20.0% |
| Xception | 0.804 (0.794 to 0.812) | 27.2% |

By race: AUC, and the false-positive rate (FPR) on real faces at each model's global threshold.

| Group  | Real / fake | GenD AUC | GenD FPR | SBI AUC | SBI FPR | Xception AUC | Xception FPR |
| ------ | ----------: | -------: | -------: | ------: | ------: | -----------: | -----------: |
| Asian  | 215 / 206   | 0.963    | 13%      | 0.897   | 21%     | 0.844        | 31%          |
| Black  | 456 / 241   | 0.966    | 6%       | 0.891   | 22%     | 0.789        | 20%          |
| Indian | 90 / 65     | 0.940    | **28%**  | 0.865   | 21%     | 0.782        | 32%          |
| Latino | 194 / 365   | 0.954    | 11%      | 0.879   | 20%     | 0.851        | 21%          |
| White  | 3,406 / 3,484 | 0.951  | 12%      | 0.876   | 20%     | 0.797        | 28%          |

- **Ranking barely moves.** "Indian" has the lowest AUC for all three models, but its intervals are wide (GenD 0.901 to 0.972) and overlap the rest. Across groups GenD spreads 2.6 points and SBI 3.2, so race doesn't meet decision 4's AUC rule.
- **False alarms do.** At GenD's threshold, 28% of real Indian faces (25 of 90) are flagged fake, against 6 to 13% for every other group; the white group's rate would predict about 10. AUC can't show this ([evaluation.md](evaluation.md#metrics)). SBI is the most even, about 21% for every group.
- **Provisional.** The group is small, the direction check is pending and HiDF isn't South Asian. It is a reason to measure false-positive rates on real South Asian faces once the request-only sets arrive, not a result in itself.

## What the results can't say yet

- **Lip-sync with cloned voices**, v1's focus: nothing scored yet. MAVOS-DD Hindi is next ([Progress](progress.md#next)).
- **Sinhala.** Its only fakes are in a cross-corpus set waiting on the shortcut probe.
- **In-the-wild South Asian audio.** Every South Asian audio set so far is read or studio speech.
- **Demographics in audio.** These sets carry gender at most.
