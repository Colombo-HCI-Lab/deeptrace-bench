---
title: "Results"
description: "What each experiment asked, what the zero-shot runs so far show, and how far to trust each number"
---

## Read this first

- **Two snapshots.** The first zero-shot round (2026-10-12) and the first full benchmark round (2026-10-27 to 29, [below](#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)), paused on 2026-10-29 with 84 audio pairs left. The latest number for every run is in the [all-runs table](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs).
- **Zero-shot.** No model was trained or fine-tuned on these evalsets.
- **Provisional rules.** The evaluation setup isn't settled ([decision 5](progress.md#open-decisions-that-shape-the-results)) and nothing has gone through `evaluate.py --publish`, so numbers can move.
- **`clean` isn't proof the data is new to a model.** The guard compares datasets, not the generators behind them ([verdicts](evaluation.md#contamination-verdicts)).
- **Video is provisional.** No video model has a settled score direction, and GenD, GenD PE-L, Effort, LipForensics and HAVIC have no parity check ([Models](models.md#status)).

## Experiments

Each experiment is a set of model and evalset pairs that answers one question. What comes next is in [Progress](progress.md#next).

| Experiment | Question | Models | Evalsets | Status |
| --- | --- | --- | --- | --- |
| E1: reproductions | Do our adapters reproduce the published figures, with scores pointing the right way? | the ten audio models below; video models later | ASVspoof 2019 LA eval, or In-the-Wild where the guard refuses ASVspoof 2019 (DF Arena, AntiDeepfake) | audio done for the seven first-round models, AASIST3, XLS-R + SLS and XLSR-Mamba queued; video waits on FF++ or Celeb-DF v2 |
| E2: zero-shot audio | Do detectors trained only on English spoofing data hold up on South Asian speech? Does multilingual pretraining help? Do language and gender shift the scores? | the ten audio models | the eight same-corpus South Asian audio sets below | seven models done on all eight; AASIST3, XLS-R + SLS and XLSR-Mamba on four, the rest queued |
| E3: zero-shot video on HiDF | Does a detector's accuracy, or its false-positive rate on real faces, differ by race? | the 18 video models; the race breakdown so far for Xception, SBI and GenD | `hidf_video` | done, provisional |
| E4: talking heads on MAVOS-DD Hindi | Do face detectors hold up on Hindi video, including audio-driven talking heads? Do audio-visual detectors catch what they miss? | the 18 video models, HAVIC, AVH-Align | `mavos_dd_hi_video`, `mavos_dd_hi_av` | done, provisional |
| E5: cross-corpus audio | Do the audio detectors hold up when the real and fake clips come from different recordings, and could a trivial classifier tell those apart? | the audio models the guard allows | MLAAD Bengali, Hindi and Sinhala, BD-GRF6; SpeechFake queued | partly done: 20 pairs scored, none a result until the shortcut probe |
| E6: fine-tuned baselines | How much does training on South Asian data help, on [identity-safe splits](evaluation.md#fine-tuning-splits)? | to be chosen | to be chosen | planned |
| E7: data from deeptrace | How do the detectors do on what deeptrace collects? | to be chosen | deeptrace's data | planned, once collection starts |

To run a pair again, score in shards, then evaluate. Two shards keep one GPU busy when audio is read from shared storage:

```bash
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 0/2
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 1/2
uv run scripts/evaluate.py --model xlsr_aasist --evalset bangla_voices
```

On one 24 GB GPU with two shards, XLS-R-sized audio models score about 30 clips a second, w2v2 small about 60, DF Arena 1B about 8. Video is bound by CPU face detection: the first model on HiDF took about 4.5 hours, and later ones reuse the cached detections.

## The datasets

Every evalset a detector was scored on, plus FF++, the training set of most video detectors. The cross-corpus sets take their fakes and their real clips from different recordings, so they don't count as results until the [shortcut probe](evaluation.md#shortcut-probe-planned) runs.

| Dataset | Type | Role | Language | Real / fake | Fakes made by | Groups | Licence |
| ------- | ---- | ---- | -------- | ----------: | ------------- | ------ | ------- |
| ASVspoof 2019 LA (eval) | Audio · TTS and voice conversion | Reproduction | English | 7,355 / 63,882 | 13 attack systems (A07 to A19) | attack | open |
| In-the-Wild | Audio · real-world deepfakes | Reproduction | English | 19,963 / 11,816 | collected from the web | none | CC BY-SA 4.0 |
| Urdu CSALT | Audio · TTS | South Asian | Urdu | 3,398 / 3,396 | VITS, Tacotron | generator | CC BY-NC |
| BanglaFake | Audio · TTS | South Asian | Bengali | 12,796 / 12,796 | VITS | sub-corpus | none declared |
| Bangla voices | Audio · TTS, each fake paired with its real twin | South Asian | Bengali | 2,250 / 2,250 | not named | gender | CC BY 4.0 |
| IndicTTS challenge | Audio · TTS | South Asian | 16 Indian languages, Nepali among them | 15,558 / 15,544 | not named | language, gender | CC BY 4.0 |
| IndicSynth Hindi | Audio · TTS and voice conversion | South Asian | Hindi | 5,059 / 10,000 | XTTS-v2, FreeVC24 | generator, gender | CC BY-NC |
| IndicSynth Bengali | Audio · TTS and voice conversion | South Asian | Bengali | 4,589 / 15,000 | XTTS-v2, VITS, FreeVC24 | generator, gender | CC BY-NC |
| IndicSynth Urdu | Audio · TTS and voice conversion | South Asian | Urdu | 5,129 / 10,000 | XTTS-v2, FreeVC24 | generator, gender | CC BY-NC |
| MAVOS-DD Hindi, audio | Audio · voice conversion | South Asian | Hindi | 3,632 / 831 | kNN-VC | generator | CC BY-NC-SA 4.0 |
| HiDF | Video · face swap | Race-labelled, not South Asian | n/a | 4,361 / 4,361 | a commercial face-swap tool | race, gender, age | CC BY-NC 4.0 |
| MAVOS-DD Hindi, video | Video · face swap, talking heads, reenactment | South Asian | Hindi | 4,463 / 3,050 | InSwapper, HifiFace, Roop, EchoMimic, Sonic, MEMO, LivePortrait | generator | CC BY-NC-SA 4.0 |
| MAVOS-DD Hindi, audio-visual | Video and audio · either track changed | South Asian | Hindi | 3,632 / 3,881 | the seven above and kNN-VC | generator | CC BY-NC-SA 4.0 |
| MLAAD Bengali | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Bengali | 4,589 / 4,000 | four TTS systems (IndicF5, Bark, two VITS voices) | generator | CC BY-NC (fakes), CC BY 4.0 (real) |
| MLAAD Hindi | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Hindi | 5,059 / 9,000 | nine TTS systems (XTTS-v2, Bark, IndicF5 and six more) | generator | CC BY-NC (fakes), CC BY 4.0 (real) |
| MLAAD Sinhala | Audio · TTS, real speech from OpenSLR | South Asian, cross-corpus | Sinhala | 11,550 / 1,000 | Edge-TTS | generator | CC BY-NC (fakes), CC BY-SA 4.0 (real) |
| BD-GRF6 | Audio · not stated | South Asian, cross-corpus | Bengali | 6,783 / 6,783 | not stated | gender, third gender included | CC BY 4.0 |
| FF++ | Video · face swap and reenactment | Training set only | n/a | 1,000 / 4,000 | Deepfakes, Face2Face, FaceSwap, NeuralTextures | none | on request |

IndicSynth's fakes are a seeded sample of 5,000 per generator (from 205,938 in Hindi): all of them would outnumber the real clips about 40 to 1 and cost about a day of GPU time per large model, without changing AUC or EER.

MAVOS-DD's labels follow the track a model sees. The video set counts the kNN-VC clips, whose video is untouched, as real; the audio-visual set counts them as fake; and the audio set scores only real and kNN-VC clips, the only ones whose audio label is settled so far.

## The detectors

| Detector | Type | Backbone | Trained on |
| -------- | ---- | -------- | ---------- |
| `aasist` | Graph attention network, 0.3 M parameters | none, trained from scratch | ASVspoof 2019 LA |
| `aasist_l` | Graph attention network, 85 K parameters | none, trained from scratch | ASVspoof 2019 LA |
| `xlsr_aasist` | Speech foundation model with a graph attention head | XLS-R 300M, 128 languages | ASVspoof 2019 LA |
| `xlsr_sls` | Speech foundation model with attention over all 24 of its layers (SLS) | XLS-R 300M, 128 languages | ASVspoof 2019 LA |
| `xlsr_mamba` | Speech foundation model with a bidirectional Mamba head | XLS-R 300M, 128 languages | ASVspoof 2019 LA |
| `aasist3` | Graph attention network with KAN layers, on speech foundation model features | wav2vec 2.0 XLSR-53, 53 languages | ASVspoof 2019 and 5, MLAAD, M-AILABS |
| `antideepfake_w2v_small` | Speech foundation model with a linear head | wav2vec 2.0 base, English only | 74k hours of real and fake speech, 100+ languages |
| `antideepfake_mms_300m` | Speech foundation model with a linear head | MMS 300M, about 1,400 languages | 74k hours of real and fake speech, 100+ languages |
| `df_arena_500m` | Speech foundation model with a conformer head | XLS-R 300M | ASVspoof 2019 and 5, MLAAD and six more |
| `df_arena_1b` | Speech foundation model with a conformer head | XLS-R 1B | ASVspoof 2019 and 5, MLAAD and six more |
| `xception` | CNN, 22 M parameters | ImageNet | FF++ |
| `sbi` | CNN (EfficientNet-B4), 18 M parameters | ImageNet | FF++ real videos, with fakes made by blending each face into itself |
| `gend` | Vision foundation model, only its LayerNorms tuned | CLIP ViT-L/14 | FF++ |
| `gend_pe_l` | Vision foundation model, only its LayerNorms tuned | Perception Encoder L/14 | FF++ |
| `effort` | Vision foundation model, its weights split by SVD and only the residual tuned | CLIP ViT-L/14 | FF++ |
| `realforensics` | Video network (CSN) on face clips | self-supervised on real talking faces and their audio (LRW) | FF++ |
| `ftcn` | 3D CNN with no spatial reach, plus a temporal transformer, on face clips | none | FF++ |
| `lipforensics` | ResNet-18 and a temporal CNN on grayscale mouth clips | lipreading (LRW) | FF++ |
| `efficientnet_b4` | CNN (EfficientNet-B4), 18 M parameters | ImageNet | FF++ |
| `ucf` | CNN (Xception) scoring only the forgery traces every method shares, 47 M parameters | ImageNet | FF++ |
| `core` | CNN (Xception) trained to agree across augmented views, 22 M parameters | ImageNet | FF++ |
| `recce` | CNN (Xception) that reconstructs real faces and attends to what it can't, 48 M parameters | ImageNet | FF++ |
| `f3net` | CNN (Xception) on DCT frequency bands, 23 M parameters | ImageNet | FF++ |
| `ffd` | CNN (Xception) with an attention map over manipulated regions, 22 M parameters | ImageNet | FF++ |
| `srm` | Two-stream CNN (Xception), colour and high-frequency noise residuals, 55 M parameters | ImageNet | FF++ |
| `spsl` | CNN (Xception) with the phase spectrum as a fourth channel, 22 M parameters | ImageNet | FF++ |
| `dfdc_b7` | Seven CNNs (EfficientNet-B7) averaged, the DFDC challenge winner | ImageNet | DFDC |
| `genconvit` | ConvNeXt and Swin hybrid with an autoencoder branch, the only branch scored | ImageNet | DFDC, FF++, DeepfakeTIMIT |
| `havic` | Audio and video transformers fused over 16-frame windows | real videos (LRS2) | FakeAVCeleb |
| `avh_align` | Small network that scores how far lips and voice drift apart | AV-HuBERT large (LRS3, VoxCeleb2), frozen | real AV-Deepfake1M videos only |

## Insights

1. **Multilingual pretraining helps, with training held fixed.** The two AntiDeepfake models share their post-training data and differ only in pretraining. Going from English only (w2v2 small) to about 1,400 languages (MMS-300M) cuts the remaining error (1 minus AUC) by 15 to 45 times, lifting IndicTTS from 0.895 to 0.997. This is the cleanest comparison in the grid.
2. **Language matters only for the detectors trained on ASVspoof.** Across the 16 IndicTTS languages, XLS-R + AASIST runs from 0.49 (Gujarati) to 0.86 (English, its training language) and AASIST from 0.23 to 0.52, while MMS-300M stays at 0.996 or above. Under [decision 4's proposed rule](evaluation.md#metrics), language counts for the classic pair only.
3. **In faces, ranking is even but false alarms aren't.** On HiDF, GenD scores 0.952, SBI 0.877 and Xception 0.804; SBI beats Xception without seeing a real deepfake in training. Across race groups AUC barely moves, yet at its global threshold GenD flags 28% of real Indian faces as fake, against 6 to 13% for the other groups. Provisional (see the HiDF section).
4. **Doing well on its home set says nothing about South Asian speech.** AASIST scores 0.999 on ASVspoof, 0.578 on In-the-Wild and 0.048 on Bangla voices, where it is confidently backwards. It most likely learned studio-clean audio as a sign of real speech, which flips on crowd-sourced recordings.
5. **Training data sorts the audio detectors into three tiers, and matters more than size.** Trained on ASVspoof 2019 LA only, AASIST and AASIST-L fall below chance on every same-corpus South Asian set except Urdu CSALT (0.86) and MAVOS-DD's Hindi voice conversions (0.64). XLS-R + AASIST, a 300M multilingual backbone with the same training, is uneven: 0.80 on Urdu and 0.83 on IndicSynth Hindi, but 0.467 on BanglaFake and 0.389 on Bangla voices. XLS-R + SLS, XLSR-Mamba and AASIST3, added in the October round, land in the same middle tier, 0.65 to 0.99 on the four sets they have so far, with no parity check yet. Post-trained on large multilingual collections of real and fake speech, AntiDeepfake w2v2 small, a third of XLS-R's size, scores 0.955 and 1.000 on BanglaFake and Bangla voices. AASIST-L, at 85K parameters, moves no same-corpus score of AASIST's by more than 0.03 (on the cross-corpus BD-GRF6 and MLAAD Sinhala, by about 0.1).
6. **The open South Asian sets no longer separate the top tier.** MMS-300M and both DF Arena models score 0.978 or more on all eight same-corpus sets, w2v2 small 0.895 or more. Only BD-GRF6 pulls the AntiDeepfake pair down, to near chance (0.51 and 0.59), and it is cross-corpus, so not a result yet. Their training collections, such as MLAAD, likely share generators (VITS, XTTS) with these sets, which the guard can't see, so part of the ceiling may be familiarity. The dataset track is moving to harder material: lip-sync with cloned voices, in-the-wild clips, Sinhala and Nepali.
7. **Past a point, bigger doesn't help.** DF Arena 1B beats 500M on IndicTTS (0.988 against 0.978) and loses on Bangla voices (0.993 against 0.998).
8. **Urdu CSALT is the odd one out, a warning about dataset artefacts.** The classic detectors score 0.86 on Urdu CSALT, 0.21 to 0.40 on the IndicSynth, BanglaFake and IndicTTS sets and 0.05 on Bangla voices. IndicSynth's Urdu, the same language from another corpus, gets 0.34, so the gap most likely comes from how CSALT was recorded, not from the language. The planned [shortcut probe](evaluation.md#shortcut-probe-planned) will test this.
9. **Voice conversion is the classic detectors' blind spot.** On IndicSynth Hindi, at the overall EER threshold, XLS-R + AASIST misses 46% of FreeVC24 voice conversions but 3% of XTTS-v2 clips; AASIST misses 92% against 27%. The multilingual detectors miss at most 5% of either.
10. **Ranking isn't calibration.** DF Arena 1B reaches AUC 0.993 on Bangla voices and 0.995 on Urdu CSALT, yet its EER there is 4.3% and 4.0%, against 0.7% on In-the-Wild. A fixed threshold needs per-group error rates, not just AUC.
11. **Gender barely shifts the audio scores.** The exception is XLS-R + AASIST on Bangla voices: AUC 0.29 for women, 0.48 for men.

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

Every score direction here is settled. The gaps to the published figures come from protocol differences, explained per model in [Models](models.md#per-model).

## Audio: South Asian speech

Each cell is AUC (EER). * marks a run from 2026-10-05 under the previous setup version, which differs only in a video setting. "so" marks a `source_overlap` verdict ([verdicts](evaluation.md#contamination-verdicts)).

| Model                   | Urdu CSALT    | BanglaFake       | Bangla voices | IndicTTS        | IndicSynth Hindi |
| ----------------------- | ------------- | ---------------- | ------------- | --------------- | ---------------- |
| AASIST                  | 0.863 (22%)*  | 0.365 (60%)*     | 0.048 (87%)   | 0.403 (58%)     | 0.393 (59%)      |
| AASIST-L                | 0.861 (23%)*  | 0.379 (58%)*     | 0.055 (87%)   | 0.377 (59%)     | 0.390 (59%)      |
| XLS-R + AASIST          | 0.802 (28%)   | 0.467 (55%) so   | 0.389 (58%)   | 0.581 (46%)     | 0.828 (25%)      |
| AntiDeepfake w2v2 small | 0.969 (9.3%)  | 0.955 (11%)      | 1.000 (0.2%)  | 0.895 (17%) so  | 0.997 (2.5%)     |
| AntiDeepfake MMS-300M   | 0.998 (2.2%)  | 0.999 (1.1%) so  | 1.000 (0.5%)  | 0.997 (2.3%) so | 1.000 (0.1%)     |
| DF Arena 500M           | 0.981 (7.8%)* | 0.997 (2.2%)* so | 0.998 (2.1%)  | 0.978 (6.9%)    | 0.997 (2.6%)     |
| DF Arena 1B             | 0.995 (4.0%)* | 0.999 (0.7%)* so | 0.993 (4.3%)  | 0.988 (5.1%)    | 0.999 (1.7%)     |

### Multilingual pretraining helps

The two AntiDeepfake models, AUC.

| Evalset    | w2v2 small (English pretraining) | MMS-300M (about 1,400 languages) |
| ---------- | -------------------------------: | -------------------------------: |
| IndicTTS   | 0.895                            | 0.997                            |
| BanglaFake | 0.955                            | 0.999                            |
| Urdu CSALT | 0.969                            | 0.998                            |

### Language matters for the classic models only

AUC by language on the IndicTTS challenge set.

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

## Video: HiDF

All three detectors trained on FF++, so HiDF's commercial face swaps are a cross-dataset test.

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
- **False alarms do.** GenD's 28% on real Indian faces is 25 of 90, where the white group's rate would predict about 10. AUC can't show this. SBI is the most even, about 21% for every group.
- **Provisional.** The group is small, the score direction is unchecked and HiDF isn't South Asian. It is a reason to measure false-positive rates on real South Asian faces once the request-only sets arrive, not a result in itself.

## The 2026-10-28 round: full HiDF and MAVOS-DD Hindi

Every video model with weights was scored on all of HiDF (8,722 clips) and all of MAVOS-DD Hindi video (7,513 clips), the audio-visual models on MAVOS-DD's audio and video together, through `evaluate.py --publish`, on one RTX 3090 on SPARK and two 24 GB GPUs outside it. Every pair's time, memory and energy are in `results/ledger.csv`. TPR at 5% FPR is the share of fakes caught when 5% of real clips are flagged.

| Model | Parity | HiDF AUC | HiDF EER | HiDF TPR@5% | MAVOS-DD AUC | MAVOS-DD EER | MAVOS-DD TPR@5% |
| ----- | :----: | -------: | -------: | ----------: | -----------: | -----------: | --------------: |
| GenD | no | 0.952 | 11.5% | 0.78 | 0.741 | 33.4% | 0.48 |
| GenD PE-L | no | 0.940 | 13.4% | 0.70 | 0.798 | 28.7% | 0.48 |
| RealForensics | yes | 0.931 | 14.5% | 0.64 | 0.779 | 29.3% | 0.36 |
| Effort | no | 0.919 | 15.8% | 0.69 | 0.603 | 42.7% | 0.42 |
| SBI | yes | 0.877 | 20.0% | 0.45 | 0.777 | 29.3% | 0.29 |
| LipForensics | no | 0.870 | 21.6% | 0.50 | 0.703 | 35.4% | 0.32 |
| FTCN | yes | 0.865 | 21.8% | 0.40 | **0.854** | 22.6% | 0.54 |
| DFDC winner (7 × EfficientNet-B7) | yes | 0.859 | 21.8% | 0.49 | 0.390 | 59.2% | 0.12 |
| Xception | yes | 0.804 | 27.2% | 0.31 | 0.628 | 41.5% | 0.19 |
| UCF | yes | 0.782 | 29.0% | 0.29 | 0.546 | 47.9% | 0.19 |
| GenConViT (ED branch) | yes | 0.779 | 29.1% | 0.31 | 0.541 | 49.1% | 0.17 |
| CORE | yes | 0.777 | 29.6% | 0.28 | 0.534 | 48.2% | 0.10 |
| RECCE | yes | 0.770 | 29.7% | 0.29 | 0.539 | 47.4% | 0.15 |
| F3Net | yes | 0.761 | 30.6% | 0.25 | 0.527 | 48.1% | 0.15 |
| FFD | yes | 0.760 | 30.5% | 0.24 | 0.554 | 46.5% | 0.09 |
| EfficientNet-B4 | yes | 0.742 | 32.1% | 0.22 | 0.583 | 43.7% | 0.14 |
| SRM | yes | 0.716 | 34.4% | 0.22 | 0.556 | 46.5% | 0.11 |
| SPSL | yes | 0.680 | 36.6% | 0.15 | 0.598 | 43.4% | 0.09 |
| HAVIC (audio and video) | no | n/a | n/a | n/a | 0.692 | 37.5% | 0.27 |
| AVH-Align (audio and video) | yes | n/a | n/a | n/a | 0.520 | 48.4% | 0.05 |

- **MAVOS-DD Hindi is the hard set.** Every model but FTCN drops from HiDF to MAVOS-DD, by 0.08 (SPSL) to 0.47 (the DFDC winner); FTCN loses 0.01. The eight DeepfakeBench detectors fall to 0.53 to 0.63, close to chance. Its fakes include audio-driven talking heads and lip-sync, which frame-level face-swap detectors weren't trained on.
- **Motion over time is what holds up.** FTCN, which looks at short clips rather than single frames, leads MAVOS-DD (0.854, with parity) and is the only model there that catches more than half the fakes at a 5% false-alarm rate (0.54). RealForensics, also temporal, is second on HiDF (0.931, behind GenD) and keeps 0.779 on MAVOS-DD. Of the frame-level models, GenD PE-L (0.798) and SBI (0.777) do best.
- **The DFDC winner is backwards on MAVOS-DD.** It scores 0.859 on HiDF but 0.390 on MAVOS-DD: at its EER threshold it flags 59% of real Hindi videos and misses 98% of EchoMimic's and 94% of Sonic's talking heads. It matches upstream within 1e-5 per face crop, so this is the model, not the adapter. GenConViT also falls to chance (0.541).
- **AVH-Align doesn't transfer.** The audio-visual alignment model, trained on AV-Deepfake1M (English), sits at chance on MAVOS-DD (0.520): it catches 92% of Roop face swaps and 87% of MEMO talking heads, but misses 76% of Sonic's and 61% of LivePortrait's. Its parity check passes, so this is how it behaves on Hindi.
- **Talking heads are the frame models' blind spot, and the temporal and audio-visual models' strength.** At each model's EER threshold, GenD PE-L misses 78% of EchoMimic's audio-driven talking heads while catching 82 to 95% of face swaps. FTCN catches every EchoMimic clip and 98% of LivePortrait's, though it still misses about half of MEMO's and Sonic's. HAVIC, which reads audio and video together, misses only 4 to 6% of the talking heads but about half the swaps, so it sits mid-table overall (0.692) with LipForensics (0.703). The two cover each other's gaps (miss rates per generator are in each run's `by_method.csv`).
- **The pattern holds across all 18 video models.** Share of each generator's fakes missed at each model's own EER threshold on MAVOS-DD Hindi, averaged over the 18:

  | Generator | Type | Mean miss | Worst model |
  | --------- | ---- | --------: | ----------- |
  | EchoMimic | audio-driven talking head | 76% | DFDC winner, 98% |
  | MEMO | audio-driven talking head | 71% | DFDC winner, 88% |
  | Sonic | audio-driven talking head | 63% | DFDC winner, 94% |
  | Roop | face swap | 33% | FTCN, 61% |
  | HifiFace | face swap | 28% | DFDC winner, 52% |
  | InSwapper | face swap | 26% | SPSL, 47% |
  | LivePortrait | face reenactment | 15% | Effort, 57% |

  Every model misses more talking heads than face swaps on average, and 16 of the 18 miss each talking-head generator more often than any face swap. MAVOS-DD's fakes are mostly talking heads, so this round can't yet separate "Hindi" from "talking head": that needs the same models on a non-South-Asian set with the same generators ([Progress](progress.md#next)).
- **Reproducible across machines.** The 2026-10-12 HiDF numbers for GenD, SBI and Xception came out the same, and Effort on MAVOS-DD scored on two different GPUs agreed within 1e-4.
- **Provisional.** No video model has a settled score direction (that needs FF++ or Celeb-DF v2), and the models marked "no" haven't passed parity, so their numbers don't count as results yet.

Audio was paused on 2026-10-29 with 84 pairs left ([Progress](progress.md#next)). These 40 same-corpus pairs are done, AUC:

| Model | Parity | Urdu CSALT | MAVOS-DD Hindi audio (EER) | IndicSynth Bengali | IndicSynth Urdu |
| ----- | :----: | ---------: | -------------------------: | -----------------: | --------------: |
| AASIST | yes | 0.863 | 0.635 (38.2%) | 0.214 | 0.343 |
| AASIST-L | yes | 0.861 | 0.644 (38.4%) | 0.234 | 0.336 |
| XLS-R + AASIST | yes | 0.802 | 0.886 (16.9%) | 0.654 | 0.777 |
| AntiDeepfake w2v2 small | yes | 0.969 | 0.999 (1.3%) | 0.998 | 0.996 |
| AntiDeepfake MMS-300M | yes | 0.998 | 1.000 (0.1%) | 0.999 | 1.000 |
| DF Arena 500M | yes | 0.981 | 1.000 (0.1%) | 0.995 | 0.999 |
| DF Arena 1B | yes | 0.995 | 1.000 (0.1%) | 1.000 | 1.000 |
| AASIST3 | no | 0.806 | 0.907 (18.2%) | 0.785 | 0.781 |
| XLSR-Mamba | no | 0.691 | 0.976 (6.3%) | 0.654 | 0.790 |
| XLS-R + SLS | no | 0.855 | 0.994 (3.1%) | 0.803 | 0.942 |

- **Urdu CSALT reproduces.** All seven models with parity matched their 2026-10-12 runs (AUC and EER within 0.001, same counts) on a different GPU.
- **MAVOS-DD's audio is easy for the post-trained models.** AntiDeepfake and DF Arena separate it almost perfectly, while the ASVspoof-only AASIST pair stays near 0.64. This is insight 6's ceiling again: their training collections may share MAVOS-DD's voice generators, which the guard can't check.
- **IndicSynth Bengali and Urdu repeat the Hindi picture.** The ASVspoof-only AASIST pair is confidently backwards on Bengali (0.21 and 0.23) and Urdu (0.34 for both); XLS-R + AASIST is in between (0.65 and 0.78); every post-trained model scores 0.995 or more. Of the three models without parity, XLS-R + SLS does best (0.80 and 0.94).
- **Bengali is the hardest of IndicSynth's three languages.** IndicSynth uses the same two generators for Hindi, Bengali and Urdu, so this comparison holds generation fixed. Every model below the post-trained ceiling scores lowest on Bengali: AASIST 0.21 against 0.34 (Urdu) and 0.39 (Hindi), XLS-R + AASIST 0.65 against 0.78 and 0.83, XLSR-Mamba 0.65 against 0.79, XLS-R + SLS 0.80 against 0.94. The same holds across corpora: XLS-R + AASIST scores 0.39 to 0.65 on all four Bengali sets and 0.78 to 0.89 on all five Hindi and Urdu ones.

Cross-corpus pairs, where the real and fake clips come from different recordings, so a model can score well by telling the sources apart. They are scored and recorded but don't count as results until the shortcut probe runs. AUC, then TPR at 5% FPR; "guard" means the model trained on MLAAD, so the contamination guard refused the pair, and a dash is still queued:

| Model | MLAAD Bengali | MLAAD Hindi | MLAAD Sinhala | BD-GRF6 |
| ----- | ------------: | ----------: | ------------: | ------: |
| AASIST | 0.195 (0.02) | 0.378 (0.04) | 0.900 (0.30) | 0.608 (0.19) |
| AASIST-L | 0.185 (0.02) | 0.406 (0.05) | 0.812 (0.03) | 0.709 (0.22) |
| XLS-R + AASIST | 0.527 (0.07) | 0.877 (0.53) | 0.941 (0.77) | 0.509 (0.07) |
| XLSR-Mamba | 0.521 (0.13) | 0.812 (0.48) | 0.944 (0.65) | – |
| XLS-R + SLS | 0.791 (0.35) | 0.875 (0.61) | **0.997** (0.99) | – |
| AntiDeepfake w2v2 small, MMS-300M | guard | guard | guard | 0.592, 0.512 |
| DF Arena 500M, 1B; AASIST3 | guard | guard | guard | – |

- **XLS-R + SLS carries over best**, and is the only model that catches more than a third of MLAAD Bengali's fakes at a 5% false-alarm rate.
- **The ASVspoof-only AASIST pair is backwards on MLAAD Bengali and Hindi** (0.19 to 0.41), as on IndicSynth Bengali, but not on Sinhala. MLAAD Sinhala has 11,550 real clips to 1,000 fakes, and AASIST-L's 0.812 there catches only 3% of fakes at 5% FPR.
- **BD-GRF6 defeats every model so far.** The best is AASIST-L at 0.709; the post-trained AntiDeepfake models sit at chance (0.51 and 0.59).

## What the results can't say yet

- **Lip-sync with cloned voices**, v1's focus: only MAVOS-DD Hindi so far, and with no video score direction settled ([above](#the-2026-10-28-round-full-hidf-and-mavos-dd-hindi)).
- **Sinhala.** Its only fakes are in MLAAD Sinhala, a cross-corpus set: scored above, but not a result until the shortcut probe.
- **In-the-wild South Asian audio.** Every South Asian audio set so far is read or studio speech.
- **Demographics in audio.** These sets carry gender at most.
