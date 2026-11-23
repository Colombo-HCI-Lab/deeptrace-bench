---
title: "Results"
description: "What each experiment asked, what the zero-shot runs so far show, and how far to trust each number"
---

## Read this first

- **The grid is the record.** Every run's AUC and EER is in [the grid](https://colombo-hci-lab.github.io/deeptrace-bench/results/#all-runs), built from the published runs; this page's text holds what they show and the breakdowns the grid can't. The [Timeline](progress.md) dates each round.
- **Zero-shot.** No model was trained or fine-tuned on these evalsets.
- **Provisional rules.** The evaluation setup isn't settled ([decision 5](progress.md#open-decisions-that-shape-the-results)), so even published numbers can move.
- **`clean` isn't proof the data is new to a model.** The guard compares datasets, not the generators behind them ([verdicts](evaluation.md#contamination-verdicts)).
- **Video is provisional.** No video model has a settled score direction ([Models](models.md#status)). Every adapter passes parity.

## Experiments

Each experiment is a set of model and evalset pairs that answers one question. What comes next is in [Progress](progress.md#next).

| Experiment | Question | Models | Evalsets | Status |
| --- | --- | --- | --- | --- |
| E1: reproductions | Do our adapters reproduce the published figures, with scores pointing the right way? | the ten audio models; video models later | ASVspoof 2019 LA eval, or In-the-Wild where the guard refuses ASVspoof 2019 (DF Arena, AntiDeepfake) | audio done for all ten; video waits on FF++ or Celeb-DF v2 |
| E2: zero-shot audio | Do detectors trained only on English spoofing data hold up on South Asian speech? Does multilingual pretraining help? Do language and gender shift the scores? | the ten audio models | the eight same-corpus South Asian audio sets below | all ten done on all eight |
| E3: zero-shot video on HiDF | Does a detector's accuracy, or its false-positive rate on real faces, differ by race? | the 21 video models with weights; the race breakdown so far for Xception, SBI and GenD | `hidf_video` | done, provisional |
| E4: talking heads on MAVOS-DD Hindi | Do face detectors hold up on Hindi video, including audio-driven talking heads? Do audio-visual detectors catch what they miss? | the 21 video models with weights and the four audio-visual ones (HAVIC, AVH-Align, LipFD, AVAD) | `mavos_dd_hi_video`, `mavos_dd_hi_av` | provisional, [below](#faces-and-talking-heads-mavos-dd-hindi): done on `mavos_dd_hi_video` and for the audio-visual models; the video models on `mavos_dd_hi_av` 9 of 42 pairs, the rest scoring on SPARK |
| E4b: the language control | Is the drop on MAVOS-DD Hindi about Hindi, or about the set's fakes? | the 21 video models with weights | `mavos_dd_de_video_quick`, the same generators on German speakers, against `mavos_dd_hi_video_quick` | done: the drop is the set's, not Hindi's, [below](#the-language-control-mavos-dd-german) |
| E5: cross-corpus audio | Do the audio detectors hold up when the real and fake clips come from different recordings, and could a trivial classifier tell those apart? | the audio models the guard allows | MLAAD Bengali, Hindi and Sinhala, BD-GRF6, SpeechFake | 75 pairs scored ([below](#cross-corpus-audio)); the shortcut probe separates every set by trivial features ([below](#the-shortcut-probe)), so which numbers count is the team's call |
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

Every evalset a detector was scored on, plus FF++, the training set of most video detectors. The cross-corpus sets take their fakes and their real clips from different recordings, so a detector can score well there by telling the recordings apart: they are scored and recorded, but don't count as results ([the shortcut probe](#the-shortcut-probe)). Licences and access are in the [Dataset catalog](catalog.md#datasets).

| Dataset | Type | Role | Language | Real / fake | Fakes made by | Groups |
| ------- | ---- | ---- | -------- | ----------: | ------------- | ------ |
| ASVspoof 2019 LA (eval) | Audio · TTS and voice conversion | Reproduction | English | 7,355 / 63,882 | 13 attack systems (A07 to A19) | attack |
| In-the-Wild | Audio · real-world deepfakes | Reproduction | English | 19,963 / 11,816 | collected from the web | none |
| Urdu CSALT | Audio · TTS | South Asian | Urdu | 3,398 / 3,396 | VITS, Tacotron | generator |
| BanglaFake | Audio · TTS | South Asian | Bengali | 12,796 / 12,796 | VITS | sub-corpus |
| Bangla voices | Audio · TTS, each fake paired with its real twin | South Asian | Bengali | 2,250 / 2,250 | not named | gender |
| IndicTTS challenge | Audio · TTS | South Asian | 16 Indian languages, Nepali among them | 15,558 / 15,544 | not named | language, gender |
| IndicSynth Hindi | Audio · TTS and voice conversion | South Asian | Hindi | 5,059 / 10,000 | XTTS-v2, FreeVC24 | generator, gender |
| IndicSynth Bengali | Audio · TTS and voice conversion | South Asian | Bengali | 4,589 / 15,000 | XTTS-v2, VITS, FreeVC24 | generator, gender |
| IndicSynth Urdu | Audio · TTS and voice conversion | South Asian | Urdu | 5,129 / 10,000 | XTTS-v2, FreeVC24 | generator, gender |
| MAVOS-DD Hindi, audio | Audio · voice conversion | South Asian | Hindi | 3,632 / 831 | kNN-VC | generator |
| HiDF | Video · face swap | Race-labelled, not South Asian | n/a | 4,361 / 4,361 | a commercial face-swap tool | race, gender, age |
| MAVOS-DD Hindi, video | Video · face swap, talking heads, reenactment | South Asian | Hindi | 4,463 / 3,050 | InSwapper, HifiFace, Roop, EchoMimic, Sonic, MEMO, LivePortrait | generator |
| MAVOS-DD Hindi, audio-visual | Video and audio · either track changed | South Asian | Hindi | 3,632 / 3,881 | the seven above and kNN-VC | generator |
| MLAAD Bengali | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Bengali | 4,589 / 4,000 | four TTS systems (IndicF5, Bark, two VITS voices) | generator |
| MLAAD Hindi | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Hindi | 5,059 / 9,000 | nine TTS systems (XTTS-v2, Bark, IndicF5 and six more) | generator |
| MLAAD Sinhala | Audio · TTS, real speech from OpenSLR | South Asian, cross-corpus | Sinhala | 11,550 / 1,000 | Edge-TTS | generator |
| BD-GRF6 | Audio · not stated | South Asian, cross-corpus | Bengali | 6,783 / 6,783 | not stated | gender, third gender included |
| SpeechFake Bengali | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Bengali | 4,589 / 20,000 | Edge-TTS | generator |
| SpeechFake Hindi | Audio · TTS and voice conversion, real speech from its own Common Voice | South Asian, cross-corpus | Hindi | 4,630 / 30,015 | Edge-TTS, SeedVC | generator |
| SpeechFake Malayalam | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Malayalam | 2,983 / 3,726 | Edge-TTS | generator |
| SpeechFake Marathi | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Marathi | 4,349 / 20,000 | Edge-TTS | generator |
| SpeechFake Tamil | Audio · TTS, real speech from IndicSUPERB | South Asian, cross-corpus | Tamil | 4,401 / 20,000 | Edge-TTS | generator |
| MAVOS-DD German, video (quick share) | Video · face swap, talking heads, reenactment | Language control, not South Asian | German | 800 / 1,901 | the same generators as the Hindi subset | generator |
| FF++ | Video · face swap and reenactment | Training set only | n/a | 1,000 / 4,000 | Deepfakes, Face2Face, FaceSwap, NeuralTextures | none |

IndicSynth's fakes are a seeded sample of 5,000 per generator (from 205,938 in Hindi): all of them would outnumber the real clips about 40 to 1 and cost about a day of GPU time per large model, without changing AUC or EER.

MAVOS-DD's labels follow the track they judge. The video set counts the kNN-VC clips, whose video is untouched, as real; the audio-visual set counts them as fake; and the audio set scores only real and kNN-VC clips, the only ones whose audio label is settled so far. Video models are scored on the audio-visual set too, to measure what a frames-only detector misses when voice-only fakes count.

## Insights

1. **Multilingual pretraining helps, with training held fixed.** The two AntiDeepfake models share their post-training data and differ only in pretraining. Going from English only (w2v2 small) to about 1,400 languages (MMS-300M) cuts the remaining error (1 minus AUC) by 15 to 45 times: IndicTTS goes from 0.895 to 0.997, BanglaFake from 0.955 to 0.999, Urdu CSALT from 0.969 to 0.998. This is the cleanest comparison in the grid.
2. **Language matters only for the detectors trained on ASVspoof.** Across the 16 IndicTTS languages, XLS-R + AASIST runs from 0.49 (Gujarati) to 0.86 (English, its training language) and AASIST from 0.23 to 0.52, while MMS-300M stays at 0.996 or above. Under [decision 4's proposed rule](evaluation.md#metrics), language counts for the classic pair only.
3. **In faces, ranking is even but false alarms aren't.** On HiDF, GenD scores 0.952, SBI 0.877 and Xception 0.804; SBI beats Xception without seeing a real deepfake in training. Across race groups AUC barely moves, yet at its global threshold GenD flags 28% of real Indian faces as fake, against 6 to 13% for the other groups ([by race](#video-hidf)).
4. **Doing well on its home set says nothing about South Asian speech.** AASIST scores 0.999 on ASVspoof, 0.578 on In-the-Wild and 0.048 on Bangla voices, where it is confidently backwards. It most likely learned studio-clean audio as a sign of real speech, which flips on crowd-sourced recordings.
5. **Training data sorts the audio detectors into three tiers, and matters more than size.** Trained on ASVspoof 2019 LA only, AASIST and AASIST-L fall below chance on every same-corpus South Asian set except Urdu CSALT (0.86) and MAVOS-DD's Hindi voice conversions (0.64). XLS-R + AASIST, a 300M multilingual backbone with the same training, is uneven: 0.80 on Urdu and 0.83 on IndicSynth Hindi, but 0.467 on BanglaFake and 0.389 on Bangla voices. XLS-R + SLS, XLSR-Mamba and AASIST3, added in the October round, are just as uneven: XLS-R + SLS is the steadiest (0.67 on Bangla voices to 0.99 on MAVOS-DD's Hindi audio), XLSR-Mamba is confidently backwards on Bangla voices (0.18), and AASIST3 scores 0.98 there but 0.30 on IndicTTS. Post-trained on large multilingual collections of real and fake speech, AntiDeepfake w2v2 small, a third of XLS-R's size, scores 0.955 and 1.000 on BanglaFake and Bangla voices. AASIST-L, at 85K parameters, moves no same-corpus score of AASIST's by more than 0.03 (on the cross-corpus BD-GRF6 and MLAAD Sinhala, by about 0.1).
6. **The open South Asian sets no longer separate the top tier.** MMS-300M and both DF Arena models score 0.978 or more on all eight same-corpus sets, w2v2 small 0.895 or more. Only BD-GRF6 pulls all four down, to near chance (0.51 to 0.59), and it is cross-corpus. Part of the ceiling may be familiarity with the generators, though the [generator check](#generator-level-contamination-check) finds unseen ones caught as well. The dataset track is moving to harder material: lip-sync with cloned voices, in-the-wild clips, Sinhala and Nepali.
7. **Past a point, bigger doesn't help.** DF Arena 1B beats 500M on IndicTTS (0.988 against 0.978) and loses on Bangla voices (0.993 against 0.998).
8. **Urdu CSALT is the odd one out, a warning about dataset artefacts.** The classic detectors score 0.86 on Urdu CSALT, 0.21 to 0.40 on the IndicSynth, BanglaFake and IndicTTS sets and 0.05 on Bangla voices. IndicSynth's Urdu, the same language from another corpus, gets 0.34, so the gap comes from the corpus, its recordings or its generators (VITS and Tacotron against XTTS-v2 and FreeVC24), not from the language. The [shortcut probe](#the-shortcut-probe) agrees: loudness alone separates Urdu CSALT's real clips from its fakes at 0.935.
9. **Voice conversion is the classic detectors' blind spot.** On IndicSynth Hindi, at the overall EER threshold, XLS-R + AASIST misses 46% of FreeVC24 voice conversions but 3% of XTTS-v2 clips; AASIST misses 92% against 27%. XLSR-Mamba (52% against 4%) and XLS-R + SLS (23% against 2%) lean the same way, AASIST3 the other (2% against 59%). The post-trained detectors miss at most 5% of either.
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
| XLS-R + SLS             | ASVspoof 2019 LA eval | 0.23% (0.16 to 0.27)  | 1.000  | trained on its train split |
| XLS-R + SLS             | In-the-Wild           | 6.73% (6.45 to 7.02)  | 0.981  | 7.46% (paper)           |
| XLSR-Mamba              | ASVspoof 2019 LA eval | 0.79% (0.69 to 0.88)  | 0.9998 | trained on its train split |
| XLSR-Mamba              | In-the-Wild           | 6.29% (5.98 to 6.55)  | 0.981  | 6.71% (paper)           |
| AASIST3                 | ASVspoof 2019 LA eval | 9.47% (9.21 to 9.75)  | 0.968  | weights aren't the paper's |
| AASIST3                 | In-the-Wild           | 30.8% (30.3 to 31.3)  | 0.766  | 28 to 32% (Speech DF Arena) |

Every score direction here is settled. XLS-R + SLS and XLSR-Mamba land slightly better than their papers on In-the-Wild, and AASIST3's released weights match what the Speech DF Arena measured for them. The gaps to the published figures come from protocol differences, explained per model in [Models](models.md#per-model).

Reruns on other machines matched the first round: seven audio models on Urdu CSALT and on IndicTTS within 0.001 with the same counts, GenD, SBI and Xception on HiDF, and Effort on MAVOS-DD across two GPUs within 1e-4.

## Audio: South Asian speech

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

### IndicSynth Bengali and Urdu repeat the Hindi picture

The ASVspoof-only AASIST pair is confidently backwards on Bengali (0.21 and 0.23) and Urdu (0.34 for both); XLS-R + AASIST is in between (0.65 and 0.78); every post-trained model scores 0.995 or more. Of the three models that passed parity last, XLS-R + SLS does best (0.80 and 0.94).

### Bengali is lowest on IndicSynth, mostly because of VITS

IndicSynth makes its Hindi and Urdu fakes with XTTS-v2 and FreeVC24, and its Bengali fakes with those two plus VITS. On the two shared generators alone (each generator's fakes against all the real clips, from the [archived](store_layout.md#archive) per-item scores), the ASVspoof-only AASIST pair still does worst on Bengali: AASIST 0.25 against 0.34 (Urdu) and 0.39 (Hindi), AASIST-L 0.29 against 0.34 and 0.39. The multilingual models barely differ there: XLS-R + AASIST 0.75 against 0.78 and 0.83, XLSR-Mamba 0.78 against 0.79, AASIST3 0.79 against 0.78, and XLS-R + SLS scores Bengali above Urdu (0.95 against 0.94). Their low Bengali totals come from VITS, where they score 0.41 to 0.51.

### VITS is the hardest generator wherever it appears

So the Bengali sets may look hard as much for their generators as for their language. In Urdu CSALT, XLS-R + AASIST scores 0.70 on VITS against 0.91 on Tacotron, and XLSR-Mamba 0.47 against 0.92. BanglaFake is all VITS and MLAAD Bengali half. XLS-R + AASIST scores 0.39 to 0.65 on all five Bengali sets and 0.78 to 0.89 on all five Hindi and Urdu ones (three of the ten are cross-corpus), a gap that generator and language both feed.

### MAVOS-DD's audio is easy for the post-trained models

AntiDeepfake and DF Arena separate MAVOS-DD's Hindi voice conversions almost perfectly, while the ASVspoof-only AASIST pair stays near 0.64. This is insight 6's ceiling again.

## Video: HiDF

Nearly every detector here trained on FF++ and none on HiDF, so its commercial face swaps are a cross-dataset test.

By race, for GenD, SBI and Xception: AUC, and the false-positive rate (FPR) on real faces at each model's global threshold.

| Group  | Real / fake | GenD AUC | GenD FPR | SBI AUC | SBI FPR | Xception AUC | Xception FPR |
| ------ | ----------: | -------: | -------: | ------: | ------: | -----------: | -----------: |
| Asian  | 215 / 206   | 0.963    | 13%      | 0.897   | 21%     | 0.844        | 31%          |
| Black  | 456 / 241   | 0.966    | 6%       | 0.891   | 22%     | 0.789        | 20%          |
| Indian | 90 / 65     | 0.940    | **28%**  | 0.865   | 21%     | 0.782        | 32%          |
| Latino | 194 / 365   | 0.954    | 11%      | 0.879   | 20%     | 0.851        | 21%          |
| White  | 3,406 / 3,484 | 0.951  | 12%      | 0.876   | 20%     | 0.797        | 28%          |

- **Ranking barely moves.** "Indian" has the lowest AUC for all three models, but its intervals are wide (GenD 0.901 to 0.972) and overlap the rest. Across groups GenD spreads 2.6 points and SBI 3.2, so race doesn't meet decision 4's AUC rule.
- **False alarms do.** GenD's 28% on real Indian faces is 25 of 90, where the white group's rate would predict about 10. AUC can't show this. SBI is the most even, about 21% for every group.
- **Provisional.** The group is small and HiDF isn't South Asian. It is a reason to measure false-positive rates on real South Asian faces once the request-only sets arrive, not a result in itself.

## Faces and talking heads: MAVOS-DD Hindi

Every video model with weights on all of HiDF (8,722 clips) and all of MAVOS-DD Hindi video (7,513 clips), the audio-visual models on MAVOS-DD's audio and video together: 18 video models, HAVIC and AVH-Align in the 2026-10-28 round, then FakeSTormer, LAA-Net, UnivFD, LipFD and AVAD by 2026-11-08. From 2026-11-23 the video models are scored on the audio-visual set as well, nine pairs so far. Every pair's time, memory and energy are in `results/ledger.csv`. TPR at 5% FPR is the share of fakes caught when 5% of real clips are flagged.

- **MAVOS-DD Hindi is the hard set.** Every model but FTCN drops from HiDF to MAVOS-DD, by 0.08 (SPSL) to 0.47 (the DFDC winner); FTCN loses 0.01. The eight DeepfakeBench detectors fall to 0.53 to 0.63, close to chance. Most of its fakes are audio-driven talking heads, which frame-level face-swap detectors weren't trained on. It has no lip-sync fakes.
- **Motion over time is what holds up.** FTCN, which looks at short clips rather than single frames, leads MAVOS-DD (0.854) and is the only model there that catches more than half the fakes at a 5% false-alarm rate (0.54). RealForensics, also temporal, is second on HiDF (0.931, behind GenD) and keeps 0.779 on MAVOS-DD. Of the frame-level models, GenD PE-L (0.798) and SBI (0.777) do best.
- **The DFDC winner is backwards on MAVOS-DD.** It scores 0.859 on HiDF but 0.390 on MAVOS-DD: at its EER threshold it flags 59% of real Hindi videos and misses 98% of EchoMimic's and 94% of Sonic's talking heads. It matches upstream within 1e-5 per face crop, so this is the model, not the adapter. GenConViT also falls to chance (0.541).
- **AVH-Align doesn't transfer.** The audio-visual alignment model, trained on AV-Deepfake1M (English), sits at chance on MAVOS-DD (0.520): it catches 92% of Roop face swaps and 87% of MEMO talking heads, but misses 76% of Sonic's and 61% of LivePortrait's. Its parity check passes, so this is how it behaves on Hindi.
- **Talking heads are the frame models' blind spot, and the temporal and audio-visual models' strength.** At each model's EER threshold, GenD PE-L misses 78% of EchoMimic's audio-driven talking heads while catching 82 to 95% of face swaps. FTCN catches every EchoMimic clip and 98% of LivePortrait's, though it still misses about half of MEMO's and Sonic's. HAVIC, which reads audio and video together, misses only 4 to 6% of the talking heads but about half the swaps, so it sits mid-table overall (0.692) with LipForensics (0.703). The two cover each other's gaps (miss rates per generator are in each run's `by_method.csv`).
- **The pattern holds across the 18 video models of the 2026-10-28 round.** Share of each generator's fakes missed at each model's own EER threshold on MAVOS-DD Hindi, averaged over the 18:

  | Generator | Type | Mean miss | Worst model |
  | --------- | ---- | --------: | ----------- |
  | EchoMimic | audio-driven talking head | 76% | DFDC winner, 98% |
  | MEMO | audio-driven talking head | 71% | DFDC winner, 88% |
  | Sonic | audio-driven talking head | 63% | DFDC winner, 94% |
  | Roop | face swap | 33% | FTCN, 61% |
  | HifiFace | face swap | 28% | DFDC winner, 52% |
  | InSwapper | face swap | 26% | SPSL, 47% |
  | LivePortrait | face reenactment | 15% | Effort, 57% |

  Every model misses more talking heads than face swaps on average, and 16 of the 18 miss each talking-head generator more often than any face swap. MAVOS-DD's fakes are mostly talking heads, so this set alone can't separate "Hindi" from "talking head"; the [language control](#the-language-control-mavos-dd-german) does.
- **AVAD ranks MAVOS-DD's fakes below its real videos.** Its AUC of 0.433 (95% CI 0.419 to 0.445) is below chance. Like AVH-Align it learned only from real talking faces, and neither finds MAVOS-DD Hindi's fakes stranger than its real clips.
- **The SBI-style detectors don't carry SBI's lead onto MAVOS-DD.** LAA-Net, trained like SBI on self-blended FF++ faces, scores 0.839 on HiDF but 0.633 on MAVOS-DD, and FakeSTormer 0.772 and 0.577, against SBI's 0.877 and 0.777.
- **LipFD, the lip-sync specialist, barely separates MAVOS-DD's fakes (0.544) and misses the talking heads most.** At its equal-error threshold it flags 82% of LivePortrait and 84% of Roop fakes but only 15% of EchoMimic, 33% of Sonic and 40% of MEMO, the audio-driven talking heads; it learned Wav2Lip-style mouth edits over English LRS3, not diffusion talking heads in Hindi. MAVOS-DD has no lip-sync fakes, so this says nothing about how LipFD does on its own type.
- **UnivFD is the one model better on MAVOS-DD than on HiDF.** The generic AI-image detector, which never saw a face swap, scores 0.685 on HiDF and 0.710 on MAVOS-DD, where many fakes are fully generated talking heads rather than swapped faces.
- **On the audio-visual set, the video models treat voice-only fakes as real, as they must.** The nine pairs scored so far (CORE, F3Net, SRM, FFD, Effort and GenConViT on the full set; CORE, F3Net and RealForensics on the quick share) each score a little below the same model on the video set. At each model's EER threshold they flag 33 to 48% of the kNN-VC clips, whose video is untouched, against 37 to 51% of real videos. kNN-VC is about a fifth of the audio-visual set's fakes, so the drop is the expected one. The strong video models (FTCN, GenD PE-L, SBI) are still to come.
- **The audio-visual models don't catch the voice-only fakes either.** In their October runs, HAVIC flags 40% of kNN-VC clips against 38% of real videos, and AVH-Align (51% against 48%), LipFD (47% and 47%) and AVAD (49% against 55%) are no better. None of the four separates a converted voice over untouched video from a real clip, so on MAVOS-DD a voice-only fake needs an audio detector: the post-trained ones separate MAVOS-DD's kNN-VC clips from its real ones almost perfectly.

## The language control: MAVOS-DD German

MAVOS-DD's German subset has the same eight generators as its Hindi one, so it tells "Hindi" apart from "a hard set of fakes". Waves `s6_language_control` to `s6d_effort` (SPARK, 2026-11-08 to 13) scored all 21 video models with weights on its quick share, up to 300 videos per generator (2,701 scored), and on MAVOS-DD Hindi's quick share with the same cap and seed (2,494). Both are columns of the grid, marked quick.

- **The drop isn't about Hindi.** 18 of the 21 models score lower on German than on Hindi, by 0.05 (GenD PE-L, the DFDC winner) to 0.19 (FFD). LAA-Net scores the same, and two score higher on German: FakeSTormer by 0.02, inside the intervals, and Effort by 0.05. What makes MAVOS-DD hard is the set's fakes and its real YouTube clips, not South Asian speakers. HiDF stays the only sign of a South Asian effect in video.
- **On German the frame detectors rank fakes below real videos.** The nine DeepfakeBench detectors released with v1.0.1 and GenConViT score 0.35 to 0.44, below chance with their confidence intervals clear of 0.5, and each flags 54 to 67% of German real videos at its equal-error threshold, against 43 to 49% of Hindi's: the German real clips look faker to them than the fakes do. The DFDC winner, 0.859 on HiDF, is below chance in both languages (0.324 and 0.373). A pattern, not proof of inverted scores, while video direction is unsettled.
- **The generator pattern holds in both languages.** Averaged over EchoMimic, Sonic and MEMO, the frame detectors miss 72 to 93% of German talking heads (54 to 83% in Hindi) while catching most face swaps; FTCN and RealForensics still catch nearly every LivePortrait reenactment (3 to 4% missed). The ranking at the top is the same: GenD PE-L, FTCN and SBI lead in both.
- **Effort, the strongest CLIP detector on HiDF after GenD (0.919), drops to 0.54 to 0.59 here,** like the other FF++-trained models; GenD PE-L keeps more of its lead.

## Generator-level contamination check

Why DF Arena (500M, 1B) and AntiDeepfake (MMS-300M, wav2vec 2.0 small) score 0.978 or more on almost every open South Asian audio set: the [contamination guard](evaluation.md#contamination-verdicts) compares datasets, so a set made with a TTS system these models trained on still counts as clean. Desk check of 2026-11-08: each evalset's generators, from its paper or card, against the generators in every corpus the four models list ([Models](models.md#per-model)).

| Evalset | Generators | Seen in training |
| --- | --- | --- |
| IndicSynth (bn, hi, ur) | XTTS-v2 (base, and fine-tuned on IndicSUPERB for bn and ur), Coqui VITS (bn), FreeVC24 | XTTS-v2 exactly (MLAAD, ASVspoof 5 A29); Bengali VITS probably (MLAAD `bn` VITS); FreeVC24 not found |
| BanglaFake | VITS trained on the SUST corpus | family only (VITS in MLAAD, ASVspoof 5, SpoofCeleb) |
| Urdu CSALT | Tacotron and VITS trained on PRUS | family only |
| MAVOS-DD Hindi audio | kNN-VC | not found |
| SpeechFake (5 languages) | Edge-TTS; SeedVC in Hindi | Edge-TTS is in MLAAD, version unknown; SeedVC not found |
| MLAAD (bn, hi, si) | Bark, XTTS-v2, Coqui `bn` VITS, Edge-TTS (si), and in Hindi IndicF5, Veena, Indri, Ringg Squirrel, Voxtral, OmniVoice, VoxCPM2 | the older systems likely; the store holds MLAAD v11 (2026-08), the models saw about v5 to v8, and Voxtral, OmniVoice and VoxCPM2 (March and April 2026) postdate both models |
| Bangla voices, IndicTTS challenge, BD-GRF6 | unstated | can't be judged |

- **Exact matches are few; family matches are everywhere.** XTTS-v2 in IndicSynth is the clear case. Most other sets share only an architecture (VITS, Tacotron) with something in training, trained on other speakers and languages.
- **Familiarity doesn't explain the ceiling by itself.** The generators found in no training corpus are caught as well as XTTS-v2: at its EER threshold DF Arena 1B misses 0% of IndicSynth Bengali's XTTS-v2 fakes, 0.9% of FreeVC24's and 0.7% of VITS's; DF Arena 500M misses 1.6% of SpeechFake Hindi's SeedVC and 13% of its Edge-TTS; DF Arena 1B misses 0.1% of MAVOS-DD's kNN-VC (`by_method.csv` of each run). The voice-conversion lists of four AntiDeepfake corpora (DSD, CVoiceFake, DECRO, DiffSSD) couldn't be read, so "not found" is provisional there. Every fake here still passes through a neural vocoder of a kind these models trained on.
- **The clean test is on disk but behind the guard.** MLAAD Hindi's Voxtral, OmniVoice and VoxCPM2 folders are systems no model here can have seen, scored against the same real speech as the older ones. The guard refuses MLAAD for all four models, rightly at the dataset level, so per-generator numbers need the team to agree a documented carve-out ([Progress](progress.md#open-decisions-that-shape-the-results)). SEA-Spoof (Hindi and Tamil, ten named generators, five of them likely unseen, on request) is the other route.

## Cross-corpus audio

Scored and recorded, but not results. The guard refuses MLAAD for the five models trained on it. SpeechFake's Bengali, Tamil, Marathi and Malayalam fakes run against real read speech from IndicSUPERB, because SpeechFake has no real speech in those languages; its Hindi fakes run against its own Common Voice Hindi.

- **XLS-R + SLS carries over best** on MLAAD, and is the only model that catches more than a third of MLAAD Bengali's fakes at a 5% false-alarm rate.
- **The ASVspoof-only AASIST pair is backwards on MLAAD Bengali and Hindi** (0.19 to 0.41), as on IndicSynth Bengali, but not on Sinhala. MLAAD Sinhala has 11,550 real clips to 1,000 fakes, and AASIST-L's 0.812 there catches only 3% of fakes at 5% FPR.
- **SpeechFake splits the models, and the split is the warning.** All four post-trained models score 0.978 or more on all five languages, 1.000 on most, while the other six run from 0.148 to 0.986 on the same clips. The real and fake clips come from different corpora, so a score of 1.000 here may be the corpus and not the fake.
- **The same set runs backwards for the classic models.** AASIST, AASIST-L, XLS-R + AASIST and XLSR-Mamba all fall below 0.33 on Hindi and Malayalam, the pattern insight 4 describes.
- **BD-GRF6 defeats every model but AASIST3**, which scores 0.952 and catches 71% of its fakes at 5% FPR. The next best is AASIST-L at 0.709; the post-trained AntiDeepfake and DF Arena models sit at chance (0.51 to 0.59). AASIST3 may be telling the recordings apart rather than the fakes: BD-GRF6's fakes were recorded at 8 kHz ([below](#the-shortcut-probe)).

## The shortcut probe

Run on SPARK on 2026-11-13 (deeptrace-experiments' `shortcut` stage; aggregates in `results/shortcut/`): a logistic regression on six trivial features ([Evaluation](evaluation.md#shortcut-probe)), on every cross-corpus audio set and two same-corpus baselines. The last column is the best detector's AUC on the same set.

| Evalset | Pairing | Probe AUC | Mean sample rate, fake / real | Strongest other feature | Best detector |
| --- | --- | --: | --- | --- | --: |
| MLAAD Bengali | cross-corpus | 1.000 | 22.05 / 44.1 kHz | duration 0.829 | 0.791 |
| MLAAD Sinhala | cross-corpus | 1.000 | 22.05 / 16 kHz | bandwidth 0.963 | 0.997 |
| MLAAD Hindi | cross-corpus | 1.000 | 22.05 / 44.1 kHz | duration 0.752 | 0.877 |
| BD-GRF6 | cross-corpus | 1.000 | 8 / 37.5 kHz | bandwidth 0.770 | 0.952 |
| SpeechFake Bengali | cross-corpus | 1.000 | 16 / 44.1 kHz | silence 0.878 | 1.000 |
| SpeechFake Malayalam | cross-corpus | 1.000 | 16 / 44.1 kHz | duration 0.988 | 1.000 |
| SpeechFake Marathi | cross-corpus | 1.000 | 16 / 44.1 kHz | duration 1.000 | 1.000 |
| SpeechFake Tamil | cross-corpus | 1.000 | 16 / 44.1 kHz | silence 0.930 | 1.000 |
| SpeechFake Hindi | cross-corpus | 0.875 | 16 / 16 kHz | rolloff 0.814 | 0.999 |
| BanglaFake | same-corpus | 0.953 | 22.05 / 24.2 kHz | silence 0.855 | 0.999 |
| Urdu CSALT | same-corpus | 0.945 | 18 / 16.1 kHz | loudness 0.935 | 0.998 |

- **On every cross-corpus set but SpeechFake Hindi, the file's sample rate alone separates real from fake.** The detectors hear everything resampled to 16 kHz, so a rate above that leaves little direct trace; an 8 kHz original does: BD-GRF6's fakes carry nothing above 4 kHz, which bandwidth partly picks up (0.770).
- **What the detectors can hear separates them almost as well.** Leaving the sample rate out, one feature alone reaches 0.75 to 1.00 on every cross-corpus set: clip length on SpeechFake Marathi and Malayalam, bandwidth on MLAAD Sinhala, silence on SpeechFake Tamil. A near-perfect detector score on these sets may measure the corpora, not the fakes. A probe on the audible features alone hasn't been run; the strongest single feature is its lower bound.
- **The same-corpus baselines are high too** (0.953 and 0.945), so a high probe score doesn't by itself mark a set unusable, and no cut-off follows from the probe alone. Urdu CSALT's loudness gap fits the classic detectors' outlying 0.86 there ([insight 8](#insights)).
- **Which E5 numbers count is the team's call** ([Progress](progress.md#next)): drop the sets a trivial feature separates as well as the detectors do, report each detector beside the probe, or match length, loudness and bandwidth before scoring.

## What the results can't say yet

- **Lip-sync with cloned voices**, v1's focus: no evalset has lip-sync fakes yet. MAVOS-DD Hindi's are talking heads, face swaps, reenactment and voice conversion ([above](#faces-and-talking-heads-mavos-dd-hindi)), so the talking-head blind spot says nothing yet about lip-sync.
- **Sinhala.** Its only fakes are in MLAAD Sinhala, which the shortcut probe separates by sample rate and bandwidth alone ([above](#the-shortcut-probe)), so no Sinhala number counts yet.
- **In-the-wild South Asian audio.** Every South Asian audio set so far is read or studio speech.
- **Demographics in audio.** These sets carry gender at most.
