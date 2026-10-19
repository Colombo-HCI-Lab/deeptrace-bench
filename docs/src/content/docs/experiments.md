---
title: "Experiments"
description: "Each experiment's question, what ran, and the command to run it again"
group: findings
order: 1
---

Each experiment is a set of model and evalset pairs that answers one question. Read this to see what has run and how to repeat it; numbers are in [Results](results.md), dates in [Progress](progress.md).

## How an experiment runs

- **Evalsets, not datasets.** An evalset fixes which clips are real and fake and how they pair. Only same-corpus evalsets (real and fake from the same recordings or speakers) feed headline numbers ([evaluation.md](evaluation.md#evalsets-not-datasets)).
- **One setup**, [`configs/eval/default.yaml`](/configs/eval/default.yaml): 16 kHz audio in 4-second windows, 32 face frames per video, a clip scored by the mean over them. Still `proposed`, so numbers can move.
- **Parity before results.** Numbers count once the adapter scores like the upstream code ([models.md](models.md#parity)) and the score direction (higher means fake) checks out on a reproduction set.
- **Contamination guard.** Pairs the guard can't vouch for, because the model may have trained on them, are refused and recorded, never forced. `source_overlap` flags a shared source corpus such as Common Voice.
- **Run ids** hash model, evalset, weights and setup: change any of them and a new run starts; rerun a pair unchanged and it resumes.

## E1: Reproductions

**Question:** under the shared setup, do our adapters reproduce the published figures, with scores pointing the right way?

| Model                                     | Reproduction evalset           | Published figure          |
| ----------------------------------------- | ------------------------------ | ------------------------- |
| AASIST, AASIST-L                          | ASVspoof 2019 LA eval          | paper EER                 |
| XLS-R + AASIST                            | ASVspoof 2019 LA eval, In-the-Wild | literature EER on In-the-Wild |
| DF Arena 500M and 1B                      | In-the-Wild                    | model card EER            |
| AntiDeepfake MMS-300M and w2v2 small      | In-the-Wild                    | model card EER            |
| Video models (Xception, SBI, GenD, others) | FF++ test or Celeb-DF v2      | waiting on access requests |

A reproduction counts when AUC is above 0.5 (the direction check) and EER is close to the published one; otherwise the gap is examined first. DF Arena and AntiDeepfake trained on ASVspoof 2019 LA, so the guard refuses them there and In-the-Wild stands in.

**Headline:** every audio score direction is settled ([Results](results.md#reproductions)).

## E2: Zero-shot audio on South Asian speech

**Questions:** do detectors trained only on English spoofing data (ASVspoof 2019 LA) hold up on South Asian synthetic speech? Does multilingual pretraining help? Do language and gender shift the scores? The two AntiDeepfake models are the controlled pair for pretraining: same post-training data, English-only (w2v2 small) against about 1,400 languages (MMS-300M).

**Models:** AASIST, AASIST-L and XLS-R + AASIST (ASVspoof 2019 LA only); AntiDeepfake MMS-300M and w2v2 small, DF Arena 500M and 1B (post-trained on large multilingual collections of real and fake speech).

**Evalsets**, all same-corpus: `urdu_csalt`, `banglafake`, `bangla_voices`, `indictts_challenge`, `indicsynth_hi` (details in [Results](results.md#the-datasets)). IndicSynth Hindi uses a seeded 5,000 fakes per generator: all 205,938 would outnumber its real side about 40 to 1 and cost about a day of GPU time per large model, without changing AUC or EER.

**Headline:** training data decides. AASIST and AASIST-L fall below chance on four of the five sets, XLS-R + AASIST is uneven, and the multilingually trained models are near the ceiling ([Results](results.md#three-tiers)).

## E3: Zero-shot video on HiDF

**Question:** does a detector's accuracy, or its false-positive rate on real faces, differ by race?

**Models:** Xception (a 2019 baseline), SBI (self-blended images, 2022) and GenD (CLIP-based, 2026), all trained on FF++.

**Evalset:** `hidf_video`, face-swapped and real videos of about 3 seconds, every face labelled for race, gender and age ([Results](results.md#the-datasets)). Not South Asian, but its "indian" group (90 real, 65 fake) can be compared with the others.

**Caveats:** no video model has a settled score direction (that needs FF++ or Celeb-DF v2) and GenD has no parity check, so the numbers are provisional; the runs prove the video path end to end.

**Headline:** AUC barely moves by race, but GenD's false-positive rate on real Indian faces doubles ([Results](results.md#video-hidf)).

## Planned

- **E4: talking heads on MAVOS-DD Hindi**, the only open South Asian video: face swaps, audio-driven talking heads (echomimic, sonic, memo) and voice conversion. Tests v1's focus, lip-sync with cloned voices. Waiting on its Hugging Face terms.
- **E5: cross-corpus sets** (MLAAD, SpeechFake, BD-GRF6, `mlaad_si`). Unreadable until the shortcut probe shows whether a trivial classifier can tell their real and fake sources apart.
- **E6: fine-tuned baselines** on identity-safe splits ([evaluation.md](evaluation.md#fine-tuning-splits)).
- **E7: deeptrace's own data**, once collection starts.

## Running one again

Score in shards, then evaluate. Two shards keep one GPU busy when audio is read from shared storage:

```bash
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 0/2
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 1/2
uv run scripts/evaluate.py --model xlsr_aasist --evalset bangla_voices
```

On one 24 GB GPU with two shards, XLS-R-sized audio models score about 30 clips a second, w2v2 small about 60, DF Arena 1B about 8. Video is bound by CPU face detection: the first model on HiDF took about 4.5 hours, and later ones reuse the cached detections. On a SLURM cluster the same steps run as array jobs ([cluster.md](cluster.md)).
