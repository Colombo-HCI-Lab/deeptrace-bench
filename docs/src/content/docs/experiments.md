---
title: "Experiments"
description: "What has been run, the question each run answers, and how to run it again"
group: findings
order: 1
---

Each experiment is a set of model and evalset pairs that answers one question. Numbers are in [Results](results.md), the chronology in [Progress](progress.md).

## How an experiment runs

- Models are scored only on **evalsets**, which fix which clips are real and fake and how they pair ([evaluation.md](evaluation.md#evalsets-not-datasets)). Only same-corpus evalsets (real and fake from the same recordings or speakers) feed headline numbers.
- Every run uses the **one evaluation setup**, [`configs/eval/default.yaml`](/configs/eval/default.yaml): 16 kHz audio in 4-second windows, 32 face frames per video, a clip's score the mean over its windows or frames. Still `proposed`, so every number can move.
- **Parity before results.** Numbers count only once the adapter scores like upstream code ([models.md](models.md#parity)) and the score direction checks out on a reproduction set.
- The **contamination guard** checks each pair against everything the model trained on and refuses pairs it can't vouch for. Refusals are recorded, never forced.
- Run ids hash model, evalset, weights and setup, so changing any of them starts a new run.

## E1: Reproductions

**Question:** under the shared setup, do our adapters reproduce the published figures, with scores pointing the right way?

| Model                                     | Reproduction evalset           | Published figure          |
| ----------------------------------------- | ------------------------------ | ------------------------- |
| AASIST, AASIST-L                          | ASVspoof 2019 LA eval          | paper EER                 |
| XLS-R + AASIST                            | ASVspoof 2019 LA eval, In-the-Wild | literature EER on In-the-Wild |
| DF Arena 500M and 1B                      | In-the-Wild                    | model card EER            |
| AntiDeepfake MMS-300M and w2v2 small      | In-the-Wild                    | model card EER            |
| Video models (Xception, SBI, GenD, others) | FF++ test or Celeb-DF v2      | waiting on access requests |

DF Arena and AntiDeepfake trained on ASVspoof 2019 LA, so the guard refuses them on its eval set; In-the-Wild stands in. A reproduction counts when AUC is above 0.5 (the direction check) and EER is close to the published one. Otherwise the gap is examined before the model's other numbers are read.

## E2: Zero-shot audio on South Asian speech

**Questions:**

1. Do detectors trained only on English spoofing data (ASVspoof 2019 LA) hold up on South Asian synthetic speech?
2. Does multilingual pretraining help? The two AntiDeepfake models are the controlled pair: same post-training data, English-only (wav2vec 2.0 small) against about 1,400-language (MMS-300M) pretraining.
3. Do language and gender shift the scores?

**Models:** AASIST, AASIST-L and XLS-R + AASIST (trained on ASVspoof 2019 LA only); AntiDeepfake MMS-300M and w2v2 small, DF Arena 500M and 1B (post-trained on large multilingual collections of real and fake speech).

**Evalsets**, all same-corpus:

| Evalset              | Language                            | Real   | Fake   | Fakes made by                     | Groups             |
| -------------------- | ----------------------------------- | -----: | -----: | --------------------------------- | ------------------ |
| `urdu_csalt`         | Urdu                                | 3,398  | 3,396  | VITS, Tacotron                    | method             |
| `banglafake`         | Bengali                             | 12,796 | 12,796 | VITS                              | sub-corpus         |
| `bangla_voices`      | Bengali                             | 2,250  | 2,250  | not named; paired per speaker and sentence | gender    |
| `indictts_challenge` | 16 Indian languages, Nepali among them | 15,558 | 15,544 | TTS, not named                 | language, gender   |
| `indicsynth_hi`      | Hindi                               | 5,059  | 10,000 | XTTS-v2, FreeVC24 (voice conversion) | method, gender  |

IndicSynth's Hindi fakes are a seeded sample of 5,000 per generator out of 205,938. The full set would outnumber its real side (IndicSUPERB's Hindi test clips) about 40 to 1 and cost about a day of GPU time per large model, changing neither AUC nor EER.

`source_overlap` marks pairs where a model's pretraining or post-training shares a source corpus (such as Common Voice) with the evalset. Those numbers are reported with the flag.

## E3: Zero-shot video on HiDF

**Question:** does a detector's accuracy, or its false-positive rate on real faces, differ by race?

**Models:** Xception (a 2019 baseline), SBI (self-blended images, 2022) and GenD (CLIP-based, 2026), all trained on FF++.

**Evalset:** `hidf_video`, 4,361 real and 4,361 face-swapped videos of about 3 seconds, every face labelled for race, gender and age. Not South Asian; included because its "indian" group (90 real and 65 fake videos) can be compared with the others.

**Caveats:** no video model has a settled score direction (that needs FF++ or Celeb-DF v2), and GenD has no parity check. These runs prove the video path end to end on real data; the numbers are provisional.

## Planned

- **E4: talking heads on MAVOS-DD Hindi**, the only open South Asian video: face swaps, audio-driven talking heads (echomimic, sonic, memo) and voice conversion. Tests v1's focus, lip-sync with cloned voices, and is the first data likely to trouble the strongest audio models. Waiting on its Hugging Face terms.
- **E5: cross-corpus sets behind the shortcut probe** (MLAAD, SpeechFake, BD-GRF6, `mlaad_si`). Unreadable until the probe shows whether a trivial classifier can tell the two sources apart.
- **E6: fine-tuned baselines**, on identity-safe splits ([evaluation.md](evaluation.md#fine-tuning-splits)).
- **E7: deeptrace's own data**, once collection starts.

## Running one again

Score in shards, then evaluate. Two shards keep one GPU busy when audio is read from shared storage:

```bash
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 0/2
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 1/2
uv run scripts/evaluate.py --model xlsr_aasist --evalset bangla_voices
```

Rerunning a pair resumes its run. On one 24 GB GPU with two shards, XLS-R-sized audio models score about 30 clips a second, w2v2 small about 60, DF Arena 1B about 8. For video, CPU face detection dominates: the first model on HiDF took about 4.5 hours; later models reuse the cached detections. On a SLURM cluster the same steps run as array jobs ([cluster.md](cluster.md)).
