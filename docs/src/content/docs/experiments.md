---
title: "Experiments"
description: "What has been run, the question each run answers, and how to run it again"
group: findings
order: 1
---

Each experiment below is a set of model and evalset pairs that answers one question. The numbers are in [Results](results.md); the order things happened in is in [Progress](progress.md).

## How an experiment runs

- A model is scored only on an **evalset**, which says which clips count as real and fake and how the two were paired ([evaluation.md](evaluation.md#evalsets-not-datasets)). Only same-corpus evalsets, where real and fake come from the same recordings or speakers, feed headline numbers.
- Every run uses the **one evaluation setup** in [`configs/eval/default.yaml`](/configs/eval/default.yaml): 16 kHz audio cut into 4-second windows, 32 face frames per video, a clip's score the mean of its windows or frames. It is still `proposed`, so every number so far can move.
- **Parity before results.** A model's numbers count only once its adapter scores like its upstream code ([models.md](models.md#parity)) and its score direction is right on a reproduction set.
- The **contamination guard** checks each pair against everything the model trained on and refuses pairs it can't vouch for. Refused pairs are recorded as refused, never forced.
- A run's id hashes the model, evalset, weights and setup, so changing any of them starts a new run rather than mixing results.

## E1: Reproductions

**Question:** do our adapters, under the shared setup, reproduce what each model's authors published, and do their scores point the right way?

| Model                                     | Reproduction evalset           | Published figure          |
| ----------------------------------------- | ------------------------------ | ------------------------- |
| AASIST, AASIST-L                          | ASVspoof 2019 LA eval          | paper EER                 |
| XLS-R + AASIST                            | ASVspoof 2019 LA eval, In-the-Wild | literature EER on In-the-Wild |
| DF Arena 500M and 1B                      | In-the-Wild                    | model card EER            |
| AntiDeepfake MMS-300M and w2v2 small      | In-the-Wild                    | model card EER            |
| Video models (Xception, SBI, GenD, others) | FF++ test or Celeb-DF v2      | waiting on access requests |

DF Arena and AntiDeepfake trained on ASVspoof 2019 LA, so the guard refuses them on its eval set and In-the-Wild stands in. A reproduction counts when the AUC is above 0.5 (the direction check) and the EER is close to the published one. Where it isn't, the gap is examined before the model's other numbers are read.

## E2: Zero-shot audio on South Asian speech

**Questions:**

1. Do detectors trained only on English spoofing data (ASVspoof 2019 LA) hold up on South Asian synthetic speech?
2. Does multilingual pretraining help? The two AntiDeepfake models are the controlled comparison: the same post-training data, with English-only (wav2vec 2.0 small) against about 1,400-language (MMS-300M) pretraining.
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

IndicSynth's Hindi fakes are a seeded sample of 5,000 per generator out of 205,938. The full set would outnumber its real side, IndicSUPERB's Hindi test clips, about 40 to 1, and cost about a day of GPU time per large model while changing neither AUC nor EER.

The guard returned `source_overlap` for some pairs, where a model's pretraining or post-training shares a source corpus (such as Common Voice) with the evalset. Those numbers are reported with the flag.

## E3: Zero-shot video on HiDF

**Question:** does a detector's accuracy, or its false-positive rate on real faces, differ by race?

**Models:** Xception (a 2019 baseline), SBI (self-blended images, 2022) and GenD (CLIP-based, 2026), all trained on FF++.

**Evalset:** `hidf_video`, 4,361 real and 4,361 face-swapped videos of about 3 seconds, every face labelled for race, gender and age. HiDF isn't a South Asian dataset: it is here because its "indian" group (90 real and 65 fake videos) can be compared with the other groups.

**Caveats:** no video model has a settled score direction yet (that needs FF++ or Celeb-DF v2), and GenD has no parity check. These runs show the video path works end to end on real data; their numbers are provisional.

## Planned

- **E4: talking heads on MAVOS-DD Hindi**, the only open South Asian video, with face swaps, audio-driven talking heads (echomimic, sonic, memo) and voice conversion. It tests v1's focus, lip-sync with cloned voices, and is the first data likely to trouble the strongest audio models. Waiting on its Hugging Face terms.
- **E5: cross-corpus sets behind the shortcut probe** (MLAAD, SpeechFake, BD-GRF6, `mlaad_si`). Their numbers can't be read until the probe shows whether a trivial classifier can tell the two sources apart.
- **E6: fine-tuned baselines**, on identity-safe splits ([evaluation.md](evaluation.md#fine-tuning-splits)).
- **E7: deeptrace's own data**, once collection starts.

## Running one again

Each pair is scored in shards and then evaluated. Two shards keep one GPU busy when the audio is read from shared storage:

```bash
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 0/2
uv run scripts/score.py --model xlsr_aasist --evalset bangla_voices --device cuda --shard 1/2
uv run scripts/evaluate.py --model xlsr_aasist --evalset bangla_voices
```

Rerunning the same pair resumes the same run. Observed on one 24 GB GPU with two shards: the XLS-R-sized audio models score about 30 clips a second, w2v2 small about 60, DF Arena 1B about 8. For video, CPU face detection dominates: the first model on HiDF took about 4.5 hours, and later models reuse the cached detections. On a SLURM cluster the same steps run as array jobs ([cluster.md](cluster.md)).
