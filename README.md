# deeptrace-bench

Benchmark harness for deepfake detection in South Asia. It runs open video and audio detectors on South Asian deepfake datasets under one evaluation setup and reports results per group, so numbers from different datasets and models can be compared.

Status: set up on 2026-10-03, no code yet.

## Scope

1. Run existing detectors on existing South Asian datasets: reproduce their published numbers, then test them zero-shot and find the attributes behind their misses.
2. Benchmark the existing datasets and models against each other, then fine-tune the detectors on the existing South Asian sets as the baseline new data must beat.
3. Evaluate and fine-tune on the data collected through [deeptrace](https://github.com/Colombo-HCI-Lab/deeptrace). This harness becomes the benchmark released with the dataset.

## Evaluation

Every model and dataset runs through the same setup: the same face detection and crop and a fixed number of frames per clip for video, fixed-length 16 kHz segments for audio, one score per clip, and AUC, EER and the gap between groups as the metrics. The exact settings are fixed before the first comparison run and recorded here.

## Data stays out of git

Datasets, model weights, checkpoints, run outputs and participant data are never committed (see `.gitignore`). Most South Asian dataset licences are non-commercial research only and can't be passed on, so downloaded data lives in one shared store on the compute where the runs happen. Data collected through deeptrace stays in deeptrace's own storage.

## Development

A [uv](https://docs.astral.sh/uv/) project on Python 3.12.

```bash
uv sync
```
