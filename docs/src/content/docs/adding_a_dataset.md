---
title: "Adding a dataset"
description: "Recipe: config, builder, sampling, evalset (the two pipeline-test datasets as worked examples)"
---

In plain words: adding a dataset means teaching the harness to find every clip and know whether it is real or fake, with nobody labelling anything by hand. You describe where the data comes from, its licence and what it was made from ([step 1](#1-config)), write a builder that reads each clip's label from its path or the dataset's metadata and lists every clip in a manifest ([step 2](#2-builder)), try it on a few clips before downloading the lot ([step 3](#3-sample-it-then-build-the-manifest)), say exactly which clips get scored and how real and fake pair up in an evalset ([step 4](#4-evalset)), and run the smoke test ([step 5](#5-prove-it-runs)). Reading a label from a path before anything is downloaded is what makes sampling possible, so a new dataset can be tried in minutes.

A dataset is a config, a builder that turns its files into a manifest, and at least one evalset saying how it is scored. Worked examples: the [pipeline-test datasets](getting_started.md#the-pipeline-test-datasets) `mendeley_roop_akool` (a zip at a URL, image items) and `unidatapro_videos` (Hugging Face, video items).

## 1. Config

Add `configs/datasets/<id>.yaml`:

- `access`: `hf` / `hf_gated` with `repo` and a pinned `revision` (and `allow_patterns`); `url` with `urls` (and `sha256` per archive); `manual` with `request` instructions and `expect` paths for `--from`. One zip inside a Hugging Face repo (BanglaFake, In-the-Wild) is `url` access to the zip's pinned `resolve/<revision>/` URL with the Hub's LFS sha256: the full fetch is hash-checked, and a sample reads members by range requests.
- `licence` and `licence_terms`, `contains` (`real`, `fake`), `languages`, `size_gb`.
- `derived_from`: every dataset or corpus its items came from, for the contamination guard.
- `role`: `reproduction`, `south_asian`, `real_reference`, `in_the_wild`, or `pipeline_test` (never mixed into a real evalset).
- `notes`: facts, each with the date checked.

## 2. Builder

Write `src/deeptrace_bench/datasets/<id>.py` against the real files:

```python
def label_from_path(rel_path: str) -> str | None:
    """The label from the path alone: real, fake, or None for anything that isn't an item."""


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per item under root, in the schema of deeptrace_bench.manifest."""
```

- `label_from_path` makes a dataset samplable: the sampler calls it on the remote file list before downloading and `build_manifest` on the local files, so the two can't disagree. Return None for metadata, archive junk (`__MACOSX/`) and anything else that isn't an item.
- Labels in a file, not the path (In-the-Wild's `meta.csv`, ASVspoof's protocols): define `METADATA_FILES` (relative paths) and `labels_from_metadata(files: dict[str, bytes]) -> {rel_path: label}` instead. The sampler reads those files first and copies them into the sample; `build_manifest` reads the same files. A builder that only reads speakers from a file can list it in `METADATA_FILES` next to `label_from_path` so samples carry it.
- Audio embedded in parquet rows (IndicSynth, IndicTTS challenge): items are rows. Define `ROW_COLUMNS` and `label_from_row(row) -> label` (optionally `row_stratum`). The sampler reads those columns from a few shards over HTTP, picks rows, fetches only the row groups holding them and writes each shard back with just those rows plus their `__source_row` numbers. `build_manifest` reads rows with `datasets._parquet.iter_rows` and writes each clip once with `write_audio`, so item ids match between a sample and the full copy.
- Archives inside an archive (BD-GRF6's per-class zips): define `nested_zip_label(member)`. The sampler fetches the smallest inner zip per label and samples its members; `build_manifest` extracts inner zips itself. Plain metadata files next to zips (HiDF's `metadata.csv`) can go in `access.urls` too. Tar-only datasets (IndicSUPERB) can't be sampled remotely; samples come from a full local copy.
- Single-label datasets (MLAAD fake only, OpenSLR Sinhala real only) sample on the labels their config `contains`. Where one kind of item is rare, `sample_stratum(rel_path)` takes N items per stratum: MAVOS-DD samples per generator, so its voice-conversion fakes are always in.
- `build_manifest` must work on a partial tree (a sample): iterate with `datasets.iter_files` (skips hidden files such as `.sample.json`) and never assume the full count.
- Columns: `item_id` (`<id>/<local path without extension>`), `dataset`, `rel_path`, `modality` (`video`, `image`, `audio`, `audio_video`), `label`; optional `method`, `method_family`, `language`, `subject_id`, `source_subject_id`, `split`, `duration_s`; for audio-video items, `label_video` and `label_audio` (which track was changed; null where nobody has checked); group attributes as `g_<attr>` plus `g_<attr>_src`. Unknown values are null, never "unknown".
- Document the layout as checked, with the date, in the module docstring. Facts only: if the dataset doesn't say which generator is which, keep neutral names (Mendeley's `tech_1`, `tech_2`).

## 3. Sample it, then build the manifest

```bash
uv run scripts/setup_datasets.py <id> --sample 2 --manifest     # smoke namespace, a few MB
uv run scripts/setup_datasets.py <id> --manifest                # the full dataset
uv run scripts/setup_datasets.py <id> --from /path/to/copy --manifest   # request-only datasets
```

`--sample N` picks N items per label (per stratum, if the builder defines them) deterministically by a seeded hash of each path, fetches only those (individual Hugging Face files, listing only folders the patterns can match; zip members by HTTP range; or symlinks into a `--from` copy) and records the choice in `.sample.json`. It always writes to the smoke namespace.

## 4. Evalset

Add `configs/evalsets/<id>.yaml`: components (dataset, filter, label), `pairing` (`same_corpus` or `cross_corpus`), `group_by`, `status`. An evalset needs real and fake items, so a fake-only dataset needs a real source as a second component (`mlaad_si` pairs MLAAD's Sinhala fakes with OpenSLR's real Sinhala speech, hence `cross_corpus`). A component's `label` keeps the manifest's (`from_manifest`), forces one (`real`, `fake`), or, for an audio-video dataset scored by one modality, takes that track's label (`from_video`, `from_audio`) and drops items whose track label is unknown (`mavos_dd_hi_audio`).

## 5. Prove it runs

```bash
uv run pytest
uv run scripts/smoke.py --model <model> --evalsets <evalset>   # samples it, scores, reports
```

Add a builder test with an invented tree (no real file names; they can contain people's names), like `tests/test_builders.py`. Then add the dataset's access row to [Datasets](datasets.md#at-a-glance).
