# Adding a dataset

A dataset is a config, a builder that turns its files into a manifest, and at least one evalset that says how it is scored. The two pipeline-test datasets are the worked examples: `mendeley_roop_akool` (a zip at a URL, image items) and `unidatapro_videos` (Hugging Face, video items).

## 1. Config

Add `configs/datasets/<id>.yaml`:

- `access`: how to get it. `hf` / `hf_gated` with `repo` and a pinned `revision` (and `allow_patterns`); `url` with `urls` (and `sha256` per archive); `manual` with `request` instructions and `expect` paths for `--from`. A dataset that is one zip inside a Hugging Face repo (BanglaFake, In-the-Wild) is `url` access to the zip's pinned `resolve/<revision>/` URL, with the Hub's LFS sha256: the full fetch is hash-checked and a sample reads members by range requests.
- `licence` and `licence_terms`, `contains` (`real`, `fake`), `languages`, `size_gb`.
- `derived_from`: every dataset or corpus its items came from, so the contamination guard can follow it.
- `role`: `reproduction`, `south_asian`, `real_reference`, `in_the_wild`, or `pipeline_test` for data that only proves the code works (never reported, never mixed into a real evalset).
- `notes`: facts with the date they were checked.

## 2. Builder

Write `src/deeptrace_bench/datasets/<id>.py` against the real files:

```python
def label_from_path(rel_path: str) -> str | None:
    """The label from the path alone: real, fake, or None for anything that isn't an item."""


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per item under root, in the schema of deeptrace_bench.manifest."""
```

- `label_from_path` is what makes a dataset samplable: the sampler calls it on the remote file list before downloading, and `build_manifest` calls it on the local files, so the two can't disagree. Return None for metadata, archive junk (`__MACOSX/`), and anything else that isn't an item.
- When labels live in a file rather than the path (In-the-Wild's `meta.csv`, ASVspoof's protocols), define `METADATA_FILES` (relative paths) and `labels_from_metadata(files: dict[str, bytes]) -> {rel_path: label}` instead. The sampler reads those files first and copies them into the sample, and `build_manifest` reads the same files. A builder that only reads speakers from a file can list it in `METADATA_FILES` next to `label_from_path` so samples carry it.
- A dataset with one label (MLAAD is fake only, OpenSLR Sinhala real only) samples on the labels its config `contains`. Where one kind of item is rare, `sample_stratum(rel_path)` takes N items from every stratum: MAVOS-DD samples per generator, so its voice-conversion fakes are always in.
- `build_manifest` must work on a partial tree (a sample): iterate with `datasets.iter_files`, which skips hidden files such as `.sample.json`, and never assume the full count.
- Columns: `item_id` (`<id>/<local path without extension>`), `dataset`, `rel_path`, `modality` (`video`, `image`, `audio`, `audio_video`), `label`; optional `method`, `method_family`, `language`, `subject_id`, `source_subject_id`, `split`, `duration_s`; for audio-video items, `label_video` and `label_audio` (which track was changed; null where nobody has checked); group attributes as `g_<attr>` plus `g_<attr>_src`. Unknown values are null, never "unknown".
- Document the layout in the module docstring, as checked, with the date. Write facts only: if the dataset doesn't say which generator is which, keep neutral names (Mendeley's `tech_1`, `tech_2`).

## 3. Sample it, then build the manifest

```bash
uv run scripts/setup_datasets.py <id> --sample 2 --manifest     # smoke namespace, a few MB
uv run scripts/setup_datasets.py <id> --manifest                # the full dataset
uv run scripts/setup_datasets.py <id> --from /path/to/copy --manifest   # request-only datasets
```

`--sample N` picks N items of each label the dataset contains (of each stratum, if the builder defines them) deterministically, by a seeded hash of each path. It fetches only those (individual Hugging Face files, listing only the folders the patterns can match; zip members by HTTP range; or symlinks into a `--from` copy) and records the choice in `.sample.json`. It always writes to the smoke namespace.

## 4. Evalset

Add `configs/evalsets/<id>.yaml`: components (dataset, filter, label), `pairing` (`same_corpus` or `cross_corpus`), `group_by`, `status`. An evalset needs real and fake items; a fake-only dataset needs a real source as a second component (`mlaad_si` pairs MLAAD's Sinhala fakes with OpenSLR's real Sinhala speech, so it is `cross_corpus`). A component's `label` keeps the manifest's (`from_manifest`), forces one (`real`, `fake`), or, for an audio-video dataset scored by one modality, takes that track's label (`from_video`, `from_audio`), filtering out items whose track label is unknown (`mavos_dd_hi_audio`).

## 5. Prove it runs

```bash
uv run pytest
uv run scripts/smoke.py --model <model> --evalsets <evalset>   # samples it, scores, reports
```

Add a builder test with an invented tree (no real file names, which can contain people's names), like `tests/test_builders.py`. Dataset access routes and licences are in [datasets.md](datasets.md).
