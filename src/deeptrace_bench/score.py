"""Resumable scoring: items in, one score per item out, safe to kill at any point.

Scores are appended as parquet parts in the run directory, keyed by ``item_id``. Each
process names its parts with its own prefix (``part-003of008-00000.parquet`` for shard 3 of
8), so array jobs that start together never write the same file. On restart the items
already in any part are skipped, so a job that hits Curnagl's 3-day limit or gets pre-empted
picks up where it stopped. Items that can't be preprocessed get a status instead of a score,
so failures can be counted per group.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models.base import Detector
from .paths import dataset_dir, weights_dir
from .preprocess import PreprocessError
from .preprocess.audio import load_audio, segment
from .preprocess.av import AudioVideoLoader
from .preprocess.faces import FaceLoader
from .preprocess.mouths import MouthLoader
from .registry import Modality, ModelConfig, Registry
from .upstream import checkout_dir

log = logging.getLogger(__name__)

AGGREGATIONS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda s: float(np.mean(s)),
    "max": lambda s: float(np.max(s)),
    "median": lambda s: float(np.median(s)),
}


def shard_items(items: pd.DataFrame, index: int, count: int) -> pd.DataFrame:
    """Return shard ``index`` of ``count``, by a stable hash of ``item_id``.

    Stable across runs and machines (unlike Python's ``hash``), so array job ``i`` always
    gets the same items.
    """
    if not 0 <= index < count:
        raise ValueError(f"shard {index} out of range for {count} shards")
    keys = items["item_id"].map(lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16))
    return items[keys % count == index]


def part_prefix(index: int, count: int) -> str:
    """The part-file prefix for shard ``index`` of ``count``."""
    return f"part-{index:03d}of{count:03d}"


def read_scores(directory: Path) -> pd.DataFrame:
    """All score parts in a run directory, as one table."""
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["item_id", "score", "n_windows", "status"])
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def score_items(
    detector: Detector,
    items: pd.DataFrame,
    load_inputs: Callable[[pd.Series], np.ndarray | Sequence[np.ndarray]],
    directory: Path,
    aggregate: str = "mean",
    flush_every: int = 256,
    max_items: int | None = None,
    prefix: str = "part",
) -> int:
    """Score every item not yet scored in ``directory``.

    Args:
        detector: a loaded detector.
        items: rows with at least ``item_id``; passed whole to ``load_inputs``.
        load_inputs: turns a row into the detector's inputs (an array of windows, or a list
            of face crops), or raises ``PreprocessError``.
        directory: the run directory.
        aggregate: how window scores become one item score (``mean``, ``max``, ``median``).
        flush_every: rows per parquet part.
        max_items: stop after this many new items (tests use it to simulate a kill).
        prefix: this process's part-file prefix (see ``part_prefix``).

    Returns:
        Number of items scored or marked failed in this call.
    """
    directory.mkdir(parents=True, exist_ok=True)
    done = set(read_scores(directory)["item_id"])
    todo = items[~items["item_id"].isin(done)]
    if max_items is not None:
        todo = todo.head(max_items)
    log.info("%d items already scored, %d to go", len(done), len(todo))

    combine = AGGREGATIONS[aggregate]
    next_part = len(list(directory.glob(f"{prefix}-*.parquet")))
    buffer: list[dict] = []
    for _, row in todo.iterrows():
        try:
            inputs = load_inputs(row)
            window_scores = np.asarray(detector.score(inputs), dtype=np.float64)
            buffer.append(
                {
                    "item_id": row["item_id"],
                    "score": combine(window_scores),
                    "n_windows": int(len(window_scores)),
                    "status": "ok",
                }
            )
        except PreprocessError as exc:
            buffer.append(
                {"item_id": row["item_id"], "score": np.nan, "n_windows": 0, "status": exc.reason}
            )
        if len(buffer) >= flush_every:
            _flush(buffer, directory, prefix, next_part)
            next_part += 1
            buffer = []
    if buffer:
        _flush(buffer, directory, prefix, next_part)
    return len(todo)


def _flush(rows: list[dict], directory: Path, prefix: str, part: int) -> None:
    path = directory / f"{prefix}-{part:05d}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.rename(path)
    log.info("wrote %s (%d rows)", path.name, len(rows))


def load_audio_inputs(row: pd.Series, segment_samples: int, min_samples: int) -> np.ndarray:
    """Read one audio item and cut it into model windows."""
    path = dataset_dir(row["dataset"]) / row["rel_path"]
    return segment(load_audio(path), segment_samples=segment_samples, min_samples=min_samples)


def build_loader(
    registry: Registry,
    model: ModelConfig,
    device: str,
    crops_dir: Path | None = None,
    save_crops: int = 0,
    prefix: str = "part",
) -> tuple[Callable[[pd.Series], Any], str]:
    """The input loader for ``model`` and the rule that aggregates its scores into one per item.

    An audio model on an audio-video evalset hears the audio track; a video model sees the
    frames. So the model, not the evalset, decides which loader runs.
    """
    if model.modality == Modality.AUDIO:
        audio = registry.eval.audio
        loader = partial(
            load_audio_inputs,
            segment_samples=audio["segment_samples"],
            min_samples=int(audio["min_seconds"] * audio["sample_rate"]),
        )
        return loader, audio["aggregation"]
    loader = FaceLoader.from_registry(
        registry, model, save_crops_to=crops_dir, save_crops=save_crops, part_prefix=prefix
    )
    if model.modality == Modality.AUDIO_VIDEO:
        # an audio-visual model gets its faces and the audio track together
        loader = AudioVideoLoader(loader, sample_rate=registry.eval.audio["sample_rate"])
    elif model.input.get("inputs") == "mouths":
        # a lip-based model gets mouth crops cut from landmarks, not face crops
        loader = MouthLoader(
            loader,
            fan_path=weights_dir(model.id) / model.input["landmarks"],
            upstream_dir=checkout_dir(model.upstream),
            device=device,
            min_frames=int(model.input["clip_frames"]),
        )
    return loader, registry.eval.video["aggregation"]
