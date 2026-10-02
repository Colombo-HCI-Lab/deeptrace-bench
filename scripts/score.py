"""Score one model on one evalset, resumably.

Examples:
    uv run scripts/score.py --model aasist --evalset asvspoof2019_la_eval
    uv run scripts/score.py --model aasist --evalset urdu_csalt --shard 3/8 --device cuda

Checks contamination first (a contaminated pair is refused unless ``--force``), creates or
resumes the run (``DTB_ROOT/scores/<run_id>/run.json``), then scores every item not yet
scored. ``--shard i/n`` scores one stable slice, for SLURM array jobs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from functools import partial

import numpy as np
import pandas as pd

from deeptrace_bench import evalset as evalsets
from deeptrace_bench.eval.contamination import Verdict, check
from deeptrace_bench.fetch import read_lock
from deeptrace_bench.models.base import load_detector
from deeptrace_bench.paths import dataset_dir
from deeptrace_bench.preprocess.audio import load_audio, segment
from deeptrace_bench.registry import Registry
from deeptrace_bench.runs import config_hash, start_run
from deeptrace_bench.score import score_items, shard_items

log = logging.getLogger("score")


def load_audio_inputs(row: pd.Series, segment_samples: int, min_samples: int) -> np.ndarray:
    """Read one audio item and cut it into model windows."""
    path = dataset_dir(row["dataset"]) / row["rel_path"]
    return segment(load_audio(path), segment_samples=segment_samples, min_samples=min_samples)


def main() -> int:
    """Parse arguments and run scoring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--shard", default="0/1", help="i/n: score shard i of n")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocessing", choices=["shared", "native"], default="shared")
    parser.add_argument("--force", action="store_true", help="score even if contaminated")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    registry = Registry.load()
    model = registry.model(args.model)
    evalset = registry.evalset(args.evalset)

    verdict = check(model, evalset, registry)
    for reason in verdict.reasons:
        log.info("contamination: %s", reason)
    if verdict.verdict == Verdict.CONTAMINATED and not args.force:
        log.error(
            "%s on %s is contaminated; refusing (use --force to override)", model.id, evalset.id
        )
        return 2

    items = evalsets.load_items(evalset)
    if verdict.exclude_items_files:
        raise NotImplementedError("item exclusion lists aren't wired up yet")
    index, count = (int(x) for x in args.shard.split("/"))
    items = shard_items(items, index, count)

    record, directory = start_run(
        model_id=model.id,
        evalset_id=evalset.id,
        upstream_commit=model.upstream.commit if model.upstream else None,
        weights_sha256=read_lock().get(model.id, {}),
        config_hashes={
            "model": config_hash(model),
            "evalset": config_hash(evalset),
            "eval": config_hash(registry.eval),
        },
        preprocessing=args.preprocessing,
    )
    log.info("run %s, shard %d/%d, %d items", record.run_id, index, count, len(items))

    if evalset.modality.value == "audio":
        audio = registry.eval.audio
        loader = partial(
            load_audio_inputs,
            segment_samples=audio["segment_samples"],
            min_samples=int(audio["min_seconds"] * audio["sample_rate"]),
        )
        aggregate = audio["aggregation"]
    else:
        raise NotImplementedError("video scoring waits on preprocess.faces; see docs/evaluation.md")

    detector = load_detector(model)
    detector.load(args.device)
    n = score_items(detector, items, loader, directory, aggregate=aggregate)
    log.info("scored %d items into %s", n, directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
