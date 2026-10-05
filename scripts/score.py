"""Score one model on one evalset, resumably.

Examples:
    uv run scripts/score.py --model aasist --evalset asvspoof2019_la_eval
    uv run scripts/score.py --model aasist --evalset urdu_csalt --shard 3/8 --device cuda
    uv run scripts/score.py --smoke --model gend --evalset unidatapro_videos --save-crops 4

Refuses a model that can't read the evalset's modality (exit 2), then checks contamination
(a contaminated pair is refused unless ``--force``), creates or resumes the run
(``scores/<run_id>/run.json`` in the store), and scores every item not yet scored. Video and
image items go through the shared face pipeline (``preprocess/faces.py``); audio models get
each item's audio (an audio-video item's audio track) cut into windows. ``--shard i/n``
scores one stable slice, for SLURM array jobs.
``--save-crops K`` keeps up to K face crops per item under the run's ``crops/`` folder, for
people to look at. ``--smoke`` works in the smoke namespace (``DTB_ROOT/smoke/``).
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
from deeptrace_bench.fetch import locked_hashes
from deeptrace_bench.models.base import load_detector, resolve_device, set_tf32
from deeptrace_bench.paths import dataset_dir, namespace, prepare_store, use_namespace
from deeptrace_bench.preprocess.audio import load_audio, segment
from deeptrace_bench.preprocess.av import AudioVideoLoader
from deeptrace_bench.preprocess.faces import FaceLoader
from deeptrace_bench.registry import Modality, Registry, accepts
from deeptrace_bench.runs import config_hash, start_run
from deeptrace_bench.score import part_prefix, score_items, shard_items

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
    parser.add_argument("--device", default="auto", help="auto (cuda, mps, cpu), or a device")
    parser.add_argument("--preprocessing", choices=["shared", "native"], default="shared")
    parser.add_argument("--force", action="store_true", help="score even if contaminated")
    parser.add_argument("--save-crops", type=int, default=0, metavar="K", help="keep K crops")
    parser.add_argument("--smoke", action="store_true", help="use the smoke namespace")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.smoke:
        use_namespace("smoke")

    registry = Registry.load()
    model = registry.model(args.model)
    evalset = registry.evalset(args.evalset)

    # Everything that can refuse the pair runs before a run folder is created.
    if not accepts(model, evalset):
        log.error(
            "%s (%s) can't score %s (%s)", model.id, model.modality, evalset.id, evalset.modality
        )
        return 2
    # An audio model on an audio-video evalset hears the audio track; a video model sees the
    # frames. So the model, not the evalset, decides which loader runs.
    is_audio = model.modality == Modality.AUDIO
    if args.preprocessing == "native" and not is_audio:
        log.error("native preprocessing for video and image models isn't built yet")
        return 2

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
    prefix = part_prefix(index, count)

    # The face detector's weights are part of what defines a video or image run.
    weights = dict(locked_hashes(model.id))
    if not is_audio:
        tool = registry.face_detector()
        weights.update({f"{tool.id}/{k}": v for k, v in locked_hashes(tool.id).items()})

    prepare_store()
    device = resolve_device(args.device)
    set_tf32(registry.eval.compute["tf32"])
    record, directory = start_run(
        model_id=model.id,
        evalset_id=evalset.id,
        upstream_commit=model.upstream.commit if model.upstream else None,
        weights_sha256=weights,
        config_hashes={
            "model": config_hash(model),
            "evalset": config_hash(evalset),
            "eval": config_hash(registry.eval),
        },
        preprocessing=args.preprocessing,
        session={"device": device, "namespace": namespace(), "shard": args.shard},
    )
    log.info("run %s, shard %d/%d, %d items", record.run_id, index, count, len(items))

    if is_audio:
        audio = registry.eval.audio
        loader = partial(
            load_audio_inputs,
            segment_samples=audio["segment_samples"],
            min_samples=int(audio["min_seconds"] * audio["sample_rate"]),
        )
        aggregate = audio["aggregation"]
    else:
        loader = FaceLoader.from_registry(
            registry,
            model,
            save_crops_to=directory / "crops" if args.save_crops else None,
            save_crops=args.save_crops,
            part_prefix=prefix,
        )
        if model.modality == Modality.AUDIO_VIDEO:
            # an audio-visual model gets its faces and the audio track together
            loader = AudioVideoLoader(loader, sample_rate=registry.eval.audio["sample_rate"])
        aggregate = registry.eval.video["aggregation"]

    detector = load_detector(model)
    detector.load(device)
    try:
        n = score_items(detector, items, loader, directory, aggregate=aggregate, prefix=prefix)
    finally:
        if hasattr(loader, "close"):
            loader.close()
    log.info("scored %d items into %s", n, directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
