"""Check an adapter against its upstream code on the same inputs (see ``deeptrace_bench.parity``).

Examples:
    uv run scripts/setup_datasets.py urdu_csalt --sample 100 --manifest
    uv run scripts/parity.py export --model aasist --evalset urdu_csalt --smoke
    uv run python scripts/parity_reference/aasist.py <run dir>
    uv run scripts/parity.py compare --model aasist --run <run dir>

``export`` picks about 200 items of an evalset (half of each label, by seeded rank), cuts
them into model inputs with the same loaders scoring uses, saves them with a ``parity.json``
the reference scripts read, and scores them with the adapter on CPU. It prints the command
that runs the matching reference script: in this environment for AASIST and DF Arena, in an
old-PyTorch virtualenv for DeepfakeBench (docs/models.md says how to make it). ``compare``
checks every score within 1e-3 and writes ``results/parity/<model>.json``; it exits 1 when
parity fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from deeptrace_bench import evalset as evalsets
from deeptrace_bench import parity
from deeptrace_bench.fetch import locked_hashes
from deeptrace_bench.models import _upstream
from deeptrace_bench.models.base import load_detector
from deeptrace_bench.paths import (
    PUBLISHED_RESULTS_DIR,
    REPO_ROOT,
    dataset_dir,
    dtb_root,
    namespace,
    use_namespace,
)
from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.audio import load_audio, segment
from deeptrace_bench.preprocess.faces import FaceLoader
from deeptrace_bench.registry import Modality, ModelConfig, Registry

log = logging.getLogger("parity")

# adapter class -> reference script under scripts/parity_reference/, and the Python that runs it
REFERENCES = {
    "AASISTDetector": ("aasist.py", "uv run python"),
    "DFArenaDetector": ("df_arena.py", "uv run python"),
    "DeepfakeBenchDetector": ("deepfakebench.py", "{dtb_root}/parity/envs/dfb/bin/python"),
}


def _inputs_loader(registry: Registry, model: ModelConfig):
    if model.modality == Modality.AUDIO:
        audio = registry.eval.audio
        min_samples = int(audio["min_seconds"] * audio["sample_rate"])

        def load(row):
            wave = load_audio(dataset_dir(row["dataset"]) / row["rel_path"])
            return segment(wave, audio["segment_samples"], min_samples=min_samples)

        return load
    return FaceLoader.from_registry(registry, model)


def export(args: argparse.Namespace) -> int:
    """Save inputs for ~200 items and the adapter's scores on them."""
    registry = Registry.load()
    model = registry.model(args.model)
    items = parity.pick_items(evalsets.load_items(registry.evalset(args.evalset)), args.items, 0)
    loader = _inputs_loader(registry, model)
    per_item, kept = [], []
    for _, row in items.iterrows():
        try:
            per_item.append(np.asarray(loader(row)))
            kept.append(row["item_id"])
        except PreprocessError as exc:
            log.info("%s skipped: %s", row["item_id"], exc.reason)
    if hasattr(loader, "close"):
        loader.close()
    units, item_index = parity.pack_inputs(per_item)

    run = parity.parity_root(model.id) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True)
    np.savez(run / parity.INPUTS, units=units, item_index=item_index)
    detector = load_detector(model)
    reference, python = REFERENCES[type(detector).__name__]
    spec = {
        "model": model.id,
        "evalset": args.evalset,
        "namespace": namespace(),
        "items": kept,
        "upstream": model.upstream.model_dump() if model.upstream else None,
        "checkout": str(detector.upstream_dir),
        "weights_dir": str(detector.weights_dir),
        "weights": locked_hashes(model.id),
        "input": model.input,
        "helper": str(Path(_upstream.__file__).resolve()),
    }
    (run / parity.RUN_INFO).write_text(json.dumps(spec, indent=2) + "\n")

    detector.load("cpu")
    scores = [detector.score(units[item_index == i]) for i in range(len(kept))]
    np.save(run / parity.ADAPTER_SCORES, np.concatenate(scores).astype(np.float64))
    command = python.format(dtb_root=dtb_root())
    print(f"{len(kept)} items, {len(units)} inputs in {run}")
    print(f"next: {command} {REPO_ROOT / 'scripts' / 'parity_reference' / reference} {run}")
    print(f"then: uv run scripts/parity.py compare --model {model.id} --run {run}")
    return 0


def compare(args: argparse.Namespace) -> int:
    """Compare the adapter's and the reference's scores and write the summary."""
    import torch

    run = Path(args.run)
    spec = json.loads((run / parity.RUN_INFO).read_text())
    reference = json.loads((run / parity.REFERENCE_INFO).read_text())
    result = parity.compare(
        np.load(run / parity.ADAPTER_SCORES), np.load(run / parity.REFERENCE_SCORES)
    )
    summary = {
        "model": spec["model"],
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "evalset": spec["evalset"],
        "namespace": spec["namespace"],
        "items": len(spec["items"]),
        **result,
        "upstream": spec["upstream"],
        "weights_sha256": spec["weights"],
        "adapter_env": {"python": platform.python_version(), "torch": torch.__version__},
        "reference_env": reference.get("env", {}),
        "reference": reference.get("method", ""),
    }
    out = PUBLISHED_RESULTS_DIR / "parity" / f"{spec['model']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"{verdict}: {result['n']} inputs, max |diff| {result['max_abs_diff']:.2e}, mean "
        f"{result['mean_abs_diff']:.2e} (tolerance {result['tolerance']:g}); wrote {out}"
    )
    return 0 if result["passed"] else 1


def main() -> int:
    """Parse arguments and run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="save inputs and the adapter's scores")
    exp.add_argument("--model", required=True)
    exp.add_argument("--evalset", required=True)
    exp.add_argument("--items", type=int, default=200)
    exp.add_argument("--smoke", action="store_true", help="use the smoke namespace")
    cmp = sub.add_parser("compare", help="compare adapter and reference scores")
    cmp.add_argument("--model", required=True)
    cmp.add_argument("--run", required=True, help="the run folder export printed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if getattr(args, "smoke", False):
        use_namespace("smoke")
    return export(args) if args.command == "export" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
