"""Set up datasets: download what is open, link what was obtained by hand, build manifests.

Examples:
    uv run scripts/setup_datasets.py --list
    uv run scripts/setup_datasets.py urdu_csalt banglafake
    uv run scripts/setup_datasets.py indicsynth --languages hi bn ur
    uv run scripts/setup_datasets.py fakeavceleb --from /work/.../FakeAVCeleb_v1.2
    uv run scripts/setup_datasets.py urdu_csalt --manifest

Open datasets are downloaded into ``DTB_ROOT/datasets/<id>/``. Request-only datasets can't
be downloaded by a script: once someone has a copy, ``--from PATH`` checks it and links it
into the store. ``--manifest`` runs the dataset's builder and writes the manifest to
``DTB_ROOT/manifests/<id>.parquet``, plus a committable summary (counts only) under
``results/manifests/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from deeptrace_bench.fetch import ManualStepRequiredError, fetch_dataset, verify_supplied
from deeptrace_bench.manifest import summarize, write_manifest
from deeptrace_bench.paths import PUBLISHED_RESULTS_DIR, dataset_dir
from deeptrace_bench.registry import DatasetConfig, Registry, resolve

log = logging.getLogger("setup_datasets")


def print_table(registry: Registry) -> None:
    """Print every dataset with role, status, access route and size."""
    order = {"open": 0, "click_through": 1, "request": 2, "deferred": 3, "dead": 4}
    rows = []
    for d in sorted(registry.datasets.values(), key=lambda d: (order[d.status], d.role, d.id)):
        rows.append(
            (
                d.id,
                d.modality.value,
                d.role,
                d.status,
                d.access.kind,
                "+".join(d.contains),
                f"{d.size_gb:g}" if d.size_gb is not None else "?",
            )
        )
    header = ("dataset", "modality", "role", "status", "access", "contains", "GB")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, tuple("-" * w for w in widths), *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))


def build(config: DatasetConfig) -> None:
    """Run a dataset's builder, write its manifest and a committable summary."""
    if config.builder is None:
        raise ManualStepRequiredError(f"{config.id} has no builder")
    df = resolve(config.builder)(dataset_dir(config.id))
    path = write_manifest(df, config.id)
    summary_path = PUBLISHED_RESULTS_DIR / "manifests" / f"{config.id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summarize(df), indent=2, sort_keys=True) + "\n")
    log.info("%s: %d items -> %s (summary %s)", config.id, len(df), path, summary_path)


def main() -> int:
    """Parse arguments and set up the requested datasets."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", help="dataset ids")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    parser.add_argument("--languages", nargs="+", help="languages to fetch, where supported")
    parser.add_argument("--from", dest="source", type=Path, help="link a copy obtained by hand")
    parser.add_argument("--manifest", action="store_true", help="build the manifest")
    parser.add_argument("--skip-download", action="store_true", help="only build manifests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    registry = Registry.load()
    if args.list:
        print_table(registry)
        return 0
    if not args.datasets:
        parser.error("give dataset ids or --list")
    if args.source and len(args.datasets) != 1:
        parser.error("--from takes exactly one dataset")

    failed = []
    for dataset_id in args.datasets:
        config = registry.dataset(dataset_id)
        dest = dataset_dir(dataset_id)
        try:
            if args.source:
                verify_supplied(config, args.source, dest)
            elif not args.skip_download:
                fetch_dataset(config, dest, args.languages)
            if args.manifest:
                build(config)
        except ManualStepRequiredError as exc:
            log.warning("%s", exc)
            failed.append(dataset_id)
        except NotImplementedError as exc:
            log.warning("%s: builder not written yet (%s)", dataset_id, exc)
            failed.append(dataset_id)
    if failed:
        log.warning("needs attention: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
