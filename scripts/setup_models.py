"""Set up detectors: pin upstream code, fetch weights, check hashes.

Examples:
    uv run scripts/setup_models.py --list
    uv run scripts/setup_models.py aasist aasist_l --record
    uv run scripts/setup_models.py --wave 1

For each model this clones the upstream repo at its pinned commit into ``third_party/``,
archives it to ``DTB_ROOT/upstream/``, downloads every weight file into
``DTB_ROOT/weights/<id>/`` and checks it against ``configs/weights.lock.yaml``. Use
``--record`` the first time a weight is fetched to pin its hash. Downloads belong on a
machine with internet access (on Curnagl, the login node).
"""

from __future__ import annotations

import argparse
import logging
import sys

from deeptrace_bench.fetch import (
    ChecksumMismatchError,
    ManualStepRequiredError,
    fetch_weight,
    read_lock,
)
from deeptrace_bench.models._stub import PendingDetector
from deeptrace_bench.paths import weights_dir
from deeptrace_bench.registry import ModelConfig, Registry, resolve
from deeptrace_bench.upstream import ensure_upstream

log = logging.getLogger("setup_models")

STATUS_ORDER = ["ready", "workable", "later", "blocked", "dropped", "test"]


def adapter_state(model: ModelConfig) -> str:
    """``yes`` for a working adapter, ``stub`` for a documented placeholder, else ``no``."""
    if model.adapter is None:
        return "no"
    return "stub" if issubclass(resolve(model.adapter), PendingDetector) else "yes"


def print_table(registry: Registry) -> None:
    """Print every model with status, wave and where its weights come from."""
    lock = read_lock()
    rows = []

    def order(m: ModelConfig) -> tuple:
        return (m.wave or 99, STATUS_ORDER.index(m.status), m.modality.value, m.id)

    for m in sorted(registry.models.values(), key=order):
        hosts = ",".join(sorted({w.kind for w in m.weights})) or "-"
        pinned = sum(1 for w in m.weights if w.name in lock.get(m.id, {}))
        rows.append(
            (
                m.id,
                m.modality.value,
                m.status,
                str(m.wave or "-"),
                hosts,
                f"{pinned}/{len(m.weights)}",
                adapter_state(m),
            )
        )
    header = ("model", "modality", "status", "wave", "weights from", "hashes", "adapter")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, tuple("-" * w for w in widths), *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))


def setup_model(model: ModelConfig, record: bool, skip_upstream: bool) -> bool:
    """Set up one model. Returns False if a manual step or a failure stopped it."""
    log.info("== %s (%s)", model.id, model.status)
    if model.status in ("dropped", "test"):
        log.info("%s is %s; skipping", model.id, model.status)
        return True
    ok = True
    if model.upstream and not skip_upstream:
        path = ensure_upstream(model.upstream)
        log.info("upstream at %s", path)
    for spec in model.weights:
        try:
            path = fetch_weight(model.id, spec, weights_dir(model.id), record=record)
            log.info("weight ready: %s", path)
        except ManualStepRequiredError as exc:
            log.warning("manual step needed: %s", exc)
            ok = False
        except ChecksumMismatchError as exc:
            log.error("%s", exc)
            ok = False
    return ok


def main() -> int:
    """Parse arguments and set up the requested models."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", help="model ids")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    parser.add_argument("--wave", type=int, help="set up every model in this wave")
    parser.add_argument("--record", action="store_true", help="pin hashes of new weights")
    parser.add_argument("--skip-upstream", action="store_true", help="weights only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    registry = Registry.load()
    if args.list:
        print_table(registry)
        return 0
    ids = list(args.models)
    if args.wave is not None:
        ids += [m.id for m in registry.models.values() if m.wave == args.wave]
    if not ids:
        parser.error("give model ids, --wave N or --list")

    results = {
        mid: setup_model(registry.model(mid), args.record, args.skip_upstream) for mid in ids
    }
    failed = [mid for mid, ok in results.items() if not ok]
    if failed:
        log.warning("needs attention: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
