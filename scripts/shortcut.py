"""The shortcut probe on one evalset: can trivial features alone tell real clips from fakes?

    uv run scripts/shortcut.py --evalset mlaad_hi [--workers N] [--publish] [--smoke]

Reads every item of the evalset (audio evalsets only), computes the probe's features
(``deeptrace_bench.eval.shortcut``), caches them per item under ``DTB_ROOT/shortcut/<evalset>/``
(item-level, never in git), and fits a cross-validated logistic regression on them. Prints the
probe's AUC, its 95% bootstrap interval and each feature's own AUC. ``--publish`` writes those
aggregates to ``results/shortcut/<evalset>.json``; pipeline tests and the smoke namespace are
refused. Items whose audio can't be read get a status and are counted, never dropped silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import pandas as pd

from deeptrace_bench import evalset as evalsets
from deeptrace_bench.eval import shortcut
from deeptrace_bench.eval.metrics import auc, bootstrap_ci
from deeptrace_bench.eval.publish import publish_refusal
from deeptrace_bench.paths import (
    PUBLISHED_RESULTS_DIR,
    dataset_dir,
    namespace,
    store_root,
    use_namespace,
)
from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.preprocess.audio import SAMPLE_RATE, load_audio
from deeptrace_bench.registry import Modality, Registry

log = logging.getLogger("shortcut")
FOLDS = 5
SEED = 0


def features_of(task: tuple[str, str]) -> dict:
    """One item's features, or its failure status."""
    item_id, path = task
    try:
        wave = load_audio(path)
        rate = shortcut.original_sample_rate(path)
        return {
            "item_id": item_id,
            "status": "ok",
            **shortcut.trivial_features(wave, SAMPLE_RATE, rate),
        }
    except PreprocessError as exc:
        return {"item_id": item_id, "status": exc.reason}
    except Exception as exc:  # noqa: BLE001 - any other failure is still a counted status
        return {"item_id": item_id, "status": f"error: {type(exc).__name__}"}


def compute(items: pd.DataFrame, cache_path, workers: int) -> pd.DataFrame:
    """Features for every item, reusing the cache and adding what's missing to it."""
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    done = set(cached["item_id"]) if len(cached) else set()
    todo = [
        (r.item_id, str(dataset_dir(r.dataset) / r.rel_path))
        for r in items.itertuples()
        if r.item_id not in done
    ]
    hits = len(done & set(items["item_id"]))
    print(f"[shortcut] {len(items)} items, {hits} cached, {len(todo)} to read", flush=True)
    rows, start = [], time.perf_counter()
    step = max(1, len(todo) // 50)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(features_of, todo, chunksize=64), 1):
            rows.append(row)
            if i % step == 0 or i == len(todo):
                rate = i / (time.perf_counter() - start)
                eta = (len(todo) - i) / rate if rate else 0
                print(
                    f"[shortcut] [{i}/{len(todo)}] {100 * i / len(todo):.0f}% "
                    f"{rate:.1f} items/s, eta {eta / 60:.0f} min",
                    flush=True,
                )
            if len(rows) >= 5000:  # checkpoint, so a stopped job resumes
                cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cached.to_parquet(cache_path)
                rows = []
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    if len(cached):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path)
    return items[["item_id", "label"]].merge(cached, on="item_id", how="left")


def summarize(table: pd.DataFrame, evalset) -> dict:
    """The probe's aggregates: AUC with its interval, per-feature AUCs and class means."""
    ok = table[table["status"] == "ok"]
    x = ok[shortcut.FEATURES].to_numpy(dtype=float)
    y = (ok["label"] == "fake").to_numpy(dtype=int)
    scores = shortcut.probe_scores(x, y, FOLDS, SEED)
    lo, hi = bootstrap_ci(y, scores, auc)
    failed = table[table["status"] != "ok"]
    means = ok.groupby("label")[shortcut.FEATURES].mean()
    return {
        "evalset": evalset.id,
        "pairing": evalset.pairing,
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": int(len(ok)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "n_failed": int(len(failed)),
        "failed_by_status": failed["status"].value_counts().to_dict(),
        "probe_auc": auc(y, scores),
        "probe_auc_lo": lo,
        "probe_auc_hi": hi,
        "folds": FOLDS,
        "seed": SEED,
        "model": "standardised logistic regression, out-of-fold scores",
        "features": shortcut.FEATURES,
        "feature_auc": shortcut.feature_aucs(x, y),
        "class_means": {lab: means.loc[lab].round(4).to_dict() for lab in means.index},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--publish", action="store_true", help="write results/shortcut/<id>.json")
    parser.add_argument("--smoke", action="store_true", help="use the smoke namespace")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.smoke:
        use_namespace("smoke")
    registry = Registry.load()
    evalset = registry.evalset(args.evalset)
    if evalset.modality != Modality.AUDIO:
        print(f"{evalset.id} is {evalset.modality}; the probe reads audio evalsets only")
        return 2
    items = evalsets.load_items(evalset)
    cache = store_root() / "shortcut" / evalset.id / "features.parquet"
    table = compute(items, cache, args.workers)
    summary = summarize(table, evalset)
    print(
        f"{evalset.id}: probe AUC {summary['probe_auc']:.3f} "
        f"({summary['probe_auc_lo']:.3f} to {summary['probe_auc_hi']:.3f}), "
        f"{summary['n_real']} real, {summary['n_fake']} fake, {summary['n_failed']} failed"
    )
    for name, value in sorted(summary["feature_auc"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24} {value:.3f}")
    (cache.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.publish:
        refusal = publish_refusal(evalset, namespace())
        if refusal:
            print(f"not published: {refusal}")
            return 2
        out = PUBLISHED_RESULTS_DIR / "shortcut" / f"{evalset.id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
