"""Turn a run's scores into metrics: overall, per group, with intervals and a verdict.

Examples:
    uv run scripts/evaluate.py --model aasist --evalset urdu_csalt
    uv run scripts/evaluate.py --run aasist__urdu_csalt__1a2b3c4d --publish
    uv run scripts/evaluate.py --smoke --model gend --evalset unidatapro_videos

Writes full results (including per-item rows) to ``results/<run_id>/`` in the store. With
``--publish`` it also copies the aggregate tables, which name no items, into the repo's
``results/<run_id>/`` so they can be committed; pipeline tests and smoke runs are refused
(exit 2). ``--smoke`` works in the smoke namespace (``DTB_ROOT/smoke/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys

from deeptrace_bench import evalset as evalsets
from deeptrace_bench.eval.contamination import check
from deeptrace_bench.eval.metrics import per_group, score_direction_ok, summarize_scores
from deeptrace_bench.eval.publish import publish_refusal
from deeptrace_bench.paths import (
    PUBLISHED_RESULTS_DIR,
    namespace,
    results_dir,
    run_dir,
    scores_root,
    use_namespace,
)
from deeptrace_bench.registry import Registry
from deeptrace_bench.score import read_scores

log = logging.getLogger("evaluate")


def find_run(model_id: str, evalset_id: str) -> str:
    """Return the only run for a (model, evalset) pair, or fail if there are none or several."""
    runs = sorted(p.name for p in scores_root().glob(f"{model_id}__{evalset_id}__*"))
    if len(runs) != 1:
        raise SystemExit(
            f"expected one run for {model_id} on {evalset_id}, found {runs}; use --run"
        )
    return runs[0]


def main() -> int:
    """Parse arguments and evaluate one run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run")
    parser.add_argument("--model")
    parser.add_argument("--evalset")
    parser.add_argument("--publish", action="store_true", help="copy aggregates into results/")
    parser.add_argument("--smoke", action="store_true", help="use the smoke namespace")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.smoke:
        use_namespace("smoke")

    run_id = args.run or find_run(args.model, args.evalset)
    record = json.loads((run_dir(run_id) / "run.json").read_text())
    registry = Registry.load()
    model = registry.model(record["model_id"])
    evalset = registry.evalset(record["evalset_id"])
    metric_cfg = registry.eval.metrics

    items = evalsets.load_items(evalset)
    scores = read_scores(run_dir(run_id))
    df = items.merge(scores, on="item_id", how="inner")
    missing = len(items) - len(df)
    if missing:
        log.warning("%d items have no score yet; evaluating the %d that do", missing, len(df))

    overall = summarize_scores(
        df, n_boot=metric_cfg["bootstrap"]["n"], seed=metric_cfg["bootstrap"]["seed"]
    )
    verdict = check(model, evalset, registry)
    ok = df[df["status"] == "ok"]
    summary = {
        "run_id": run_id,
        "model": model.id,
        "evalset": evalset.id,
        "role": evalset.role,
        "namespace": namespace(),
        "pairing": evalset.pairing,
        "verdict": verdict.verdict.label,
        "verdict_reasons": verdict.reasons,
        "score_direction_ok": score_direction_ok(ok["label"], ok["score"]),
        "eval_config_status": registry.eval.status,
        "overall": overall,
        "unscored_items": missing,
    }

    out = results_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    for col in evalset.group_by:
        if col not in df.columns:
            log.warning("group column %s not in the items; skipping", col)
            continue
        table = per_group(
            df,
            col,
            threshold=overall["threshold"],
            min_n=metric_cfg["min_group_n"],
            n_boot=metric_cfg["bootstrap"]["n"],
            seed=metric_cfg["bootstrap"]["seed"],
        )
        table.to_csv(out / f"by_{col}.csv", index=False)
    df[["item_id", "label", "score", "status"]].to_parquet(out / "items.parquet", index=False)
    log.info("results in %s (verdict %s)", out, verdict.verdict.label)

    if args.publish:
        refusal = publish_refusal(evalset, namespace())
        if refusal:
            log.error("not publishing: %s", refusal)
            return 2
        public = PUBLISHED_RESULTS_DIR / run_id
        public.mkdir(parents=True, exist_ok=True)
        for path in out.glob("*"):
            # Per-subject tables name speakers or faces, which counts as derived data under
            # some licences, so they stay in DTB_ROOT with the per-item rows.
            if path.name.startswith(("by_subject_id", "by_source_subject_id")):
                continue
            if path.suffix in (".json", ".csv"):
                shutil.copy2(path, public / path.name)
        log.info("published aggregates to %s", public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
