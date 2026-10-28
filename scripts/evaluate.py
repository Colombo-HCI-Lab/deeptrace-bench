"""Turn a run's scores into metrics: overall, per group, with intervals and a verdict.

Examples:
    uv run scripts/evaluate.py --model aasist --evalset urdu_csalt
    uv run scripts/evaluate.py --run aasist__urdu_csalt__1a2b3c4d --publish
    uv run scripts/evaluate.py --smoke --model gend --evalset unidatapro_videos

Writes full results (including per-item rows) to ``results/<run_id>/`` in the store. The
summary also carries what scoring cost (``compute``: time per phase and per item, throughput,
peak memory, GPU load, energy and the machine, from the run's telemetry), failures by
reason, and whether the model's adapter passed parity. With ``--publish`` it also copies the
aggregate tables, which name no items, into the repo's ``results/<run_id>/`` with a
``provenance.json`` (configs, weights, commits; no host names) so they can be committed;
pipeline tests and smoke runs are refused (exit 2). ``--smoke`` works in the smoke namespace
(``DTB_ROOT/smoke/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from deeptrace_bench import evalset as evalsets
from deeptrace_bench.eval.contamination import check
from deeptrace_bench.eval.metrics import (
    FPR_TARGETS,
    per_group,
    score_direction_ok,
    summarize_scores,
)
from deeptrace_bench.eval.publish import publish_refusal
from deeptrace_bench.paths import (
    PUBLISHED_RESULTS_DIR,
    namespace,
    results_dir,
    run_dir,
    scores_root,
    use_namespace,
)
from deeptrace_bench.registry import Modality, Registry
from deeptrace_bench.score import read_scores
from deeptrace_bench.telemetry import read_sessions, summarize_compute

log = logging.getLogger("evaluate")


def find_run(model_id: str, evalset_id: str) -> str:
    """Return the only run for a (model, evalset) pair, or fail if there are none or several."""
    runs = sorted(p.name for p in scores_root().glob(f"{model_id}__{evalset_id}__*"))
    if len(runs) != 1:
        raise SystemExit(
            f"expected one run for {model_id} on {evalset_id}, found {runs}; use --run"
        )
    return runs[0]


def provenance(record: dict) -> dict:
    """What defines a run and which code scored it, safe to publish (no host names)."""
    sessions = record["sessions"]
    return {
        **{k: record[k] for k in ("run_id", "model_id", "evalset_id", "upstream_commit")},
        **{k: record[k] for k in ("weights_sha256", "config_hashes", "preprocessing")},
        "first_started_utc": sessions[0]["started_utc"],
        "last_started_utc": sessions[-1]["started_utc"],
        "repo_commits": sorted({s["repo_commit"] for s in sessions if s.get("repo_commit")}),
        "repo_dirty": any(s.get("repo_dirty") for s in sessions),
        "devices": sorted({s["device"] for s in sessions if s.get("device")}),
    }


def replace_file(path: Path, write: Callable[[Path], object]) -> None:
    """Write ``path`` through a sibling temporary file and rename it into place.

    Over the SMB mount the workers use, overwriting a file with shorter content can keep the old
    file's tail (a 2186-byte summary.json copied over a 2187-byte one kept its last byte), so a
    file is never rewritten in place.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    write(tmp)
    os.replace(tmp, path)


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

    fpr_targets = sorted({*FPR_TARGETS, metric_cfg["fixed_fpr_reference"]})
    overall = summarize_scores(
        df,
        n_boot=metric_cfg["bootstrap"]["n"],
        seed=metric_cfg["bootstrap"]["seed"],
        fpr_targets=fpr_targets,
    )
    verdict = check(model, evalset, registry)
    ok = df[df["status"] == "ok"]
    audio = registry.eval.audio
    is_audio = model.modality == Modality.AUDIO
    window_s = audio["segment_samples"] / audio["sample_rate"] if is_audio else None
    parity = PUBLISHED_RESULTS_DIR / "parity" / f"{model.id}.json"
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
        "parity_passed": bool(parity.exists() and json.loads(parity.read_text())["passed"]),
        "overall": overall,
        "unscored_items": missing,
        "failures": {
            str(k): int(v) for k, v in df.loc[df["status"] != "ok", "status"].value_counts().items()
        },
        "compute": summarize_compute(scores, read_sessions(run_dir(run_id)), window_s),
    }

    out = results_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, default=float) + "\n"
    replace_file(out / "summary.json", lambda p: p.write_text(text))
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
            fpr_targets=fpr_targets,
        )
        replace_file(out / f"by_{col}.csv", lambda p, t=table: t.to_csv(p, index=False))
    items = df[["item_id", "label", "score", "status"]]
    replace_file(out / "items.parquet", lambda p: items.to_parquet(p, index=False))
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
                replace_file(public / path.name, lambda p, src=path: shutil.copy2(src, p))
        text = json.dumps(provenance(record), indent=2) + "\n"
        replace_file(public / "provenance.json", lambda p: p.write_text(text))
        log.info("published aggregates to %s", public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
