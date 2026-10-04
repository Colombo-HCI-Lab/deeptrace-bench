"""Run the whole pipeline on a few items per dataset, and write a report a person can read.

Examples:
    uv run scripts/smoke.py
    uv run scripts/smoke.py --model gend --sample 3 --device cpu
    uv run scripts/smoke.py --evalsets unidatapro_videos --keep

Everything happens in the smoke namespace, ``DTB_ROOT/smoke/``, which never mixes with real
runs. The steps are the same scripts a real run uses, each run as a subprocess with its output
in the log:

1. ``setup_models.py <model>``: upstream checkout and weights (plus the face detector for a
   video model), checked against the lock.
2. ``setup_datasets.py <datasets> --sample N --manifest``: N real and N fake items per dataset.
3. ``score.py --smoke ... --save-crops 4`` for each evalset.
4. ``evaluate.py --smoke ...`` for each evalset.

Earlier smoke faces, scores and results are deleted first (``--keep`` keeps them), so a fixed
bug can never pass on stale scores; sampled datasets are kept, since they don't change. The
report goes to ``smoke/reports/<UTC time>__<model>.md`` (``latest.md`` points at the newest),
with one table per evalset: every item's label, score, status, faces found, and a face crop.

The run passes when every step exits 0, every item has a score row with a known status,
each evalset has at least one scored item, and every score is a probability. That proves the
plumbing, not the model: the numbers are never reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from deeptrace_bench import evalset as evalsets
from deeptrace_bench.fetch import is_pinned, locked_hashes
from deeptrace_bench.paths import (
    REPO_ROOT,
    faces_dir,
    namespace,
    prepare_store,
    reports_dir,
    results_dir,
    run_dir,
    scores_root,
    store_root,
    use_namespace,
)
from deeptrace_bench.preprocess.faces import FaceLoader
from deeptrace_bench.registry import EvalsetConfig, Modality, ModelConfig, Registry, accepts
from deeptrace_bench.runs import git_state
from deeptrace_bench.score import read_scores

log = logging.getLogger("smoke")

KNOWN_STATUSES = {"ok", "no_face", "unreadable", "too_short", "empty_audio", "no_audio"}
# No open audio set is a pipeline test, so audio models smoke-test on a small open South Asian
# set instead. It lives in the smoke namespace, which can never be published.
AUDIO_FALLBACK = ["urdu_csalt"]
SCRIPTS = REPO_ROOT / "scripts"


def default_evalsets(registry: Registry, model: ModelConfig) -> list[EvalsetConfig]:
    """Every ready pipeline-test evalset the model can score (audio: ``AUDIO_FALLBACK``)."""
    chosen = [
        e
        for e in registry.evalsets.values()
        if e.role == "pipeline_test" and e.status == "ready" and accepts(model, e)
    ]
    if not chosen and model.modality == Modality.AUDIO:
        chosen = [registry.evalset(e) for e in AUDIO_FALLBACK]
    return chosen


def unpinned(registry: Registry, model: ModelConfig) -> list[str]:
    """Weights of the model (and its face detector) with no hash in the lock yet."""
    owners = [model]
    if model.modality in (Modality.VIDEO, Modality.AUDIO_VIDEO):
        owners.append(registry.face_detector())
    return [f"{o.id}/{w.name}" for o in owners for w in o.weights if not is_pinned(o.id, w)]


def run_step(args: list[str], log_file: Path) -> int:
    """Run one script in the smoke namespace, appending its output to the log."""
    cmd = [sys.executable, str(SCRIPTS / args[0]), *args[1:]]
    log.info("$ %s", " ".join(args))
    with log_file.open("a") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env={**os.environ, "DTB_NAMESPACE": "smoke"},
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode:
        log.error("%s exited %d; see %s", args[0], proc.returncode, log_file)
    return proc.returncode


def latest_run(model_id: str, evalset_id: str) -> str | None:
    """The most recently written run of a pair in the store, if any."""
    runs = sorted(
        scores_root().glob(f"{model_id}__{evalset_id}__*"),
        key=lambda p: (p / "run.json").stat().st_mtime if (p / "run.json").exists() else 0,
    )
    return runs[-1].name if runs else None


def face_counts(registry: Registry, model: ModelConfig, datasets: list[str]) -> pd.DataFrame:
    """Frames sampled and frames with a face, per item, from the detection cache."""
    if model.modality == Modality.AUDIO:
        return pd.DataFrame(columns=["item_id", "frames", "faces"])
    key = FaceLoader.from_registry(registry, model).key
    parts = [p for d in datasets for p in sorted(faces_dir(d, key).glob("part-*.parquet"))]
    if not parts:
        return pd.DataFrame(columns=["item_id", "frames", "faces"])
    rows = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return (
        rows.assign(face=rows["kps"].notna())
        .groupby("item_id")
        .agg(frames=("frame_index", "size"), faces=("face", "sum"))
        .reset_index()
    )


def check_items(df: pd.DataFrame, n_items: int) -> list[str]:
    """Why an evalset's scores fail the smoke test (empty when they pass)."""
    problems = []
    if len(df) != n_items:
        problems.append(f"{n_items - len(df)} of {n_items} items have no score row")
    unknown = set(df["status"]) - KNOWN_STATUSES
    if unknown:
        problems.append(f"unknown statuses {sorted(unknown)}")
    ok = df[df["status"] == "ok"]
    if ok.empty:
        problems.append("no item was scored")
    elif not all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in ok["score"]):
        problems.append("a score is not a probability in [0, 1]")
    return problems


def _short(digest: str | None) -> str:
    return f"`{digest[:8]}`" if digest else "not pinned"


def _weights_line(hashes: dict[str, str]) -> str:
    """Pinned weights, one entry per file and one per folder (with its file count)."""
    files = {k: v for k, v in hashes.items() if "/" not in k}
    folders: dict[str, int] = {}
    for key in hashes:
        if "/" in key:
            folders[key.split("/")[0]] = folders.get(key.split("/")[0], 0) + 1
    parts = [f"{k} {_short(v)}" for k, v in sorted(files.items())]
    parts += [f"{k}/ ({n} files, pinned)" for k, n in sorted(folders.items())]
    return ", ".join(parts) or "none"


def _versions() -> str:
    import cv2
    import onnxruntime
    import torch
    import transformers

    return (
        f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers "
        f"{transformers.__version__}, onnxruntime {onnxruntime.__version__}, opencv "
        f"{cv2.__version__}"
    )


def evalset_section(
    registry: Registry, model: ModelConfig, evalset: EvalsetConfig, report: Path
) -> tuple[list[str], list[str], dict]:
    """Markdown lines for one evalset, its problems, and a row for the console table."""
    lines = [f"## {evalset.id} ({evalset.modality})", ""]
    run_id = latest_run(model.id, evalset.id)
    if run_id is None:
        lines += ["No run was written.", ""]
        return lines, ["no run was written"], {"evalset": evalset.id, "result": "FAIL"}

    record = json.loads((run_dir(run_id) / "run.json").read_text())
    device = record["sessions"][-1].get("device", "?")
    items = evalsets.load_items(evalset)
    scores = read_scores(run_dir(run_id))
    df = items.merge(scores, on="item_id", how="inner")
    faces = face_counts(registry, model, sorted({c.dataset for c in evalset.components}))
    df = df.merge(faces, on="item_id", how="left")
    problems = check_items(df, len(items))

    summary_path = results_dir(run_id) / "summary.json"
    overall = json.loads(summary_path.read_text())["overall"] if summary_path.exists() else {}
    rel = os.path.relpath(store_root(), report.parent)
    lines += [
        f"Run `{run_id}` on {device}: [scores]({rel}/scores/{run_id}/), "
        f"[results]({rel}/results/{run_id}/)",
        "",
        f"AUC {overall.get('auc', float('nan')):.3f} and EER {overall.get('eer', float('nan')):.3f}"
        f" over {len(df)} items. With this few items both are noise; they only show the "
        "metrics code ran.",
        "",
        "| item | label | method | score | status | faces / frames | crop |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in df.sort_values(["label", "item_id"], ascending=[False, True]).itertuples():
        crop_dir = run_dir(run_id) / "crops" / row.item_id
        crops = sorted(crop_dir.glob("*.png")) if crop_dir.exists() else []
        crop = f'<img src="{os.path.relpath(crops[0], report.parent)}" width="96">' if crops else ""
        method = getattr(row, "method", None)
        score = f"{row.score:.4f}" if row.status == "ok" else ""
        frames = "" if pd.isna(row.frames) else f"{int(row.faces)} / {int(row.frames)}"
        local = row.item_id.split("/", 1)[1]
        lines.append(
            f"| `{local}` | {row.label} | {method if isinstance(method, str) else ''} "
            f"| {score} | {row.status} | {frames} | {crop} |"
        )
    lines.append("")
    if problems:
        lines += [f"**Problems:** {'; '.join(problems)}", ""]
    ok = df[df["status"] == "ok"]
    table_row = {
        "evalset": evalset.id,
        "items": len(items),
        "scored": len(ok),
        "failed": len(df) - len(ok),
        "result": "FAIL" if problems else "pass",
    }
    return lines, problems, table_row


def write_report(
    registry: Registry,
    model: ModelConfig,
    chosen: list[EvalsetConfig],
    args: argparse.Namespace,
    report: Path,
    failed_steps: list[str],
) -> tuple[bool, list[dict]]:
    """Write the Markdown report. Returns (passed, console rows)."""
    commit, dirty = git_state()
    detector = registry.face_detector() if model.modality != Modality.AUDIO else None
    body: list[str] = []
    rows: list[dict] = []
    all_problems = [f"step failed: {s}" for s in failed_steps]
    for evalset in chosen:
        lines, problems, row = evalset_section(registry, model, evalset, report)
        body += lines
        rows.append(row)
        all_problems += [f"{evalset.id}: {p}" for p in problems]
    passed = not all_problems

    model_hashes = locked_hashes(model.id)
    head = [
        f"# Smoke test: {model.id}, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        "> **Pipeline test, not a result.** A handful of items chosen to exercise the code. "
        "The scores say nothing about the model, and its parity with upstream hasn't been "
        "checked.",
        "",
        f"**{'PASS' if passed else 'FAIL'}**" + ("" if passed else ": " + "; ".join(all_problems)),
        "",
        "| | |",
        "|---|---|",
        f"| Repo | `{(commit or 'unknown')[:8]}`{' with uncommitted changes' if dirty else ''} |",
        f"| Sample | {args.sample} real + {args.sample} fake per dataset, seed {args.seed} |",
        f"| Model | {model.id}, upstream "
        f"`{model.upstream.commit[:8] if model.upstream else '-'}`, weights "
        f"{_weights_line(model_hashes)} |",
    ]
    if detector is not None:
        det_hash = next(iter(locked_hashes(detector.id).values()), None)
        head.append(f"| Face detector | {detector.id} {_short(det_hash)} |")
    head += [
        f"| Versions | {_versions()} |",
        f"| Log | [{report.with_suffix('.log').name}]({report.with_suffix('.log').name}) |",
        "",
    ]
    report.write_text("\n".join(head + body) + "\n")
    latest = report.parent / "latest.md"
    latest.unlink(missing_ok=True)
    latest.symlink_to(report.name)
    return passed, rows


def main() -> int:
    """Parse arguments, run the pipeline in the smoke namespace and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="gend")
    parser.add_argument("--evalsets", nargs="+", help="default: every pipeline test it accepts")
    parser.add_argument("--sample", type=int, default=2, help="real and fake items per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep", action="store_true", help="keep earlier faces and scores")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    use_namespace("smoke")
    assert namespace() == "smoke"
    registry = Registry.load()
    model = registry.model(args.model)
    chosen = (
        [registry.evalset(e) for e in args.evalsets]
        if args.evalsets
        else default_evalsets(registry, model)
    )
    if not chosen:
        log.error("no pipeline-test evalset fits %s", model.id)
        return 1
    missing = unpinned(registry, model)
    if missing:
        log.error(
            "weights not pinned yet: %s. Fetch and pin them once with: uv run "
            "scripts/setup_models.py %s --record",
            ", ".join(missing),
            model.id,
        )
        return 1

    root = prepare_store()
    if not args.keep:
        for sub in ("faces", "scores", "results"):
            shutil.rmtree(root / sub, ignore_errors=True)
    reports = reports_dir()
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}__{model.id}.md"
    log_file = report.with_suffix(".log")

    datasets = sorted({c.dataset for e in chosen for c in e.components})
    steps = [["setup_models.py", model.id]]
    steps.append(
        ["setup_datasets.py", *datasets, "--sample", str(args.sample), "--seed", str(args.seed)]
        + ["--manifest"]
    )
    for e in chosen:
        steps.append(
            ["score.py", "--smoke", "--model", model.id, "--evalset", e.id]
            + ["--device", args.device, "--save-crops", "4"]
        )
        steps.append(["evaluate.py", "--smoke", "--model", model.id, "--evalset", e.id])
    failed = []
    for step in steps:
        if run_step(step, log_file):
            failed.append(" ".join(step[:1] + [s for s in step[1:] if not s.startswith("-")][:2]))
            if step[0] in ("setup_models.py", "setup_datasets.py"):
                break

    passed, rows = write_report(registry, model, chosen, args, report, failed)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(f"\n{'PASS' if passed else 'FAIL'}: report at {report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
