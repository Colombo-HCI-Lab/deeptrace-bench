"""The Markdown reference, ``docs/catalog.md``: every model and dataset described for newcomers.

Rendered from the same data as the status page (``status.collect``), so the two never
disagree. The prose comes from ``docs/catalog.yaml``; sizes, costs and citation counts are
measured or looked up; the features a dataset records come from its manifest.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .paths import dtb_root
from .registry import ModelConfig

LANGUAGES = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri", "en": "English",
    "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam", "mni": "Manipuri",
    "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit",
    "si": "Sinhala", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}  # fmt: skip
ACCESS = {
    "open": "Open download",
    "click_through": "Accept terms online, then download",
    "request": "On request from the authors",
    "deferred": "Deferred (too large for now)",
    "dead": "Unavailable",
}
ROLES = {
    "south_asian": "South Asian",
    "reproduction": "Reproduction",
    "real_reference": "Real reference",
    "in_the_wild": "In the wild",
    "pipeline_test": "Pipeline tests only",
}
MODALITIES = {"audio": "Audio", "video": "Video", "audio_video": "Audio-visual", "image": "Image"}
LEVELS = {"pretrain": "pretrained on", "train": "trained on", "finetune": "fine-tuned on"}
# Manifest columns, in the order and words the reference uses for them.
FEATURES = {
    "method": "generator",
    "method_family": "kind of fake",
    "language": "language",
    "subject_id": "speaker or face id",
    "source_subject_id": "source identity of each fake",
    "split": "official split",
    "label_video": "which track was faked",
    "g_gender": "gender",
    "g_age": "age",
    "g_race": "race",
    "g_skin_tone": "skin tone",
    "g_subcorpus": "sub-corpus",
}
GLANCE = [
    "generator",
    "language",
    "speaker or face id",
    "source identity of each fake",
    "gender",
    "age",
    "race",
    "sub-corpus",
    "official split",
]


def model_input(model: ModelConfig) -> str | None:
    """What one model reads, in words."""
    spec = model.input
    if "segment_samples" in spec:
        seconds = spec["segment_samples"] / spec["sample_rate"]
        return f"{seconds:.2f} s windows of raw {spec['sample_rate'] / 1000:g} kHz audio"
    if spec.get("inputs") == "mouths":
        return (
            f"mouth crops from {spec['clip_frames']} consecutive frames at a time "
            f"(up to {spec['max_frames']} frames per video)"
        )
    crop = spec.get("crop")
    if not crop:
        return None
    size = f"{crop['size']} px" if crop.get("size") else "native-size"
    kind = "aligned face crops" if crop.get("align") == "five_point" else "face-box crops"
    if "frame_rate" in spec:
        return (
            f"{size} {kind} at {spec['frame_rate']} frames per second "
            f"(up to {spec['max_frames']}), with the audio under them"
        )
    return f"{size} {kind}, {spec.get('frames_per_clip', 32)} frames per video"


def recorded_features(dataset_id: str) -> tuple[str, list[str]] | None:
    """Features the dataset's manifest fills in, from the full manifest or else the sample.

    Returns ``(source, features)`` with source ``full`` or ``sample``, or None if no manifest
    has been built. Reads column contents only to see which are filled.
    """
    folders = {"full": dtb_root() / "manifests", "sample": dtb_root() / "smoke" / "manifests"}
    for source, folder in folders.items():
        path = folder / f"{dataset_id}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            return source, [FEATURES[c] for c in FEATURES if c in df and df[c].notna().any()]
    return None


def _languages(codes: list[str]) -> str:
    return ", ".join(LANGUAGES.get(c, c) for c in codes) or "not stated"


def _paper(paper: dict[str, Any] | None) -> str:
    if not paper:
        return "no paper"
    venue = paper.get("venue") or paper.get("year")
    cites = f" · {paper['citations']:,} citations" if paper.get("citations") is not None else ""
    return f"[{paper['title']}]({paper['url']}), {venue}{cites}"


def _params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M" if n < 1e7 else f"{n / 1e6:.0f} M"
    return f"{n / 1e3:.0f} K"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(rows: list[tuple[str, str]]) -> list[str]:
    lines = ["| | |", "| --- | --- |"]
    lines += [f"| {k} | {_cell(v)} |" for k, v in rows if v]
    return lines


def _model(m: dict[str, Any], runs: int) -> list[str]:
    out = [f'<a id="model-{m["id"]}"></a>', "", f"#### {m['name']}", ""]
    out += [m.get("description") or m.get("summary") or "No description yet.", ""]
    training = "; ".join(
        f"{LEVELS[t['level']]} {t['name']}" + (f" ({t['split']} split)" if t.get("split") else "")
        for t in m["training"]
    )
    cost = None
    if p := m.get("profile"):
        unit = "window" if m["modality"] == "audio" else p["unit"].replace("face ", "")
        gflops = f"{p['gflops_per_unit']:.1f} GFLOPs and " if p.get("gflops_per_unit") else ""
        cost = f"{gflops}{p['ms_per_unit']:.1f} ms per {unit} on a {p['gpu']} ({p['precision']})"
    status = "blocked: weights unreachable" if m["blocked"] else "adapter written"
    if m.get("parity"):
        status += ", parity checked"
    if runs:
        status += f", scored on {runs} evalset{'s' if runs > 1 else ''}"
    out += _table(
        [
            ("How it works", m.get("architecture") or ""),
            ("Input", m.get("input") or ""),
            ("Paper", _paper(m.get("paper"))),
            ("Training", training),
            ("Size", f"{_params(m['profile']['params'])} parameters" if m.get("profile") else ""),
            ("Cost", cost or ""),
            ("Licence", m["licence"]),
            ("Status here", status),
        ]
    )
    return [*out, ""]


def _dataset(d: dict[str, Any]) -> list[str]:
    out = [f'<a id="dataset-{d["id"]}"></a>', "", f"#### {d['name']}", ""]
    out += [d.get("summary") or "No description yet.", ""]
    generators = d.get("generators")
    if generators is not None:
        generators = "; ".join(generators) if generators else "none: real clips only"
    recorded = d.get("recorded")
    if recorded:
        source, features = recorded
        recorded_text = ", ".join(["real or fake", *features])
        if source == "sample":
            recorded_text += " (from a sampled build)"
    else:
        recorded_text = "not built yet"
    out += _table(
        [
            ("Kind", f"{MODALITIES[d['modality']]}, {' and '.join(d['contains'])}"),
            ("Languages", _languages(d["languages"])),
            ("Size", d.get("scale") or ""),
            ("How fakes were made", generators or ""),
            ("Labels it ships", "; ".join(d.get("annotations") or [])),
            ("Format", d.get("format") or ""),
            ("Features in our manifest", recorded_text),
            ("Paper", _paper(d.get("paper"))),
            ("Licence", d["licence"]),
            ("Access", ACCESS.get(d["status"], d["status"])),
            ("Used in evalsets", ", ".join(f"`{e}`" for e in d["evalsets"])),
        ]
    )
    if d.get("notes"):
        out += ["", f"Note: {d['notes']}"]
    return [*out, ""]


def render(data: dict[str, Any]) -> str:
    """The whole reference as Markdown."""
    runs: dict[str, int] = {}
    for r in data["runs"]:
        runs[r["model"]] = runs.get(r["model"], 0) + 1
    out = [
        "---",
        "title: Models and datasets",
        "---",
        "",
        "# Models and datasets",
        "",
        "A plain-language reference for every detector and dataset in deeptrace-bench. "
        "New to the terms? Start with [How to read the status page]"
        "(https://colombo-hci-lab.github.io/deeptrace-bench/#start).",
        "",
        f"Generated by `scripts/build_site.py` on {data['built']} from the configs, "
        "`docs/catalog.yaml` (the descriptions), measured compute profiles and Semantic "
        f"Scholar citation counts (checked {data.get('citations_checked') or 'never'}). "
        "Edit those, not this file.",
        "",
        "- [Models](#models): what each detector looks at, its paper, size and cost",
        "- [Datasets](#datasets): what each dataset contains and which features it records",
        "",
        "## Models",
        "",
    ]
    for modality in ("audio", "video", "audio_video"):
        models = [m for m in data["models"] if m["modality"] == modality]
        out += [f"### {MODALITIES[modality]} ({len(models)})", ""]
        for m in sorted(models, key=lambda m: m["name"].lower()):
            out += _model(m, runs.get(m["id"], 0))

    datasets = data["datasets"]
    built = [d for d in datasets if d.get("recorded")]
    out += [
        "## Datasets",
        "",
        "### Features at a glance",
        "",
        "Which features each built dataset records per clip, read from its manifest, so these "
        "are the groups results can be split by today. Every dataset also records real or fake.",
        "",
        "| Dataset | " + " | ".join(GLANCE) + " |",
        "| --- | " + " | ".join(":---:" for _ in GLANCE) + " |",
    ]
    for d in sorted(built, key=lambda d: d["name"].lower()):
        features = d["recorded"][1]
        marks = " | ".join("✓" if f in features else "" for f in GLANCE)
        out.append(f"| [{_cell(d['name'])}](#dataset-{d['id']}) | {marks} |")
    unbuilt = sorted(d["name"] for d in datasets if not d.get("recorded"))
    out += [
        "",
        f"Not built yet, so not in the table: {', '.join(unbuilt)}. Their sections below "
        "list what the dataset itself provides.",
        "",
    ]
    for role, label in ROLES.items():
        group = [d for d in datasets if d["role"] == role]
        if not group:
            continue
        out += [f"### {label} ({len(group)})", ""]
        for d in sorted(group, key=lambda d: d["name"].lower()):
            out += _dataset(d)
    return "\n".join(out).rstrip() + "\n"
