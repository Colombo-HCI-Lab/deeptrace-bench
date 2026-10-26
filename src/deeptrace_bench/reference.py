"""The site's model and dataset reference: every model and dataset described for newcomers.

Written to ``docs/src/content/docs/catalog.md`` from the same data as the status page
(``status.collect``), so the two never disagree. It is for reading on GitHub: the site builds its
Model and Dataset catalogs, a page per entry, from ``status.json`` directly. The prose comes from
``docs/src/reference/catalog.yaml``; sizes, costs and citation counts are
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


def held_languages(dataset_id: str) -> list[str] | None:
    """Languages the full manifest holds, or None without a full build or a language column.

    A partial download (IndicSynth's Hindi of its 12 languages) then reads as what it is.
    """
    path = dtb_root() / "manifests" / f"{dataset_id}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "language" not in df or df["language"].isna().all():
        return None
    return sorted(df["language"].dropna().unique())


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


def _why(entry: dict[str, Any]) -> str:
    """Why an entry is here, flagged when it is a cut candidate."""
    why = entry.get("why") or ""
    return f"Cut candidate. {why}" if entry.get("cut") else why


def _published(rows: list[dict[str, Any]]) -> str:
    return "; ".join(f"{r['test']}: {r['metric']} {r['value']} ({r['source']})" for r in rows)


def _ours(runs: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"`{r['evalset']}`: AUC {r['auc']:.3f}, EER {100 * r['eer']:.1f}%" for r in runs
    )


def _kinds(ids: list[str] | None, kinds: dict[str, str]) -> str:
    """Kinds of fake by label; None for an untagged entry, real only for an empty list."""
    if ids is None:
        return ""
    return ", ".join(kinds.get(k, k) for k in ids) if ids else "none: real clips only"


def _model(m: dict[str, Any], model_runs: list[dict[str, Any]], kinds: dict[str, str]) -> list[str]:
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
    if runs := len(model_runs):
        status += f", scored on {runs} evalset{'s' if runs > 1 else ''}"
    out += _table(
        [
            ("Why it's here", _why(m)),
            ("Kinds of fake it trained on", _kinds(m.get("catches"), kinds)),
            ("What it looks at", m.get("cue") or ""),
            ("How it works", m.get("architecture") or ""),
            ("Input", m.get("input") or ""),
            ("Paper", _paper(m.get("paper"))),
            ("Training", training),
            ("Reported by the authors", _published(m.get("published") or [])),
            ("Our runs", _ours(model_runs)),
            ("Size", f"{_params(m['profile']['params'])} parameters" if m.get("profile") else ""),
            ("Cost", cost or ""),
            ("Licence", m["licence"]),
            ("Status here", status),
        ]
    )
    return [*out, ""]


def _dataset(d: dict[str, Any], kinds: dict[str, str]) -> list[str]:
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
            ("Why it's here", _why(d)),
            ("Media", f"{MODALITIES[d['modality']]}, {' and '.join(d['contains'])}"),
            ("Languages", _languages(d["languages"])),
            ("Size", d.get("scale") or ""),
            ("Kinds of fake", _kinds(d.get("fakes"), kinds)),
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
    kinds = {t["id"]: t["label"] for t in data.get("fake_types") or []}
    runs: dict[str, list[dict[str, Any]]] = {}
    for r in data["runs"]:
        runs.setdefault(r["model"], []).append(r)
    out = [
        "---",
        "title: Models and datasets",
        'description: "Every detector and dataset: training data, published and our numbers, '
        'size, cost, what each records"',
        "---",
        "",
        f"Generated by `scripts/build_site.py` on {data['built']} from the configs, "
        "`docs/src/reference/catalog.yaml` (the descriptions), measured compute profiles and "
        f"Semantic Scholar citation counts (checked {data.get('citations_checked') or 'never'}). "
        "Edit those, not this file.",
        "",
        "- [Models](#models): architecture, training data, published and our numbers, size, cost",
        "- [Datasets](#datasets): what each dataset contains and which features it records",
        "",
        "## Models",
        "",
    ]
    for modality in ("audio", "video", "audio_video"):
        models = [m for m in data["models"] if m["modality"] == modality]
        out += [f"### {MODALITIES[modality]} ({len(models)})", ""]
        for m in sorted(models, key=lambda m: m["name"].lower()):
            out += _model(m, runs.get(m["id"], []), kinds)

    datasets = data["datasets"]
    built = [d for d in datasets if d.get("recorded")]
    out += ["## Datasets", ""]
    if data.get("fake_types"):
        out += [
            "### Kinds of fake",
            "",
            "What each kind of fake changes and what it tends to leave behind. A detector trained "
            "on one kind often misses another.",
            "",
            "| Kind | Track | What's changed | What it leaves | Datasets |",
            "| --- | --- | --- | --- | --- |",
        ]
        for t in data["fake_types"]:
            having = sorted(
                d["name"]
                for d in datasets
                if t["id"] in (d.get("fakes") or []) and d["role"] != "pipeline_test"
            )
            out.append(
                f"| {t['label']} | {t['modality']} | {_cell(t['changed'])} | {_cell(t['trace'])} "
                f"| {_cell(', '.join(having)) or 'none'} |"
            )
        out.append("")
    out += [
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
            out += _dataset(d, kinds)
    return "\n".join(out).rstrip() + "\n"
