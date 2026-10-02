"""Typed loading and cross-checking of everything under ``configs/``.

Four kinds of config, one YAML file each, named after its ``id``:

- ``configs/models/<id>.yaml``: a detector, where its code and weights come from, and every
  dataset its released weights saw (``training_data``), which drives the contamination guard.
- ``configs/datasets/<id>.yaml``: a dataset, how to get it, its licence terms and what it
  was derived from.
- ``configs/evalsets/<id>.yaml``: what actually gets scored. One or more dataset components,
  each filtered and labelled, plus whether real and fake come from the same corpus.
- ``configs/eval/default.yaml``: the one evaluation setup every run uses.

``configs/corpora.yaml`` lists corpora that models were trained on but that we never
evaluate on (VCTK, LRS2, Common Voice and so on), so training data can be checked by id.
"""

from __future__ import annotations

import importlib
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .paths import CONFIG_DIR


class Modality(StrEnum):
    """Input modality of a model, dataset or evalset."""

    VIDEO = "video"
    AUDIO = "audio"
    AUDIO_VIDEO = "audio_video"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- models -------------------------------------------------------------------------------

ModelStatus = Literal["ready", "workable", "later", "blocked", "dropped", "test"]
WeightKind = Literal[
    "github_release", "github_raw", "url", "hf", "gdrive", "gdrive_folder", "manual"
]
TrainingLevel = Literal["pretrain", "train", "finetune"]


class Upstream(_Strict):
    """A GitHub repo pinned to one commit."""

    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")

    @property
    def slug(self) -> str:
        """Directory name for the checkout, shared by models that use the same repo."""
        return f"{self.repo.split('/')[1]}-{self.commit[:12]}"


class WeightSpec(_Strict):
    """One weight file and where to fetch it."""

    name: str
    kind: WeightKind
    url: str | None = None
    repo: str | None = None
    filename: str | None = None
    revision: str | None = None
    drive_id: str | None = None
    instructions: str | None = None
    size_bytes: int | None = None

    @model_validator(mode="after")
    def _kind_fields(self) -> WeightSpec:
        required = {
            "github_release": ["url"],
            "github_raw": ["url"],
            "url": ["url"],
            "hf": ["repo", "filename"],
            "gdrive": ["drive_id"],
            "gdrive_folder": ["drive_id"],
            "manual": ["instructions"],
        }[self.kind]
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(f"weight {self.name!r} of kind {self.kind} needs {missing}")
        return self


class TrainingData(_Strict):
    """A dataset the released weights saw, and how."""

    dataset: str
    level: TrainingLevel
    split: str | None = None
    items_file: str | None = Field(
        default=None,
        description="Path under DTB_ROOT listing the item ids seen in training. When set, "
        "evaluation can exclude them instead of refusing the dataset outright.",
    )
    note: str | None = None


class ModelConfig(_Strict):
    """A detector."""

    id: str
    name: str
    modality: Modality
    status: ModelStatus
    wave: int | None = None
    licence: str
    paper: str | None = None
    upstream: Upstream | None = None
    weights: list[WeightSpec] = []
    training_data: list[TrainingData] = []
    adapter: str | None = Field(default=None, pattern=r"^[\w.]+:\w+$")
    input: dict[str, Any] = {}
    notes: str | None = None


# --- datasets -----------------------------------------------------------------------------

AccessKind = Literal["hf", "hf_gated", "url", "manual", "modelscope", "none"]
DatasetStatus = Literal["open", "click_through", "request", "deferred", "dead"]
DatasetRole = Literal["reproduction", "south_asian", "real_reference", "in_the_wild"]


class Access(_Strict):
    """How to obtain a dataset."""

    kind: AccessKind
    page: str | None = None
    repo: str | None = None
    allow_patterns: list[str] | None = None
    language_patterns: dict[str, list[str]] | None = None
    default_languages: list[str] | None = None
    urls: dict[str, str] | None = None
    request: str | None = None
    expect: list[str] = []


class DatasetConfig(_Strict):
    """A dataset."""

    id: str
    name: str
    modality: Modality
    role: DatasetRole
    status: DatasetStatus
    access: Access
    licence: str
    licence_terms: list[str] = []
    contains: list[Literal["real", "fake"]]
    languages: list[str] = []
    size_gb: float | None = None
    derived_from: list[str] = []
    builder: str | None = Field(default=None, pattern=r"^[\w.]+:\w+$")
    notes: str | None = None


# --- evalsets -----------------------------------------------------------------------------


class Component(_Strict):
    """One dataset's contribution to an evalset."""

    dataset: str
    filter: dict[str, Any] = {}
    label: Literal["from_manifest", "real", "fake"] = "from_manifest"


class EvalsetConfig(_Strict):
    """What gets scored: components, how real and fake are paired, and what to group by."""

    id: str
    modality: Modality
    role: Literal["reproduction", "south_asian"]
    pairing: Literal["same_corpus", "cross_corpus"]
    components: list[Component] = Field(min_length=1)
    match_on: list[str] = []
    group_by: list[str] = []
    status: Literal["ready", "pending_access", "needs_real_source"]
    notes: str | None = None


class EvalConfig(BaseModel):
    """The shared evaluation setup. Sections are free-form so the settings can evolve."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["proposed", "settled"]
    video: dict[str, Any]
    audio: dict[str, Any]
    metrics: dict[str, Any]
    reporting: dict[str, Any]


class Corpus(_Strict):
    """A corpus that appears only as training or source data."""

    id: str
    name: str
    note: str | None = None


# --- registry -----------------------------------------------------------------------------


def _load_dir(directory: Path, cls: type[BaseModel]) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        obj = cls.model_validate(data)
        if obj.id != path.stem:  # type: ignore[attr-defined]
            raise ValueError(f"{path}: id {obj.id!r} must match the file name")  # type: ignore[attr-defined]
        items[path.stem] = obj
    return items


class Registry:
    """All configs, loaded and cross-checked."""

    def __init__(
        self,
        models: dict[str, ModelConfig],
        datasets: dict[str, DatasetConfig],
        evalsets: dict[str, EvalsetConfig],
        corpora: dict[str, Corpus],
        eval_config: EvalConfig,
    ) -> None:
        self.models = models
        self.datasets = datasets
        self.evalsets = evalsets
        self.corpora = corpora
        self.eval = eval_config

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> Registry:
        """Load every config under ``config_dir``.

        Raises:
            pydantic.ValidationError: if a file doesn't match its schema.
            ValueError: if a file's ``id`` doesn't match its name.
        """
        corpora_raw = yaml.safe_load((config_dir / "corpora.yaml").read_text()) or []
        corpora = {c["id"]: Corpus.model_validate(c) for c in corpora_raw}
        return cls(
            models=_load_dir(config_dir / "models", ModelConfig),
            datasets=_load_dir(config_dir / "datasets", DatasetConfig),
            evalsets=_load_dir(config_dir / "evalsets", EvalsetConfig),
            corpora=corpora,
            eval_config=EvalConfig.model_validate(
                yaml.safe_load((config_dir / "eval" / "default.yaml").read_text())
            ),
        )

    def model(self, model_id: str) -> ModelConfig:
        """Return a model config, with a helpful error for unknown ids."""
        try:
            return self.models[model_id]
        except KeyError:
            raise KeyError(f"unknown model {model_id!r}; known: {sorted(self.models)}") from None

    def dataset(self, dataset_id: str) -> DatasetConfig:
        """Return a dataset config, with a helpful error for unknown ids."""
        try:
            return self.datasets[dataset_id]
        except KeyError:
            raise KeyError(
                f"unknown dataset {dataset_id!r}; known: {sorted(self.datasets)}"
            ) from None

    def evalset(self, evalset_id: str) -> EvalsetConfig:
        """Return an evalset config, with a helpful error for unknown ids."""
        try:
            return self.evalsets[evalset_id]
        except KeyError:
            raise KeyError(
                f"unknown evalset {evalset_id!r}; known: {sorted(self.evalsets)}"
            ) from None

    def known_sources(self) -> set[str]:
        """Ids that training data and ``derived_from`` may refer to."""
        return set(self.datasets) | set(self.corpora)

    def problems(self, import_adapters: bool = False) -> list[str]:
        """Cross-check the configs and return a list of problems (empty when all is well).

        Args:
            import_adapters: also import every adapter and builder to check it resolves.
        """
        found: list[str] = []
        sources = self.known_sources()
        overlap = set(self.datasets) & set(self.corpora)
        if overlap:
            found.append(f"ids are both datasets and corpora: {sorted(overlap)}")

        for m in self.models.values():
            for td in m.training_data:
                if td.dataset not in sources:
                    found.append(f"model {m.id}: training_data {td.dataset!r} is unknown")
            if m.wave is not None:
                if m.status not in ("ready", "workable"):
                    found.append(f"model {m.id}: wave {m.wave} but status {m.status}")
                if m.adapter is None:
                    found.append(f"model {m.id}: in wave {m.wave} but has no adapter")
            if m.status not in ("dropped", "test") and m.upstream is None:
                found.append(f"model {m.id}: no upstream pinned")
            if import_adapters and m.adapter:
                found.extend(_check_import(f"model {m.id}", m.adapter))

        for d in self.datasets.values():
            for src in d.derived_from:
                if src not in sources:
                    found.append(f"dataset {d.id}: derived_from {src!r} is unknown")
            if d.access.kind in ("hf", "hf_gated") and not d.access.repo:
                found.append(f"dataset {d.id}: access kind {d.access.kind} needs a repo")
            if d.access.kind == "url" and not d.access.urls:
                found.append(f"dataset {d.id}: access kind url needs urls")
            if d.access.kind == "manual" and not d.access.request:
                found.append(f"dataset {d.id}: manual access needs request instructions")
            if import_adapters and d.builder:
                found.extend(_check_import(f"dataset {d.id}", d.builder))

        for e in self.evalsets.values():
            for comp in e.components:
                if comp.dataset not in self.datasets:
                    found.append(f"evalset {e.id}: component dataset {comp.dataset!r} is unknown")
                    continue
                ds = self.datasets[comp.dataset]
                if not _modality_fits(e.modality, ds.modality):
                    found.append(f"evalset {e.id}: {e.modality} evalset uses {ds.modality} {ds.id}")
                if comp.label != "from_manifest" and comp.label not in ds.contains:
                    found.append(
                        f"evalset {e.id}: {ds.id} has no {comp.label} items ({ds.contains})"
                    )
            forced = {c.label for c in e.components}
            if "from_manifest" not in forced and len(forced) < 2 and e.status == "ready":
                found.append(f"evalset {e.id}: ready but only {forced} items")
        return found


def _modality_fits(evalset: Modality, dataset: Modality) -> bool:
    if evalset == dataset:
        return True
    # A video evalset can take the frames of an audio-video dataset, and an audio evalset
    # its audio track.
    return dataset == Modality.AUDIO_VIDEO


def _check_import(owner: str, target: str) -> list[str]:
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report any import failure as a problem
        return [f"{owner}: cannot import {module_name}: {exc}"]
    if not hasattr(module, attr):
        return [f"{owner}: {module_name} has no {attr}"]
    return []


def resolve(target: str) -> Any:
    """Import ``package.module:attr`` and return the attribute."""
    module_name, _, attr = target.partition(":")
    if not re.fullmatch(r"[\w.]+", module_name) or not attr:
        raise ValueError(f"bad import target {target!r}")
    return getattr(importlib.import_module(module_name), attr)
