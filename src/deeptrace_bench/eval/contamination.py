"""Whether a model may be evaluated on an evalset, given what its weights were trained on.

Zero-shot results are meaningless on data the model saw in training, and the overlap is easy
to miss (HAVIC was fine-tuned on 70% of FakeAVCeleb; AASIST3 was trained on MLAAD). Every
result row carries the verdict, worst first:

- ``contaminated``: an evalset dataset was in the model's train or finetune data and there
  is no list of the items it saw. Refused unless forced, and never a headline number.
- ``overlap_excluded``: it was, but ``items_file`` lists the items seen, so evaluation runs
  on the rest only.
- ``source_overlap``: the model's pretraining data, or a corpus its training data was built
  from, is also a source of the evalset (e.g. Common Voice speech inside XLS-R's
  pretraining). Reported with a warning.
- ``clean``: no overlap we know of.

Official splits are respected: a model trained on FF++ ``train`` evaluated on FF++ ``test``
is clean, which is the standard reproduction setup. "Built from" follows ``derived_from``:
FakeAVCeleb is derived from VoxCeleb2, so a model trained on VoxCeleb2 has seen FakeAVCeleb's
real identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from ..registry import Component, EvalsetConfig, ModelConfig, Registry, TrainingData


class Verdict(IntEnum):
    """Ordered so ``max`` gives the worst verdict."""

    CLEAN = 0
    SOURCE_OVERLAP = 1
    OVERLAP_EXCLUDED = 2
    CONTAMINATED = 3

    @property
    def label(self) -> str:
        """Lower-case name for result tables."""
        return self.name.lower()


@dataclass
class Check:
    """The verdict for one (model, evalset) pair, with reasons and exclusion lists."""

    verdict: Verdict = Verdict.CLEAN
    reasons: list[str] = field(default_factory=list)
    exclude_items_files: list[str] = field(default_factory=list)

    def raise_to(self, verdict: Verdict, reason: str) -> None:
        """Record a finding, keeping the worst verdict."""
        self.verdict = max(self.verdict, verdict)
        self.reasons.append(reason)


def _lineage(source: str, registry: Registry) -> set[str]:
    """A source plus everything it was derived from, transitively."""
    seen: set[str] = set()
    stack = [source]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in registry.datasets:
            stack.extend(registry.datasets[current].derived_from)
    return seen


def _split_disjoint(td: TrainingData, comp: Component) -> bool:
    """True when training used one official split and the component filters to others."""
    wanted = comp.filter.get("split")
    if td.split is None or wanted is None:
        return False
    values = wanted if isinstance(wanted, list) else [wanted]
    return td.split not in values


def check(model: ModelConfig, evalset: EvalsetConfig, registry: Registry) -> Check:
    """Work out the contamination verdict for running ``model`` on ``evalset``."""
    result = Check()
    for comp in evalset.components:
        eval_lineage = _lineage(comp.dataset, registry)
        for td in model.training_data:
            if td.dataset == comp.dataset:
                if _split_disjoint(td, comp):
                    result.reasons.append(
                        f"{model.id} trained on {td.dataset} {td.split}; evaluated on "
                        f"{comp.filter['split']}"
                    )
                elif td.level == "pretrain":
                    result.raise_to(
                        Verdict.SOURCE_OVERLAP, f"{model.id} pretrained on {td.dataset}"
                    )
                elif td.items_file:
                    result.raise_to(
                        Verdict.OVERLAP_EXCLUDED,
                        f"{model.id} {td.level} data includes {td.dataset}; excluding "
                        f"{td.items_file}",
                    )
                    result.exclude_items_files.append(td.items_file)
                else:
                    result.raise_to(
                        Verdict.CONTAMINATED,
                        f"{model.id} {td.level} data includes {td.dataset}, with no list of "
                        "the items it saw",
                    )
                continue
            shared = eval_lineage & _lineage(td.dataset, registry)
            if shared:
                result.raise_to(
                    Verdict.SOURCE_OVERLAP,
                    f"{model.id} {td.level} data {td.dataset} shares {sorted(shared)} with "
                    f"{comp.dataset}",
                )
    return result
