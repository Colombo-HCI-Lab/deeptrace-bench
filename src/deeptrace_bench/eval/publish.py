"""What may be published to the repo's ``results/``.

Publishing copies a run's aggregate tables into git, where they read as results. Pipeline
tests and anything in a namespace prove the plumbing works and nothing more, so they are
refused here rather than relying on someone remembering not to.
"""

from __future__ import annotations

from ..registry import EvalsetConfig


def publish_refusal(evalset: EvalsetConfig, namespace: str | None) -> str | None:
    """Why a run may not be published, or None if it may."""
    if evalset.role == "pipeline_test":
        return f"{evalset.id} is a pipeline test; its numbers are never reported"
    if namespace is not None:
        return f"runs in the {namespace!r} namespace are pipeline tests, never reported"
    return None
