"""Shared fixtures. Tests use synthetic data only: nothing derived from a real dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from deeptrace_bench.registry import Registry


@pytest.fixture(scope="session")
def registry() -> Registry:
    """The repo's real configs."""
    return Registry.load()


def make_manifest(dataset: str, rows: list[dict]) -> pd.DataFrame:
    """Build a small valid manifest; each row needs at least ``local`` and ``label``."""
    records = []
    for r in rows:
        rec = {
            "item_id": f"{dataset}/{r['local']}",
            "dataset": dataset,
            "rel_path": f"{r['local']}.wav",
            "modality": "audio",
            "label": r["label"],
        }
        rec.update({k: v for k, v in r.items() if k not in ("local", "label")})
        records.append(rec)
    return pd.DataFrame(records)
