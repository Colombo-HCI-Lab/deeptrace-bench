"""Identity-safe train/val/test splits for fine-tuning.

Splitting items at random leaks identities: the same speaker or face lands on both sides, and
a fake made from person A's voice can sit in test while A's real clips sit in train. Here
items are linked whenever they share a ``subject_id`` or a ``source_subject_id``, each linked
group (a connected component) goes to one split as a whole, and groups are dealt out until
each split reaches its share of items.

Assignments are data (they name items), so they're written under ``DTB_ROOT/splits/``; the
seed and this code are what live in git.
"""

from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def identity_components(df: pd.DataFrame) -> pd.Series:
    """Label each row with the id of its identity component.

    Rows with neither identity column set form their own component.
    """
    uf = _UnionFind()
    keys: list[str] = []
    for item_id, subj, src in zip(
        df["item_id"],
        df.get("subject_id", pd.Series([None] * len(df), index=df.index)),
        df.get("source_subject_id", pd.Series([None] * len(df), index=df.index)),
        strict=True,
    ):
        node = f"item:{item_id}"
        for ident in (subj, src):
            if pd.notna(ident):
                uf.union(f"id:{ident}", node)
        keys.append(node)
    return pd.Series([uf.find(k) for k in keys], index=df.index, name="component")


def identity_safe_split(
    df: pd.DataFrame,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> pd.Series:
    """Assign every row to a split so that no identity spans two splits.

    Args:
        df: a manifest (needs ``item_id``; uses ``subject_id`` and ``source_subject_id``).
        fractions: target share of items per split; must sum to 1. Defaults to 70/10/20.
        seed: shuffles the order components are dealt in.

    Returns:
        A Series of split names aligned with ``df``.

    Raises:
        ValueError: if the fractions don't sum to 1.
    """
    fractions = fractions or {"train": 0.7, "val": 0.1, "test": 0.2}
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    components = identity_components(df)
    sizes: dict[str, int] = defaultdict(int)
    for comp in components:
        sizes[comp] += 1
    order = sorted(sizes)
    random.Random(seed).shuffle(order)

    total = len(df)
    targets = {name: share * total for name, share in fractions.items()}
    filled = dict.fromkeys(fractions, 0)
    assigned: dict[str, str] = {}
    for comp in order:
        # Give each component to the split furthest below its target, relative to its size.
        name = max(targets, key=lambda n: (targets[n] - filled[n]) / max(targets[n], 1e-9))
        assigned[comp] = name
        filled[name] += sizes[comp]
    return components.map(assigned).rename("split")
