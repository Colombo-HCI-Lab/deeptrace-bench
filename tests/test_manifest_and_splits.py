"""Manifest validation, summaries and identity-safe splits."""

from __future__ import annotations

import pandas as pd
import pytest

from deeptrace_bench.manifest import ManifestError, content_hash, summarize, validate_manifest
from deeptrace_bench.splits import identity_safe_split

from .conftest import make_manifest


def test_valid_manifest_passes():
    df = make_manifest(
        "toy",
        [
            {"local": "a", "label": "real", "g_gender": "f", "g_gender_src": "dataset"},
            {"local": "b", "label": "fake", "g_gender": None, "g_gender_src": None},
        ],
    )
    validate_manifest(df, "toy")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"local": "a", "label": "bonafide"}], "label must be real or fake"),
        ([{"local": "a", "label": "real"}, {"local": "a", "label": "fake"}], "unique"),
        ([{"local": "a", "label": "real", "g_gender": "f"}], "needs a g_gender_src"),
        (
            [{"local": "a", "label": "real", "g_gender": "unknown", "g_gender_src": "dataset"}],
            "placeholder",
        ),
        ([{"local": "a", "label": "real", "gender": "f"}], "g_ prefix"),
    ],
)
def test_invalid_manifests_are_rejected(rows, message):
    with pytest.raises(ManifestError, match=message):
        validate_manifest(make_manifest("toy", rows), "toy")


def test_summary_names_no_items_and_hash_ignores_order():
    df = make_manifest(
        "toy",
        [
            {"local": "a", "label": "real", "g_gender": "f", "g_gender_src": "dataset"},
            {"local": "b", "label": "fake", "g_gender": "m", "g_gender_src": "dataset"},
        ],
    )
    summary = summarize(df)
    assert summary["rows"] == 2
    assert "toy/a" not in str(summary)
    assert content_hash(df) == content_hash(df.iloc[::-1])


def test_splits_never_share_an_identity():
    rows = []
    for speaker in range(30):
        rows.append({"local": f"r{speaker}", "label": "real", "subject_id": f"s{speaker}"})
        # A fake of this speaker, made from the next speaker's voice, links the two.
        rows.append(
            {
                "local": f"f{speaker}",
                "label": "fake",
                "subject_id": f"s{speaker}",
                "source_subject_id": f"s{(speaker + 1) % 30 if speaker % 3 else speaker}",
            }
        )
    df = make_manifest("toy", rows)
    split = identity_safe_split(df, seed=3)
    # An identity may appear as a subject in one row and a source in another; either way it
    # must sit in one split only.
    both = pd.concat(
        [
            pd.DataFrame({"id": df["subject_id"], "split": split}),
            pd.DataFrame({"id": df["source_subject_id"], "split": split}),
        ]
    ).dropna()
    assert (both.groupby("id")["split"].nunique() == 1).all()
    assert split.nunique() > 1


def test_splits_are_deterministic_and_roughly_sized():
    df = make_manifest(
        "toy", [{"local": str(i), "label": "real", "subject_id": f"s{i}"} for i in range(200)]
    )
    a = identity_safe_split(df, seed=7)
    assert a.equals(identity_safe_split(df, seed=7))
    shares = a.value_counts(normalize=True)
    assert abs(shares["train"] - 0.7) < 0.05
    assert abs(shares["test"] - 0.2) < 0.05


@pytest.mark.parametrize(
    "row",
    [
        {"local": "a", "label": "real", "label_video": "fake", "label_audio": "real"},
        {"local": "b", "label": "fake", "label_video": "real", "label_audio": "real"},
        {"local": "c", "label": "fake", "label_video": "maybe", "label_audio": None},
    ],
)
def test_track_labels_must_agree_with_the_item_label(row):
    with pytest.raises(ManifestError, match="track"):
        validate_manifest(make_manifest("av", [row]), "av")


def test_track_labels_may_be_unknown():
    rows = [
        {"local": "a", "label": "fake", "label_video": "fake", "label_audio": None},
        {"local": "b", "label": "fake", "label_video": "real", "label_audio": "fake"},
        {"local": "c", "label": "real", "label_video": "real", "label_audio": "real"},
    ]
    validate_manifest(make_manifest("av", rows), "av")
