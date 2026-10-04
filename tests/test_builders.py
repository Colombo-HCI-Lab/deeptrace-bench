"""The smoke datasets' builders read their layouts (invented names; no real files)."""

from __future__ import annotations

import pytest

from deeptrace_bench.datasets import mendeley_roop_akool, unidatapro_videos
from deeptrace_bench.manifest import validate_manifest


def _tree(root, paths):
    for rel in paths:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("deepfake_dataset/real/7.mp4_frame_1_face_0.jpg", "real"),
        ("deepfake_dataset/deepfake/PersonA_Vid-1_Tech-2_frame_3_face_0.jpg", "fake"),
        ("__MACOSX/deepfake_dataset/real/._7.mp4_frame_1_face_0.jpg", None),
        ("deepfake_dataset/real/.DS_Store", None),
        ("deepfake_dataset/notes.txt", None),
    ],
)
def test_mendeley_labels(path, label):
    assert mendeley_roop_akool.label_from_path(path) == label


def test_mendeley_manifest(tmp_path):
    _tree(
        tmp_path,
        [
            "deepfake_dataset/real/7.mp4_frame_1_face_0.jpg",
            "deepfake_dataset/real/male_male_2.mp4_frame_3_face_0.jpg",
            "deepfake_dataset/deepfake/PersonA_Vid-1_Tech-2_frame_3_face_0.jpg",
            "deepfake_dataset/deepfake/PersonB_4_Tech-01.mp4_frame_9_face_0.jpg",
            "__MACOSX/deepfake_dataset/real/._7.mp4_frame_1_face_0.jpg",
            ".sample.json",
        ],
    )
    df = mendeley_roop_akool.build_manifest(tmp_path)
    validate_manifest(df, "mendeley_roop_akool")
    assert len(df) == 4 and set(df["modality"]) == {"image"}
    fakes = df[df["label"] == "fake"].set_index("rel_path")["method"]
    assert sorted(fakes) == ["tech_1", "tech_2"]
    assert "mendeley_roop_akool/real/7.mp4_frame_1_face_0" in set(df["item_id"])


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("video/1.mp4", "real"),
        ("video/5.MOV", "real"),
        ("deepfake/5.mov", "fake"),
        ("image/1.jpg", None),
        ("DeepFake Videos Dataset.csv", None),
    ],
)
def test_unidatapro_labels(path, label):
    assert unidatapro_videos.label_from_path(path) == label


def test_unidatapro_manifest(tmp_path):
    _tree(tmp_path, ["video/1.mp4", "video/5.MOV", "deepfake/5.mov", "image/1.jpg", "meta.csv"])
    df = unidatapro_videos.build_manifest(tmp_path)
    validate_manifest(df, "unidatapro_videos")
    assert sorted(df["item_id"]) == [
        "unidatapro_videos/deepfake/5",
        "unidatapro_videos/video/1",
        "unidatapro_videos/video/5",
    ]
