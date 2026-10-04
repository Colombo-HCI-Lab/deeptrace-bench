"""Dataset builders read their layouts (invented names; no real files)."""

from __future__ import annotations

import pytest

from deeptrace_bench.datasets import (
    in_the_wild,
    mendeley_roop_akool,
    unidatapro_videos,
    urdu_csalt,
)
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


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("Bonafide/Speaker_01/Part 1/7.wav", "real"),
        ("Spoofed_TTS/Speaker_01/7.wav", "fake"),
        ("Spoofed_Tacotron/Speaker_02/9.wav", "fake"),
        ("README.md", None),
        ("Other/Speaker_01/7.wav", None),
    ],
)
def test_urdu_labels(path, label):
    assert urdu_csalt.label_from_path(path) == label


def test_urdu_manifest(tmp_path):
    _tree(
        tmp_path,
        [
            "Bonafide/Speaker_01/Part 1/7.wav",
            "Bonafide/Speaker_01/Part 2/7.wav",
            "Spoofed_TTS/Speaker_01/7.wav",
            "Spoofed_Tacotron/Speaker_02/9.wav",
            "README.md",
            ".sample.json",
        ],
    )
    df = urdu_csalt.build_manifest(tmp_path)
    validate_manifest(df, "urdu_csalt")
    assert len(df) == 4 and set(df["language"]) == {"ur"}
    # The same file name recurs across parts, so the part stays in the id.
    assert df["item_id"].is_unique
    by_path = df.set_index("rel_path")
    assert by_path.loc["Spoofed_TTS/Speaker_01/7.wav", "method"] == "vits"
    assert by_path.loc["Spoofed_Tacotron/Speaker_02/9.wav", "method"] == "tacotron"
    assert by_path.loc["Bonafide/Speaker_01/Part 2/7.wav", "subject_id"] == "Speaker_01"


_ITW_META = "file,speaker,label\n1.wav,Speaker A,bona-fide\n2.wav,Speaker A,spoof\n3.wav,Speaker B,spoof\n"


def test_in_the_wild_labels_come_from_its_metadata():
    labels = in_the_wild.labels_from_metadata({in_the_wild.META: _ITW_META.encode()})
    assert labels == {
        "release_in_the_wild/1.wav": "real",
        "release_in_the_wild/2.wav": "fake",
        "release_in_the_wild/3.wav": "fake",
    }


def test_in_the_wild_manifest_covers_only_files_present(tmp_path):
    _tree(tmp_path, ["release_in_the_wild/1.wav", "release_in_the_wild/3.wav", ".sample.json"])
    (tmp_path / in_the_wild.META).write_text(_ITW_META)
    df = in_the_wild.build_manifest(tmp_path)
    validate_manifest(df, "in_the_wild")
    assert sorted(df["label"]) == ["fake", "real"]
    assert set(df["subject_id"]) == {"Speaker A", "Speaker B"}
