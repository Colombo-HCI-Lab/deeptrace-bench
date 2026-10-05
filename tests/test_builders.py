"""Dataset builders read their layouts (invented names; no real files)."""

from __future__ import annotations

import pandas as pd
import pytest

from deeptrace_bench.datasets import (
    asvspoof2019_la,
    banglafake,
    celeb_df_v2,
    in_the_wild,
    mavos_dd_hi,
    mendeley_roop_akool,
    mlaad,
    openslr_sinhala,
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


_ITW_META = "\n".join(
    [
        "file,speaker,label",
        "1.wav,Speaker A,bona-fide",
        "2.wav,Speaker A,spoof",
        "3.wav,Speaker B,spoof",
    ]
)


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


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("final_data/deepfake_data_sust/real_wav/01001.wav", "real"),
        ("final_data/deepfake_data_mozilla/deepfake_wav/common_voice_s2_7.wav", "fake"),
        ("final_data/deepfake_data_news/real_wav/3.wav", "real"),
        ("final_data/deepfake_data_news/metadata.csv", None),
    ],
)
def test_banglafake_labels(path, label):
    assert banglafake.label_from_path(path) == label


def test_banglafake_manifest(tmp_path):
    _tree(
        tmp_path,
        [
            "final_data/deepfake_data_sust/real_wav/01001.wav",
            "final_data/deepfake_data_sust/deepfake_wav/10001.wav",
            "final_data/deepfake_data_mozilla/real_wav/common_voice_s2_7.wav",
            "final_data/deepfake_data_mozilla/deepfake_wav/common_voice_s2_7.wav",
            "final_data/deepfake_data_news/real_wav/3.wav",
        ],
    )
    df = banglafake.build_manifest(tmp_path)
    validate_manifest(df, "banglafake")
    assert len(df) == 5 and df["item_id"].is_unique
    assert set(df["g_subcorpus"]) == {"sust", "mozilla", "news"}
    mozilla = df[df["g_subcorpus"] == "mozilla"]
    assert set(mozilla["subject_id"]) == {"mozilla_s2"}
    assert df.loc[df["label"] == "fake", "method"].eq("vits").all()


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("fake/si/Edge-TTS/story_01_f000001.wav", "fake"),
        ("fake/hi/tts_models_hi_x/story_02_f000002.wav", "fake"),
        ("fake/si/Edge-TTS/meta.csv", None),
        ("README.md", None),
    ],
)
def test_mlaad_labels(path, label):
    assert mlaad.label_from_path(path) == label


def test_mlaad_manifest(tmp_path):
    _tree(
        tmp_path,
        [
            "fake/si/Edge-TTS/story_01_f000001.wav",
            "fake/hi/tts_models_hi_x/story_02_f000002.wav",
            "fake/si/Edge-TTS/meta.csv",
        ],
    )
    df = mlaad.build_manifest(tmp_path)
    validate_manifest(df, "mlaad")
    assert set(df["label"]) == {"fake"}
    assert dict(zip(df["language"], df["method"], strict=True)) == {
        "si": "edge-tts",
        "hi": "tts_models_hi_x",
    }


def test_openslr_sinhala_manifest(tmp_path):
    _tree(
        tmp_path,
        ["asr_sinhala/data/00/00aa.flac", "asr_sinhala/data/0b/0bcd.flac", "asr_sinhala/LICENSE"],
    )
    (tmp_path / "asr_sinhala/utt_spk_text.tsv").write_text(
        '00aa\tspk1\t"opens a quote\n0bcd\tspk2\ttext\n'
    )
    assert openslr_sinhala.label_from_path("asr_sinhala/data/00/00aa.flac") == "real"
    assert openslr_sinhala.label_from_path("asr_sinhala/LICENSE") is None
    df = openslr_sinhala.build_manifest(tmp_path)
    validate_manifest(df, "openslr_sinhala")
    assert set(df["label"]) == {"real"} and set(df["language"]) == {"si"}
    assert set(df["subject_id"]) == {"spk1", "spk2"}


_PROTOCOLS = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.txt"


def _asvspoof_protocols() -> dict[str, bytes]:
    return {
        _PROTOCOLS.format(
            "train.trn"
        ): b"LA_0001 LA_T_1 - - bonafide\nLA_0001 LA_T_2 - A01 spoof\n",
        _PROTOCOLS.format("dev.trl"): b"",
        _PROTOCOLS.format("eval.trl"): b"LA_0002 LA_E_3 - A17 spoof\n",
    }


def test_asvspoof_labels_come_from_its_protocols():
    labels = asvspoof2019_la.labels_from_metadata(_asvspoof_protocols())
    assert labels == {
        "LA/ASVspoof2019_LA_train/flac/LA_T_1.flac": "real",
        "LA/ASVspoof2019_LA_train/flac/LA_T_2.flac": "fake",
        "LA/ASVspoof2019_LA_eval/flac/LA_E_3.flac": "fake",
    }


def test_asvspoof_manifest(tmp_path):
    for name, text in _asvspoof_protocols().items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(text)
    _tree(
        tmp_path,
        ["LA/ASVspoof2019_LA_train/flac/LA_T_1.flac", "LA/ASVspoof2019_LA_eval/flac/LA_E_3.flac"],
    )
    df = asvspoof2019_la.build_manifest(tmp_path)
    validate_manifest(df, "asvspoof2019_la")
    rows = df.set_index("split")
    assert rows.loc["train", "label"] == "real" and pd.isna(rows.loc["train", "method"])
    assert rows.loc["eval", "method"] == "A17" and rows.loc["eval", "method_family"] == "vc"
    assert rows.loc["eval", "subject_id"] == "LA_0002"


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("hindi/real/0001-aa.mp4", "real"),
        ("hindi/knnvc/0002-bb.mp4", "fake"),
        ("hindi/inswapper/0003-cc.mp4", "fake"),
        ("english/real/0004-dd.mp4", None),
        ("hindi/unknown_method/0005-ee.mp4", None),
    ],
)
def test_mavos_labels(path, label):
    assert mavos_dd_hi.label_from_path(path) == label


def test_mavos_manifest_labels_each_track(tmp_path):
    _tree(
        tmp_path,
        [
            "hindi/real/0001-aa.mp4",
            "hindi/knnvc/0002-bb.mp4",
            "hindi/inswapper/0003-cc.mp4",
            "hindi/liveportrait/0004-dd.mp4",
        ],
    )
    df = mavos_dd_hi.build_manifest(tmp_path)
    validate_manifest(df, "mavos_dd_hi")
    tracks = df.set_index("method")[["label_video", "label_audio"]]
    # Voice conversion leaves the video real; nobody has checked a face swap's audio yet.
    assert tracks.loc["knnvc"].tolist() == ["real", "fake"]
    assert tracks.loc["inswapper", "label_video"] == "fake"
    assert pd.isna(tracks.loc["inswapper", "label_audio"])
    assert set(df["language"]) == {"hi"} and set(df["modality"]) == {"audio_video"}


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("Celeb-real/id0_0001.mp4", "real"),
        ("YouTube-real/00001.mp4", "real"),
        ("Celeb-synthesis/id0_id1_0001.mp4", "fake"),
        ("List_of_testing_videos.txt", None),
    ],
)
def test_celeb_df_labels(path, label):
    assert celeb_df_v2.label_from_path(path) == label


def test_celeb_df_marks_the_official_test_list(tmp_path):
    _tree(tmp_path, ["Celeb-real/id0_0001.mp4", "Celeb-synthesis/id0_id1_0001.mp4"])
    (tmp_path / "List_of_testing_videos.txt").write_text("1 Celeb-real/id0_0001.mp4\n")
    df = celeb_df_v2.build_manifest(tmp_path)
    validate_manifest(df, "celeb_df_v2")
    assert df.set_index("rel_path")["split"].to_dict() == {
        "Celeb-real/id0_0001.mp4": "test",
        "Celeb-synthesis/id0_id1_0001.mp4": "train",
    }


def _parquet(path, columns: dict, row_group_size: int = 2):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=row_group_size)


def _wav(tag: int) -> dict:
    return {"bytes": b"RIFF" + bytes([tag]) * 8, "path": None}


def test_indicsynth_manifest_writes_audio_and_reads_speakers(tmp_path):
    from deeptrace_bench.datasets import indicsynth

    _parquet(
        tmp_path / "Hindi" / "train-00000-of-00002.parquet",
        {
            "audio": [_wav(1), _wav(2), _wav(3)],
            "Generative Model": ["xtts_v2", "freevc24", "vits"],
            "Target Speaker ID": [11, 12, 13],
            "Source Speaker_ID": [None, 21.0, None],
            "Gender": ["Female", "Male", "Female"],
        },
    )
    df = indicsynth.build_manifest(tmp_path)
    validate_manifest(df, "indicsynth")
    assert list(df.label) == ["fake"] * 3
    assert list(df.method_family) == ["tts", "vc", "tts"]
    vc = df[df.method == "freevc24"].iloc[0]
    assert (vc.subject_id, vc.source_subject_id, vc.g_gender, vc.language) == (
        "12",
        "21",
        "male",
        "hi",
    )
    assert (tmp_path / vc.rel_path).read_bytes() == _wav(2)["bytes"]
    assert vc.item_id == "indicsynth/Hindi/train-00000-of-00002/1"


def test_indictts_challenge_labels_and_skips_the_unlabelled(tmp_path):
    from deeptrace_bench.datasets import indictts_challenge

    _parquet(
        tmp_path / "data" / "train-00000-of-00001.parquet",
        {
            "id": ["NEP_F_HAPPY_00001", "NEP_M_SAD_00002", "en_f_lj_LJ000-0001", "XX_F_Y_1"],
            "language": ["Nepali", "Nepali", "English", "Nepali"],
            "is_tts": [1, 0, 0, -1],
            "text": ["a", "b", "c", "d"],
            "audio": [_wav(1), _wav(2), _wav(3), _wav(4)],
        },
    )
    df = indictts_challenge.build_manifest(tmp_path)
    validate_manifest(df, "indictts_challenge")
    assert list(df.label) == ["fake", "real", "real"]  # is_tts -1 is no item
    assert list(df.language) == ["ne", "ne", "en"]
    assert list(df.subject_id) == ["NEP_F", "NEP_M", "EN_F"]
    assert list(df.g_gender) == ["female", "male", "female"]


def test_a_sampled_shard_keeps_full_shard_row_numbers(tmp_path):
    from deeptrace_bench.datasets import indictts_challenge

    _parquet(
        tmp_path / "data" / "train-00000-of-00001.parquet",
        {
            "id": ["TAM_F_X_1"],
            "language": ["Tamil"],
            "is_tts": [1],
            "audio": [_wav(9)],
            "__source_row": [417],
        },
    )
    df = indictts_challenge.build_manifest(tmp_path)
    assert df.item_id.tolist() == ["indictts_challenge/train-00000-of-00001/417"]
