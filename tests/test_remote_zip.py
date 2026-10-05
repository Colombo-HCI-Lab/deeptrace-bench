"""Members of a remote zip come out whole, checked, and without fetching the whole archive."""

from __future__ import annotations

import io
import os
import zipfile

import pytest

from deeptrace_bench import remote_zip
from deeptrace_bench.remote_zip import (
    HttpRangeFile,
    RangeNotSupportedError,
    extract_members,
    open_zip,
)


def _zip(members: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _served(data: bytes) -> tuple[HttpRangeFile, list[tuple[int, int]]]:
    """A range-backed file over ``data`` that records every range it was asked for."""
    asked: list[tuple[int, int]] = []

    def fetch(start: int, end: int) -> bytes:
        asked.append((start, end))
        return data[start : end + 1]

    return HttpRangeFile(fetch, len(data)), asked


def test_members_come_out_whole_from_a_zip64_archive(tmp_path, monkeypatch):
    # A low entry limit makes zipfile write the ZIP64 end records, as the real 110k-entry
    # archive has, with only a few members.
    monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 2)
    members = {
        f"d/{label}/{i}.jpg": os.urandom(500) for label in ("real", "fake") for i in range(3)
    }
    raw, _ = _served(_zip(members))
    with open_zip(raw) as zf:
        assert sorted(zf.namelist()) == sorted(members)
        written = extract_members(zf, ["d/real/1.jpg", "d/fake/2.jpg"], tmp_path)
    assert [p.read_bytes() for p in written] == [members["d/real/1.jpg"], members["d/fake/2.jpg"]]


def test_only_what_is_needed_is_fetched(tmp_path):
    big = os.urandom(6 << 20)  # incompressible, so the archive really is this big
    data = _zip({"big.bin": big, "small.txt": b"hello"})
    raw, asked = _served(data)
    with open_zip(raw) as zf:
        extract_members(zf, ["small.txt"], tmp_path)
    fetched = sum(end - start + 1 for start, end in asked)
    assert (tmp_path / "small.txt").read_bytes() == b"hello"
    assert fetched < len(data) / 2


def test_a_corrupt_member_leaves_nothing_behind(tmp_path):
    data = bytearray(_zip({"a.txt": b"A" * 1000}, method=zipfile.ZIP_STORED))
    offset = data.index(b"A" * 1000)
    data[offset + 10] ^= 0xFF
    raw, _ = _served(bytes(data))
    with open_zip(raw) as zf, pytest.raises(zipfile.BadZipFile):
        extract_members(zf, ["a.txt"], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_members_cannot_escape(tmp_path):
    raw, _ = _served(_zip({"../escaped.txt": b"x"}))
    dest = tmp_path / "dataset"
    dest.mkdir()
    with open_zip(raw) as zf, pytest.raises(ValueError, match="escapes"):
        extract_members(zf, ["../escaped.txt"], dest)
    assert not (tmp_path / "escaped.txt").exists()


class _Reply:
    def __init__(self, status: int, headers: dict[str, str], url: str = "https://x/y") -> None:
        self.status_code = status
        self.headers = headers
        self.url = url
        self.content = b""

    def close(self) -> None:
        pass

    def raise_for_status(self) -> None:
        pass


class _Session:
    def __init__(self, reply: _Reply) -> None:
        self.reply = reply

    def get(self, *args, **kwargs) -> _Reply:
        return self.reply


def test_a_host_without_ranges_is_reported():
    with pytest.raises(RangeNotSupportedError):
        remote_zip.open_remote("https://x/y", session=_Session(_Reply(200, {})))


def test_the_size_comes_from_content_range():
    reply = _Reply(206, {"content-range": "bytes 0-0/4114687567"})
    assert remote_zip.open_remote("https://x/y", session=_Session(reply)).size == 4114687567


@pytest.mark.parametrize("level", ["-0", "-6"], ids=["stored", "deflated"])
def test_split_archives_are_read_member_by_member(tmp_path, level):
    import shutil
    import subprocess

    import numpy as np

    from deeptrace_bench.remote_zip import SplitZip

    if shutil.which("zip") is None:
        pytest.skip("Info-ZIP's zip makes the split archive")
    rng = np.random.default_rng(0)
    folder = tmp_path / "MD" / "hi"
    folder.mkdir(parents=True)
    members = {}
    for i in range(6):  # incompressible, so members straddle the 64 KB parts
        data = rng.integers(0, 256, 30_000 + 7_000 * i, dtype=np.uint8).tobytes()
        (folder / f"{i}.wav").write_bytes(data)
        members[f"MD/hi/{i}.wav"] = data
    (folder / "note.txt").write_text("text " * 2_000)
    subprocess.run(
        ["zip", "-q", level, "-s", "64k", "-r", str(tmp_path / "MD.zip"), "MD"],
        cwd=tmp_path,
        check=True,
    )
    parts = sorted(tmp_path.glob("MD.z0*")) + [tmp_path / "MD.zip"]
    assert len(parts) > 2
    split = SplitZip([p.open("rb") for p in parts], keep=lambda name: name.endswith(".wav"))
    assert sorted(split.namelist()) == sorted(members)
    for name, data in members.items():
        assert split.read(name) == data
