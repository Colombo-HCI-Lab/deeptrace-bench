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
