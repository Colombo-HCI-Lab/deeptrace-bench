"""Reading single members of a remote zip with HTTP range requests.

A zip's table of contents (the central directory) sits at its end, and each member can be
read on its own, so with a host that honours ``Range`` we can list a 4 GB archive and pull
three files out of it for a few MB. ``zipfile`` does the parsing (including ZIP64) and checks
each member's CRC as it is read; this module only gives it a seekable file backed by range
requests.

Hosts that ignore ``Range`` (they answer 200 with the whole body) raise
``RangeNotSupportedError``, and callers fall back to a full download.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import requests

from .fetch import ensure_inside

log = logging.getLogger(__name__)

Fetch = Callable[[int, int], bytes]
_CONTENT_RANGE = re.compile(r"bytes \d+-\d+/(\d+)")
_BUFFER = 1 << 20


class RangeNotSupportedError(RuntimeError):
    """The server answered a range request with the whole file."""


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file whose bytes come from ``fetch(start, end_inclusive)``."""

    def __init__(self, fetch: Fetch, size: int) -> None:
        super().__init__()
        self._fetch = fetch
        self._size = size
        self._pos = 0

    @property
    def size(self) -> int:
        """Total size in bytes."""
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer) -> int:  # noqa: ANN001 - any writable buffer
        if self._pos >= self._size:
            return 0
        end = min(self._pos + len(buffer), self._size) - 1
        data = self._fetch(self._pos, end)
        n = len(data)
        buffer[:n] = data
        self._pos += n
        return n


def _http_fetcher(url: str, session: requests.Session) -> tuple[Fetch, int]:
    """Resolve redirects once, check range support, and return (fetch, size)."""

    def resolve() -> tuple[str, int]:
        resp = session.get(url, headers={"Range": "bytes=0-0"}, timeout=60, stream=True)
        resp.close()
        if resp.status_code == 200:
            raise RangeNotSupportedError(f"{url} ignores range requests")
        resp.raise_for_status()
        match = _CONTENT_RANGE.fullmatch(resp.headers.get("content-range", ""))
        if resp.status_code != 206 or match is None:
            raise RangeNotSupportedError(f"{url}: no usable Content-Range in the reply")
        return resp.url, int(match.group(1))

    final_url, size = resolve()
    state = {"url": final_url}

    def fetch(start: int, end: int) -> bytes:
        for attempt in range(2):
            resp = session.get(state["url"], headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            # A presigned redirect target can expire mid-run; resolve the original URL again.
            if resp.status_code in (401, 403) and attempt == 0:
                state["url"], _ = resolve()
                continue
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RangeNotSupportedError(f"{url}: range request answered {resp.status_code}")
            return resp.content
        raise RuntimeError("unreachable")

    return fetch, size


def open_remote(url: str, session: requests.Session | None = None) -> HttpRangeFile:
    """A seekable file over ``url`` (redirects resolved once).

    Raises:
        RangeNotSupportedError: if the host doesn't support range requests.
    """
    fetch, size = _http_fetcher(url, session or requests.Session())
    return HttpRangeFile(fetch, size)


def open_zip(raw: HttpRangeFile) -> zipfile.ZipFile:
    """A ``ZipFile`` over a range-backed file, buffered so small reads don't each cost a request."""
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=_BUFFER))


def open_remote_zip(url: str, session: requests.Session | None = None) -> zipfile.ZipFile:
    """List and read members of the zip at ``url`` without downloading all of it."""
    return open_zip(open_remote(url, session))


def extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> Path:
    """Write one member to ``dest`` (via a ``.part`` file), checking its CRC on the way.

    Raises:
        zipfile.BadZipFile: on a CRC mismatch; nothing is left at ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with zf.open(member) as src, part.open("wb") as out:
            while chunk := src.read(_BUFFER):
                out.write(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


def extract_members(zf: zipfile.ZipFile, members: list[str], dest_root: Path) -> list[Path]:
    """Extract several members under ``dest_root``, keeping their paths inside the archive."""
    written = []
    for member in members:
        target = ensure_inside(dest_root, dest_root / member)
        written.append(extract_member(zf, member, target))
    return written


# --- split archives -------------------------------------------------------------------------


class SplitZip:
    """A split zip (``name.z01``, ``name.z02``, ..., ``name.zip``) read by range requests.

    ``zipfile`` doesn't read multi-part archives, so this parses the little it needs: the
    end-of-central-directory records (ZIP64 or not) in the last part, the central directory,
    and local headers, then reads a member's compressed bytes (which may run on into the next
    part), inflates them and checks the CRC. Parts are given in order, the ``.zip`` last; each
    is any seekable file (``open_remote`` for a URL). ``keep`` filters names while the
    central directory is scanned, so a directory of millions of entries needn't be held.
    """

    _CHUNK = 16 << 20

    def __init__(self, parts: list, keep: Callable[[str], bool] | None = None) -> None:
        self.parts = parts
        self.sizes = [_size(p) for p in parts]
        self.entries: dict[str, tuple[int, int, int, int, int]] = {}
        self._read_directory(keep or (lambda name: True))

    def _read(self, disk: int, offset: int, length: int) -> bytes:
        """``length`` bytes from ``offset`` of part ``disk``, running on into later parts."""
        out = bytearray()
        while length > 0:
            if disk >= len(self.parts):
                raise zipfile.BadZipFile("member runs past the last part")
            take = min(length, self.sizes[disk] - offset)
            if take > 0:
                self.parts[disk].seek(offset)
                out += self.parts[disk].read(take)
                length -= take
            disk, offset = disk + 1, 0
        return bytes(out)

    def _read_directory(self, keep: Callable[[str], bool]) -> None:
        import struct

        last = len(self.parts) - 1
        tail_len = min(self.sizes[last], 1 << 16)
        tail = self._read(last, self.sizes[last] - tail_len, tail_len)
        at = tail.rfind(b"PK\x05\x06")
        if at < 0:
            raise zipfile.BadZipFile("no end-of-central-directory record in the last part")
        _, _, cd_disk, _, _, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[at : at + 22])
        locator = tail.rfind(b"PK\x06\x07", 0, at)
        if locator >= 0:  # ZIP64: the real values are in the ZIP64 end record
            _, z64_disk, z64_offset, _ = struct.unpack("<IIQI", tail[locator : locator + 20])
            record = self._read(z64_disk, z64_offset, 56)
            if record[:4] != b"PK\x06\x06":
                raise zipfile.BadZipFile("bad ZIP64 end-of-central-directory record")
            (cd_disk,) = struct.unpack("<I", record[20:24])
            cd_size, cd_offset = struct.unpack("<QQ", record[40:56])
        pending = b""
        position = 0
        while position < cd_size or pending:
            if position < cd_size:
                take = min(self._CHUNK, cd_size - position)
                pending += self._read(cd_disk, cd_offset + position, take)
                position += take
            used = self._parse_entries(pending, keep, final=position >= cd_size)
            pending = pending[used:]
            if position >= cd_size:
                break

    def _parse_entries(self, data: bytes, keep: Callable[[str], bool], final: bool) -> int:
        import struct

        at = 0
        while at + 46 <= len(data):
            if data[at : at + 4] != b"PK\x01\x02":
                if final:
                    break
                raise zipfile.BadZipFile(f"bad central directory entry at {at}")
            fields = struct.unpack("<IHHHHHHIIIHHHHHII", data[at : at + 46])
            method, crc, csize, usize = fields[4], fields[7], fields[8], fields[9]
            name_len, extra_len, comment_len, disk = fields[10], fields[11], fields[12], fields[13]
            offset = fields[16]
            end = at + 46 + name_len + extra_len + comment_len
            if end > len(data):
                break
            name = data[at + 46 : at + 46 + name_len].decode("utf-8", "replace")
            if keep(name) and not name.endswith("/"):
                extra = data[at + 46 + name_len : at + 46 + name_len + extra_len]
                usize, csize, offset, disk = _zip64_extra(extra, usize, csize, offset, disk)
                self.entries[name] = (disk, offset, csize, method, crc)
            at = end
        return at

    def namelist(self) -> list[str]:
        return list(self.entries)

    def compress_size(self, name: str) -> int:
        return self.entries[name][2]

    def read(self, name: str) -> bytes:
        """A member's bytes, inflated and CRC-checked.

        Raises:
            zipfile.BadZipFile: on a bad local header or a CRC mismatch.
        """
        import struct
        import zlib

        disk, offset, csize, method, crc = self.entries[name]
        header = self._read(disk, offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise zipfile.BadZipFile(f"{name}: bad local header")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        start = offset + 30 + name_len + extra_len
        while start >= self.sizes[disk]:  # the header ends exactly at a part boundary
            start -= self.sizes[disk]
            disk += 1
        raw = self._read(disk, start, csize)
        if method == 0:
            data = raw
        elif method == 8:
            data = zlib.decompressobj(-15).decompress(raw)
        else:
            raise zipfile.BadZipFile(f"{name}: compression method {method} not supported")
        if zlib.crc32(data) & 0xFFFFFFFF != crc:
            raise zipfile.BadZipFile(f"{name}: CRC mismatch")
        return data


def _size(part) -> int:  # noqa: ANN001 - a seekable file
    if hasattr(part, "size"):
        return part.size
    position = part.tell()
    end = part.seek(0, io.SEEK_END)
    part.seek(position)
    return end


def _zip64_extra(extra: bytes, usize: int, csize: int, offset: int, disk: int) -> tuple:
    """Replace the 0xFFFF... placeholders of a central directory entry from its ZIP64 field."""
    import struct

    at = 0
    while at + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[at : at + 4])
        body = extra[at + 4 : at + 4 + size]
        if tag == 0x0001:
            pos = 0
            if usize == 0xFFFFFFFF:
                (usize,) = struct.unpack("<Q", body[pos : pos + 8])
                pos += 8
            if csize == 0xFFFFFFFF:
                (csize,) = struct.unpack("<Q", body[pos : pos + 8])
                pos += 8
            if offset == 0xFFFFFFFF:
                (offset,) = struct.unpack("<Q", body[pos : pos + 8])
                pos += 8
            if disk == 0xFFFF:
                (disk,) = struct.unpack("<I", body[pos : pos + 4])
        at += 4 + size
    return usize, csize, offset, disk
