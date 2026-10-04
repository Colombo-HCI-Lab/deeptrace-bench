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
