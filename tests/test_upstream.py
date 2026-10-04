"""Pinned upstream checkouts: where each host's repo is cloned from, and how."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deeptrace_bench import upstream
from deeptrace_bench.registry import Upstream

GITHUB = Upstream(repo="owner/tool", commit="0" * 40)
HF = Upstream(host="hf", repo="owner/Tool_V1", commit="1" * 40)


def test_clone_url_per_host():
    assert GITHUB.url == "https://github.com/owner/tool.git"
    assert HF.url == "https://huggingface.co/owner/Tool_V1"
    assert HF.slug == "Tool_V1-111111111111"


def test_unknown_host_is_refused():
    with pytest.raises(ValidationError):
        Upstream(host="gitlab", repo="a/b", commit="0" * 40)


@pytest.mark.parametrize("pinned", [GITHUB, HF])
def test_clone_uses_the_host_url_and_skips_lfs_blobs(pinned, tmp_path, monkeypatch):
    calls = []

    def fake_git(*args, env=None):
        calls.append((args, env))
        if args[0] == "clone":
            (tmp_path / pinned.slug / ".git").mkdir(parents=True)
        return ""

    monkeypatch.setattr(upstream, "THIRD_PARTY_DIR", tmp_path)
    monkeypatch.setattr(upstream, "_git", fake_git)
    upstream.ensure_upstream(pinned, archive=False)

    clone_args, clone_env = calls[0]
    assert clone_args[0] == "clone" and pinned.url in clone_args
    # Weights live in LFS on Hugging Face; the code checkout must never pull them.
    assert clone_env["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert calls[1][0][-2:] == ("--detach", pinned.commit)
