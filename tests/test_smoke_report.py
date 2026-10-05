"""The smoke report's ``latest.md`` pointer, on stores shared between machines."""

from __future__ import annotations

import errno
import importlib.util
from pathlib import Path

import pytest

from deeptrace_bench.paths import REPO_ROOT


def _smoke():
    spec = importlib.util.spec_from_file_location("smoke_script", REPO_ROOT / "scripts/smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_is_a_plain_copy_that_replaces_an_old_link(tmp_path):
    old = tmp_path / "old.md"
    old.write_text("old")
    (tmp_path / "latest.md").symlink_to("old.md")
    new = tmp_path / "new.md"
    new.write_text("new")

    _smoke().point_latest(new)

    latest = tmp_path / "latest.md"
    assert not latest.is_symlink()
    assert latest.read_text() == "new"
    assert old.read_text() == "old"


def test_a_share_that_refuses_the_pointer_does_not_fail_the_run(tmp_path, monkeypatch):
    # The Mac's SMB server refuses (EAGAIN) to let a Linux client delete a link the Mac made.
    report = tmp_path / "new.md"
    report.write_text("new")
    (tmp_path / "latest.md").symlink_to("new.md")

    def refuse(self, missing_ok=False):
        raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable", str(self))

    monkeypatch.setattr(Path, "unlink", refuse)
    _smoke().point_latest(report)
    assert report.read_text() == "new"


@pytest.fixture(autouse=True)
def _no_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DTB_ROOT", str(tmp_path / "store"))


def test_a_video_model_is_smoked_once_per_sample(registry):
    smoke = _smoke()
    chosen = {e.id for e in smoke.default_evalsets(registry, registry.model("xception"))}
    assert "unidatapro_videos" in chosen and "unidatapro_av" not in chosen
    assert {e.id for e in smoke.default_evalsets(registry, registry.model("havic"))} == {
        "unidatapro_av"
    }


def test_a_lip_model_skips_still_images(registry):
    chosen = {e.id for e in _smoke().default_evalsets(registry, registry.model("lipforensics"))}
    assert chosen == {"unidatapro_videos"}
