"""Importing upstream code from a checkout without installing it."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from deeptrace_bench.models._upstream import load_module, load_package, scoped_modules


def test_a_package_of_loose_files_imports_with_relative_imports(tmp_path):
    (tmp_path / "head.py").write_text("from .body import WIDTH\nSIZE = WIDTH * 2\n")
    (tmp_path / "body.py").write_text("WIDTH = 3\n")
    (tmp_path / "__init__.py").write_text("raise RuntimeError('must not run')\n")
    try:
        load_package(tmp_path, "dtb_test_pkg")
        assert importlib.import_module("dtb_test_pkg.head").SIZE == 6
    finally:
        for name in [n for n in sys.modules if n.startswith("dtb_test_pkg")]:
            del sys.modules[name]


def test_a_single_file_is_registered_once(tmp_path):
    (tmp_path / "model.py").write_text("COUNT = []\nCOUNT.append(1)\n")
    try:
        first = load_module(tmp_path / "model.py", "dtb_test_model")
        assert load_module(tmp_path / "model.py", "dtb_test_model") is first
        assert first.COUNT == [1]
    finally:
        sys.modules.pop("dtb_test_model", None)


def test_a_missing_file_is_not_left_registered(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_module(tmp_path / "absent.py", "dtb_test_absent")
    assert "dtb_test_absent" not in sys.modules


def test_scoped_modules_restores_sys_modules():
    import json.decoder  # noqa: F401

    before = sys.modules.get("json")
    stub = ModuleType("networks")
    with scoped_modules({"networks": stub, "json": ModuleType("json")}, purge=["detectors"]):
        sys.modules["detectors"] = ModuleType("detectors")
        sys.modules["detectors.x"] = ModuleType("detectors.x")
        assert sys.modules["networks"] is stub
    assert "networks" not in sys.modules
    assert "detectors" not in sys.modules and "detectors.x" not in sys.modules
    assert sys.modules.get("json") is before
    # A stub replaces one name only; a real package's submodules stay loaded.
    assert "json.decoder" in sys.modules
