"""Importing upstream code from a pinned checkout without installing it.

Adapters load the one or two files they need from ``third_party/<repo>-<commit>/`` (see
``upstream.py``), never a copy. Three helpers cover the cases met so far:

- ``load_module``: one file, registered under a fixed name (transformers looks a model
  class's module up in ``sys.modules``).
- ``load_package``: a folder of loose files that import each other relatively
  (``from .backbone import X``), as Hugging Face custom-code repos do, without running any
  ``__init__.py``.
- ``scoped_modules``: stand-in modules for the duration of an import, removed afterwards,
  for upstream files whose imports would otherwise drag in a whole training framework.

Standard library and Python 3.9 syntax only: the parity scripts load this file by path in an
upstream's original environment.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType


def load_module(path: Path, name: str) -> ModuleType:
    """Import one file as module ``name``; a second call returns the same module.

    Raises:
        FileNotFoundError: if the file is missing (usually: setup_models.py hasn't run).
    """
    if name in sys.modules:
        return sys.modules[name]
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_package(directory: Path, name: str) -> ModuleType:
    """Register ``directory`` as package ``name`` so ``name.<file>`` can be imported.

    The package itself is empty: an ``__init__.py`` in the folder is not run.

    Raises:
        FileNotFoundError: if the folder is missing.
    """
    if name in sys.modules:
        return sys.modules[name]
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    return module


@contextlib.contextmanager
def scoped_modules(stubs: dict[str, ModuleType], purge: Iterable[str] = ()) -> Iterator[None]:
    """Install ``stubs`` in ``sys.modules`` for the duration of the block.

    On exit every stub name gets its previous entry back (or none), and every module whose
    top-level package is one of ``purge`` is removed, so generic upstream names like
    ``networks`` or ``detectors`` never outlive the import that needed them. Objects created
    inside the block keep working; only the names are gone.
    """
    saved = {name: sys.modules.get(name) for name in stubs}
    purged = set(purge)
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n.split(".")[0] in purged]:
            del sys.modules[name]
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
