"""Rebuild the status page, docs/index.html, from the configs and the store's run summaries.

Example:
    uv run scripts/build_site.py

Rewrites only the JSON block inside the page, so markup and data stay in one file that GitHub
Pages, an artifact viewer or a plain browser can open. Needs DTB_ROOT for the run summaries.
"""

from __future__ import annotations

import sys

from deeptrace_bench.paths import REPO_ROOT
from deeptrace_bench.registry import Registry
from deeptrace_bench.status import collect, write_page

PAGE = REPO_ROOT / "docs" / "index.html"


def main() -> int:
    """Collect the status and write it into the page."""
    data = collect(Registry.load())
    write_page(PAGE, data)
    print(f"{'model':<16} {'evalset':<22} {'n':>7} {'auc':>6}  flags")
    for r in data["runs"]:
        flags = ", ".join(r["flags"])
        print(f"{r['model']:<16} {r['evalset']:<22} {r['n']:>7} {r['auc']:>6.3f}  {flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
