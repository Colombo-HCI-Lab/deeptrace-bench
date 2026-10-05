"""Rebuild the status page (docs/index.html) and the model and dataset reference (docs/catalog.md).

Examples:
    uv run scripts/build_site.py
    uv run scripts/build_site.py --citations   # also refresh citation counts first

Rewrites only the JSON block inside the page, so markup and data stay in one file that GitHub
Pages, an artifact viewer or a plain browser can open, and regenerates docs/catalog.md from the
same data. Needs DTB_ROOT for the run summaries and manifests.
``--citations`` looks up every paper in docs/catalog.yaml on Semantic Scholar and rewrites
results/citations.json; without it the last saved counts are used.
"""

from __future__ import annotations

import argparse
import sys

from deeptrace_bench import reference
from deeptrace_bench.paths import REPO_ROOT
from deeptrace_bench.registry import Registry
from deeptrace_bench.status import collect, fetch_citations, load_catalog, write_page

PAGE = REPO_ROOT / "docs" / "index.html"
REFERENCE = REPO_ROOT / "docs" / "catalog.md"


def main() -> int:
    """Collect the status and write it into the page."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--citations", action="store_true", help="refresh citation counts")
    args = parser.parse_args()
    registry = Registry.load()
    if args.citations:
        found = fetch_citations(load_catalog(registry))
        print(f"{len(found['papers'])} papers found on {found['source']}")
    data = collect(registry)
    write_page(PAGE, data)
    REFERENCE.write_text(reference.render(data))
    print(f"{'model':<16} {'evalset':<22} {'n':>7} {'auc':>6}  flags")
    for r in data["runs"]:
        flags = ", ".join(r["flags"])
        print(f"{r['model']:<16} {r['evalset']:<22} {r['n']:>7} {r['auc']:>6.3f}  {flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
