"""Archive every evaluated run's per-item results and run record outside DTB_ROOT.

Examples:
    uv run scripts/archive.py --to /path/to/archive
    uv run scripts/archive.py --to /path/to/archive --from /path/to/another/store

Copies results/<run_id>/, the run's run.json and telemetry, and the wave ledgers from DTB_ROOT
(and any --from store, such as a copy of another machine's) into <archive>/runs/ and
<archive>/waves/, then rewrites <archive>/index.csv. Runs on datasets whose terms forbid copies
are left out. See store_layout.md, "Archive".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deeptrace_bench.archive import archive
from deeptrace_bench.paths import REPO_ROOT, store_root
from deeptrace_bench.registry import Registry


def main() -> int:
    """Archive the stores and print what was copied."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--to", type=Path, required=True, help="archive folder")
    parser.add_argument(
        "--from",
        dest="sources",
        type=Path,
        action="append",
        default=[],
        help="another store root to archive as well (repeatable)",
    )
    args = parser.parse_args()
    published = {
        p.name for p in (REPO_ROOT / "results").glob("*") if (p / "summary.json").is_file()
    }
    sources = [store_root(), *args.sources]
    report = archive(sources, args.to, Registry.load(), published)
    print(
        f"{report.runs} runs and {report.waves} wave ledgers from {len(sources)} store(s), "
        f"{report.copied} files copied to {args.to}"
    )
    if report.skipped_licence:
        print(f"left out under licence terms: {', '.join(report.skipped_licence)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
