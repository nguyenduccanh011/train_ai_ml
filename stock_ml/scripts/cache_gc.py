"""Cache garbage collector CLI.

Examples:
    # Report orphan caches without touching anything (default)
    python -m stock_ml.scripts.cache_gc

    # Quarantine orphans into results/cache/_trash/<timestamp>/
    python -m stock_ml.scripts.cache_gc --apply

    # Quarantine orphans AND purge trash batches older than 7 days
    python -m stock_ml.scripts.cache_gc --apply --purge --older-than 7

    # Only purge old trash
    python -m stock_ml.scripts.cache_gc --purge --older-than 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache garbage collector")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help="Results dir (default: resolved via src.env.get_results_dir)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Quarantine orphans (default: dry-run report only)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Also delete trash batches older than --older-than days",
    )
    parser.add_argument(
        "--older-than",
        type=float,
        default=7.0,
        help="Purge trash batches older than this many days (default: 7)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List orphan file paths in the report",
    )
    args = parser.parse_args()

    from src.cache.garbage_collector import sweep
    from src.env import get_results_dir

    results_dir = args.results_dir or get_results_dir()

    report = sweep(
        results_dir,
        dry_run=not args.apply,
        purge_older_than_days=args.older_than if args.purge else None,
    )
    print(report.summary())

    if args.list and report.orphan_files:
        print("\nOrphan files:")
        for p in report.orphan_files:
            print(f"  {p}")

    if report.dry_run and report.orphan_count:
        print("\nRun with --apply to quarantine these orphans.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
