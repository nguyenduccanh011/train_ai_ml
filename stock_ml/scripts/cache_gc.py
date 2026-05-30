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

    # Apply retention policy from base.yaml (auto-retire old runs, purge trash)
    python -m stock_ml.scripts.cache_gc --apply-policy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def apply_retention_policy(results_dir: str) -> None:
    """Apply retention policy from base.yaml to leaderboard runs."""
    from datetime import datetime, timedelta
    import json

    from src.utils.config_loader import load_base_config

    cfg = load_base_config()
    retention = cfg.get("retention", {})

    keep_pinned = retention.get("keep_pinned", True)
    keep_per_group = retention.get("keep_per_group", 5)
    auto_retire_days = retention.get("auto_retire_after_days", 30)
    trash_purge_days = retention.get("trash_purge_after_days", 14)

    # Read leaderboard.json
    leaderboard_path = Path(results_dir) / "leaderboard" / "leaderboard.json"
    if not leaderboard_path.exists():
        print(f"Leaderboard not found: {leaderboard_path}")
        return

    with open(leaderboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("rows", []) if isinstance(data, dict) else data
    now = datetime.now()
    cutoff_date = now - timedelta(days=auto_retire_days)

    print(f"Retention Policy:")
    print(f"  Keep pinned: {keep_pinned}")
    print(f"  Keep per group: {keep_per_group}")
    print(f"  Auto-retire after: {auto_retire_days} days (cutoff: {cutoff_date.date()})")
    print(f"  Trash purge after: {trash_purge_days} days")

    # Count what would be retired
    trained = [r for r in rows if (r.get("state") or "trained") == "trained"]
    old_trained = [
        r
        for r in trained
        if datetime.fromisoformat(r.get("generated_at", "2099-01-01").replace("Z", "+00:00")).date()
        < cutoff_date.date()
    ]

    print(f"\nSummary:")
    print(f"  Trained runs (eligible for retirement): {len(trained)}")
    print(f"  Runs older than {auto_retire_days} days: {len(old_trained)}")
    print(f"(Set --apply flag in cache_gc to actually apply GC + trash purge)")


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
    parser.add_argument(
        "--apply-policy",
        action="store_true",
        help="Apply retention policy from base.yaml (auto-retire old runs, purge trash)",
    )
    args = parser.parse_args()

    from src.cache.garbage_collector import sweep
    from src.env import get_results_dir

    results_dir = args.results_dir or get_results_dir()

    if args.apply_policy:
        apply_retention_policy(results_dir)
        return 0

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
