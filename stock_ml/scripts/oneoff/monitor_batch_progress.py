"""
Monitor batch retrain progress.

Usage:
    python monitor_batch_progress.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    log_path = ROOT / "results/leakage_check/batch_retrain_log.json"

    if not log_path.exists():
        print("No batch retrain log found yet.")
        print(f"Expected: {log_path}")
        return 1

    data = json.loads(log_path.read_text(encoding="utf-8"))

    print("=" * 80)
    print("BATCH RETRAIN PROGRESS")
    print("=" * 80)
    print(f"Last update: {data['timestamp']}")
    print(f"Target: {data['batch_size']} models")
    print(f"Completed: {data['completed']}")
    print(f"Skipped: {data['skipped']}")
    print(f"Success: {data['success']}")
    print(f"Failed: {data['failed']}")
    print(f"Total time: {data['total_time_minutes']:.1f} minutes")

    if data["completed"] > 0:
        avg_time = data["total_time_minutes"] * 60 / data["completed"]
        remaining = data["batch_size"] - data["completed"] - data["skipped"]
        eta_minutes = remaining * avg_time / 60
        print(f"\nAvg time per model: {avg_time:.1f}s")
        print(f"Remaining: {remaining} models")
        print(f"ETA: {eta_minutes:.1f} minutes ({eta_minutes / 60:.1f} hours)")

    # Recent results
    print("\nRecent 5 results:")
    for r in data["results"][-5:]:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} {r['bundle'][:40]:40} | {r['elapsed']:.0f}s")
        if r["status"] == "success":
            m = r["metrics"]
            print(
                f"      WR={m['wr']:.2f}% PF={m['pf']:.2f} Trades={m['trades']} Score={m['composite_score']:.1f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
