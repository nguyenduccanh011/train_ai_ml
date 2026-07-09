"""
Xoa 315 leaky entries khoi leaderboard.csv va rebuild.

Usage:
    python cleanup_leaky_leaderboard.py --dry-run  # Preview only
    python cleanup_leaky_leaderboard.py --execute  # Actually delete
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.leaderboard import rebuild_leaderboard


def main():
    parser = argparse.ArgumentParser(description="Cleanup leaky leaderboard entries")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't modify")
    parser.add_argument("--execute", action="store_true", help="Actually delete leaky entries")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("ERROR: Must specify --dry-run or --execute")
        return 1

    leaderboard_path = ROOT / "results/leaderboard/leaderboard.csv"
    backup_path = ROOT / "results/leaderboard/leaderboard_backup_before_cleanup.csv"

    # Load leaderboard
    df = pd.read_csv(leaderboard_path)
    print(f"Total entries: {len(df)}")

    # Filter leaky
    leaky_sets = ["leading_v2", "leading_v3", "leading_v4", "leading_deriv", "leading"]
    mask = df["feature_set"].isin(leaky_sets)
    leaky = df[mask].copy()
    clean = df[~mask].copy()

    print(f"Leaky entries: {len(leaky)}")
    print(f"Clean entries: {len(clean)}")
    print("\nLeaky feature sets:")
    print(leaky["feature_set"].value_counts().to_string())

    if args.dry_run:
        print("\n[DRY RUN] Would delete these entries:")
        print(leaky[["bundle", "run_name", "feature_set", "wr", "pf", "composite_score"]].head(20).to_string(index=False))
        print(f"\n... and {len(leaky) - 20} more")
        print(f"\nBackup would be saved to: {backup_path}")
        print(f"Clean leaderboard would have {len(clean)} entries")
        return 0

    # Execute
    print(f"\n[EXECUTE] Deleting {len(leaky)} leaky entries...")

    # Backup
    df.to_csv(backup_path, index=False)
    print(f"Backup saved: {backup_path}")

    # Save clean leaderboard
    clean.to_csv(leaderboard_path, index=False)
    print(f"Clean leaderboard saved: {leaderboard_path} ({len(clean)} entries)")

    # Rebuild leaderboard từ experiments dir (sẽ pick up retrained models)
    print("\nRebuilding leaderboard from experiments artifacts...")
    experiments_dir = ROOT / "results/experiments"
    output_dir = ROOT / "results/leaderboard"

    try:
        rows = rebuild_leaderboard(experiments_dir, output_dir)
        print(f"Leaderboard rebuilt: {len(rows)} total entries")

        # Count clean vs leaky after rebuild
        df_new = pd.read_csv(leaderboard_path)
        mask_new = df_new["feature_set"].isin(leaky_sets)
        print(f"  Clean entries: {(~mask_new).sum()}")
        print(f"  Leaky entries (retrained): {mask_new.sum()}")
    except Exception as e:
        print(f"ERROR rebuilding leaderboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n[OK] Cleanup completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
