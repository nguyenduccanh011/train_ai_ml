#!/usr/bin/env python3
"""
Retrain v22_exit_ablation_round25 with pivot leakage fix.

This script:
1. Backs up current leaderboard as leaderboard_pre_fix.json
2. Runs the full round25 matrix with fixed pivot features
3. Compares metrics before/after fix
4. Generates impact report
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def backup_leaderboard():
    """Backup current leaderboard."""
    lb_path = Path("results/leaderboard/leaderboard.json")
    if not lb_path.exists():
        print("Warning: leaderboard.json not found")
        return None

    backup_path = Path("results/leaderboard/leaderboard_pre_fix.json")
    with open(lb_path, 'r', encoding='utf-8') as f:
        lb = json.load(f)

    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(lb, f, indent=2, ensure_ascii=False)

    print(f"[OK] Backed up leaderboard to {backup_path}")
    print(f"  Total models: {len(lb)}")

    # Count round25 models
    round25 = [m for m in lb if 'round25' in m.get('run_name', '')]
    print(f"  Round25 models: {len(round25)}")

    return lb

def run_matrix():
    """Run the round25 matrix with fixed features."""
    print("\n=== Running v22_exit_ablation_round25 with fix ===\n")

    cmd = [
        sys.executable, "-m", "stock_ml", "run-matrix",
        "matrix/v22_exit_ablation_round25"
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run from parent dir (stock_ml is a package there)
    parent = Path.cwd().parent
    result = subprocess.run(cmd, cwd=parent)

    if result.returncode != 0:
        print(f"\n[FAIL] Matrix run failed with exit code {result.returncode}")
        return False

    print("\n[OK] Matrix run completed")
    return True

def compare_results(pre_fix_lb):
    """Compare metrics before/after fix."""
    print("\n=== Comparing Results ===\n")

    # Load new leaderboard
    lb_path = Path("results/leaderboard/leaderboard.json")
    if not lb_path.exists():
        print("[FAIL] New leaderboard not found")
        return

    with open(lb_path, 'r', encoding='utf-8') as f:
        post_fix_lb = json.load(f)

    # Find round25 models in both
    pre_round25 = {m['config_hash']: m for m in pre_fix_lb if 'round25' in m.get('run_name', '')}
    post_round25 = {m['config_hash']: m for m in post_fix_lb if 'round25' in m.get('run_name', '')}

    common_hashes = set(pre_round25.keys()) & set(post_round25.keys())

    print(f"Models in both: {len(common_hashes)}")
    print()

    if not common_hashes:
        print("No common models found for comparison")
        return

    # Compare metrics
    print(f"{'Config':12} | {'WR Pre':8} | {'WR Post':8} | {'Delta':8} | {'PF Pre':8} | {'PF Post':8} | {'Delta':8}")
    print("-" * 90)

    wr_deltas = []
    pf_deltas = []

    for ch in sorted(common_hashes, key=lambda x: pre_round25[x].get('composite_score', 0), reverse=True)[:20]:
        pre = pre_round25[ch]
        post = post_round25[ch]

        wr_pre = pre.get('wr', 0)
        wr_post = post.get('wr', 0)
        wr_delta = wr_post - wr_pre

        pf_pre = pre.get('pf', 0)
        pf_post = post.get('pf', 0)
        pf_delta = pf_post - pf_pre

        wr_deltas.append(wr_delta)
        pf_deltas.append(pf_delta)

        print(f"{ch[:10]:12} | {wr_pre:7.2f}% | {wr_post:7.2f}% | {wr_delta:+7.2f}% | {pf_pre:8.2f} | {pf_post:8.2f} | {pf_delta:+8.2f}")

    print()
    print("SUMMARY:")
    print(f"  Avg WR delta: {sum(wr_deltas)/len(wr_deltas):+.2f}%")
    print(f"  Avg PF delta: {sum(pf_deltas)/len(pf_deltas):+.2f}")
    print()

    # Save comparison report
    report = {
        "timestamp": datetime.now().isoformat(),
        "models_compared": len(common_hashes),
        "avg_wr_delta": sum(wr_deltas)/len(wr_deltas),
        "avg_pf_delta": sum(pf_deltas)/len(pf_deltas),
        "wr_deltas": wr_deltas,
        "pf_deltas": pf_deltas,
    }

    report_path = Path("results/leaderboard/pivot_fix_impact_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Saved impact report to {report_path}")

def main():
    print("=" * 80)
    print("Retrain v22_exit_ablation_round25 with Pivot Leakage Fix")
    print("=" * 80)
    print()

    # Step 1: Backup
    pre_fix_lb = backup_leaderboard()
    if pre_fix_lb is None:
        print("[FAIL] Cannot proceed without leaderboard backup")
        return 1

    # Step 2: Run matrix
    success = run_matrix()
    if not success:
        print("\n[FAIL] Retrain failed")
        return 1

    # Step 3: Compare
    compare_results(pre_fix_lb)

    print("\n" + "=" * 80)
    print("[OK] Retrain and comparison complete")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
