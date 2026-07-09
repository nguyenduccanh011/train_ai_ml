"""
So sanh 3 versions:
1. Leaderboard (pivot leak + boundary leak)
2. Pivot fix only (gap_days=0)
3. Pivot fix + boundary fix (gap_days=25)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 1. Leaderboard metrics (leaky)
leaderboard = {
    "version": "Leaderboard (pivot leak + boundary leak)",
    "wr": 78.2,
    "pf": 15.16,
    "trades": 1225,
    "total_pnl": 15977.05,
    "max_dd": 149.86,
    "avg_hold": 36.3,
}

# 2. Pivot fix only (gap_days=0)
pivot_fix_file = ROOT / "results/leakage_check/top1_retrain_metrics.json"
if pivot_fix_file.exists():
    with open(pivot_fix_file) as f:
        data = json.load(f)
        pivot_fix = {
            "version": "Pivot fix only (gap_days=0, boundary leak con)",
            **data["retrain_metrics"]
        }
else:
    pivot_fix = None

# 3. Pivot + boundary fix (gap_days=25)
boundary_fix_file = ROOT / "results/leakage_check/top1_retrain_gap25_metrics.json"
if boundary_fix_file.exists():
    with open(boundary_fix_file) as f:
        data = json.load(f)
        boundary_fix = {
            "version": "Pivot + boundary fix (gap_days=25)",
            **data["retrain_metrics"]
        }
else:
    boundary_fix = None

print("=" * 80)
print("SO SANH 3 VERSIONS")
print("=" * 80)
print()

versions = [leaderboard]
if pivot_fix:
    versions.append(pivot_fix)
if boundary_fix:
    versions.append(boundary_fix)

# Print table
print(f"{'Version':<50} {'WR':>8} {'PF':>8} {'Trades':>8} {'PnL':>12} {'MaxDD':>10}")
print("-" * 100)
for v in versions:
    print(f"{v['version']:<50} {v['wr']:>8.2f} {v['pf']:>8.2f} {v['trades']:>8} {v['total_pnl']:>12.2f} {v.get('max_dd', v.get('max_drawdown', 0)):>10.2f}")

print()
print("=" * 80)
print("DELTA ANALYSIS")
print("=" * 80)

if pivot_fix:
    print()
    print("Leaderboard -> Pivot fix only:")
    print(f"  WR: {leaderboard['wr']:.2f}% -> {pivot_fix['wr']:.2f}% (delta: {pivot_fix['wr'] - leaderboard['wr']:+.2f}%)")
    print(f"  PF: {leaderboard['pf']:.2f} -> {pivot_fix['pf']:.2f} (delta: {pivot_fix['pf'] - leaderboard['pf']:+.2f})")
    print(f"  => Pivot leak inflation: ~{leaderboard['wr'] - pivot_fix['wr']:.1f}% WR, ~{leaderboard['pf'] - pivot_fix['pf']:.1f} PF")

if boundary_fix and pivot_fix:
    print()
    print("Pivot fix only -> Pivot + boundary fix:")
    print(f"  WR: {pivot_fix['wr']:.2f}% -> {boundary_fix['wr']:.2f}% (delta: {boundary_fix['wr'] - pivot_fix['wr']:+.2f}%)")
    print(f"  PF: {pivot_fix['pf']:.2f} -> {boundary_fix['pf']:.2f} (delta: {boundary_fix['pf'] - pivot_fix['pf']:+.2f})")
    print(f"  => Boundary leak inflation: ~{pivot_fix['wr'] - boundary_fix['wr']:.1f}% WR, ~{pivot_fix['pf'] - boundary_fix['pf']:.1f} PF")

if boundary_fix:
    print()
    print("Leaderboard -> Pivot + boundary fix (TOTAL):")
    print(f"  WR: {leaderboard['wr']:.2f}% -> {boundary_fix['wr']:.2f}% (delta: {boundary_fix['wr'] - leaderboard['wr']:+.2f}%)")
    print(f"  PF: {leaderboard['pf']:.2f} -> {boundary_fix['pf']:.2f} (delta: {boundary_fix['pf'] - leaderboard['pf']:+.2f})")
    print(f"  => TOTAL leak inflation: ~{leaderboard['wr'] - boundary_fix['wr']:.1f}% WR, ~{leaderboard['pf'] - boundary_fix['pf']:.1f} PF")

print()
print("=" * 80)
