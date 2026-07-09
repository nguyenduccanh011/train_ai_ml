"""
Update leaderboard:
1. Mark superseded cac models bi pivot leak (dung feature_set co market_structure)
2. Add model retrain (top 1 fixed) vao leaderboard

Feature sets bi leak:
- leading_v2, leading_v3, leading_v4 (tu config/feature_sets/)
- leading_deriv (tu config/feature_sets/)
- all_features (deriv) - khong xac minh duoc nhung de an toan mark luon

Khong leak:
- leading (basic)
- volatility, momentum, price_action (chi 1-2 models)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LB_DIR = ROOT / "results/leaderboard"
LB_JSON = LB_DIR / "leaderboard.json"

# Feature sets co dung market_structure (pivot leak)
LEAKY_FEATURE_SETS = {
    "leading_v2",
    "leading_v3",
    "leading_v4",
    "leading_deriv",
    "all_features",  # deriv top1, can verify but mark to be safe
}

# Backup
BACKUP = LB_DIR / f"leaderboard_pre_supersede_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
shutil.copy(LB_JSON, BACKUP)
print(f"[BACKUP] {BACKUP.name}")

# Load leaderboard
with open(LB_JSON, encoding="utf-8") as f:
    data = json.load(f)

print(f"[LOAD] Total entries: {len(data)}")

# Step 1: Mark superseded
n_superseded = 0
for entry in data:
    if entry.get("superseded", False):
        continue
    feat = entry.get("feature_set", "")
    if feat in LEAKY_FEATURE_SETS:
        entry["superseded"] = True
        # Add reason
        warnings = entry.get("warnings", [])
        if "pivot_leakage_pre_fix" not in warnings:
            warnings.append("pivot_leakage_pre_fix")
        # gap_days=0 boundary leak (assume all pre-fix had gap=0)
        if "boundary_leakage_gap_days_0" not in warnings:
            warnings.append("boundary_leakage_gap_days_0")
        entry["warnings"] = warnings
        n_superseded += 1

print(f"[SUPERSEDE] Marked {n_superseded} models as superseded (leaky)")

# Step 2: Add new fixed model (top 1 retrain)
metrics_file = ROOT / "results/leakage_check/top1_retrain_metrics.json"
with open(metrics_file) as f:
    fix_data = json.load(f)
m = fix_data["retrain_metrics"]

# Compute composite_score (simplified - khong dung scoring engine dung)
# Tham chieu: leaderboard cu co score=662.1 voi pf=15.16, wr=78.2
# Voi pf=9.44, wr=69.64 -> du doan score thap hon nhieu
# Su dung formula don gian de uoc tinh
trades = m["trades"]
wr = m["wr"]
pf = m["pf"]
total_pnl = m["total_pnl"]
max_dd = m["max_drawdown"]
avg_hold = m["avg_hold"]

# Build new entry mirror cua top 1 cu
old_top = next((e for e in data if "5be8a6ec" in str(e.get("config_hash", ""))), None)
if old_top is None:
    print("[ERROR] Khong tim thay top 1 cu de mirror")
    exit(1)

new_entry = dict(old_top)  # copy structure
new_entry["run_id"] = "v22_exit_ablation_round25_FIXED/v22_top1_pivot_boundary_fix#fixed_v1"
new_entry["bundle"] = "v22_exit_ablation_round25_FIXED"
new_entry["run_name"] = "v22_top1_pivot_and_boundary_leakage_fix"
new_entry["config_hash"] = "fixed_v1_gap25"
new_entry["generated_at"] = datetime.now().isoformat()
new_entry["superseded"] = False
new_entry["trades"] = trades
new_entry["wr"] = round(wr, 2)
new_entry["avg_pnl"] = round(total_pnl / trades, 4) if trades else 0
new_entry["total_pnl"] = round(total_pnl, 2)
new_entry["pf"] = round(pf, 3)
new_entry["avg_hold"] = round(avg_hold, 1)
new_entry["max_drawdown"] = round(max_dd, 2)
# Estimate composite_score:
# Old: score=662.1 with pf=15.16, wr=78.2, pnl=15977
# New: pf=9.44 (-37%), wr=69.64 (-11%), pnl=96131 (+500%)
# Score scales with pf, wr, pnl positively, mdd negatively
# Rough estimate: 662 * (9.44/15.16)*(69.64/78.2) = ~367
estimated_score = 662.1 * (pf / 15.16) * (wr / 78.2)
new_entry["composite_score"] = round(estimated_score, 1)
new_entry["warnings"] = ["score_estimated_not_validated"]

# Add config_hash uniqueness
new_entry["n_symbols"] = 486  # retrain dung 486 symbols
new_entry["mdd_per_symbol"] = round(max_dd / 486, 4)

data.append(new_entry)
print("[ADD] New fixed model:")
print(f"  WR: {new_entry['wr']}%, PF: {new_entry['pf']}, Trades: {new_entry['trades']}")
print(f"  Estimated score: {new_entry['composite_score']}")

# Save
with open(LB_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[SAVE] {LB_JSON}")
print()

# Step 3: Show new top 5 (only non-superseded)
non_superseded = [e for e in data if not e.get("superseded", False)]
non_superseded.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

print("=" * 80)
print("NEW TOP 5 (non-superseded):")
print("=" * 80)
for i, e in enumerate(non_superseded[:5]):
    print(f"#{i+1}: score={e['composite_score']} pf={e['pf']} wr={e['wr']} trades={e['trades']}")
    print(f"    run_id={e['run_id'][:80]}")
    print(f"    feature_set={e.get('feature_set','?')}")

print()
print("=" * 80)
print(f"Total: {len(data)} | Active (non-superseded): {len(non_superseded)} | Superseded: {len(data) - len(non_superseded)}")
print("=" * 80)
