"""
Add boundary leakage warning cho tat ca models co gap_days < forward_window
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LB_DIR = ROOT / "results/leaderboard"
LB_JSON = LB_DIR / "leaderboard.json"

# Backup
BACKUP = LB_DIR / f"leaderboard_pre_boundary_warn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
shutil.copy(LB_JSON, BACKUP)
print(f"[BACKUP] {BACKUP.name}")

# Load
with open(LB_JSON, encoding="utf-8") as f:
    data = json.load(f)

print(f"[LOAD] Total: {len(data)}")

# Add boundary leak warning
n_warned = 0
for entry in data:
    if entry.get("superseded", False):
        continue  # skip superseded

    # Extract forward_windows
    target_dict = entry.get("target", {})
    if isinstance(target_dict, str):
        # Parse from string if needed
        import ast

        try:
            target_dict = ast.literal_eval(target_dict)
        except:
            target_dict = {}

    target_fw = target_dict.get("forward_window", 0) if isinstance(target_dict, dict) else 0
    exit_fw = entry.get("exit_model_forward_window", 0)
    gap_days = entry.get("gap_days", 0)

    # Check boundary leak
    max_fw = max(target_fw, exit_fw) if exit_fw else target_fw
    if gap_days < max_fw:
        warnings = entry.get("warnings", [])
        warn_msg = f"boundary_leak_gap{gap_days}_fw{max_fw}"
        if warn_msg not in warnings and "boundary_leakage" not in str(warnings):
            warnings.append(warn_msg)
            entry["warnings"] = warnings
            n_warned += 1

print(f"[WARN] Added boundary leak warning to {n_warned} models")

# Save
with open(LB_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[SAVE] {LB_JSON}")

# Show top 5 with warnings
non_sup = [e for e in data if not e.get("superseded", False)]
non_sup.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

print()
print("=" * 80)
print("TOP 5 - BOUNDARY LEAK CHECK:")
print("=" * 80)
for i, e in enumerate(non_sup[:5]):
    target_dict = e.get("target", {})
    if isinstance(target_dict, str):
        import ast

        try:
            target_dict = ast.literal_eval(target_dict)
        except:
            target_dict = {}
    target_fw = target_dict.get("forward_window", 0) if isinstance(target_dict, dict) else 0
    exit_fw = e.get("exit_model_forward_window", 0)
    gap = e.get("gap_days", 0)
    warns = e.get("warnings", [])

    print(f"#{i + 1}: score={e['composite_score']} pf={e['pf']} wr={e['wr']}")
    print(f"    gap_days={gap}, target_fw={target_fw}, exit_fw={exit_fw}")
    print(f"    warnings={warns}")
    print()
