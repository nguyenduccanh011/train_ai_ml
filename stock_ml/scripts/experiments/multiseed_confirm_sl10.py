"""Multi-seed confirm sl10 vs champ. config_hash ignores seed -> every seed clobbers the same
run_id, so we READ composite from the DB right after each run (before the next overwrites it).
Seed 42 is run LAST for each template to restore the canonical leaderboard row.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

SEEDS = [7, 99, 314, 42]   # 42 last -> leaves canonical row correct
TEMPLATES = {1586: "champ", 1674: "sl10"}


def read_composite(run_id: str) -> float:
    out = subprocess.run(
        ["docker", "exec", "stock-ml-postgres", "psql", "-U", "stockml", "-d", "stockml",
         "-t", "-A", "-c",
         f"SELECT composite_score||','||trades||','||round(total_pnl::numeric,1)||','||round(sharpe::numeric,3) "
         f"FROM leaderboard_runs WHERE run_id='{run_id}'"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


rows = []
for tid, label in TEMPLATES.items():
    for seed in SEEDS:
        r = run_template_experiment(template_id=tid, seed=seed)
        rid = r.get("run_id")
        vals = read_composite(rid) if rid else "ERR"
        rows.append((label, seed, vals))
        print(f"RESULT {label} seed={seed} -> comp,trd,pnl,sharpe = {vals}")

print("\n=== SUMMARY ===")
for label, seed, vals in rows:
    print(f"{label:6} seed={seed:4} : {vals}")
print("MULTISEED_TABLE_DONE")
