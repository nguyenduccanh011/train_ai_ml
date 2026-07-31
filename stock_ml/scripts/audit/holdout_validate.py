"""HOLDOUT validation: confirm the final champion (1760, pb045_w50) beats the original
champion (1586) on seeds NOT used during optimization [7,99,42]. If the gap holds on fresh
seeds, the +26 gain is real, not 3-seed overfit. seed 42 run LAST per template to restore
the canonical leaderboard row.
"""

from __future__ import annotations
import subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

HOLDOUT = [123, 777, 2025]
SEEDS = HOLDOUT + [42]  # 42 last restores canonical
CONFIGS = {1586: "orig_champ", 1760: "new_champ_pb045w50"}


def read_comp(run_id: str):
    out = subprocess.run(
        [
            "docker",
            "exec",
            "stock-ml-postgres",
            "psql",
            "-U",
            "stockml",
            "-d",
            "stockml",
            "-t",
            "-A",
            "-c",
            f"SELECT round(composite_score::numeric,1)||'/'||trades||'/'||round(total_pnl::numeric,1)||'/'||round(sharpe::numeric,3)||'/'||round(mdd_per_symbol::numeric,3) "
            f"FROM leaderboard_runs WHERE run_id='{run_id}'",
        ],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


data = {}
for tid, label in CONFIGS.items():
    rows = []
    for seed in SEEDS:
        r = run_template_experiment(template_id=tid, seed=seed)
        v = read_comp(r.get("run_id")) if r.get("run_id") else "ERR"
        rows.append((seed, v))
        print(f"  {label} seed={seed}: {v}")
    hold_comps = []
    for seed, v in rows:
        if seed in HOLDOUT:
            try:
                hold_comps.append(float(v.split("/")[0]))
            except Exception:
                pass
    mean = sum(hold_comps) / len(hold_comps) if hold_comps else 0.0
    data[label] = (mean, hold_comps)
    print(f"== {label}: HOLDOUT MEAN={mean:.1f} comps={hold_comps}")

print("\n=== HOLDOUT RESULT (seeds 123/777/2025) ===")
for label, (mean, comps) in sorted(data.items(), key=lambda kv: -kv[1][0]):
    print(f"{label:20} holdout_mean={mean:6.1f} seeds={comps}")
if "new_champ_pb045w50" in data and "orig_champ" in data:
    print(
        f"GAP (new - orig) on holdout = {data['new_champ_pb045w50'][0] - data['orig_champ'][0]:+.1f}"
    )
print("HOLDOUT_DONE")
