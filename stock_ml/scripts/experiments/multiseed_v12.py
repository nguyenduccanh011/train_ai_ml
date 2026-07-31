"""Multi-seed v12 pullback-sweep candidates vs base 1751 (pbw35, 423.4)."""

from __future__ import annotations
import subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

SEEDS = [7, 99, 42]
CONFIGS = {
    1751: "pbw35_base",
    1752: "pb04_w35",
    1753: "pb05_w35",
    1754: "pb06_w35",
    1755: "pb03_w45",
    1756: "pb04_w45",
    1757: "pb05_w45",
    1758: "pb04_w50",
    1759: "pb05_w40",
}


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
            f"SELECT round(composite_score::numeric,1)||'/'||trades||'/'||round(total_pnl::numeric,1) "
            f"FROM leaderboard_runs WHERE run_id='{run_id}'",
        ],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


data = {}
for tid, label in CONFIGS.items():
    comps = []
    for seed in SEEDS:
        r = run_template_experiment(template_id=tid, seed=seed)
        v = read_comp(r.get("run_id")) if r.get("run_id") else "ERR"
        try:
            comps.append(float(v.split("/")[0]))
        except Exception:
            pass
        print(f"  {label} seed={seed}: {v}")
    mean = sum(comps) / len(comps) if comps else 0.0
    data[label] = (mean, comps)
    print(f"== {label}: MEAN={mean:.1f} comps={comps}")

print("\n=== V12 MULTI-SEED RANKING ===")
for label, (mean, comps) in sorted(data.items(), key=lambda kv: -kv[1][0]):
    print(f"{label:14} mean={mean:6.1f} seeds={comps}")
print("V12_MS_DONE")
