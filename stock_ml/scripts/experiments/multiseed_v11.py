"""Multi-seed v11 entry-axis candidates vs base 1743 (minhold2, 417.3)."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

SEEDS = [7, 99, 42]
CONFIGS = {
    1743: "minhold2_base",
    1744: "emt08", 1745: "emt14", 1746: "emw3", 1747: "emw10",
    1748: "pb02", 1749: "pb04", 1750: "pbw15", 1751: "pbw35",
}


def read_comp(run_id: str):
    out = subprocess.run(
        ["docker", "exec", "stock-ml-postgres", "psql", "-U", "stockml", "-d", "stockml",
         "-t", "-A", "-c",
         f"SELECT round(composite_score::numeric,1)||'/'||trades||'/'||round(total_pnl::numeric,1) "
         f"FROM leaderboard_runs WHERE run_id='{run_id}'"],
        capture_output=True, text=True)
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

print("\n=== V11 MULTI-SEED RANKING ===")
for label, (mean, comps) in sorted(data.items(), key=lambda kv: -kv[1][0]):
    print(f"{label:14} mean={mean:6.1f} seeds={comps}")
print("V11_MS_DONE")
