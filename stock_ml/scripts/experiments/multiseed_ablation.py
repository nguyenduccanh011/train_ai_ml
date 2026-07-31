"""Multi-seed evaluate champion + 10 ablations. config_hash ignores --seed -> each seed
clobbers the same run_id, so read composite from DB right after each run. seed 42 LAST per
template -> canonical row left intact. Output: mean composite per config (the TRUE ranking,
free of seed-42 selection bias). A lever whose removal keeps/raises the mean is overfit.
"""

from __future__ import annotations
import subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

SEEDS = [7, 99, 42]  # 42 last -> canonical row restored
CONFIGS = {
    1586: "champ",
    1695: "no_cooldown",
    1696: "no_incubate",
    1697: "no_nonbull",
    1698: "no_exitmkt",
    1699: "no_pullback",
    1700: "no_entrymkt",
    1701: "no_upleggate",
    1702: "no_trailactiv",
    1703: "no_atrtrail",
    1704: "no_absfloor",
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
    vals = []
    for seed in SEEDS:
        r = run_template_experiment(template_id=tid, seed=seed)
        rid = r.get("run_id")
        v = read_comp(rid) if rid else "ERR"
        comp = None
        try:
            comp = float(v.split("/")[0])
        except Exception:
            pass
        vals.append((seed, v, comp))
        print(f"  {label} seed={seed}: {v}")
    comps = [c for _, _, c in vals if c is not None]
    mean = sum(comps) / len(comps) if comps else 0.0
    data[label] = (mean, comps)
    print(f"== {label}: MEAN={mean:.1f}  comps={comps}")

print("\n=== MULTI-SEED MEAN RANKING (vs champ) ===")
champ_mean = data.get("champ", (0, []))[0]
for label, (mean, comps) in sorted(data.items(), key=lambda kv: -kv[1][0]):
    print(f"{label:14} mean={mean:6.1f}  delta_vs_champ={mean - champ_mean:+5.1f}  seeds={comps}")
print("ABLATION_DONE")
