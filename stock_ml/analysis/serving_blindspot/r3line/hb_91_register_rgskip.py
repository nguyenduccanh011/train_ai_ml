# -*- coding: utf-8 -*-
"""hb_91: dang ky winner x2_struct_rgskip (t3237) len leaderboard. Multi-seed (hb_90): full NAV
26.47+/-0.47 (+2.4% vs base 25.85, thang base 3/3 seed; >= struct_to moi seed). Re-run seeds
[21,123,42] (42 CUOI -> headline row + trades seed42, cache-hit), ghi seed_stats, update description.
NAV-score chay rieng: score_nav_leaderboard.py --run-like x2_struct_rgskip."""
from __future__ import annotations
import sys, os, statistics, json, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2
from scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
TID = 3237
SEEDS = [21, 123, 42]  # 42 last -> headline
DESC = ("double-RS (entry_recov_rs + exit_vol_rs) + struct-trail donch20 trend_only + regime-skip "
        "MA300 winner_only. Multi-seed NAV@K25 (42/21/123): full x26.47+/-0.47 (+2.4% vs base ft_rs "
        "x25.85, thang base 3/3 seed), f22 x4.73 (+2.1%); >= struct_to (x26.23) moi seed. Regime-skip "
        "stack truc giao len struct-trail (giu winner trong bull tape thay vi ban tin hieu).")


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("update strategy_templates set description=%s, seed=42 where id=%s", (DESC, TID))
    con.commit()
    comps = {}; headline = None
    for sd in SEEDS:
        r = run_template_experiment(template_id=TID, seed=sd); rid = r.get("run_id"); headline = rid
        cur.execute("select composite_score from leaderboard_runs where run_id=%s", (rid,))
        row = cur.fetchone(); comps[sd] = float(row[0]) if row and row[0] is not None else None
        print(f"  seed {sd}: {rid} comp={comps[sd]}", flush=True)
    vals = [comps[s] for s in SEEDS if comps.get(s) is not None]
    cur.execute("""insert into leaderboard_seed_stats
        (template_id, run_name, headline_composite, headline_seed, mean_composite, std_composite,
         min_composite, max_composite, n_seeds, seeds_json, mean_minus_headline, computed_at)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
        on conflict (template_id) do update set run_name=excluded.run_name,
         headline_composite=excluded.headline_composite, headline_seed=excluded.headline_seed,
         mean_composite=excluded.mean_composite, std_composite=excluded.std_composite,
         min_composite=excluded.min_composite, max_composite=excluded.max_composite,
         n_seeds=excluded.n_seeds, seeds_json=excluded.seeds_json,
         mean_minus_headline=excluded.mean_minus_headline, computed_at=now()""",
        (TID, "x2_struct_rgskip", comps.get(42), 42, statistics.mean(vals),
         statistics.pstdev(vals) if len(vals) > 1 else 0.0, min(vals), max(vals), len(vals),
         json.dumps({str(s): comps.get(s) for s in SEEDS}), statistics.mean(vals) - (comps.get(42) or 0.0)))
    con.commit()
    print(f"seed_stats: mean_comp={statistics.mean(vals):.1f} headline(42)={comps.get(42)} seeds={comps}", flush=True)
    print(f"HEADLINE run_id = {headline}", flush=True)
    con.close(); print("HB_91_DONE", flush=True)


if __name__ == "__main__":
    main()
