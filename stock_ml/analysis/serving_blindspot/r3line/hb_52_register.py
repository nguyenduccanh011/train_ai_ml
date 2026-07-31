# -*- coding: utf-8 -*-
"""hb_52: dang ky winner mh14_oxt03 (t2998) len leaderboard DUNG chuan:
 1) rename template -> ten lineage n2_2783_noT_mh14_oxt03
 2) chay 3 seed [99,7,42] (cache-hit), seed 42 CHAY CUOI -> headline row + trades seed42
 3) ghi leaderboard_seed_stats (mean/std, headline seed 42)
 4) supersede toan bo row sweep 'xd_*' (don rac khoi board)
NAV-score + verify chay rieng sau.
"""
from __future__ import annotations
import sys, shutil, statistics
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2  # noqa: E402
from scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
FPMAP = {42: "6fcf9ecc03", 7: "8f4b582b6d", 99: "4c2fdb103c", 555: "0df2e1ea08"}
RESULTS = REPO / "results"
TID = 2998
NEWNAME = "n2_2783_noT_mh14_oxt03"
SEEDS = [99, 7, 42]  # 42 last -> headline


def seed_cache(tid, fp):
    src = RESULTS / f"tmpl_2936_{fp}" / "folds"; dst = RESULTS / f"tmpl_{tid}_{fp}" / "folds"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.parquet"):
        if not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    # 1) rename
    cur.execute("update strategy_templates set name=%s, seed=42, "
                "description=%s where id=%s",
                (NEWNAME,
                 "ab_noT lineage + max_hold 14 (16->14) + overext_trail_pct 0.03 (0.04->0.03): "
                 "NAV-frontier winner. 4-seed NAV x23.89 (+7.4% vs t2936 mh16), f22 x4.42 (+1.2%), "
                 "composite flat. Huong nguoc composite (giu ngan hon giai phong slot).",
                 TID))
    con.commit()
    print(f"renamed t{TID} -> {NEWNAME}", flush=True)

    # 2) run seeds (cache-hit), capture composite
    comps = {}; headline_rid = None
    for sd in SEEDS:
        seed_cache(TID, FPMAP[sd])
        r = run_template_experiment(template_id=TID, seed=sd)
        rid = r.get("run_id"); headline_rid = rid
        cur.execute("select composite_score from leaderboard_runs where run_id=%s", (rid,))
        row = cur.fetchone()
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        print(f"  seed {sd}: {rid} comp={comps[sd]}", flush=True)

    # 3) seed_stats
    vals = [comps[s] for s in (42, 7, 99) if comps.get(s) is not None]
    import json as _json
    cur.execute("""insert into leaderboard_seed_stats
        (template_id, run_name, headline_composite, headline_seed, mean_composite,
         std_composite, min_composite, max_composite, n_seeds, seeds_json,
         mean_minus_headline, computed_at)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
        on conflict (template_id) do update set
         run_name=excluded.run_name, headline_composite=excluded.headline_composite,
         headline_seed=excluded.headline_seed, mean_composite=excluded.mean_composite,
         std_composite=excluded.std_composite, min_composite=excluded.min_composite,
         max_composite=excluded.max_composite, n_seeds=excluded.n_seeds,
         seeds_json=excluded.seeds_json, mean_minus_headline=excluded.mean_minus_headline,
         computed_at=now()""",
        (TID, NEWNAME, comps.get(42), 42, statistics.mean(vals),
         statistics.pstdev(vals) if len(vals) > 1 else 0.0, min(vals), max(vals),
         len(vals), _json.dumps({str(s): comps.get(s) for s in (42, 7, 99)}),
         statistics.mean(vals) - (comps.get(42) or 0.0)))
    con.commit()
    print(f"seed_stats: mean={statistics.mean(vals):.1f} std={statistics.pstdev(vals):.2f} "
          f"headline(42)={comps.get(42)} seeds={comps}", flush=True)

    # 4) supersede sweep junk rows (xd_*), keep headline untouched
    cur.execute("update leaderboard_runs set superseded=true "
                "where run_name like 'xd\\_%' and run_id<>%s", (headline_rid,))
    n = cur.rowcount; con.commit()
    print(f"superseded {n} sweep rows (xd_*)", flush=True)
    print(f"HEADLINE run_id = {headline_rid}", flush=True)
    con.close()
    print("HB_52_DONE", flush=True)


if __name__ == "__main__":
    main()
