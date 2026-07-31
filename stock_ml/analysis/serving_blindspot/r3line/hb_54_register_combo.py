# -*- coding: utf-8 -*-
"""hb_54: dang ky winner combo t3056 (mh14+oxt03 + regime-skip MA200 winner-only) len
leaderboard. Rename lineage, 3-seed headline (42 last), seed_stats, NAV-score, supersede
sweep junk cua session nay (xd_/xr_/xg_/xc_mh14o3/xc_mh16_sk — KHONG dung xc_ctl/coh/thr
cua user)."""
from __future__ import annotations
import sys, shutil, statistics, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2  # noqa: E402
from scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
FPMAP = {42: "6fcf9ecc03", 7: "8f4b582b6d", 99: "4c2fdb103c", 555: "0df2e1ea08"}
RESULTS = REPO / "results"
TID = 3064
NEWNAME = "n2_2783_noT_mh14o3_rgskip300"
SEEDS = [99, 7, 42]


def seed_cache(tid, fp):
    src = RESULTS / f"tmpl_2936_{fp}" / "folds"; dst = RESULTS / f"tmpl_{tid}_{fp}" / "folds"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.parquet"):
        if not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("update strategy_templates set name=%s, seed=42, description=%s where id=%s",
                (NEWNAME,
                 "ab_noT mh14+overext_trail0.03 (NAV-frontier) + REGIME-SKIP hard rule "
                 "(signal_exit_skip_if_mkt_above_ma=300, winner_only): skip signal-exit khi "
                 "VNINDEX>MA300 (bull tape = shakeout hoi, AUC 0.78) -> defer trail/overext. "
                 "4-seed NAV x25.32 (+13.9% vs t2936), f22 x4.57 (+4.7%), regime-robust "
                 "(chi giu bull, f22 khong sap nhu ML approach). shakeout_vs_top hb_60/61.",
                 TID))
    con.commit(); print(f"renamed t{TID} -> {NEWNAME}", flush=True)

    comps = {}; headline = None
    for sd in SEEDS:
        seed_cache(TID, FPMAP[sd])
        r = run_template_experiment(template_id=TID, seed=sd); rid = r.get("run_id"); headline = rid
        cur.execute("select composite_score from leaderboard_runs where run_id=%s", (rid,))
        row = cur.fetchone(); comps[sd] = float(row[0]) if row and row[0] is not None else None
        print(f"  seed {sd}: {rid} comp={comps[sd]}", flush=True)

    vals = [comps[s] for s in (42, 7, 99) if comps.get(s) is not None]
    cur.execute("""insert into leaderboard_seed_stats
        (template_id,run_name,headline_composite,headline_seed,mean_composite,std_composite,
         min_composite,max_composite,n_seeds,seeds_json,mean_minus_headline,computed_at)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
        on conflict (template_id) do update set run_name=excluded.run_name,
         headline_composite=excluded.headline_composite,headline_seed=excluded.headline_seed,
         mean_composite=excluded.mean_composite,std_composite=excluded.std_composite,
         min_composite=excluded.min_composite,max_composite=excluded.max_composite,
         n_seeds=excluded.n_seeds,seeds_json=excluded.seeds_json,
         mean_minus_headline=excluded.mean_minus_headline,computed_at=now()""",
        (TID, NEWNAME, comps.get(42), 42, statistics.mean(vals),
         statistics.pstdev(vals) if len(vals) > 1 else 0.0, min(vals), max(vals), len(vals),
         json.dumps({str(s): comps.get(s) for s in (42, 7, 99)}),
         statistics.mean(vals) - (comps.get(42) or 0.0)))
    con.commit()
    print(f"seed_stats headline(42)={comps.get(42)} mean={statistics.mean(vals):.1f} seeds={comps}", flush=True)

    # supersede session sweep junk (chi cua toi, khong dung user xc_ctl/coh/thr)
    cur.execute("""update leaderboard_runs set superseded=true where run_id<>%s and (
        run_name like 'xd\\_%' or run_name like 'xr\\_%' or run_name like 'xg\\_%'
        or run_name like 'xc_mh14o3%' or run_name like 'xc_mh16_sk%' or run_name like 'xh\_%' or run_name like 'xi\_%' or run_name like 'n2_2783_noT_mh14o3_rgskip200%'
        or run_name like 'n2_2783_noT_exitregime%' or run_name like 'n2_2783_noT_topshake%')""",
        (headline,))
    print(f"superseded {cur.rowcount} session junk rows", flush=True); con.commit(); con.close()
    print(f"HEADLINE={headline}", flush=True); print("HB_54_DONE", flush=True)


if __name__ == "__main__":
    main()
