# -*- coding: utf-8 -*-
"""hb_75: multi-seed NAV — exit-RS (ft_rs_exitrs t3112) vs ft_rs (frontier). exit_features
KHAC ft_rs -> KHONG cache reuse; retrain t3112 seed 7/99 (seed42 da co). ft_rs 3-seed lay
tu run_trades cua ftrs_s42/s7/s99 (3102/3106/3107, da train). Dump + NavSim truc tiep.
"""
from __future__ import annotations
import os, sys, statistics
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import pandas as pd, psycopg2  # noqa: E402
from scripts.run_template import run_template_experiment  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent
# ft_rs 3-seed = separate trained templates
FTRS = {42: 3102, 7: 3106, 99: 3107}
EXITRS_TID = 3112


def dump_navsim(run_id, tag):
    con = psycopg2.connect(**PG)
    t = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                    "where run_id=%s and exit_date is not null", con, params=(run_id,))
    con.close()
    p = HERE / f"_ms75_{tag}.csv"; t.to_csv(p, index=False)
    out = {}
    for lo, k in (("2020-01-01", "full"), ("2022-01-01", "f22")):
        out[k] = shuffle_stats(NavSim2(str(p), date_lo=lo), K=25, roundtrip=0.006,
                               settle_lag=2, advance_fee=0.0008, n=20)["mean"]
    return out, len(t)


def runid_of(tid):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("select run_id from leaderboard_runs where template_id=%s order by run_seed desc limit 1", (tid,))
    r = cur.fetchone(); con.close(); return r[0] if r else None


agg = {"ft_rs": {"full": [], "f22": []}, "exitrs": {"full": [], "f22": []}}
for sd in (42, 7, 99):
    # ft_rs: read existing run_trades
    rid = runid_of(FTRS[sd])
    o, n = dump_navsim(rid, f"ftrs_s{sd}")
    agg["ft_rs"]["full"].append(o["full"]); agg["ft_rs"]["f22"].append(o["f22"])
    print(f"  ft_rs   s{sd}: full x{o['full']:.2f} f22 x{o['f22']:.2f} ({n} tr)", flush=True)
    # exit-rs: retrain (no cache) if not seed42
    r = run_template_experiment(template_id=EXITRS_TID, seed=sd)
    o2, n2 = dump_navsim(r.get("run_id"), f"exitrs_s{sd}")
    agg["exitrs"]["full"].append(o2["full"]); agg["exitrs"]["f22"].append(o2["f22"])
    print(f"  exitrs  s{sd}: full x{o2['full']:.2f} f22 x{o2['f22']:.2f} ({n2} tr)", flush=True)

print("\n== MEAN over 3 seeds ==", flush=True)
for name in ("ft_rs", "exitrs"):
    mf = statistics.mean(agg[name]["full"]); ff = statistics.mean(agg[name]["f22"])
    print(f"  {name:7s} full x{mf:.2f}  f22 x{ff:.2f}", flush=True)
bf = statistics.mean(agg["ft_rs"]["full"]); bff = statistics.mean(agg["ft_rs"]["f22"])
ef = statistics.mean(agg["exitrs"]["full"]); eff = statistics.mean(agg["exitrs"]["f22"])
print(f"  exitrs vs ft_rs: full {(ef/bf-1)*100:+.1f}%  f22 {(eff/bff-1)*100:+.1f}%", flush=True)
print("HB_75_DONE", flush=True)
