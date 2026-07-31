# -*- coding: utf-8 -*-
"""hb_51: multi-seed NAV cho winner (mh14_oxt03[_smg20]) vs base (mh16) tren 4 seed.
run_id KHONG chua seed -> cac seed cung template ghi de cung run_id; nen DUMP trades
ngay sau moi run truoc khi seed ke tiep de. Roi NAV-sim truc tiep bang nh_nav2
(K=25, roundtrip 0.006, settle_lag 2, adv 0.0008, n=20; full 2020 + f22 2022).
"""
from __future__ import annotations
import os, sys, shutil, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from scripts.run_template import run_template_experiment  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
FPMAP = {42: "6fcf9ecc03", 7: "8f4b582b6d", 99: "4c2fdb103c", 555: "0df2e1ea08"}
RESULTS = REPO / "results"
HERE = Path(__file__).parent
CANDS = [(2946, "base_mh16"), (3056, "sk200w"), (3064, "sk300w")]
SEEDS = [42, 7, 99, 555]


def seed_cache(tid, fp):
    src = RESULTS / f"tmpl_2936_{fp}" / "folds"; dst = RESULTS / f"tmpl_{tid}_{fp}" / "folds"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.parquet"):
        if not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)


def dump_trades(run_id, out_csv):
    con = psycopg2.connect(**PG)
    t = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                    "where run_id=%s and exit_date is not null and entry_price is not null "
                    "and exit_price is not null", con, params=(run_id,))
    con.close()
    t.to_csv(out_csv, index=False)
    return len(t)


def nav(csv, lo):
    sim = NavSim2(str(csv), date_lo=lo)
    s = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
    return s["mean"]


def main():
    # 1) run + dump per seed
    csvs = {}  # (lab,seed) -> csv
    for tid, lab in CANDS:
        for sd in SEEDS:
            seed_cache(tid, FPMAP[sd])
            r = run_template_experiment(template_id=tid, seed=sd)
            rid = r.get("run_id")
            out = HERE / f"_ms_{lab}_s{sd}.csv"
            n = dump_trades(rid, out) if rid else 0
            csvs[(lab, sd)] = out
            print(f"RAN {lab} s{sd} -> {rid} ({n} tr)", flush=True)

    # 2) NAV-sim per seed (full + f22), aggregate
    print("\n== NAV per seed (adv, K=25) ==", flush=True)
    agg = {}
    for _, lab in CANDS:
        full = []; f22 = []
        for sd in SEEDS:
            nf = nav(csvs[(lab, sd)], "2020-01-01")
            n22 = nav(csvs[(lab, sd)], "2022-01-01")
            full.append(nf); f22.append(n22)
            print(f"  {lab:18s} s{sd:<3d} full x{nf:6.2f}  f22 x{n22:5.2f}", flush=True)
        agg[lab] = (statistics.mean(full), statistics.pstdev(full),
                    statistics.mean(f22), statistics.pstdev(f22))
    print("\n== MEAN over 4 seeds (vs base) ==", flush=True)
    bf, _, bff, _ = agg["base_mh16"]
    for _, lab in CANDS:
        mf, sf, mff, sff = agg[lab]
        print(f"  {lab:18s} full x{mf:6.2f}±{sf:4.2f} ({(mf/bf-1)*100:+5.1f}%)  "
              f"f22 x{mff:5.2f}±{sff:4.2f} ({(mff/bff-1)*100:+5.1f}%)", flush=True)
    print("HB_51_DONE", flush=True)


if __name__ == "__main__":
    main()
