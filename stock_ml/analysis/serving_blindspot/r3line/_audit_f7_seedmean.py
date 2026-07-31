# -*- coding: utf-8 -*-
"""Audit finding #7: registered cagr_t2/maxdd_t2 (3-seed means) vs stored seed-42 portfolio."""
import psycopg2, pandas as pd
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
RIDS = [
 ("gtos",  "template/x2_struct_to_k10_cs5ma50_r7ec_gtos-69338138"),
 ("osdef", "template/x2_struct_to_k10_cs5ma50_r7ec_osdef-69338138"),
 ("gtrail","template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138"),
 ("earlycut","template/x2_struct_to_k10_cs5ma50_r7earlycut-69338138"),
]
for name, rid in RIDS:
    ln = pd.read_sql("select cagr_t2, maxdd_t2, nav_adv, cagr_adv, maxdd_nav, years, n_trades_sim from leaderboard_nav where run_id=%s", con, params=(rid,))
    eq = pd.read_sql("select date, nav from run_equity where run_id=%s order by date", con, params=(rid,))
    nt = pd.read_sql("select count(*) c from run_trades where run_id=%s", con, params=(rid,))
    if ln.empty or eq.empty:
        print(f"{name}: MISSING ln_empty={ln.empty} eq_empty={eq.empty}"); continue
    eq["date"] = pd.to_datetime(eq["date"])
    yrs = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.25
    fin = float(eq["nav"].iloc[-1])
    cagr_eq = fin ** (1/yrs) - 1
    dd_eq = float((eq["nav"] / eq["nav"].cummax() - 1).min())
    r = ln.iloc[0]
    print(f"{name}: registered cagr_t2={100*r.cagr_t2:.2f}% maxdd_t2={100*r.maxdd_t2:.2f}% n_trades_sim={r.n_trades_sim} years_col={r.years}")
    print(f"    stored equity (seed42 track): navx{fin:.2f} yrs={yrs:.2f} -> CAGR={100*cagr_eq:.2f}% DD={100*dd_eq:.2f}% | run_trades rows={int(nt.c[0])}")
    print(f"    delta: cagr {100*(float(r.cagr_t2)-cagr_eq):+.2f}pp, dd {100*(float(r.maxdd_t2)-dd_eq):+.2f}pp")
con.close()
