# -*- coding: utf-8 -*-
"""hb_104: toi uu TIN HIEU priority (extend C). blended score cho +9.7%@K16. Do TRAN oracle (fill theo
pnl THUC = lookahead, max co the) + san (worst) + cac signal causal khac (exit_score-inv, score x -exit).
Neu oracle >> score -> con du dia tim signal tot hon; neu oracle ~ score -> da gan toi uu. K16 seed42."""
from __future__ import annotations
import os, sys, warnings
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, pandas as pd
from nh_nav2 import NavSim2, shuffle_stats, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; RID = "template/x2_struct_to-69338138"; K = 16


def prun(sim, pm, desc=True, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), 0.0)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=desc)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"]-leg["ratio0"])*(j-i0)/(i1-i0)
        v = leg["invested"] * (sc[s][j]*r)/leg["p0"]; leg["last_val"] = v; return v
    cash = 1.0; pending = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            cash += leg["invested"]*(1.0+leg["net"])*(1.0-advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash+pt+pos
        for t in entries.get(dt, ()):
            size = nav_now/K
            if cash+1e-12 >= size:
                s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"]*(1.0+t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"]/c0, ratio1=xe/c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash+pt+pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); return float(d["nav"].iloc[-1])


def main():
    con = psycopg2.connect(**PG)
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(RID,))
    sig = pd.read_sql("select symbol,date,score,exit_score from run_signals where run_id=%s and signal=1 and score is not null", con, params=(RID,))
    con.close()
    cv = HERE / "_k104.csv"; tr.to_csv(cv, index=False)
    sig["dt"] = pd.to_datetime(sig["date"]); tr["dt"] = pd.to_datetime(tr["entry_date"]); tr["edkey"] = tr["dt"].dt.strftime("%Y-%m-%d")
    tr["pnl"] = tr.exit_price / tr.entry_price - 1.0
    # causal prio maps
    P = {"score": {}, "exitinv": {}, "score_x_exitinv": {}, "oracle": {}}
    for s, g in sig.groupby("symbol"):
        g = g.sort_values("dt"); tg = tr[tr.symbol == s].sort_values("dt")
        if not len(tg): continue
        m = pd.merge_asof(tg[["dt", "edkey"]], g[["dt", "score", "exit_score"]], on="dt", direction="backward")
        for ed, sco, xs in zip(m["edkey"], m["score"], m["exit_score"]):
            sco = float(sco) if pd.notna(sco) else 0.0; xs = float(xs) if pd.notna(xs) else 0.0
            P["score"][(s, ed)] = sco; P["exitinv"][(s, ed)] = -xs; P["score_x_exitinv"][(s, ed)] = sco * (1.0 - min(max(xs, -3), 10) / 10)
    for _, r in tr.iterrows(): P["oracle"][(r.symbol, r.edkey)] = float(r.pnl)  # lookahead ceiling
    base_mean = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=40)["mean"]
    print(f"K={K} shuffle-mean=x{base_mean:.2f}", flush=True)
    for lab in ["score", "exitinv", "score_x_exitinv", "oracle"]:
        hi = prun(NavSim2(str(cv), date_lo="2020-01-01"), P[lab], desc=True)
        print(f"  prio={lab:16s} hi=x{hi:.2f} ({(hi/base_mean-1)*100:+.1f}% vs shuffle-mean)", flush=True)
    lo = prun(NavSim2(str(cv), date_lo="2020-01-01"), P["oracle"], desc=False)
    print(f"  oracle-WORST (floor) x{lo:.2f} ({(lo/base_mean-1)*100:+.1f}%)", flush=True)
    print("HB_104_DONE", flush=True)


if __name__ == "__main__":
    main()
