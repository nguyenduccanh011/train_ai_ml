# -*- coding: utf-8 -*-
"""hb_103: DYNAMIC-K (regime-adaptive concentration) + conviction-priority. C win = K16+priority
71.6%/DD-17.1% NHUNG DD sinh ra chu yeu khi concentrate trong chop/dead-year. Y tuong: K=16 (don von)
CHI khi index uptrend (bat song), K=25 (chia mong/giu cash) trong chop -> giu return K16 nam-tot, cat DD.
Causal regime = equal-weight index 61-ma > MA(N). So static K25 / static K16-prio / dynamic. Multi-seed."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd
from nh_nav2 import NavSim2, shuffle_stats, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


VNI_CSV = "f:/PROJECTS/train_ai_ml/portable_data/vn_stock_ai_dataset_cleaned/context_features/symbol=VNINDEX/timeframe=1D/data.csv"
_VNI = None
def regime_kmap(syms, calendar, ma=50, k_up=16, k_down=25):
    global _VNI
    if _VNI is None:
        v = pd.read_csv(VNI_CSV); v['ds'] = pd.to_datetime(v['timestamp']).dt.strftime('%Y-%m-%d')
        _VNI = v.drop_duplicates('ds', keep='last').set_index('ds')['close'].astype(float).sort_index()
    up = (_VNI > _VNI.rolling(ma, min_periods=ma // 2).mean())  # real VNINDEX regime, date-string index
    ser = up.reindex(sorted(set(list(up.index) + list(calendar)))).ffill().fillna(False)
    km = {c: (k_up if bool(ser.get(c, False)) else k_down) for c in calendar}
    frac_up = sum(1 for c in calendar if km[c] == k_up) / len(calendar)
    return km, frac_up


def prio_run(sim, pm, K=None, kmap=None, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), 0.0)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v
    cash = 1.0; pending = defaultdict(float); legs = []; exits = defaultdict(list); nav_series = []
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            pr = leg["invested"] * (1.0 + leg["net"]); cash += pr * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        kk = kmap[dt] if kmap else K
        for t in entries.get(dt, ()):
            size = nav_now / kk
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"]/c0, ratio1=xe/c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); nav_series.append((dt, cash + pt + pos))
    ns = pd.DataFrame(nav_series, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
    nav = ns["nav"]; final = float(nav.iloc[-1]); yrs = (ns["date"].iloc[-1]-ns["date"].iloc[0]).days/365.25
    return final, final**(1/yrs)-1, float((nav/nav.cummax()-1).min())


def build_prio(con, rid, tr):
    sig = pd.read_sql("select symbol,date,score from run_signals where run_id=%s and signal=1 and score is not null", con, params=(rid,))
    sig["dt"] = pd.to_datetime(sig["date"]); tr = tr.copy(); tr["dt"] = pd.to_datetime(tr["entry_date"]); tr["edkey"] = tr["dt"].dt.strftime("%Y-%m-%d")
    pm = {}
    for s, g in sig.groupby("symbol"):
        g = g.sort_values("dt"); tg = tr[tr.symbol == s].sort_values("dt")
        if not len(tg): continue
        m = pd.merge_asof(tg[["dt", "edkey"]], g[["dt", "score"]], on="dt", direction="backward")
        for ed, x in zip(m["edkey"], m["score"]): pm[(s, ed)] = float(x) if pd.notna(x) else 0.0
    return pm


def main():
    con = psycopg2.connect(**PG)
    from scripts.run_template import run_template_experiment
    data = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=3185, seed=sd); rid = r.get("run_id")
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k103_s{sd}.csv"; tr.to_csv(cv, index=False)
        data[sd] = (str(cv), build_prio(con, rid, tr), tr.symbol.unique().tolist())
    print("config                     CAGR%   DD%   (3-seed mean)", flush=True)
    # static refs
    for lab, kw in [("static K25 shuffle", dict(K=25)), ("static K16 + priority", dict(K=16)), ("static K20 + priority", dict(K=20))]:
        cs, ds = [], []
        for sd in SEEDS:
            cv, pm, _ = data[sd]
            c, cg, dd = prio_run(NavSim2(cv, date_lo="2020-01-01"), pm, **kw); cs.append(cg); ds.append(dd)
        print(f"  {lab:24s} {statistics.mean(cs)*100:5.1f}  {statistics.mean(ds)*100:5.1f}", flush=True)
    # dynamic-K variants
    for ma, ku, kd in [(50, 16, 25), (100, 16, 25), (50, 14, 25), (50, 16, 30)]:
        cs, ds = [], []
        fr = 0
        for sd in SEEDS:
            cv, pm, syms = data[sd]
            sim = NavSim2(cv, date_lo="2020-01-01")
            km, fr = regime_kmap(syms, sim.calendar, ma=ma, k_up=ku, k_down=kd)
            c, cg, dd = prio_run(sim, pm, kmap=km); cs.append(cg); ds.append(dd)
        print(f"  dyn MA{ma} K{ku}<->K{kd} +prio   {statistics.mean(cs)*100:5.1f}  {statistics.mean(ds)*100:5.1f}  (up {fr*100:.0f}%)", flush=True)
    con.close(); print("HB_103_DONE", flush=True)


if __name__ == "__main__":
    main()
