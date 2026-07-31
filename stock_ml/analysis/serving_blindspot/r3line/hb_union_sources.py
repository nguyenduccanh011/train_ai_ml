# -*- coding: utf-8 -*-
"""User idea: combine our execution (preempt + cs5_ma50 + K10/m005) with OTHER model lines (sv_dsb60_cd07,
en_pb035, top-composite un_full). All are dual-ML siblings of champion 3185. Test (a) does our stack
generalize to each source alone? (b) does UNION (pool signals, dedup, one book) beat the best single source?
seed42 directional. If union helps -> signal diversity is a real lever."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
K = 10; MARGIN = 0.005; KCONV = 2.0; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
SOURCES = [("champ3185", 3185), ("sv_dsb60_cd07", 2769), ("en_pb035", 3155), ("un_full", 2846)]


def prun(trades_df, metamap, cmap):
    """trades_df: symbol,entry_date,exit_date,entry_price,exit_price. metamap/cmap keyed (sym,entry_date_str)."""
    cvf = HERE / "_us_tmp.csv"; trades_df.to_csv(cvf, index=False)
    sim = NavSim2(str(cvf), date_lo="2020-01-01")
    s_new = (0.006 - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = metamap.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cmap.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= SKIP:
            entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def mk(t, size):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"])

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - 0.0008); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - 0.0008); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


# cs5_ma50 panel
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSMAP = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None; SRC = {}
print("=== (a) our execution on EACH source (seed42, cs5_ma50 combo K10/m005) ===", flush=True)
print(f"  {'source':16s} | n_tr | CAGR%  DD%", flush=True)
for nm, tid in SOURCES:
    rid = run_template_experiment(template_id=tid, seed=42).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cvtr["ed"] = cvtr["entry_date"].astype(str).str[:10]
    cm = {(r.symbol, r.ed): CSMAP.get((r.symbol, str(r.sigd.date())), 0.5) for r in cvtr.itertuples()}
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pmv = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    mm = {(r.symbol, r.ed): pmv.get((r.symbol, r.entry_date), -9.9) for r in cvtr.itertuples()}
    df = cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    cg, dd = prun(df, mm, cm)
    SRC[nm] = dict(df=df, mm=mm, cm=cm, cvtr=cvtr)
    print(f"  {nm:16s} | {len(df):4d} | {100*cg:5.1f} {100*dd:5.1f}", flush=True)
con.close()

# overlap: how many unique (sym,entry_date) each source adds vs champion
champ_keys = set(zip(SRC["champ3185"]["cvtr"].symbol, SRC["champ3185"]["cvtr"].ed))
print("\n=== signal overlap vs champion ===", flush=True)
for nm in [n for n, _ in SOURCES if n != "champ3185"]:
    ks = set(zip(SRC[nm]["cvtr"].symbol, SRC[nm]["cvtr"].ed))
    print(f"  {nm:16s}: {len(ks)} trades, {len(ks - champ_keys)} NOT in champ ({100*len(ks-champ_keys)/max(1,len(ks)):.0f}% unique)", flush=True)

# (b) UNION: pool champ + each other, dedup by (sym,entry_date) keep champ's, combo
print("\n=== (b) UNION champ + source (dedup keep champ), cs5_ma50 combo ===", flush=True)
for nm in [n for n, _ in SOURCES if n != "champ3185"]:
    frames = [SRC["champ3185"]["df"].assign(src="c")]
    om = dict(SRC["champ3185"]["mm"])
    a = SRC[nm]["df"].copy(); a["ed"] = a["entry_date"].astype(str).str[:10]
    seen = set(zip(SRC["champ3185"]["df"].symbol, SRC["champ3185"]["df"].entry_date.astype(str).str[:10]))
    add = a[[not ((r.symbol, r.ed) in seen) for r in a.itertuples()]]
    frames.append(add[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]])
    for k, v in SRC[nm]["mm"].items():
        om.setdefault(k, v)
    cmU = dict(SRC["champ3185"]["cm"]);
    for k, v in SRC[nm]["cm"].items():
        cmU.setdefault(k, v)
    U = pd.concat(frames, ignore_index=True)
    cg, dd = prun(U[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]], om, cmU)
    print(f"  champ + {nm:16s} | n_tr {len(U):4d} (+{len(U)-len(SRC['champ3185']['df'])}) | CAGR {100*cg:5.1f}% DD {100*dd:5.1f}%", flush=True)
print("  (champ alone baseline shown in (a))", flush=True)
print("UNION_SOURCES_DONE", flush=True)
