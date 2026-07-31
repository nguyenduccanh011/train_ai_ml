# -*- coding: utf-8 -*-
"""MEAN-REV sleeve via ML label (Phase 1 proof). Rule mean-rev failed before; test the ML-label version.
Entry universe = OVERSOLD dips (RSI14<35, NEGATIVE recent momentum -> decorrelated from momentum champion).
Label = forward 10-session return (the bounce). Walk-forward LGBM ranks which dips bounce. Equal-weight
K-slot sleeve, hold 10 sessions (auto T+2-safe). Measure standalone CAGR/DD + PER-YEAR (does it make money
in momentum dead-years 2023/24?) + ML-vs-random control. If dead-year positive -> proceed to combine."""
from __future__ import annotations
import os, sys, warnings, statistics, random
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M
from lightgbm import LGBMRegressor

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
SEEDS = [1, 2, 3]; HOLD = 10; KSL = 10; RSI_TH = 35
FCOLS = ['ret5', 'ret20', 'ret60', 'vol20', 'dist_h20', 'dist_h63', 'dist_l20', 'dist_l63',
         'ma20r', 'ma50r', 'atr_pct', 'updays10', 'volr', 'rs_mom20', 'rs_mom60']


def eqprun(sim, pm, K, advance_fee=0.0008, roundtrip=0.006):
    """equal-weight K-slot fill, prio=pm; hold baked into trades (exit_date). Returns nav df."""
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)

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
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"])
    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; held = set()
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg); held.discard(leg["symbol"])
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            if len(legs) >= K or t["symbol"] in held:
                continue
            size = nav_now / K
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg); held.add(t["symbol"])
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); return d


con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
syms = pd.read_sql("select distinct symbol from run_signals where run_id=%s", con, params=(rid,)).symbol.tolist()
con.close()
feat = M.features(syms)

cx = duckdb.connect(DUCK, read_only=True); ph = ",".join(["?"] * len(syms))
px = cx.execute(f"SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({ph}) AND date>='2018-06-01' ORDER BY symbol,date", syms).fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); rows = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True); c = g["close"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["fwdH"] = c.shift(-HOLD) / c - 1.0
    g["entry_date"] = g["date"].shift(-1); g["entry_px"] = c.shift(-1); g["exit_px"] = c.shift(-1 - HOLD); g["exit_date"] = g["date"].shift(-1 - HOLD)
    rows.append(g)
PX = pd.concat(rows, ignore_index=True)
cand = PX[(PX["rsi14"] < RSI_TH) & PX["fwdH"].notna() & PX["entry_date"].notna() & PX["exit_date"].notna()].copy()
cand["yr"] = cand["date"].dt.year
cand = cand.merge(feat, on=["symbol", "date"], how="left").dropna(subset=FCOLS)
print(f"oversold candidates (RSI<{RSI_TH}): {len(cand)} rows, {cand.symbol.nunique()} syms, yrs {cand.yr.min()}-{cand.yr.max()}", flush=True)

# --- market regime (causal, from EW index of the 61-universe) ---
piv = px.pivot(index="date", columns="symbol", values="close").sort_index()
norm = piv.div(piv.bfill().iloc[0]); idx = norm.mean(axis=1)
idx_ma100 = idx.rolling(100).mean(); idx_ma50 = idx.rolling(50).mean()
breadth50 = (piv > piv.rolling(50).mean()).mean(axis=1)
REG = pd.DataFrame({"date": idx.index, "above100": (idx > idx_ma100).values,
                    "above50": (idx > idx_ma50).values, "breadth": breadth50.values})
cand = cand.merge(REG, on="date", how="left")

# --- barrier exits (mean-rev OCO): TP +5% / SL -7% / timeout HOLD, on close path ---
TP, SL = 0.05, 0.07
CLO = {}; DTS = {}; BAR = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True); CLO[s] = g["close"].values; DTS[s] = g["date"].values
    BAR[s] = {d: i for i, d in enumerate(g["date"].values)}
bx_date, bx_px = [], []
for r in cand.itertuples():
    s = r.symbol; b0 = BAR[s].get(np.datetime64(pd.to_datetime(r.date))); c = CLO[s]
    if b0 is None or b0 + 1 >= len(c):
        bx_date.append(None); bx_px.append(None); continue
    eb = b0 + 1; ep = c[eb]; last = min(eb + HOLD, len(c) - 1); ex = last; xp = c[last]
    for k in range(eb + 1, last + 1):
        if c[k] >= ep * (1 + TP) or c[k] <= ep * (1 - SL):
            ex = k; xp = c[k]; break
    bx_date.append(pd.Timestamp(DTS[s][ex])); bx_px.append(float(xp))
cand["bexit_date"] = bx_date; cand["bexit_px"] = bx_px


def make_csv(mask, tag, barrier=False):
    sub = cand[mask].copy()
    if barrier:
        sub = sub[sub["bexit_date"].notna()]
        t = sub[["symbol", "entry_date", "bexit_date", "entry_px", "bexit_px"]].copy()
    else:
        t = sub[["symbol", "entry_date", "exit_date", "entry_px", "exit_px"]].copy()
    t.columns = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.strftime("%Y-%m-%d"); t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.strftime("%Y-%m-%d")
    f = HERE / f"_mrml_{tag}.csv"; t.to_csv(f, index=False); return str(f), len(t)


REGIMES = [
    ("none (all dips)", cand.index == cand.index),
    ("idx>MA100", cand["above100"] == True),
    ("idx>MA50", cand["above50"] == True),
    ("breadth>0.5", cand["breadth"] > 0.5),
]
cvf = HERE / "_mrml_trades.csv"
tr = cand[["symbol", "entry_date", "exit_date", "entry_px", "exit_px"]].copy()
tr.columns = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]
tr["entry_date"] = tr["entry_date"].dt.strftime("%Y-%m-%d"); tr["exit_date"] = tr["exit_date"].dt.strftime("%Y-%m-%d")
tr.to_csv(cvf, index=False)


def walkforward_pm(seed):
    pm = {}
    for ty in range(2020, 2027):
        train = cand[cand.yr < ty]; test = cand[cand.yr == ty]
        if len(train) < 200 or not len(test):
            continue
        mdl = LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                            verbose=-1, deterministic=True, force_col_wise=True, random_state=seed)
        mdl.fit(train[FCOLS], train["fwdH"])
        for (_, r), p in zip(test.iterrows(), mdl.predict(test[FCOLS])):
            pm[(r.symbol, pd.to_datetime(r.entry_date).strftime("%Y-%m-%d"))] = float(p)
    return pm


def peryear(d):
    g = d.copy(); g["y"] = g["date"].dt.year; ye = g.groupby("y")["nav"].last(); out = {}; prev = 1.0
    for y, v in ye.items():
        out[int(y)] = v / prev - 1.0; prev = v
    return out


PMS = {sd: walkforward_pm(sd) for sd in SEEDS}
YEARS = list(range(2020, 2027))
print(f"\n=== MEAN-REV sleeve + REGIME filter (ML-label K{KSL}/hold{HOLD}/RSI<{RSI_TH}, T+2, 3 ML-seeds) ===", flush=True)
print(f"  {'regime/exit':26s} | CAGR%  DD%  | n | " + " ".join(f"{y}" for y in YEARS), flush=True)
yrs = None
for barrier in (False, True):
    xtag = "barrier(TP5/SL7)" if barrier else "hold10-fixed"
    for lab, mask in REGIMES:
        f, n = make_csv(mask, lab.split()[0].replace(">", "gt").replace("(", "").replace("0.5", "05") + ("_b" if barrier else ""), barrier)
        navs = [eqprun(NavSim2(f, date_lo="2020-01-01"), PMS[sd], KSL) for sd in SEEDS]
        if yrs is None:
            yrs = (navs[0]["date"].iloc[-1] - navs[0]["date"].iloc[0]).days / 365.25
        fin = [float(d["nav"].iloc[-1]) for d in navs]
        cg = statistics.mean(f2 ** (1 / yrs) - 1 for f2 in fin); dd = statistics.mean(float((d["nav"] / d["nav"].cummax() - 1).min()) for d in navs)
        byy = defaultdict(list)
        for d in navs:
            for y, r in peryear(d).items():
                byy[y].append(r)
        yr_s = " ".join(f"{100*statistics.mean(byy[y]):+4.0f}" if y in byy else "   —" for y in YEARS)
        print(f"  {lab+'/'+('bar' if barrier else 'fix'):26s} | {100*cg:5.1f} {100*dd:5.1f} | {n:5d} | {yr_s}", flush=True)
print("MEANREV_DONE", flush=True)
