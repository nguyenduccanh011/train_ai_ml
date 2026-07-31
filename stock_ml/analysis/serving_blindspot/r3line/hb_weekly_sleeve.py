# -*- coding: utf-8 -*-
"""WEEKLY-momentum sleeve (multi-timeframe unlock). Thesis: weekly breakout + trailing exit rides the
SUSTAINED trend that the daily pullback model gives back (capture leak). Decorrelated (buy strength vs daily
buy-dip; multi-week hold vs short). Causal: signal at week-close, execute next daily session, walk-forward ML.
Standalone CAGR/DD + per-year (money in dead-years 2024/26?) + corr vs daily. If viable -> combine (<=1)."""
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
import psycopg2, pandas as pd, numpy as np, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
from lightgbm import LGBMRegressor

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
SEEDS = [1, 2, 3]; BRK = 12; HOLD_W = 12; TRAIL = 0.12; KSL = 10
FCOLS = ["wret4", "wret12", "wret26", "wrsi", "wdist52", "wvol", "wrs12", "wrs26"]


def eqprun(sim, pm, K, advance_fee=0.0008, roundtrip=0.006):
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


def peryear(d):
    g = d.copy(); g["y"] = g["date"].dt.year; ye = g.groupby("y")["nav"].last(); out = {}; prev = 1.0
    for y, v in ye.items():
        out[int(y)] = v / prev - 1.0; prev = v
    return out


con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
syms = pd.read_sql("select distinct symbol from run_signals where run_id=%s", con, params=(rid,)).symbol.tolist()
con.close()
cx = duckdb.connect(DUCK, read_only=True); ph = ",".join(["?"] * len(syms))
px = cx.execute(f"SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({ph}) AND date>='2016-06-01' ORDER BY symbol,date", syms).fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])

rows = []
DDATES = {}; DCLOSE = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True); DDATES[s] = g["date"].values; DCLOSE[s] = g["close"].values
    wc = g.set_index("date")["close"].resample("W-FRI").last().dropna()
    if len(wc) < 60:
        continue
    d = wc.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    W = pd.DataFrame({"we": wc.index, "wc": wc.values})
    W["wret4"] = wc.pct_change(4).values; W["wret12"] = wc.pct_change(12).values; W["wret26"] = wc.pct_change(26).values
    W["wrsi"] = (100 - 100 / (1 + up / (dn + 1e-9))).values
    W["wdist52"] = (wc / wc.rolling(52).max() - 1).values
    W["wvol"] = wc.pct_change().rolling(12).std().values
    W["brk"] = (wc.values >= pd.Series(wc.values).rolling(BRK).max().values - 1e-9)  # new BRK-week high
    W["fwd8"] = (wc.shift(-8) / wc - 1).values
    W["symbol"] = s
    rows.append(W)
WK = pd.concat(rows, ignore_index=True)
WK["wrs12"] = WK.groupby("we")["wret12"].rank(pct=True); WK["wrs26"] = WK.groupby("we")["wret26"].rank(pct=True)
cand = WK[WK.brk & WK[FCOLS].notna().all(axis=1)].copy()
cand["yr"] = pd.to_datetime(cand.we).dt.year

# map each weekly signal -> entry daily (next session) + trailing exit daily
tr_rows = []
for s, g in cand.groupby("symbol"):
    dts = DDATES[s]; cl = DCLOSE[s]
    wser = WK[WK.symbol == s].sort_values("we").reset_index(drop=True)
    wdt = pd.to_datetime(wser.we).values; wcl = wser.wc.values
    widx = {d: i for i, d in enumerate(wdt)}
    for r in g.itertuples():
        we = np.datetime64(pd.to_datetime(r.we))
        ei = np.searchsorted(dts, we, side="right")  # first daily AFTER week-close signal
        if ei >= len(dts):
            continue
        wi = widx.get(we)
        if wi is None:
            continue
        peak = wcl[wi]; ex_w = min(wi + HOLD_W, len(wcl) - 1)
        for k in range(wi + 1, min(wi + HOLD_W, len(wcl) - 1) + 1):
            peak = max(peak, wcl[k])
            if wcl[k] <= peak * (1 - TRAIL):
                ex_w = k; break
        ex_we = np.datetime64(pd.to_datetime(wdt[ex_w]))
        xi = np.searchsorted(dts, ex_we, side="right") - 1  # last daily <= exit week-close
        xi = max(xi, ei); xi = min(xi, len(dts) - 1)
        tr_rows.append(dict(symbol=s, entry_date=pd.Timestamp(dts[ei]).strftime("%Y-%m-%d"),
                            exit_date=pd.Timestamp(dts[xi]).strftime("%Y-%m-%d"),
                            entry_price=float(cl[ei]), exit_price=float(cl[xi]),
                            we=str(pd.Timestamp(r.we).date()), **{c: getattr(r, c) for c in FCOLS}, fwd8=r.fwd8, yr=r.yr))
T = pd.DataFrame(tr_rows).dropna(subset=["fwd8"])
print(f"weekly breakout trades: {len(T)} ({T.symbol.nunique()} syms), yrs {T.yr.min()}-{T.yr.max()}", flush=True)
trades = T[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
cvf = HERE / "_wk_trades.csv"; trades.to_csv(cvf, index=False)


def walkforward_pm(seed):
    pm = {}
    for ty in range(2020, 2027):
        train = T[T.yr < ty]; test = T[T.yr == ty]
        if len(train) < 150 or not len(test):
            continue
        m = LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7,
                          bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0, verbose=-1, deterministic=True, force_col_wise=True, random_state=seed)
        m.fit(train[FCOLS], train["fwd8"])
        for (_, r), p in zip(test.iterrows(), m.predict(test[FCOLS])):
            pm[(r.symbol, r.entry_date)] = float(p)
    return pm


navs = []
for sd in SEEDS:
    navs.append(eqprun(NavSim2(str(cvf), date_lo="2020-01-01"), walkforward_pm(sd), KSL))
fin = [float(d["nav"].iloc[-1]) for d in navs]; yrs = (navs[0]["date"].iloc[-1] - navs[0]["date"].iloc[0]).days / 365.25
cg = statistics.mean(f ** (1 / yrs) - 1 for f in fin); dd = statistics.mean(float((d["nav"] / d["nav"].cummax() - 1).min()) for d in navs)
print(f"\n=== WEEKLY sleeve standalone (breakout{BRK}w/trail{int(100*TRAIL)}/hold{HOLD_W}w, K{KSL}, T+2, 3 ML-seeds) ===", flush=True)
print(f"  CAGR {100*cg:.1f}%  DD {100*dd:.1f}%  NAV x{statistics.mean(fin):.2f}", flush=True)
print("  PER-YEAR (seed-mean) — KEY: 2024/2026 dương?:", flush=True)
byy = defaultdict(list)
for d in navs:
    for y, r in peryear(d).items():
        byy[y].append(r)
for y in sorted(byy):
    print(f"    {y}: {100*statistics.mean(byy[y]):+6.1f}%", flush=True)
import random
rng = random.Random(0); pmr = {(r.symbol, r.entry_date): rng.random() for _, r in T.iterrows()}
dr = eqprun(NavSim2(str(cvf), date_lo="2020-01-01"), pmr, KSL)
print(f"  [control] random-prio: CAGR {100*(float(dr['nav'].iloc[-1])**(1/yrs)-1):.1f}% (ML adds {100*(cg-(float(dr['nav'].iloc[-1])**(1/yrs)-1)):+.1f}pp)", flush=True)
print("WEEKLY_DONE", flush=True)
