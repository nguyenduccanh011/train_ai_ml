# -*- coding: utf-8 -*-
"""META-MODEL feature axis (highest-EV remaining: preempt oracle x1243 vs causal 89.3% = meta prediction
is the bottleneck). The churn meta-model uses 17 mostly per-symbol RAW features (+2 RS ranks). Cross-
sectional RANK is the theme that keeps winning. Augment with ~8 per-day cross-sectional ranks of the
existing raws, retrain the walk-forward churn meta (t_pnl), run the combo (preempt R2 m0.01 + cs k1.5).
3-seed [42,21,123] K16. WIN = augmented meta beats 17-feat baseline 3/3 (better churn prediction -> NAV)."""
from __future__ import annotations
import os, sys, statistics
from collections import defaultdict
from pathlib import Path
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np, duckdb
from lightgbm import LGBMRegressor
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
SEEDS = [42, 21, 123]; K = 16; KCONV = 1.5; MARGIN = 0.01; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
BASE_F = M.FCOLS  # 17
# per-day cross-sectional ranks of existing raws to add
AUGCOLS = ["csr_ret5", "csr_ma20r", "csr_ma50r", "csr_distl20", "csr_disth20", "csr_atr", "csr_vol20", "csr_volr"]


def meta_preds_fc(tr, fcols, tgt='t_pnl'):
    pm = {}
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=fcols + [tgt]); test = tr[tr.yr == ty].dropna(subset=fcols)
        if len(train) < 100 or not len(test):
            continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7,
                            bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0, verbose=-1, deterministic=True,
                            force_col_wise=True, random_state=1)
        mdl.fit(train[fcols], train[tgt])
        for (_, row), p in zip(test.iterrows(), mdl.predict(test[fcols])):
            pm[(row.symbol, row.edkey)] = float(p)
    return pm


def prun(sim, pm, cm, k_conv=KCONV, margin=MARGIN, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
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
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


# cs4 sizing panel
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                "ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    parts.append(g[["symbol", "date"] + CS4])
P = pd.concat(parts, ignore_index=True)
for col in CS4:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1)
CS = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in P.itertuples()}

# base meta feature panel + AUG cross-sectional ranks
con = psycopg2.connect(**PG); feat = None
tr_c, cv_c, cm_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                       "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    cm_c[sd] = cm
    cv = HERE / f"_mf_s{sd}.csv"
    cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist()).copy()
        # add per-day cross-sectional ranks of existing raws
        feat["csr_ret5"] = feat.groupby("date")["ret5"].rank(pct=True)
        feat["csr_ma20r"] = feat.groupby("date")["ma20r"].rank(pct=True)
        feat["csr_ma50r"] = feat.groupby("date")["ma50r"].rank(pct=True)
        feat["csr_distl20"] = feat.groupby("date")["dist_l20"].rank(pct=True)
        feat["csr_disth20"] = feat.groupby("date")["dist_h20"].rank(pct=True)
        feat["csr_atr"] = feat.groupby("date")["atr_pct"].rank(pct=True)
        feat["csr_vol20"] = feat.groupby("date")["vol20"].rank(pct=True)
        feat["csr_volr"] = feat.groupby("date")["volr"].rank(pct=True)
    tr_c[sd] = M.build_tr(con, rid, feat); cv_c[sd] = str(cv)
con.close()

print("META-FEATURE augment (combo preempt R2m01 + cs k1.5), 3-seed K16:", flush=True)
print("  feature set        | NAV     CAGR%   DD%   | vs 17-feat base", flush=True)
VARIANTS = [("17-feat (base)", BASE_F), ("17 + 8 cs-rank", BASE_F + AUGCOLS), ("17 + cs-rank ONLY-swap", None)]
store = {}
for name, fcols in VARIANTS:
    if fcols is None:  # replace the 2 existing RS with the 8 cs-rank (test cross-sectional emphasis)
        fcols = [f for f in BASE_F if f not in ("rs_mom20", "rs_mom60")] + AUGCOLS
    nv, cg, dd = [], [], []
    for sd in SEEDS:
        pm = meta_preds_fc(tr_c[sd], fcols, tgt='t_pnl')
        f, c, d = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm, cm_c[sd])
        nv.append(f); cg.append(c); dd.append(d)
    store[name] = (nv, statistics.mean(cg), statistics.mean(dd))
base = store["17-feat (base)"][0]
for name, _ in VARIANTS:
    nv, cg, dd = store[name]
    delt = [nv[i] - base[i] for i in range(len(SEEDS))]
    signs = "".join("+" if x > 0 else "-" for x in delt)
    print(f"  {name:19s} | x{statistics.mean(nv):5.2f}  {cg*100:5.1f}  {dd*100:5.1f}  | "
          f"{[f'{x:+.1f}' for x in delt]} {signs}", flush=True)
print("METAFEAT_DONE")
