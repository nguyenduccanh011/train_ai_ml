# -*- coding: utf-8 -*-
"""Can the 'runs-immediately' behavior (e3 = first-3-session return) be predicted BEFORE entry from
entry-time features -> move the early-cut signal from POST to PRE (skip/upweight at entry, no fill fee/lockup)?
Features at entry: M.features (momentum/RS/dist) + cs5_ma50 + exit_score + mw7 (all causal at signal date).
Label e3. Walk-forward LGBM -> IC/AUC. If predictive: NAV pre-filter (skip predicted-red-early) vs base 123.4
and vs post-cut 127.8. If NOT predictive -> confirms post-entry reaction is necessary."""
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
from sklearn.metrics import roc_auc_score
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, TPLUS, R7THR = 10, 0.005, 2.0, 2, 0.02
FCOLS = ['ret5', 'ret20', 'ret60', 'vol20', 'dist_h20', 'dist_h63', 'dist_l20', 'dist_l63', 'ma20r', 'ma50r',
         'atr_pct', 'updays10', 'volr', 'rs_mom20', 'rs_mom60', 'cs5_ma50', 'exit_score', 'mw7']


def prun(sim, pm, cm, r7map, e3skip=None, e3thr=None, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if (bx - be) < TPLUS:
            nb = min(be + TPLUS, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < SKIP:
            continue
        v = r7map.get((t["symbol"], t["entry_date"]), np.nan)
        if not np.isnan(v) and v < R7THR:
            continue
        if e3skip is not None:
            p = e3skip.get((t["symbol"], t["entry_date"]))
            if p is not None and p < e3thr:
                continue
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

    def mk(t, size, di):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"], ratio0=t["p0"] / c0,
                    ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di)

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
                cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= TPLUS]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []; CLO = {}; DIDX = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values; DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20c"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["mw7"] = c / c.shift(7) - 1
    g = g.rename(columns={"ret20c": "ret20"})
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "mw7"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
MW7 = {(r.symbol, str(r.date.date())): (r.mw7 if pd.notna(r.mw7) else np.nan) for r in P.itertuples()}
MW7f = {(r.symbol, str(r.date.date())): (r.mw7 if pd.notna(r.mw7) else np.nan) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c, cvtrs = {}, {}, {}, {}
sig_es = {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    sig = pd.read_sql("select symbol,date,exit_score from run_signals where run_id=%s and signal=1", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    sig["date"] = pd.to_datetime(sig["date"]); esd = {(r.symbol, str(r.date.date())): (float(r.exit_score) if pd.notna(r.exit_score) else np.nan) for r in sig.itertuples()}
    sig_es[sd] = esd
    cvf = HERE / f"_pe_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl'); cvtrs[sd] = cvtr
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
FEAT = feat.rename(columns={"date": "entry_date"})


FEATs = feat.rename(columns={"date": "sigdate"})


def build_frame(sd):
    cv = cvtrs[sd].copy(); cv["entry_date"] = pd.to_datetime(cv["entry_date"]); cv["sigdate"] = cv["sigd"]
    cv = cv.merge(FEATs, on=["symbol", "sigdate"], how="left")   # AIRTIGHT: M.features tại NGÀY TÍN HIỆU (trước khi khớp)
    cv["cs5_ma50"] = [CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in cv.itertuples()]
    cv["exit_score"] = [sig_es[sd].get((r.symbol, str(r.sigd.date())), np.nan) for r in cv.itertuples()]
    cv["mw7"] = [MW7.get((r.symbol, str(r.sigd.date())), np.nan) for r in cv.itertuples()]
    e3 = []
    for r in cv.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date.date()))
        e3.append(CLO[r.symbol][ei + 3] / r.entry_price - 1.0 if (ei is not None and ei + 3 < len(CLO[r.symbol])) else np.nan)
    cv["e3"] = e3; cv["yr"] = cv["entry_date"].dt.year
    return cv.dropna(subset=FCOLS + ["e3"])


# walk-forward predict e3
def wf_e3(sd, cv):
    pmm = {}
    for ty in range(2020, 2027):
        tr = cv[cv.yr < ty]; te = cv[cv.yr == ty]
        if len(tr) < 150 or not len(te):
            continue
        m = LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7,
                          bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0, verbose=-1, deterministic=True, force_col_wise=True, random_state=1)
        m.fit(tr[FCOLS], tr["e3"])
        for (_, r), p in zip(te.iterrows(), m.predict(te[FCOLS])):
            pmm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = float(p)
    return pmm


frames = {sd: build_frame(sd) for sd in SEEDS}
E3 = {sd: wf_e3(sd, frames[sd]) for sd in SEEDS}
# diagnostic seed42
cv = frames[42]; pr = np.array([E3[42].get((r.symbol, str(pd.to_datetime(r.entry_date).date())), np.nan) for r in cv.itertuples()])
ok = ~np.isnan(pr); act = cv["e3"].values[ok]; prd = pr[ok]
from scipy.stats import spearmanr
print(f"=== PRE-ENTRY dự báo e3 (chạy-ngay) từ feature lúc vào — walk-forward seed42, n={ok.sum()} ===", flush=True)
print(f"  IC (spearman predicted-e3 vs actual-e3) = {spearmanr(prd, act).correlation:+.3f}", flush=True)
print(f"  AUC predicted-e3 -> green-early (e3>0): {roc_auc_score((act>0).astype(int), prd):.3f} (0.5=vô dụng)", flush=True)
qs = pd.qcut(pd.Series(prd).rank(method="first"), 4, labels=["Q1 dự thấp", "Q2", "Q3", "Q4 dự cao"])
dd = pd.DataFrame({"prd": prd, "act": act, "q": qs})
for q, g in dd.groupby("q", observed=True):
    print(f"    {q:11s}: actual-e3 TB {100*g.act.mean():+5.2f}% | green {100*(g.act>0).mean():3.0f}%", flush=True)

# NAV pre-filter: skip predicted-red-early
print("\n=== NAV pre-filter (skip predicted-e3 < thr) vs base 123.4 / post-cut 127.8 ===", flush=True)
bcg = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd])[0] for sd in SEEDS]
print(f"  base: {100*statistics.mean(bcg):.1f}%", flush=True)
allpr = np.concatenate([[v for v in E3[sd].values()] for sd in SEEDS])
for p in (20, 30, 40):
    thr = float(np.nanpercentile(allpr, p))
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], e3skip=E3[sd], e3thr=thr) for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); sgn = sum(1 for i in range(3) if rr[i][0] > bcg[i])
    mk = "*" if (sgn == 3 and cg > statistics.mean(bcg)) else ("+" if sgn == 3 else " ")
    print(f"  skip predicted-e3 bottom{p}% (<{thr:+.3f}): CAGR {100*cg:.1f}%{mk} ({sgn}/3) DD {100*statistics.mean([d for c,d in rr]):.1f}%", flush=True)
print("PREENTRY_DONE", flush=True)
