# -*- coding: utf-8 -*-
"""NICHE #3: BOOK-CORRELATION. Is the K=10 book loading co-moving names (inflating DD)? No sector table ->
use causal trailing-60d return correlation as proxy. (A) measure book avg pairwise corr per year.
(B) diversity cap: skip a fill whose trailing-corr with the current book > CAP -> does de-correlation cut
DD (and at what CAGR cost)? Operating model K10/cs5_ma50+ret7, T+2, 3-seed. User metric = plain CAGR."""
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
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, TPLUS, R7THR, CW = 10, 0.005, 2.0, 2, 0.02, 60


def prun(sim, pm, cm, r7map, RM, pos_of, div_cap=None, measure=False):
    s_new = (0.006 - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
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
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"], ratio0=t["p0"] / c0,
                    ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=t["_di"])

    def avgcorr(sym, held_syms, gi):
        if gi < CW or not held_syms:
            return 0.0
        win = RM[gi - CW:gi]; cser = win[:, colof[sym]]
        cs = []
        for hs in held_syms:
            o = win[:, colof[hs]]; m = ~(np.isnan(cser) | np.isnan(o))
            if m.sum() > 20:
                a = cser[m]; b = o[m]
                if a.std() > 0 and b.std() > 0:
                    cs.append(np.corrcoef(a, b)[0, 1])
        return float(np.mean(cs)) if cs else 0.0

    colof = {s: i for i, s in enumerate(SYMS)}
    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; bc = defaultdict(list)
    for di, dt in enumerate(cal):
        for t in entries.get(dt, ()):
            t["_di"] = di
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - 0.0008); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        gi = pos_of.get(dt, -1)
        for t in entries.get(dt, ()):
            hs = [l["symbol"] for l in legs]
            if div_cap is not None and avgcorr(t["symbol"], hs, gi) > div_cap:
                continue                                        # diversity: too correlated with book
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= TPLUS]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - 0.0008); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
        if measure and len(legs) >= 2 and gi >= CW:
            hs = [l["symbol"] for l in legs]; pc = []
            for a in range(len(hs)):
                for b in range(a + 1, len(hs)):
                    w = RM[gi - CW:gi]; x = w[:, colof[hs[a]]]; y = w[:, colof[hs[b]]]; m = ~(np.isnan(x) | np.isnan(y))
                    if m.sum() > 20 and x[m].std() > 0 and y[m].std() > 0:
                        pc.append(np.corrcoef(x[m], y[m])[0, 1])
            if pc:
                bc[pd.to_datetime(dt).year].append(np.mean(pc))
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    cg = float(nav.iloc[-1]) ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    return (cg, dd, {y: np.mean(v) for y, v in bc.items()}) if measure else (cg, dd)


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["mw7"] = c / c.shift(7) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "mw7"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
MW7 = {(r.symbol, str(r.date.date())): (r.mw7 if pd.notna(r.mw7) else np.nan) for r in P.itertuples()}
# returns matrix aligned to a global calendar
RET = px.pivot(index="date", columns="symbol", values="close").sort_index().pct_change()
SYMS = list(RET.columns); RM = RET.values; pos_of = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(RET.index)}

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_bc_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

# (A) measure book pairwise corr per year (seed42)
_, _, bc = prun(NavSim2(cv_c[42], date_lo="2020-01-01"), pm[42], CMS[42], MW[42], RM, pos_of, measure=True)
print("=== (A) BOOK avg pairwise corr (trailing-60d) per year — cao = book đồng-vận-động ===", flush=True)
for y in sorted(bc):
    print(f"    {y}: {bc[y]:.3f}", flush=True)

# (B) diversity-cap sweep
print("\n=== (B) DIVERSITY CAP (skip fill nếu corr với book > cap) — 3-seed T+2 ===", flush=True)
bcg = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], RM, pos_of)[0] for sd in SEEDS]
bdd = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], RM, pos_of)[1] for sd in SEEDS]
print(f"  base (no cap): CAGR {100*statistics.mean(bcg):.1f}% DD {100*statistics.mean(bdd):.1f}%", flush=True)
for cap in (0.5, 0.6, 0.7):
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], RM, pos_of, div_cap=cap) for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr])
    sgn = sum(1 for i in range(3) if rr[i][0] > bcg[i])
    print(f"  cap {cap}: CAGR {100*cg:5.1f}% ({100*(cg-statistics.mean(bcg)):+4.1f}pp, {sgn}/3) DD {100*dd:5.1f}% ({100*(dd-statistics.mean(bdd)):+4.1f}pp)", flush=True)
print("BOOKCORR_DONE", flush=True)
