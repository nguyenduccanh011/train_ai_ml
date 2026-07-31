# -*- coding: utf-8 -*-
"""Harden the T+2 ret5-entry-gate before promote: (1) per-year breakdown (not one-year-driven),
(2) cross-K (K10/cs5_ma50, K16/cs4 champion, K12/cs4). Gate = skip entry if ret5(signal date) < thr.
All under T+2 (VN real). 3-seed."""
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
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]; TPLUS = 2


def prun(sim, pm, cm, K, margin, kconv, r5map=None, r5thr=None, want_nav=False, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if TPLUS and (bx - be) < TPLUS:
            nb = min(be + TPLUS, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + kconv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < SKIP:
            continue
        if r5map is not None:
            v = r5map.get((t["symbol"], t["entry_date"]), np.nan)
            if not np.isnan(v) and v < r5thr:
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
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di)

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
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    cg = float(nav.iloc[-1]) ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    return (cg, dd, d) if want_nav else (cg, dd)


def peryear(navdf):
    g = navdf.copy(); g["y"] = g["date"].dt.year; ye = g.groupby("y")["nav"].last()
    out = {}; prev = 1.0
    for y, v in ye.items():
        out[int(y)] = v / prev - 1.0; prev = v
    return out


# ---- price panel: cs4, cs5_ma50, ret5 ----
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
    g["ret5"] = c / c.shift(5) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "ret5"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs4"] = P[b4].mean(axis=1); P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in ("cs4", "cs5_ma50")}
R5 = {(r.symbol, str(r.date.date())): (r.ret5 if pd.notna(r.ret5) else np.nan) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None
cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_r5_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def cmap(sd, sig):
    return {(sym, ed): CS[sig].get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


def r5map(sd):
    return {(sym, ed): R5.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]}


CONFIGS = [("K10/cs5_ma50", 10, 0.005, 2.0, "cs5_ma50"), ("K16/cs4 champ", 16, 0.01, 1.5, "cs4"), ("K12/cs4", 12, 0.01, 2.0, "cs4")]
print("=== CROSS-K: ret5-gate dưới T+2, 3-seed (Δ vs no-gate) ===", flush=True)
for lab, K, mg, kc, sig in CONFIGS:
    cms = {sd: cmap(sd, sig) for sd in SEEDS}; r5s = {sd: r5map(sd) for sd in SEEDS}
    base = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cms[sd], K, mg, kc) for sd in SEEDS]
    bcg = statistics.mean([c for c, d in base])
    print(f"  {lab:14s} baseline CAGR {100*bcg:.1f}% DD {100*statistics.mean([d for c,d in base]):.1f}%", flush=True)
    for thr in (0.02, 0.03):
        rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cms[sd], K, mg, kc, r5s[sd], thr) for sd in SEEDS]
        dcg = [rr[i][0] - base[i][0] for i in range(3)]; sgn = sum(1 for x in dcg if x > 0)
        print(f"    ret5<{thr:.2f}: CAGR {100*statistics.mean([c for c,d in rr]):.1f}% DD {100*statistics.mean([d for c,d in rr]):.1f}%  (Δ {['%+.0f' % (100*x) for x in dcg]}, {sgn}/3)", flush=True)

print("\n=== PER-YEAR: K10/cs5_ma50 T+2, seed-mean return theo năm (base vs ret5<0.03) ===", flush=True)
cms = {sd: cmap(sd, "cs5_ma50") for sd in SEEDS}; r5s = {sd: r5map(sd) for sd in SEEDS}
byb, byg = defaultdict(list), defaultdict(list)
for sd in SEEDS:
    _, _, nb = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cms[sd], 10, 0.005, 2.0, want_nav=True)
    _, _, ng = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cms[sd], 10, 0.005, 2.0, r5s[sd], 0.03, want_nav=True)
    for y, r in peryear(nb).items():
        byb[y].append(r)
    for y, r in peryear(ng).items():
        byg[y].append(r)
print(f"  {'năm':6s} | base   | +ret5gate | Δ", flush=True)
for y in sorted(byb):
    b = statistics.mean(byb[y]); g = statistics.mean(byg[y]); mk = "  <-- xấu đi" if g < b - 0.005 else ""
    print(f"  {y:6d} | {100*b:+6.1f}% | {100*g:+7.1f}%  | {100*(g-b):+5.1f}pp{mk}", flush=True)
print("RET5_VERIFY_DONE", flush=True)
