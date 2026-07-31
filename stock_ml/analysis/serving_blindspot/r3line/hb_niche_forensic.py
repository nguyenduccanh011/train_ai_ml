# -*- coding: utf-8 -*-
"""NICHE forensic on operating model (K10/cs5_ma50+ret7, T+2). Two under-dissected corners:
 (1) PREEMPT-REGRET: each eviction swaps an evicted leg (low meta-prio) for a new one. Ex-post, is the
     replacement's full-trade net actually > the evicted's? If not, prio mis-ranks -> preempt destroys value.
 (2) CONV-SKIP threshold: skip conv<0.40. Sweep threshold + skipped-bucket net -> are we throwing winners?"""
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
SEEDS = [42, 21, 123]; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, TPLUS, R7THR = 10, 0.005, 2.0, 2, 0.02


def prun(sim, pm, cm, r7map, skip, evrec=None, skrec=None, advance_fee=0.0008, roundtrip=0.006):
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
        if t["conv"] < skip:
            if skrec is not None:
                skrec.append(dict(net=t["net"], conv=t["conv"]))
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

    def mk(t, size, di):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"], ratio0=t["p0"] / c0,
                    ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di, cnet=t["net"], ed=t["entry_date"])

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
                    if evrec is not None:
                        evrec.append(dict(evicted_net=c["cnet"], repl_net=t["net"], evicted_prio=c["prio"], repl_prio=t["prio"], ed=t["entry_date"]))
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

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_nf_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

# ---- (1) PREEMPT-REGRET ----
ev = []
for sd in SEEDS:
    prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], 0.40, evrec=ev)
E = pd.DataFrame(ev); E["yr"] = pd.to_datetime(E.ed).dt.year; E["gain"] = E.repl_net - E.evicted_net
print(f"=== (1) PREEMPT-REGRET: {len(E)} lần đuổi (3-seed) ===", flush=True)
print(f"  evicted net TB {100*E.evicted_net.mean():+.2f}% | replacement net TB {100*E.repl_net.mean():+.2f}% | GAIN TB {100*E.gain.mean():+.2f}%/lần", flush=True)
print(f"  % lần đuổi ĐÚNG (repl>evicted): {100*(E.gain>0).mean():.0f}%", flush=True)
print("  theo năm (gain>0 = preempt tạo giá trị):", flush=True)
for y, g in E.groupby("yr"):
    print(f"    {y}: n={len(g):4d} evicted {100*g.evicted_net.mean():+5.2f}% repl {100*g.repl_net.mean():+5.2f}% gain {100*g.gain.mean():+5.2f}% ({100*(g.gain>0).mean():3.0f}% đúng)", flush=True)

# ---- (2) CONV-SKIP threshold sweep ----
print("\n=== (2) CONV-SKIP threshold: NAV + skipped-bucket net (3-seed T+2) ===", flush=True)
b = None
for skip in (0.30, 0.35, 0.40, 0.45, 0.50):
    cgv, sk = [], []
    for sd in SEEDS:
        skr = []
        c, d = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], skip, skrec=skr)
        cgv.append(c); sk.append(pd.DataFrame(skr) if skr else pd.DataFrame({"net": []}))
    cg = statistics.mean(cgv)
    if b is None:
        b = cgv
    skdf = pd.concat(sk, ignore_index=True); skn = 100 * skdf.net.mean() if len(skdf) else 0.0
    sgn = sum(1 for i in range(3) if cgv[i] > b[i])
    mk = " (base)" if skip == 0.40 else ("  <== vượt 3/3" if (sgn == 3 and cg > statistics.mean(b)) else "")
    print(f"  skip<{skip}: CAGR {100*cg:5.1f}% | skipped {len(skdf):5d} lệnh net TB {skn:+.2f}% (thắng {100*(skdf.net>0).mean() if len(skdf) else 0:.0f}%){mk}", flush=True)
print("NICHE_DONE", flush=True)
