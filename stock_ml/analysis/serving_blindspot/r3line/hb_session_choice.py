# -*- coding: utf-8 -*-
"""Does session 1 or 3 carry the same meaning as session 2 for the red/green split (early-cut vs green-trail)?
Vary SIG (classification session): red@SIG -> early-cut sell SIG+1; green@SIG -> green-trail 8%. SIG in
{1,2,3}. Earlier=cut sooner(less bleed) but noisier; later=more info but more bleed. NAV 3-seed + per-year
on K10/cs5_ma50+ret7 T+2. Registered = SIG2 (131.7%)."""
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
K, MARGIN, KCONV, TPLUS, R7THR, GT = 10, 0.005, 2.0, 2, 0.02, 0.08


def prun(sim, pm, cm, r7map, want_nav=False, advance_fee=0.0008, roundtrip=0.006):
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
    cg = float(nav.iloc[-1]) ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    return (cg, dd, d) if want_nav else (cg, dd)


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []; CLO = {}; DIDX = {}; INV = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values; DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}; INV[s] = {i: d.strftime("%Y-%m-%d") for i, d in enumerate(g["date"])}
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

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c, cvtrs = {}, {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvtrs[sd] = cvtr
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}


def cut_csv(sd, SIG):
    """red@SIG -> early-cut sell SIG+1; green@SIG -> green-trail GT%."""
    cv = cvtrs[sd]; nd, npx = [], []
    for r in cv.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei:
            nd.append(r.exit_date); npx.append(r.exit_price); continue
        c = CLO[r.symbol]; ep = r.entry_price; ek = xi; ep_ = r.exit_price
        exe = ei + SIG + 1
        if xi > exe and c[ei + SIG] / ep - 1.0 < 0.0 and exe >= ei + TPLUS:
            ek = exe; ep_ = float(c[exe])
        else:
            peak = c[ei]
            for b in range(ei + 2, xi + 1):
                peak = max(peak, c[b])
                if c[b] <= peak * (1 - GT) and b >= ei + TPLUS:
                    if b + 1 <= xi:
                        ek = b + 1; ep_ = float(c[b + 1])
                    else:
                        ek = b; ep_ = float(c[b])
                    break
        nd.append(INV[r.symbol][ek]); npx.append(ep_)
    out = cv[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    out["exit_date"] = pd.to_datetime(nd); out["exit_price"] = npx
    f = HERE / f"_sess_s{sd}.csv"; out.to_csv(f, index=False); return str(f)


def peryear(d):
    g = d.copy(); g["y"] = g["date"].dt.year; ye = g.groupby("y")["nav"].last(); out = {}; prev = 1.0
    for y, v in ye.items():
        out[int(y)] = v / prev - 1; prev = v
    return out


print("=== Split/cut session 1 vs 2 vs 3 (green-trail 8%, K10/cs5_ma50+ret7 T+2, 3-seed) ===", flush=True)
res = {}
for SIG in (1, 2, 3):
    rr = [prun(NavSim2(cut_csv(sd, SIG), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd]) for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); res[SIG] = ([c for c, d in rr], cg)
    tag = " <== registered" if SIG == 2 else ""
    print(f"  split@phiên{SIG}: CAGR {100*cg:5.1f}% DD {100*dd:5.1f}% | seed {['%.0f' % (100*c) for c,d in rr]}{tag}", flush=True)
# per-year 1 vs 2 vs 3
print("\n=== PER-YEAR (seed-mean) split@1 / @2 / @3 ===", flush=True)
by = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
for SIG in (1, 2, 3):
    for sd in SEEDS:
        _, _, d = prun(NavSim2(cut_csv(sd, SIG), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], want_nav=True)
        for y, v in peryear(d).items():
            by[SIG][y].append(v)
for y in sorted(by[2]):
    print(f"  {y}: @1 {100*statistics.mean(by[1][y]):+6.1f}% | @2 {100*statistics.mean(by[2][y]):+6.1f}% | @3 {100*statistics.mean(by[3][y]):+6.1f}%", flush=True)
print("SESSION_DONE", flush=True)
