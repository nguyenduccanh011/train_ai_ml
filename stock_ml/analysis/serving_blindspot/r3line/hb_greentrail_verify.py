# -*- coding: utf-8 -*-
"""VERIFY the green-early wide-trailing lead (+3.4pp 3/3): for trades GREEN at session 2 (early-cut doesn't
touch), add a WIDE trailing stop (drop X% from running peak -> sell next session) to catch the residual
'rose-then-crashed' case-2. Sweep X (plateau?), per-year, cross-K. Red-early still early-cut. T+2 3-seed."""
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
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]; TPLUS, R7THR = 2, 0.02


def prun(sim, pm, cm, r7map, K, margin, kconv, want_nav=False, advance_fee=0.0008, roundtrip=0.006):
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
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + kconv * z, 0.4), 1.8); raw.append(t["_w"])
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
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1); P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in ("cs4", "cs5_ma50")}
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
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}


def cmap(sd, sig):
    return {(sym, ed): CS[sig].get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


def cut_csv(sd, trail=None):
    """early-cut (red@s2 -> sell s3); green@s2 -> wide trailing 'trail' from peak -> sell next session."""
    cv = cvtrs[sd]; nd, npx = [], []
    for r in cv.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei:
            nd.append(r.exit_date); npx.append(r.exit_price); continue
        c = CLO[r.symbol]; ep = r.entry_price; ek = xi; ep_ = r.exit_price
        if xi > ei + 3 and c[ei + 2] / ep - 1.0 < 0.0:
            ek = ei + 3; ep_ = float(c[ei + 3])
        elif trail is not None:
            peak = c[ei]
            for b in range(ei + 2, xi + 1):
                peak = max(peak, c[b])
                if c[b] <= peak * (1 - trail) and b >= ei + TPLUS:
                    if b + 1 <= xi:
                        ek = b + 1; ep_ = float(c[b + 1])
                    else:
                        ek = b; ep_ = float(c[b])
                    break
        nd.append(INV[r.symbol][ek]); npx.append(ep_)
    out = cv[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    out["exit_date"] = pd.to_datetime(nd); out["exit_price"] = npx
    f = HERE / f"_gt_s{sd}.csv"; out.to_csv(f, index=False); return str(f)


def peryear(d):
    g = d.copy(); g["y"] = g["date"].dt.year; ye = g.groupby("y")["nav"].last(); out = {}; prev = 1.0
    for y, v in ye.items():
        out[int(y)] = v / prev - 1; prev = v
    return out


# (1) threshold plateau, K10/cs5_ma50
cms = {sd: cmap(sd, "cs5_ma50") for sd in SEEDS}
base = [prun(NavSim2(cut_csv(sd, None), date_lo="2020-01-01"), pm[sd], cms[sd], MW[sd], 10, 0.005, 2.0) for sd in SEEDS]
b0 = statistics.mean([c for c, d in base])
print(f"=== (1) GREEN-TRAIL threshold sweep (K10/cs5_ma50, early-cut only = {100*b0:.1f}%) ===", flush=True)
for tr in (0.06, 0.07, 0.08, 0.09, 0.10, 0.12):
    rr = [prun(NavSim2(cut_csv(sd, tr), date_lo="2020-01-01"), pm[sd], cms[sd], MW[sd], 10, 0.005, 2.0) for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); sgn = sum(1 for i in range(3) if rr[i][0] > base[i][0])
    mk = "*" if (sgn == 3 and cg > b0) else ("+" if sgn == 3 else " ")
    print(f"  trail {int(100*tr)}%: CAGR {100*cg:5.1f}%{mk} ({100*(cg-b0):+4.1f}, {sgn}/3) DD {100*dd:5.1f}%", flush=True)

# (2) per-year @8%
print("\n=== (2) PER-YEAR K10/cs5_ma50 (early-cut vs +green-trail8%) ===", flush=True)
byb, byt = defaultdict(list), defaultdict(list)
for sd in SEEDS:
    _, _, nb = prun(NavSim2(cut_csv(sd, None), date_lo="2020-01-01"), pm[sd], cms[sd], MW[sd], 10, 0.005, 2.0, want_nav=True)
    _, _, nt = prun(NavSim2(cut_csv(sd, 0.08), date_lo="2020-01-01"), pm[sd], cms[sd], MW[sd], 10, 0.005, 2.0, want_nav=True)
    for y, v in peryear(nb).items():
        byb[y].append(v)
    for y, v in peryear(nt).items():
        byt[y].append(v)
for y in sorted(byb):
    b = statistics.mean(byb[y]); t = statistics.mean(byt[y]); print(f"  {y}: early-cut {100*b:+6.1f}% -> +trail8% {100*t:+6.1f}% ({100*(t-b):+5.1f})", flush=True)

# (3) cross-K @8%
print("\n=== (3) CROSS-K +green-trail8% (T+2 3-seed) ===", flush=True)
for lab, Kk, mg, kc, sig in [("K10/cs5_ma50", 10, 0.005, 2.0, "cs5_ma50"), ("K12/cs4", 12, 0.01, 2.0, "cs4"), ("K16/cs4", 16, 0.01, 1.5, "cs4")]:
    cm = {sd: cmap(sd, sig) for sd in SEEDS}
    bb = [prun(NavSim2(cut_csv(sd, None), date_lo="2020-01-01"), pm[sd], cm[sd], MW[sd], Kk, mg, kc) for sd in SEEDS]
    tt = [prun(NavSim2(cut_csv(sd, 0.08), date_lo="2020-01-01"), pm[sd], cm[sd], MW[sd], Kk, mg, kc) for sd in SEEDS]
    b_ = statistics.mean([c for c, d in bb]); sgn = sum(1 for i in range(3) if tt[i][0] > bb[i][0])
    print(f"  {lab:13s} early-cut {100*b_:5.1f}% -> +trail8% {100*statistics.mean([c for c,d in tt]):5.1f}% (+{100*(statistics.mean([c for c,d in tt])-b_):4.1f}, {sgn}/3) DD {100*statistics.mean([d for c,d in tt]):5.1f}", flush=True)
print("GREENTRAIL_DONE", flush=True)
