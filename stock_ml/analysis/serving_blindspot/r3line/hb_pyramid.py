# -*- coding: utf-8 -*-
"""MULTI-LOT / PYRAMID (user unlocked "nhiều lượt", total capital<=1). Memory: pyramid = +63.7u/every year
sizing-alpha, DEFERRED waiting for exactly this multi-lot capability. Add a 2nd (3rd..) lot to a WINNING
position once it is up +ADD% from fill, funded from CASH (no leverage, cash>=0 enforced -> total<=1), exit
with the parent. New entries get cash priority; adds consume leftover (the defensive idle buffer -> DD trade).
Operating model K10/cs5_ma50 + ret7, T+2, 3-seed. CAGR/DD + per-year (does pyramid help weak years too?)."""
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
K, MARGIN, KCONV, TPLUS, R7THR = 10, 0.005, 2.0, 2, 0.02


def prun(sim, pm, cm, r7map, add_trig=None, add_frac=1.0, max_add=1, add_mode="strength", want_year=False, advance_fee=0.0008, roundtrip=0.006):
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

    def mk(t, size, di, p0=None, i0=None):
        s = t["symbol"]; i1 = t["i1"]
        if i0 is None:                                            # base leg: original fill + net
            i0 = t["i0"]; p0 = t["p0"]; netv = t["net"]
        else:                                                     # add-on: recompute net from add price
            netv = (t["x_raw"] * (1.0 - s_new)) / (p0 * (1.0 + s_new)) - 1.0 - FEE
        c0, c1 = sc[s][i0], sc[s][i1]; xe = p0 * (1.0 + netv)
        return dict(symbol=s, i0=i0, i1=i1, invested=size, net=netv, p0=p0, ratio0=p0 / c0, ratio1=xe / c1,
                    last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di, src=t, nadd=0)

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        # new entries (cash priority)
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
        # PYRAMID: add to winners up +ADD% from fill, from leftover cash (total<=1)
        if add_trig is not None:
            for leg in list(legs):
                s = leg["symbol"]; j = si[s].get(dt)
                if j is None or j >= leg["i1"]:
                    continue
                cpx = sc[s][j]; leg["peak"] = max(leg.get("peak", leg["p0"]), cpx)
                if leg["nadd"] >= max_add:
                    continue
                if add_mode == "strength":
                    fire = cpx >= leg["p0"] * (1.0 + add_trig * (leg["nadd"] + 1))
                else:  # dip: đã chạy >=5% rồi hồi về -add_trig từ đỉnh, vẫn còn lời
                    fire = (leg["peak"] >= leg["p0"] * 1.05) and (cpx <= leg["peak"] * (1.0 - add_trig)) and (cpx >= leg["p0"])
                if fire:
                    add_size = (nav_now / K) * leg["src"]["w"] * add_frac
                    if cash + 1e-12 >= add_size:
                        cash -= add_size; al = mk(leg["src"], add_size, di, p0=cpx, i0=j)
                        legs.append(al); exits[al["exit_date"]].append(al); leg["nadd"] += 1
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    cg = float(nav.iloc[-1]) ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    if want_year:
        g = d.copy(); g["y"] = g["date"].dt.year; ye = g.groupby("y")["nav"].last(); yr = {}; prev = 1.0
        for y, v in ye.items():
            yr[int(y)] = v / prev - 1; prev = v
        return cg, dd, yr
    return cg, dd


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
    cvf = HERE / f"_py_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}


def ev(add_trig, add_frac, max_add, mode="strength"):
    cg = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], add_trig, add_frac, max_add, mode) for sd in SEEDS]
    return statistics.mean([c for c, d in cg]), statistics.mean([d for c, d in cg]), [c for c, d in cg]


bcg, bdd, bseed = ev(None, 0, 0)
print(f"=== MULTI-LOT (T+2, K10/cs5_ma50+ret7, tổng vốn≤1) — base {100*bcg:.1f}%/DD{100*bdd:.1f} ===", flush=True)
print(f"  {'config':32s} | CAGR%  DD%  | vs base | 3/3", flush=True)
GRID = [("strength", 0.05, 1.0, 1), ("strength", 0.08, 0.5, 1),
        ("dip", 0.03, 1.0, 1), ("dip", 0.05, 1.0, 1), ("dip", 0.05, 0.5, 1), ("dip", 0.07, 1.0, 1),
        ("dip", 0.05, 1.0, 2), ("dip", 0.03, 0.5, 2)]
best = None
for mode, at, af, ma in GRID:
    cg, dd, sv = ev(at, af, ma, mode)
    sgn = sum(1 for i in range(3) if sv[i] > bseed[i]); mk = "*" if (sgn == 3 and cg > bcg) else ("+" if sgn == 3 else " ")
    print(f"  {mode}+{100*at:.0f}% frac{af} madd{ma:<9} | {100*cg:5.1f} {100*dd:5.1f} | {100*(cg-bcg):+5.1f}{mk} | {sgn}/3", flush=True)
    if sgn == 3 and cg > bcg and (best is None or cg > best[0]):
        best = (cg, at, af, ma, mode)
if best:
    print(f"\n  BEST 3/3: {best[4]}+{100*best[1]:.0f}% frac{best[2]} maxadd{best[3]} = {100*best[0]:.1f}% (+{100*(best[0]-bcg):.1f}pp)", flush=True)
    _, _, byr = prun(NavSim2(cv_c[42], date_lo="2020-01-01"), pm[42], CMS[42], MW[42], best[1], best[2], best[3], best[4], want_year=True)
    _, _, byr0 = prun(NavSim2(cv_c[42], date_lo="2020-01-01"), pm[42], CMS[42], MW[42], None, 0, 0, want_year=True)
    print("  per-year (seed42) base vs pyramid:", flush=True)
    for y in sorted(byr):
        print(f"    {y}: {100*byr0[y]:+6.1f}% -> {100*byr[y]:+6.1f}%  ({100*(byr[y]-byr0[y]):+5.1f}pp)", flush=True)
else:
    print("\n  Không config 3/3-vượt base -> pyramid không giúp (idle-cash defensive load-bearing thắng)", flush=True)
print("PYRAMID_DONE", flush=True)
