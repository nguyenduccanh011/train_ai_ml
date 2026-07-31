# -*- coding: utf-8 -*-
"""Extend the ret5-gate idea: sweep momentum-gate lookback windows retN (N=2..10) x thresholds, under
T+2, 3-seed, on top model K10/cs5_ma50. Goal: does any window beat ret5<0.03 (121.3%) ROBUSTLY?
Print the FULL grid so a smooth plateau (real) is distinguishable from a spiky pick (overfit noise).
Also test a 2-window COMBINED gate."""
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
import psycopg2, pandas as pd, numpy as _np, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV = 10, 0.005, 2.0
NS = [2, 3, 4, 5, 7, 10, 20]; THRS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


def prun(sim, pm, cm, gates, advance_fee=0.0008, roundtrip=0.006):
    """gates = list of (map, thr): skip entry if ANY gate feature < its thr (all must pass)."""
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    tplus = 2
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if (bx - be) < tplus:
            nb = min(be + tplus, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
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
        drop = False
        for gm, gt in gates:
            v = gm.get((t["symbol"], t["entry_date"]), _np.nan)
            if not _np.isnan(v) and v < gt:
                drop = True; break
        if drop:
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
                cand = [l for l in legs if (di - l["be_di"]) >= tplus]
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


# panel: cs5_ma50 + retN
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
    for n in NS:
        g[f"mw{n}"] = c / c.shift(n) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50"] + [f"mw{n}" for n in NS]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
RN = {n: {(r.symbol, str(r.date.date())): (getattr(r, f"mw{n}") if pd.notna(getattr(r, f"mw{n}")) else _np.nan) for r in P.itertuples()} for n in NS}

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_mg_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def cmap(sd):
    return {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


def rnmap(sd, n):
    return {(sym, ed): RN[n].get((sym, sgd), _np.nan) for sym, ed, sgd in key_c[sd]}


CMS = {sd: cmap(sd) for sd in SEEDS}


def ev(gate_specs):
    """gate_specs: list of (n, thr) -> build per-seed gates."""
    cg, dd = [], []
    for sd in SEEDS:
        gates = [(rnmap(sd, n), thr) for n, thr in gate_specs]
        c, d = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], gates)
        cg.append(c); dd.append(d)
    return statistics.mean(cg), statistics.mean(dd), sum(1 for x in cg if x > 0)  # sgn placeholder


base = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], []) for sd in SEEDS]
bcg = statistics.mean([c for c, d in base]); bseed = [c for c, d in base]
print(f"=== SWEEP retN-gate (T+2, K10/cs5_ma50) — baseline {100*bcg:.1f}%, ref ret5<0.03=121.3% ===", flush=True)
print("  sgn = #seed CAGR > baseline-seed (cần 3/3). * = 3/3 VÀ mean>ref 121.3", flush=True)
print("  N \\ thr |" + "".join(f"  {t:.2f} " for t in THRS), flush=True)
best = None
for n in NS:
    cells = []
    for thr in THRS:
        cg, dd, _ = ev([(n, thr)])
        sgn = 0
        for sd_i, sd in enumerate(SEEDS):
            gates = [(rnmap(sd, n), thr)]
            c, _ = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], gates)
            if c > bseed[sd_i]:
                sgn += 1
        star = "*" if (sgn == 3 and cg > 1.213) else ("+" if sgn == 3 else " ")
        cells.append(f"{100*cg:4.0f}{star}")
        if sgn == 3 and (best is None or cg > best[2]):
            best = (n, thr, cg, dd)
    print(f"  ret{n:<2d}   |" + "".join(f" {c:>5s}" for c in cells), flush=True)
if best:
    print(f"\n  BEST 3/3 single-window: ret{best[0]}<{best[1]:.2f} = {100*best[2]:.1f}% / DD {100*best[3]:.1f}%", flush=True)
    # combined: best window + ret20 (longer trend) gate
    for n2, t2 in [(20, 0.05), (20, 0.0)]:
        cg, dd, _ = ev([(best[0], best[1]), (n2, t2)])
        sgn = 0
        for sd_i, sd in enumerate(SEEDS):
            gates = [(rnmap(sd, best[0]), best[1]), (rnmap(sd, n2), t2)]
            c, _ = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], gates)
            if c > bseed[sd_i]:
                sgn += 1
        print(f"  + ret{n2}<{t2:.2f} combined: {100*cg:.1f}% / DD {100*dd:.1f}% ({sgn}/3)", flush=True)
print("MOMGATE_SWEEP_DONE", flush=True)
