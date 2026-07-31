# -*- coding: utf-8 -*-
"""Populate a T+2-realistic CAGR column (cagr_t2/maxdd_t2) on leaderboard_nav for the 8 offline-combo
models. Keeps existing cagr_adv (T+0 theoretical) untouched for the 7 old ones; for t2ret5g fixes
cagr_adv to its T+0 (gated) value so its cagr_adv column is also T+0-consistent. 3-seed."""
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
PRE, SUF = "template/", "-69338138"
MODELS = [  # (name, K, margin, kconv, sig, gate_thr, also_fix_t0)
    ("x2_struct_to_k16preempt_cssize", 16, 0.01, 1.5, "cs4", None, False),
    ("x2_struct_to_k12size2", 12, 0.01, 2.0, "cs4", None, False),
    ("x2_struct_to_k10preempt_cssize", 10, 0.01, 1.5, "cs4", None, False),
    ("x2_struct_to_k8preempt_cssize", 8, 0.01, 2.5, "cs4", None, False),
    ("x2_struct_to_k10c2m005", 10, 0.005, 2.0, "cs4", None, False),
    ("x2_struct_to_k10c2m005_cs5", 10, 0.005, 2.0, "cs5", None, False),
    ("x2_struct_to_k10_cs5ma50", 10, 0.005, 2.0, "cs5_ma50", None, False),
    ("x2_struct_to_k10_cs5ma50_t2ret5g", 10, 0.005, 2.0, "cs5_ma50", 0.03, True),
]


def prun(sim, pm, cm, tplus, K, margin, kconv, r5map=None, r5thr=None, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if tplus and (bx - be) < tplus:
            nb = min(be + tplus, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
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
            v = r5map.get((t["symbol"], t["entry_date"]), _np.nan)
            if not _np.isnan(v) and v < r5thr:
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
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


# panels
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["ret5"] = c / c.shift(5) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "ret5"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs4"] = P[b4].mean(axis=1); P["cs5"] = P[b4 + ["atrpct_r"]].mean(axis=1); P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in ("cs4", "cs5", "cs5_ma50")}
R5 = {(r.symbol, str(r.date.date())): (r.ret5 if pd.notna(r.ret5) else _np.nan) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_t2c_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')


def cmap(sd, sig):
    return {(sym, ed): CS[sig].get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


def r5map(sd):
    return {(sym, ed): R5.get((sym, sgd), _np.nan) for sym, ed, sgd in key_c[sd]}


def ev(K, mg, kc, sig, tplus, thr):
    r5 = {sd: r5map(sd) for sd in SEEDS} if thr is not None else None
    cg, dd = [], []
    for sd in SEEDS:
        c, d = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cmap(sd, sig), tplus, K, mg, kc,
                    (r5[sd] if r5 else None), thr)
        cg.append(c); dd.append(d)
    return statistics.mean(cg), statistics.mean(dd)


cur = con.cursor()
cur.execute("ALTER TABLE leaderboard_nav ADD COLUMN IF NOT EXISTS cagr_t2 double precision")
cur.execute("ALTER TABLE leaderboard_nav ADD COLUMN IF NOT EXISTS maxdd_t2 double precision")
con.commit()
print(f"  {'model':34s} | T+0 CAGR | T+2 CAGR/DD", flush=True)
for name, K, mg, kc, sig, thr, fix_t0 in MODELS:
    rid = PRE + name + SUF
    t2cg, t2dd = ev(K, mg, kc, sig, 2, thr)
    if fix_t0:
        t0cg, t0dd = ev(K, mg, kc, sig, 0, thr)
        cur.execute("update leaderboard_nav set cagr_adv=%s, maxdd_nav=%s, cagr_t2=%s, maxdd_t2=%s where run_id=%s",
                    (t0cg, t0dd, t2cg, t2dd, rid))
        t0s = f"{100*t0cg:5.1f}%"
    else:
        cur.execute("update leaderboard_nav set cagr_t2=%s, maxdd_t2=%s where run_id=%s", (t2cg, t2dd, rid))
        t0s = "(giữ)"
    con.commit()
    print(f"  {name:34s} | {t0s:8s} | {100*t2cg:5.1f}%/{100*t2dd:5.1f}", flush=True)
con.close()
print("T2_COLUMN_DONE", flush=True)
