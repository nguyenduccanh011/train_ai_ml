# -*- coding: utf-8 -*-
"""Populate 'Danh muc' (run_equity + run_portfolio_daily + run_pending + run_skipped) for the NEW TOP models
that lack it: k10c2m005 (cs4), k10c2m005_cs5 (cs5), k10_cs5ma50 (cs5_ma50) — all K10/margin0.005/convk2.0.
Same populator as champion but parametrized (K, margin, kconv, conviction-signal). seed42."""
from __future__ import annotations
import os, sys, statistics
from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as _np, duckdb
from psycopg2.extras import execute_values
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
SEED = 42; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]; PB_PCT, PB_WIN = 0.045, 40
SPECS = [  # (run_name, K, margin, kconv, sig)
    ("x2_struct_to_k10c2m005", 10, 0.005, 2.0, "cs4"),
    ("x2_struct_to_k10c2m005_cs5", 10, 0.005, 2.0, "cs5"),
    ("x2_struct_to_k10_cs5ma50", 10, 0.005, 2.0, "cs5_ma50"),
]


def prun_daily(sim, pm, cm, K, kconv, margin, RID, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + kconv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= SKIP:
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

    def mk(t, size, dt, nav_at):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"],
                    prio=t["prio"], entry_dt=dt, conv=t["conv"], nav_at=nav_at)

    equity = []; holdings = []; held_by_date = {}
    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list)
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values()); exit_today = {}
        for leg in list(exits.get(dt, ())):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg); exit_today[leg["symbol"]] = "signal"
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos; new_today = set()
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size, dt, nav_now); legs.append(leg); exits[t["exit_date"]].append(leg); new_today.add(t["symbol"])
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    exit_today[c["symbol"]] = "preempt"
                    nav_mid = cash + pt + sum(lv(l, dt) for l in legs); size = (nav_mid / K) * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, dt, nav_mid); legs.append(leg); exits[t["exit_date"]].append(leg); new_today.add(t["symbol"])
        pos = sum(lv(l, dt) for l in legs); nav = cash + pt + pos; ds = str(pd.to_datetime(dt).date())
        equity.append((RID, ds, float(nav), float(cash + pt), float(pos / nav if nav > 0 else 0.0), len(legs)))
        for leg in legs:
            val = lv(leg, dt)
            holdings.append((RID, ds, leg["symbol"], float(val / nav if nav > 0 else 0.0),
                             float(leg["invested"] / leg["nav_at"] if leg["nav_at"] else 0.0),
                             float(val / leg["invested"] - 1.0 if leg["invested"] else 0.0),
                             str(pd.to_datetime(leg["entry_dt"]).date()), int((pd.to_datetime(dt) - pd.to_datetime(leg["entry_dt"])).days),
                             leg["symbol"] in new_today, False, None, float(leg["conv"])))
        for sym, reason in exit_today.items():
            holdings.append((RID, ds, sym, 0.0, 0.0, 0.0, ds, 0, False, True, reason, None))
        held_by_date[ds] = {leg["symbol"] for leg in legs}
    return equity, holdings, held_by_date


# panel with cs4/cs5/cs5_ma50
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close,high FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs4"] = P[b4].mean(axis=1); P["cs5"] = P[b4 + ["atrpct_r"]].mean(axis=1); P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in ("cs4", "cs5", "cs5_ma50")}

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=SEED).get("run_id")
bt = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,pnl_pct from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
bt["sigd"] = pd.to_datetime(bt["entry_signal_date"])
cv = HERE / f"_pt_s{SEED}.csv"; bt[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
feat = M.features(bt.symbol.unique().tolist()); pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
sigdf = pd.read_sql("select symbol,date from run_signals where run_id=%s and signal=1", con, params=(rid,)); sigdf["date"] = pd.to_datetime(sigdf["date"])
# run_id lookup
nm2rid = {r[0]: r[1] for r in pd.read_sql("select run_name,run_id from leaderboard_runs where run_name like %s", con, params=('x2_struct_to_k10%',)).itertuples(index=False)}


def cmap_for(sig):
    return {(r.symbol, str(pd.to_datetime(r.entry_date).date())): CS[sig].get((r.symbol, str(r.sigd.date())), 0.5) for r in bt.itertuples()}


def compute_skipped(RID, cm):
    out = []
    for r in bt.itertuples():
        conv = cm.get((r.symbol, str(pd.to_datetime(r.entry_date).date())), 0.5)
        if conv < SKIP:
            out.append((RID, r.symbol, str(r.sigd.date()) if pd.notna(r.sigd) else None, str(pd.to_datetime(r.entry_date).date()), float(r.pnl_pct or 0.0), float(conv), "conv_skip"))
    return out


def compute_pending(RID, caldates, held_by_date):
    bar_of = {d: i for i, d in enumerate(caldates)}; nD = len(caldates); low_arr, close_arr = {}, {}
    for s, g in px.groupby("symbol"):
        la = _np.full(nD, _np.nan); ca = _np.full(nD, _np.nan)
        for r in g.itertuples():
            b = bar_of.get(r.date)
            if b is not None:
                la[b] = r.low; ca[b] = r.close
        low_arr[s] = la; close_arr[s] = ca
    sig_bars = defaultdict(list)
    for r in sigdf.itertuples():
        b = bar_of.get(r.date)
        if b is not None:
            sig_bars[r.symbol].append(b)
    for d in sig_bars:
        sig_bars[d].sort()
    out = []
    for sym, sbars in sig_bars.items():
        la = low_arr.get(sym); ca = close_arr.get(sym)
        if la is None:
            continue
        for d in range(nD):
            li = bisect_right(sbars, d - PB_WIN); ri = bisect_right(sbars, d)
            if li >= ri:
                continue
            S = sbars[ri - 1]; cS = ca[S]
            if _np.isnan(cS):
                continue
            limit = cS * (1.0 - PB_PCT); seg_sd = la[S:d + 1]
            if seg_sd.size and _np.nanmin(seg_sd) <= limit:
                continue
            ds = str(caldates[d].date())
            if sym in held_by_date.get(ds, set()):
                continue
            cD = ca[d]
            if _np.isnan(cD):
                continue
            pctL = limit / cD - 1.0; hi = min(S + PB_WIN, nD - 1); seg = la[S:hi + 1]; touch = _np.where(seg <= limit)[0]
            if touch.size:
                outc, rdate = "fill", str(caldates[S + int(touch[0])].date())
            else:
                outc, rdate = "expire", str(caldates[hi].date())
            out.append((RID, ds, sym, str(caldates[S].date()), d - S, float(limit), float(cD), float(pctL), outc, rdate))
    return out


cur = con.cursor()
for nm, K, margin, kconv, sig in SPECS:
    RID = nm2rid.get(nm)
    if RID is None:
        print(f"  {nm}: run_id NOT FOUND, skip", flush=True); continue
    cm = cmap_for(sig)
    equity, holdings, hbd = prun_daily(NavSim2(str(cv), date_lo="2020-01-01"), pm, cm, K, kconv, margin, RID)
    caldates = [pd.to_datetime(e[1]) for e in equity]
    skipped = compute_skipped(RID, cm); pending = compute_pending(RID, caldates, hbd)
    for tbl in ("run_equity", "run_portfolio_daily", "run_skipped", "run_pending"):
        cur.execute(f"DELETE FROM {tbl} WHERE run_id=%s", (RID,))
    execute_values(cur, "INSERT INTO run_equity (run_id,date,nav,cash,exposure,n_positions) VALUES %s", equity)
    execute_values(cur, "INSERT INTO run_portfolio_daily (run_id,date,symbol,weight,entry_weight,unreal_pnl,entry_date,days_held,is_new,is_exit,exit_reason,conv) VALUES %s", holdings)
    if skipped:
        execute_values(cur, "INSERT INTO run_skipped (run_id,symbol,signal_date,entry_date,pnl_pct,conv,skip_reason) VALUES %s", skipped)
    if pending:
        execute_values(cur, "INSERT INTO run_pending (run_id,date,symbol,signal_date,days_waiting,limit_price,ref_price,pct_to_limit,outcome,result_date) VALUES %s", pending)
    con.commit()
    print(f"  {nm:28s} ({sig}): equity={len(equity)} holdings={len(holdings)} pending={len(pending)} skipped={len(skipped)} | NAVx{equity[-1][2]:.1f} avg_npos={statistics.mean(e[5] for e in equity):.1f}", flush=True)
con.close()
print("PORTFOLIO_TOP_DONE", flush=True)
