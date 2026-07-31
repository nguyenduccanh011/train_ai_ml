# -*- coding: utf-8 -*-
"""SCOPE-B step1: generate the champion's DAILY PORTFOLIO (holdings + actual weights) and EQUITY/exposure
timeline from the combo sim (preempt R2m01 + cs k1.5 + conv-skip 0.40, seed 42), and persist to two new
tables (run_equity, run_portfolio_daily) that the detail-page 'Danh mục' tab will read. Weight = position
value / nav that day. Marks is_new (entered today) / is_exit (closing today, incl preempt)."""
from __future__ import annotations
import os, sys, statistics
from collections import defaultdict
from pathlib import Path
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, duckdb
from psycopg2.extras import execute_values
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
CHAMP_RID = "template/x2_struct_to_k16preempt_cssize-69338138"
SEED = 42; K = 16; KCONV = 1.5; MARGIN = 0.01; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun_daily(sim, pm, cm, k_conv=KCONV, margin=MARGIN, skip=SKIP, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= skip:
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
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        exit_today = {}   # symbol -> reason
        for leg in list(exits.get(dt, ())):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
                exit_today[leg["symbol"]] = "signal"; leg["_gone"] = True
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        new_today = set()
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
        pos = sum(lv(l, dt) for l in legs); nav = cash + pt + pos
        ds = str(pd.to_datetime(dt).date())
        equity.append((CHAMP_RID, ds, float(nav), float(cash + pt), float(pos / nav if nav > 0 else 0.0), len(legs)))
        # holdings snapshot: weight = current market value / nav (drifts); entry_weight = FIXED allocation
        # at entry (invested / nav_at_entry); unreal_pnl = mark-to-market P&L of the holding as of today.
        for leg in legs:
            val = lv(leg, dt)
            holdings.append((CHAMP_RID, ds, leg["symbol"], float(val / nav if nav > 0 else 0.0),
                             float(leg["invested"] / leg["nav_at"] if leg["nav_at"] else 0.0),
                             float(val / leg["invested"] - 1.0 if leg["invested"] else 0.0),
                             str(pd.to_datetime(leg["entry_dt"]).date()), int((pd.to_datetime(dt) - pd.to_datetime(leg["entry_dt"])).days),
                             leg["symbol"] in new_today, False, None, float(leg["conv"])))
        # exit rows (is_exit=True) for names that closed today
        for sym, reason in exit_today.items():
            holdings.append((CHAMP_RID, ds, sym, 0.0, 0.0, 0.0, ds, 0, False, True, reason, None))
        held_by_date[ds] = {leg["symbol"] for leg in legs}
    return equity, holdings, held_by_date


# setup
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                "ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    parts.append(g[["symbol", "date"] + CS4])
P = pd.concat(parts, ignore_index=True)
for col in CS4:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1)
CS = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in P.itertuples()}

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=SEED).get("run_id")
bt = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,pnl_pct "
                 "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
bt["sigd"] = pd.to_datetime(bt["entry_signal_date"])
cm = {}
for r in bt.itertuples():
    cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
# SKIPPED (opportunity cost): base trades dropped by conv-skip (cs4 < SKIP) with their REALIZED base pnl.
skipped = []
for r in bt.itertuples():
    conv = cm.get((r.symbol, str(pd.to_datetime(r.entry_date).date())), 0.5)
    if conv < SKIP:
        skipped.append((CHAMP_RID, r.symbol, str(r.sigd.date()) if pd.notna(r.sigd) else None,
                        str(pd.to_datetime(r.entry_date).date()), float(r.pnl_pct or 0.0), float(conv), "conv_skip"))
cv = HERE / f"_pd_s{SEED}.csv"
bt[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
feat = M.features(bt.symbol.unique().tolist())
pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')

equity, holdings, held_by_date = prun_daily(NavSim2(str(cv), date_lo="2020-01-01"), pm, cm)
print(f"equity days={len(equity)}  holding rows={len(holdings)}", flush=True)

# CAUSAL resting pullback book. Engine: buy signal at bar S -> limit=close[S]*(1-4.5%); FILLS when a later
# LOW <= limit within 40 bars, else EXPIRES. At date D a name is "waiting" if its most-recent signal S
# (<=D, in window) hasn't been touched yet (min low[S..D] > limit) and isn't held. outcome = the PRICE-PATH
# fill/expire (engine trigger); a price-fill may still be dropped by conv-skip/capacity so it need NOT be a
# champion trade — that's why validating outcome against run_trades is wrong.
from bisect import bisect_right
import numpy as _np
PB_PCT, PB_WIN = 0.045, 40
caldates = [pd.to_datetime(e[1]) for e in equity]; bar_of = {d: i for i, d in enumerate(caldates)}
nD = len(caldates)
low_arr, close_arr = {}, {}
for s, g in px.groupby("symbol"):
    la = _np.full(nD, _np.nan); ca = _np.full(nD, _np.nan)
    for r in g.itertuples():
        b = bar_of.get(r.date)
        if b is not None:
            la[b] = r.low; ca[b] = r.close
    low_arr[s] = la; close_arr[s] = ca
sigdf = pd.read_sql("select symbol,date from run_signals where run_id=%s and signal=1", con, params=(rid,))
sigdf["date"] = pd.to_datetime(sigdf["date"])
sig_bars = defaultdict(list)
for r in sigdf.itertuples():
    b = bar_of.get(r.date)
    if b is not None:
        sig_bars[r.symbol].append(b)
for d in sig_bars:
    sig_bars[d].sort()
pending = []
for sym, sbars in sig_bars.items():
    la = low_arr.get(sym); ca = close_arr.get(sym)
    if la is None:
        continue
    for d in range(nD):
        li = bisect_right(sbars, d - PB_WIN); ri = bisect_right(sbars, d)
        if li >= ri:
            continue
        S = sbars[ri - 1]                               # most recent signal in window = current resting limit
        cS = ca[S]
        if _np.isnan(cS):
            continue
        limit = cS * (1.0 - PB_PCT)
        seg_sd = la[S:d + 1]                            # already touched between S and D -> filled, not waiting
        if seg_sd.size and _np.nanmin(seg_sd) <= limit:
            continue
        ds = str(caldates[d].date())
        if sym in held_by_date.get(ds, set()):
            continue
        cD = ca[d]
        if _np.isnan(cD):
            continue
        pctL = limit / cD - 1.0
        hi = min(S + PB_WIN, nD - 1); seg = la[S:hi + 1]  # outcome over the full window from the price path
        touch = _np.where(seg <= limit)[0]
        if touch.size:
            outc, rdate = "fill", str(caldates[S + int(touch[0])].date())
        else:
            outc, rdate = "expire", str(caldates[hi].date())
        pending.append((CHAMP_RID, ds, sym, str(caldates[S].date()), d - S, float(limit), float(cD),
                        float(pctL), outc, rdate))
print(f"pending rows={len(pending)}", flush=True)

cur = con.cursor()
# Schema is owned by alembic migration 0026 (run_equity / run_portfolio_daily). This ops script only
# refreshes the champion's rows (delete + re-insert); it assumes the migration has created the tables.
cur.execute("DELETE FROM run_equity WHERE run_id=%s", (CHAMP_RID,))
cur.execute("DELETE FROM run_portfolio_daily WHERE run_id=%s", (CHAMP_RID,))
cur.execute("DELETE FROM run_skipped WHERE run_id=%s", (CHAMP_RID,))
cur.execute("DELETE FROM run_pending WHERE run_id=%s", (CHAMP_RID,))
execute_values(cur, "INSERT INTO run_equity (run_id,date,nav,cash,exposure,n_positions) VALUES %s", equity)
execute_values(cur, "INSERT INTO run_portfolio_daily (run_id,date,symbol,weight,entry_weight,unreal_pnl,entry_date,days_held,is_new,is_exit,exit_reason,conv) VALUES %s", holdings)
if skipped:
    execute_values(cur, "INSERT INTO run_skipped (run_id,symbol,signal_date,entry_date,pnl_pct,conv,skip_reason) VALUES %s", skipped)
if pending:
    execute_values(cur, "INSERT INTO run_pending (run_id,date,symbol,signal_date,days_waiting,limit_price,ref_price,pct_to_limit,outcome,result_date) VALUES %s", pending)
con.commit()
print("run_skipped rows:", len(skipped), "| run_pending rows:", len(pending))
cur.execute("SELECT count(*) FROM run_equity WHERE run_id=%s", (CHAMP_RID,)); print("run_equity rows:", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM run_portfolio_daily WHERE run_id=%s", (CHAMP_RID,)); print("run_portfolio_daily rows:", cur.fetchone()[0])
# sample: latest day holdings
cur.execute("SELECT date FROM run_equity WHERE run_id=%s ORDER BY date DESC LIMIT 1", (CHAMP_RID,)); ld = cur.fetchone()[0]
cur.execute("SELECT symbol,(weight*100)::numeric(6,1),days_held,is_new,is_exit FROM run_portfolio_daily WHERE run_id=%s AND date=%s AND is_exit=false ORDER BY weight DESC", (CHAMP_RID, ld))
print(f"--- holdings on latest day {ld} (symbol, weight%, days_held, new) ---")
for r in cur.fetchall(): print("  ", r)
con.close()
print("PORTFOLIO_DAILY_DONE")
