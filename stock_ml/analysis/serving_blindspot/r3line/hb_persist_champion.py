# -*- coding: utf-8 -*-
"""SCOPE-A: populate the combo champion's DETAIL PAGE. Run the champion combo (preempt R2m01 + cs-sizing
k1.5 + conv-skip 0.40, seed 42), capture the ACTUAL filled legs (natural exits + preempt evictions), and
persist them as run_trades + the base t3185 signals as run_signals under the champion run_id so the
dashboard detail page shows real content. No schema change (9 core trade fields). Preempt-evicted legs get
exit_reason='preempt' (exit@eviction-date close, mark-to-market pnl)."""
from __future__ import annotations
import os, sys, asyncio, statistics
from collections import defaultdict
from pathlib import Path
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
from db.engine import async_engine
from db.repositories.trade_repo import RunTradeRepository
from db.repositories.signal_repo import RunSignalRepository
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
CHAMP_RID = "template/x2_struct_to_k16preempt_cssize-69338138"
SEED = 42; K = 16; KCONV = 1.5; MARGIN = 0.01; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun_trades(sim, pm, cm, base, k_conv=KCONV, margin=MARGIN, skip=SKIP, advance_fee=0.0008, roundtrip=0.006):
    """Returns list of filled-leg trade dicts (detail-page schema). base = (sym,edate)->base trade row."""
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

    def mk(t, size):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"],
                    prio=t["prio"], entry_dt=t["entry_date"])

    out = []

    def emit(leg, exit_dt, evicted):
        s = leg["symbol"]; b = base.get((s, str(pd.to_datetime(leg["entry_dt"]).date())), {})
        ed = pd.to_datetime(leg["entry_dt"]); xd = pd.to_datetime(exit_dt)
        if evicted:
            j = si[s].get(exit_dt); xp = float(sc[s][j]) if j is not None else float(b.get("exit_price") or 0)
            pnl = xp / leg["p0"] - 1.0 if leg["p0"] else 0.0; reason = "preempt"
        else:
            xp = float(b.get("exit_price") or leg["p0"] * (1.0 + leg["net"])); pnl = float(leg["net"]); reason = b.get("exit_reason") or "signal"
        out.append(dict(symbol=s, entry_date=str(ed.date()), entry_price=float(leg["p0"]),
                        exit_date=str(xd.date()), exit_price=xp, holding_days=float((xd - ed).days),
                        pnl_pct=float(pnl), exit_reason=reason, entry_signal_date=(str(b.get("sigd"))[:10] if b.get("sigd") is not None else None)))

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list)
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg); emit(leg, dt, False)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    emit(c, dt, True)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs)
    for leg in legs:  # residual open at end
        emit(leg, cal[-1], False)
    return out


# cs4 + meta setup (seed 42)
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
bt = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,exit_reason,entry_signal_date "
                 "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
bt["sigd"] = pd.to_datetime(bt["entry_signal_date"])
base = {}
for r in bt.itertuples():
    base[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = dict(
        exit_price=r.exit_price, exit_reason=r.exit_reason, sigd=r.sigd)
cm = {}
for r in bt.itertuples():
    cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
cv = HERE / f"_persist_s{SEED}.csv"
bt[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
feat = M.features(bt.symbol.unique().tolist())
pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
sig = pd.read_sql("select symbol,date,signal,score,exit_score from run_signals where run_id=%s", con, params=(rid,))
con.close()

trades = prun_trades(NavSim2(str(cv), date_lo="2020-01-01"), pm, cm, base)
print(f"combo filled legs: {len(trades)}  (natural={sum(1 for t in trades if t['exit_reason']!='preempt')} "
      f"preempt={sum(1 for t in trades if t['exit_reason']=='preempt')})", flush=True)

# build signal rows for champion run_id (base signals the champion acts on)
sig_rows = []
for r in sig.itertuples():
    sig_rows.append(dict(symbol=r.symbol, date=str(pd.to_datetime(r.date).date()),
                         signal=int(r.signal), score=(float(r.score) if pd.notna(r.score) else None),
                         exit_score=(float(r.exit_score) if pd.notna(r.exit_score) else None)))


async def persist():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        tr = RunTradeRepository(s); await tr.delete_by_run_id(CHAMP_RID)
        n1 = await tr.bulk_insert(CHAMP_RID, trades)
        sr = RunSignalRepository(s); await sr.delete_by_run_id(CHAMP_RID)
        n2 = await sr.bulk_insert(CHAMP_RID, sig_rows)
        await s.commit(); print(f"persisted trades={n1} signals={n2} under {CHAMP_RID}", flush=True)
    await async_engine.dispose()

asyncio.run(persist())
print("PERSIST_CHAMPION_DONE")
