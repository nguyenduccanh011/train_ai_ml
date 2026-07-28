"""AUDIT the k16/k25 leaderboard_nav rows: recompute each under a CLEAN causal exposure-neutral harness
(signal-date cs4, fair mean-1 sizing, priority-fill) and compare to the registered value. Flags rows
whose registered number does NOT reproduce (likely inflated by look-ahead/naive sizing from the old scorer).
Modes: prio_eq (equal-weight, priority by score = the valve), size (cs_mix k1.5). K in {16,25}. 3-seed.
Registered: k16prio 33.71/71.6%, k16size 57.42(fixed), k25size 41.24/77.0%, k25preempt 33.83, k16preempt 61.76, k16meta 36.03.
"""
from __future__ import annotations
import sys, statistics
from pathlib import Path
sys.path.insert(0, "F:/PROJECTS/hb2943_work")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from collections import defaultdict
import numpy as np, pandas as pd, duckdb, psycopg2
from nh_nav2 import NavSim2, FEE
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
WORK = Path("F:/PROJECTS/hb2943_work/navboard"); WORK.mkdir(exist_ok=True)
FRONTIER = 3185; SEEDS = [42, 7, 99]
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
YEARS = 6.51


def run_v(sim, K, mode, roundtrip=0.006, advance_fee=0.0008):
    """mode: 'prio_eq' (equal-weight, priority by score) or 'size' (cs_mix k1.5)."""
    k_conv = 0.0 if mode == "prio_eq" else 1.5
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    cv = [t.get("conv", 0.5) for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    sc = [t.get("score", 0.0) for t in sim.trades]; smu = statistics.mean(sc); ssd = statistics.pstdev(sc) or 1.0
    raw = []
    for t in sim.trades:
        z = (t.get("conv", 0.5) - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
        szc = (t.get("score", 0.0) - smu) / ssd
        t["_pri"] = szc if mode == "prio_eq" else szc + (t.get("conv", 0.5) - mu) / sd
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: -t["_pri"])
    scl, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (scl[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pending = defaultdict(float); pt = 0.0; legs = []; exits = defaultdict(list); navs = []
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            cash += proceeds * (1.0 - advance_fee) if advance_fee is not None else 0.0
            legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos; nopen = len(legs)
        for t in entries.get(dt, ()):
            if nopen >= K:
                break
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = scl[s][t["i0"]], scl[s][t["i1"]]; ee = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=ee / c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg); nopen += 1
        pos = sum(lv(l, dt) for l in legs); navs.append((dt, cash + pt + pos))
    ns = pd.DataFrame(navs, columns=["date", "nav"]); nav = ns["nav"]; dd = nav / nav.cummax() - 1
    return float(nav.iloc[-1]), float(dd.min())


_cx = duckdb.connect(MARKET, read_only=True)
_px = _cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                  "ORDER BY symbol,date").fetchdf(); _cx.close()
_px["date"] = pd.to_datetime(_px["date"]); _parts = []
for s, g in _px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    _parts.append(g[["symbol", "date"] + CS4])
PANEL = pd.concat(_parts, ignore_index=True)
for col in CS4:
    PANEL[col + "_r"] = PANEL.groupby("date")[col].rank(pct=True)
PANEL["cs4"] = PANEL[[c + "_r" for c in CS4]].mean(axis=1)

CONF = [("prio_eq", 16), ("size", 16), ("size", 25), ("prio_eq", 25)]
con = psycopg2.connect(**PG)
res = {c: {} for c in CONF}
for sd in SEEDS:
    rid = run_template_experiment(template_id=FRONTIER, seed=sd)["run_id"]
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                     "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    tr["sigd"] = pd.to_datetime(tr["entry_signal_date"]); sg["dd"] = pd.to_datetime(sg["date"])
    tr = tr.merge(sg[["symbol", "dd", "score"]], left_on=["symbol", "sigd"], right_on=["symbol", "dd"], how="left")
    m = tr.merge(PANEL[["symbol", "date", "cs4"]], left_on=["symbol", "sigd"], right_on=["symbol", "date"], how="left")
    m["ek"] = m["symbol"] + "|" + m["entry_date"].astype(str).str[:10]
    csk = dict(zip(m["ek"], m["cs4"].fillna(0.5)))
    slook = {x.symbol + "|" + str(x.entry_date)[:10]: (x.score if pd.notna(x.score) else 0.0) for x in tr.itertuples()}
    csv = WORK / f"_au_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk():
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0); t["conv"] = csk.get(ek, 0.5)
        return sim
    for (mode, K) in CONF:
        res[(mode, K)][sd] = run_v(mk(), K, mode)
    print("seed%d: " % sd + " ".join(f"{m}K{K}={res[(m,K)][sd][0]:.1f}" for m, K in CONF), flush=True)
con.close()

REG = {("prio_eq", 16): (33.71, "k16prio"), ("size", 16): (57.42, "k16size(fixed)"),
       ("size", 25): (41.24, "k25size"), ("prio_eq", 25): (None, "(no k25prio row)")}
print("\n=== AUDIT: clean recompute vs registered leaderboard_nav ===")
print("config        | clean NAV(3-seed) | clean CAGR | registered NAV | verdict")
for (mode, K) in CONF:
    navs = [res[(mode, K)][s][0] for s in SEEDS]; cn = sum(navs) / 3
    cagr = cn ** (1.0 / YEARS) - 1.0
    reg, nm = REG[(mode, K)]
    if reg is None:
        vd = "n/a"
    else:
        ratio = cn / reg
        vd = "OK (reproduces)" if 0.9 <= ratio <= 1.1 else f"INFLATED reg={reg} is {reg/cn:.2f}x clean"
    print(f"{mode:7s} K{K:2d} {nm:16s}| {cn:6.2f}            | {cagr*100:5.1f}%    | {str(reg):>8s}       | {vd}")
print("CLEAN k16size baseline cross-check should ~57.4")
print("AUDIT_DONE")
