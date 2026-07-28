"""PROSECUTION of the new best (cs_mix k=1.5). Two discipline gates before it can be a milestone:
(a) PER-YEAR: does the sizing/cs gain SURVIVE the dead-years 2024/2026, or is it variance-drag
    concentrated in the good years (the failure mode of conviction-sizing-per-trade-real-nav-fails-recent)?
(b) FEE sensitivity: is the edge robust as roundtrip cost rises (0.006 -> 0.010 -> 0.015)?
Compares equal-weight vs cs_mix_k15, 3-seed, per-calendar-year annual returns + full-period NAV per fee.
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
FRONTIER = 3185; SEEDS = [42, 7, 99]; K = 16
BLEND = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def run_v(sim, K, pri_key, size_key, k_conv, roundtrip=0.006, advance_fee=0.0008):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    def zof(key):
        v = [t.get(key, 0.0) for t in sim.trades]; return statistics.mean(v), (statistics.pstdev(v) or 1.0)
    smu, ssd = zof("score"); zk = {"gz": zof("gz"), "cs": zof("cs")}
    sm, ss = zk[size_key]; raw = []
    for t in sim.trades:
        z = (t.get(size_key, 0.0) - sm) / ss
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
        sz = (t.get("score", 0.0) - smu) / ssd
        if pri_key is None:
            t["_pri"] = sz
        else:
            pm, ps = zk[pri_key]; t["_pri"] = sz + (t.get(pri_key, 0.0) - pm) / ps
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
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size and nopen < K:
                s = t["symbol"]; c0, c1 = scl[s][t["i0"]], scl[s][t["i1"]]; ee = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=ee / c1, last_val=size, entry_date=dt, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg); nopen += 1
        pos = sum(lv(l, dt) for l in legs); navs.append((dt, cash + pt + pos))
    ns = pd.DataFrame(navs, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
    return ns


def annual(ns):
    ns = ns.copy(); ns["yr"] = ns["date"].dt.year; out = {}
    for y, g in ns.groupby("yr"):
        out[int(y)] = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0
    return out


_cx = duckdb.connect(MARKET, read_only=True)
_px = _cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                  "ORDER BY symbol,date").fetchdf(); _cx.close()
_px["date"] = pd.to_datetime(_px["date"]); _parts = []
for s, g in _px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    _parts.append(g[["symbol", "date"] + BLEND])
PANEL = pd.concat(_parts, ignore_index=True)
for col in BLEND:
    PANEL[col + "_z"] = (PANEL[col] - PANEL[col].mean()) / (PANEL[col].std() + 1e-9)
    PANEL[col + "_r"] = PANEL.groupby("date")[col].rank(pct=True)
PANEL["gz"] = PANEL[[c + "_z" for c in BLEND]].mean(axis=1)
PANEL["cs"] = PANEL[[c + "_r" for c in BLEND]].mean(axis=1)

con = psycopg2.connect(**PG)
YEARS = list(range(2020, 2027)); FEES = [0.006, 0.010, 0.015]
ann = {"equal": defaultdict(list), "csk15": defaultdict(list)}
feeres = {f: {"equal": [], "csk15": []} for f in FEES}
for sd in SEEDS:
    r = run_template_experiment(template_id=FRONTIER, seed=sd); rid = r.get("run_id")
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                     "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    tr["sigd"] = pd.to_datetime(tr["entry_signal_date"]); sg["dd"] = pd.to_datetime(sg["date"])
    tr = tr.merge(sg[["symbol", "dd", "score"]], left_on=["symbol", "sigd"], right_on=["symbol", "dd"], how="left")
    m = tr.merge(PANEL[["symbol", "date", "gz", "cs"]], left_on=["symbol", "sigd"], right_on=["symbol", "date"], how="left")
    m["ek"] = m["symbol"] + "|" + m["entry_date"].astype(str).str[:10]
    gzk = dict(zip(m["ek"], m["gz"].fillna(0.0))); csk = dict(zip(m["ek"], m["cs"].fillna(0.5)))
    slook = {x.symbol + "|" + str(x.entry_date)[:10]: (x.score if pd.notna(x.score) else 0.0) for x in tr.itertuples()}
    csv = WORK / f"_pr_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk():
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0); t["gz"] = gzk.get(ek, 0.0); t["cs"] = csk.get(ek, 0.5)
        return sim
    ae = annual(run_v(mk(), K, None, "cs", 0.0)); ac = annual(run_v(mk(), K, "cs", "cs", 1.5))
    for y in YEARS:
        if y in ae: ann["equal"][y].append(ae[y])
        if y in ac: ann["csk15"][y].append(ac[y])
    for f in FEES:
        feeres[f]["equal"].append(run_v(mk(), K, None, "cs", 0.0, roundtrip=f)["nav"].iloc[-1])
        feeres[f]["csk15"].append(run_v(mk(), K, "cs", "cs", 1.5, roundtrip=f)["nav"].iloc[-1])
    print(f"seed{sd} done", flush=True)
con.close()

print("\n=== (a) PER-YEAR annual return (3-seed mean) — equal vs cs_mix k1.5 ===")
print("year | equal  | csk15  | Δ(cs-eq)")
for y in YEARS:
    if ann["equal"][y] and ann["csk15"][y]:
        e = statistics.mean(ann["equal"][y]); c = statistics.mean(ann["csk15"][y])
        flag = "  <-- dead-year" if y in (2024, 2026) else ""
        print(f"{y} | {e*100:+6.1f}% | {c*100:+6.1f}% | {(c-e)*100:+6.1f}pp{flag}")
print("\n=== (b) FEE sensitivity (full-period NAV, 3-seed mean) ===")
print("roundtrip | equal  | csk15  | ratio")
for f in FEES:
    e = statistics.mean(feeres[f]["equal"]); c = statistics.mean(feeres[f]["csk15"])
    print(f"  {f:.3f}   | {e:6.2f} | {c:6.2f} | {c/e:.2f}x")
print("CSPROSECUTE_DONE")
