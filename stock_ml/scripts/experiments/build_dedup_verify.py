"""VERIFY an accidental discovery from build_split_sleeve: its mom-only 16/0 (69.3, DD-13.1) beat the
cs_mix k1.5 milestone (57.4, DD-15) — the only structural difference was a NO-SAME-NAME-CONCURRENT
constraint (skip an entry whose symbol is already held; the slot goes to the next new name instead).
Clean A/B on the momentum book (cs_mix k1.5, K16): nodedup (can hold same name in >1 concurrent leg)
vs dedup (unique names only). 3-seed. If dedup wins 3/3 with lower DD, it's a real diversification lever
stacking on the milestone; if it's a fluke/bug, they match.
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
FRONTIER = 3185; SEEDS = [42, 7, 99]; K = 16; KCONV = 1.5
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def run_v(sim, K, k_conv, dedup, roundtrip=0.006, advance_fee=0.0008):
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
        t["_pri"] = (t.get("score", 0.0) - smu) / ssd + (t.get("conv", 0.5) - mu) / sd
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
    held = defaultdict(int)
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            cash += proceeds * (1.0 - advance_fee) if advance_fee is not None else 0.0
            legs.remove(leg); held[leg["symbol"]] -= 1
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos; nopen = len(legs)
        for t in entries.get(dt, ()):
            if nopen >= K:
                break
            if dedup and held[t["symbol"]] > 0:
                continue
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = scl[s][t["i0"]], scl[s][t["i1"]]; ee = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=ee / c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg); nopen += 1; held[s] += 1
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
CSLK = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in PANEL.itertuples()}

con = psycopg2.connect(**PG)
res = {"nodedup": {}, "dedup": {}}
for sd in SEEDS:
    rid = run_template_experiment(template_id=FRONTIER, seed=sd)["run_id"]
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                     "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    tr["sigd"] = pd.to_datetime(tr["entry_signal_date"]); sg["dd"] = pd.to_datetime(sg["date"])
    tr = tr.merge(sg[["symbol", "dd", "score"]], left_on=["symbol", "sigd"], right_on=["symbol", "dd"], how="left")
    slook = {(x.symbol, str(pd.to_datetime(x.entry_date).date())): (x.score if pd.notna(x.score) else 0.0) for x in tr.itertuples()}
    csv = WORK / f"_dd_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk():
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            key = (t["symbol"], str(pd.to_datetime(t["entry_date"]).date()))
            t["score"] = slook.get(key, 0.0); t["conv"] = CSLK.get(key, 0.5)
        return sim
    res["nodedup"][sd] = run_v(mk(), K, KCONV, False)
    res["dedup"][sd] = run_v(mk(), K, KCONV, True)
    print(f"seed{sd}: nodedup={res['nodedup'][sd][0]:.2f}(dd{res['nodedup'][sd][1]*100:.1f}) "
          f"dedup={res['dedup'][sd][0]:.2f}(dd{res['dedup'][sd][1]*100:.1f})", flush=True)
con.close()

print("\n=== dedup (no same-name concurrent) vs nodedup, cs_mix k1.5 K16, sign 3/3 ===")
base = res["nodedup"]
navs = [res["dedup"][s][0] for s in SEEDS]; d = [res["dedup"][s][0] - base[s][0] for s in SEEDS]
dd = statistics.mean([res["dedup"][s][1] for s in SEEDS]) * 100
signs = "".join("+" if x > 0 else "-" for x in d)
print(f"nodedup: mean={sum(base[s][0] for s in SEEDS)/3:.3f} dd={statistics.mean([base[s][1] for s in SEEDS])*100:.1f}%")
print(f"dedup  : mean={sum(navs)/3:.3f} dd={dd:.1f}% | Δ={[f'{x:+.2f}' for x in d]} {signs}")
print("DEDUP_VERIFY_DONE")
