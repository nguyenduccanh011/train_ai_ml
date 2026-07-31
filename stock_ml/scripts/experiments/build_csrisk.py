"""Orthogonal 2nd sizing dimension: RISK. The cs4 tilt sizes by return-MAGNITUDE (extension). Stack a
fair (mean-1) RISK tilt = inverse cross-sectional volatility rank (down-size high-rvol20 names) on top,
multiplicatively, re-centered to mean-1 (exposure-neutral). Does a risk overlay lift NAV via lower DD /
less compounding drag, or is risk already captured by the magnitude tilt? Priority stays mix(score+cs4).
Variants: cs4-only (base 57.4), cs4×risk (kr=0.5/1.0), risk-only control. 3-seed, sign 3/3 vs cs4.
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
WORK = Path("F:/PROJECTS/hb2943_work/navboard")
WORK.mkdir(exist_ok=True)
FRONTIER = 3185
SEEDS = [42, 7, 99]
K = 16
KMAG = 1.5
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def run_v(sim, K, kmag, kr, risk_only=False, roundtrip=0.006, advance_fee=0.0008):
    """sizing weight = magnitude_tilt(cs4, kmag) * risk_tilt(inverse-rvol rank, kr), re-centered mean-1.
    priority = score_z + cs4_z (mix). risk_only=True -> drop magnitude tilt (kmag ignored)."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    cv = [t.get("cs4", 0.5) for t in sim.trades]
    mu = statistics.mean(cv)
    sd = statistics.pstdev(cv) or 1.0
    rv = [t.get("rrisk", 0.5) for t in sim.trades]
    rmu = statistics.mean(rv)
    rsd = statistics.pstdev(rv) or 1.0
    sc = [t.get("score", 0.0) for t in sim.trades]
    smu = statistics.mean(sc)
    ssd = statistics.pstdev(sc) or 1.0
    raw = []
    for t in sim.trades:
        mz = (t.get("cs4", 0.5) - mu) / sd
        rz = (
            t.get("rrisk", 0.5) - rmu
        ) / rsd  # rrisk = inverse-vol rank (high => low vol => size up)
        wm = 1.0 if risk_only else min(max(1.0 + kmag * mz, 0.4), 1.8)
        wr = min(max(1.0 + kr * rz, 0.4), 1.8)
        t["_w"] = wm * wr
        raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
        t["_pri"] = (t.get("score", 0.0) - smu) / ssd + (t.get("cs4", 0.5) - mu) / sd
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: -t["_pri"])
    scl, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]
        j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]
        j = min(max(j, i0), i1)
        r = (
            leg["ratio1"]
            if i1 == i0
            else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        )
        v = leg["invested"] * (scl[s][j] * r) / leg["p0"]
        leg["last_val"] = v
        return v

    cash = 1.0
    pending = defaultdict(float)
    pt = 0.0
    legs = []
    exits = defaultdict(list)
    navs = []
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0)
        pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            cash += proceeds * (1.0 - advance_fee) if advance_fee is not None else 0.0
            legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs)
        nav_now = cash + pt + pos
        nopen = len(legs)
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size and nopen < K:
                s = t["symbol"]
                c0, c1 = scl[s][t["i0"]], scl[s][t["i1"]]
                ee = t["p0"] * (1.0 + t["net"])
                leg = dict(
                    symbol=s,
                    i0=t["i0"],
                    i1=t["i1"],
                    invested=size,
                    net=t["net"],
                    p0=t["p0"],
                    ratio0=t["p0"] / c0,
                    ratio1=ee / c1,
                    last_val=size,
                    entry_date=dt,
                    exit_date=t["exit_date"],
                )
                cash -= size
                legs.append(leg)
                exits[t["exit_date"]].append(leg)
                nopen += 1
        pos = sum(lv(l, dt) for l in legs)
        navs.append((dt, cash + pt + pos))
    ns = pd.DataFrame(navs, columns=["date", "nav"])
    nav = ns["nav"]
    dd = nav / nav.cummax() - 1
    return float(nav.iloc[-1]), float(dd.min())


_cx = duckdb.connect(MARKET, read_only=True)
_px = _cx.execute(
    "SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
    "ORDER BY symbol,date"
).fetchdf()
_cx.close()
_px["date"] = pd.to_datetime(_px["date"])
_parts = []
for s, g in _px.groupby("symbol"):
    g = g.sort_values("date").copy()
    c, l = g["close"], g["low"]
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1
    g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret20"] = c / c.shift(20) - 1
    g["rvol20"] = c.pct_change().rolling(20).std()
    _parts.append(g[["symbol", "date"] + CS4 + ["rvol20"]])
PANEL = pd.concat(_parts, ignore_index=True)
for col in CS4:
    PANEL[col + "_r"] = PANEL.groupby("date")[col].rank(pct=True)
PANEL["cs4"] = PANEL[[c + "_r" for c in CS4]].mean(axis=1)
PANEL["rrisk"] = 1.0 - PANEL.groupby("date")["rvol20"].rank(
    pct=True
)  # low vol => high rank => size up

con = psycopg2.connect(**PG)
VARS = [
    ("cs4_only", 1.5, 0.0, False),
    ("cs4xrisk_k05", 1.5, 0.5, False),
    ("cs4xrisk_k10", 1.5, 1.0, False),
    ("risk_only_k15", 0.0, 1.5, True),
]
res = {nm: {} for nm, *_ in VARS}
for sd in SEEDS:
    r = run_template_experiment(template_id=FRONTIER, seed=sd)
    rid = r.get("run_id")
    tr = pd.read_sql(
        "SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
        "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
        "AND exit_price IS NOT NULL",
        con,
        params=(rid,),
    )
    sg = pd.read_sql(
        "SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,)
    )
    tr["sigd"] = pd.to_datetime(tr["entry_signal_date"])
    sg["dd"] = pd.to_datetime(sg["date"])
    tr = tr.merge(
        sg[["symbol", "dd", "score"]],
        left_on=["symbol", "sigd"],
        right_on=["symbol", "dd"],
        how="left",
    )
    m = tr.merge(
        PANEL[["symbol", "date", "cs4", "rrisk"]],
        left_on=["symbol", "sigd"],
        right_on=["symbol", "date"],
        how="left",
    )
    m["ek"] = m["symbol"] + "|" + m["entry_date"].astype(str).str[:10]
    c4 = dict(zip(m["ek"], m["cs4"].fillna(0.5)))
    rr = dict(zip(m["ek"], m["rrisk"].fillna(0.5)))
    slook = {
        x.symbol + "|" + str(x.entry_date)[:10]: (x.score if pd.notna(x.score) else 0.0)
        for x in tr.itertuples()
    }
    csv = WORK / f"_rk_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk():
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0)
            t["cs4"] = c4.get(ek, 0.5)
            t["rrisk"] = rr.get(ek, 0.5)
        return sim

    for nm, km, kr, ro in VARS:
        nav, dd = run_v(mk(), K, km, kr, ro)
        res[nm][sd] = (nav, dd)
    print(
        "seed%d: " % sd
        + " ".join(f"{nm}={res[nm][sd][0]:.2f}(dd{res[nm][sd][1] * 100:.1f})" for nm, *_ in VARS),
        flush=True,
    )
con.close()

print("\n=== risk-tilt overlay on cs4 magnitude sizing (K16), sign 3/3 vs cs4_only ===")
base = res["cs4_only"]
for nm, *_ in VARS:
    navs = [res[nm][s][0] for s in SEEDS]
    d = [res[nm][s][0] - base[s][0] for s in SEEDS]
    dd = statistics.mean([res[nm][s][1] for s in SEEDS]) * 100
    signs = "".join("+" if x > 0 else "-" for x in d)
    print(
        f"{nm:14s}: mean={sum(navs) / 3:.3f} dd={dd:.1f}% | Δvs-cs4={[f'{x:+.2f}' for x in d]} {signs}"
    )
print("CSRISK_DONE")
