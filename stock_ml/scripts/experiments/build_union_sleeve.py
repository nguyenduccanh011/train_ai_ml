"""Decorrelated SLEEVE under the valve+sizing (new condition). Merge momentum frontier (t3185) trades
with a mean-rev/reversal sleeve (x2_ovrev t3187) into ONE book; the valve priority-fills top-K by
[within-source z(score) + z(cs4)] and sizes by cs4 (cross-sectional extension magnitude — source-agnostic,
computed from price so it works for BOTH sources). Dedupe same symbol+entry_date (keep higher cs4). In
dead-years where momentum is thin, decorrelated sleeve candidates can fill slots. K=16, k_conv=1.5, 3-seed.
WIN = union book NAV beats momentum-only (57.4) 3/3 -> the sleeve adds monetizable decorrelated trades.
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
MOM = 3185
REV = 3187
SEEDS = [42, 7, 99]
K = 16
KCONV = 1.5
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def load_trades(con, rid):
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
    return tr


def run_book(csv, convmap, primap, K, k_conv, roundtrip=0.006, advance_fee=0.0008):
    sim = NavSim2(str(csv), date_lo="2020-01-01")
    for t in sim.trades:
        ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
        t["conv"] = convmap.get(ek, 0.5)
        t["pri_score"] = primap.get(ek, 0.0)
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    cv = [t["conv"] for t in sim.trades]
    mu = statistics.mean(cv)
    sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8)
        raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
        t["_pri"] = (
            t["pri_score"] + (t["conv"] - mu) / sd
        )  # within-source score-z (precomputed) + cs4 z
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
    held = set()
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0)
        pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            cash += proceeds * (1.0 - advance_fee) if advance_fee is not None else 0.0
            legs.remove(leg)
            held.discard(leg["symbol"])
        pos = sum(lv(l, dt) for l in legs)
        nav_now = cash + pt + pos
        nopen = len(legs)
        for t in entries.get(dt, ()):
            if t["symbol"] in held:  # no double position in same name across sources
                continue
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
                held.add(s)
        pos = sum(lv(l, dt) for l in legs)
        navs.append((dt, cash + pt + pos))
    ns = pd.DataFrame(navs, columns=["date", "nav"])
    nav = ns["nav"]
    dd = nav / nav.cummax() - 1
    return float(nav.iloc[-1]), float(dd.min())


# cs4 panel
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
    _parts.append(g[["symbol", "date"] + CS4])
PANEL = pd.concat(_parts, ignore_index=True)
for col in CS4:
    PANEL[col + "_r"] = PANEL.groupby("date")[col].rank(pct=True)
PANEL["cs4"] = PANEL[[c + "_r" for c in CS4]].mean(axis=1)
CSLK = {(r.symbol, r.date): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in PANEL.itertuples()}

con = psycopg2.connect(**PG)
res = {"mom_only": {}, "union": {}}
for sd in SEEDS:
    rm = run_template_experiment(template_id=MOM, seed=sd)["run_id"]
    rr = run_template_experiment(template_id=REV, seed=sd)["run_id"]
    tm = load_trades(con, rm)
    tm["src"] = "m"
    trv = load_trades(con, rr)
    trv["src"] = "r"
    # within-source score-z for priority
    for df in (tm, trv):
        mu = df["score"].mean()
        sd_ = df["score"].std() or 1.0
        df["pri_score"] = (df["score"].fillna(mu) - mu) / sd_
        df["cs4"] = [CSLK.get((s, d), 0.5) for s, d in zip(df["symbol"], df["sigd"])]
        df["ek"] = df["symbol"] + "|" + df["entry_date"].astype(str).str[:10]

    def write_and_maps(df):
        df2 = df.drop_duplicates(subset=["ek"], keep="first")
        cm = dict(zip(df2["ek"], df2["cs4"]))
        pm = dict(zip(df2["ek"], df2["pri_score"]))
        return df2, cm, pm

    # momentum-only book
    dm, cmm, pmm = write_and_maps(tm)
    csvm = WORK / f"_um_{sd}.csv"
    dm[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csvm, index=False)
    res["mom_only"][sd] = run_book(csvm, cmm, pmm, K, KCONV)
    # union book: concat, dedupe same ek keeping higher cs4
    U = pd.concat([tm, trv], ignore_index=True).sort_values("cs4", ascending=False)
    du, cmu, pmu = write_and_maps(U)
    csvu = WORK / f"_uu_{sd}.csv"
    du[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csvu, index=False)
    res["union"][sd] = run_book(csvu, cmu, pmu, K, KCONV)
    print(
        f"seed{sd}: mom_only={res['mom_only'][sd][0]:.2f}(dd{res['mom_only'][sd][1] * 100:.1f}) "
        f"union={res['union'][sd][0]:.2f}(dd{res['union'][sd][1] * 100:.1f}) "
        f"[mom_tr={len(dm)} union_tr={len(du)}]",
        flush=True,
    )
con.close()

print("\n=== union sleeve (mom+rev) under valve+cs-sizing vs mom-only, sign 3/3 ===")
base = res["mom_only"]
navs = [res["union"][s][0] for s in SEEDS]
d = [res["union"][s][0] - base[s][0] for s in SEEDS]
dd = statistics.mean([res["union"][s][1] for s in SEEDS]) * 100
signs = "".join("+" if x > 0 else "-" for x in d)
print(f"mom_only: mean={sum(base[s][0] for s in SEEDS) / 3:.3f}")
print(f"union   : mean={sum(navs) / 3:.3f} dd={dd:.1f}% | Δ={[f'{x:+.2f}' for x in d]} {signs}")
print("UNION_SLEEVE_DONE")
