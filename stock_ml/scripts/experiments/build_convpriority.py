"""Conviction-aware PRIORITY-FILL (new orthogonal lever, offline). The valve sorts oversubscribed-day
entries by SCORE (selection quality). But the blend predicts per-trade MAGNITUDE — when slots bind,
taking the higher-MAGNITUDE names first (not just higher-score) may capture more. Distinct from sizing
(which weights the taken names); this changes WHICH names are taken when K binds. Test priority keys at
EQUAL-weight (isolate the priority effect), then stack sizing. K16, 3-seed, walk-forward-clean (blend at
signal date). WIN = beats score-priority 3/3.
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
BLEND = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def run_v(sim, K, pri, k_conv, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    """pri in {'score','blend','mix'} = oversubscribed-day fill order key; k_conv = fair sizing tilt."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    # sizing weight from blend conviction (fair mean-1)
    convs = [t.get("conv", 0.0) for t in sim.trades]
    mu = statistics.mean(convs)
    sd = statistics.pstdev(convs) or 1.0
    raw = []
    for t in sim.trades:
        z = (t.get("conv", 0.0) - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8)
        raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    # priority key: z-scores of score and conv across the population
    sc = [t.get("score", 0.0) for t in sim.trades]
    smu = statistics.mean(sc)
    ssd = statistics.pstdev(sc) or 1.0
    for t in sim.trades:
        sz = (t.get("score", 0.0) - smu) / ssd
        cz = (t.get("conv", 0.0) - mu) / sd
        t["_pri"] = {"score": sz, "blend": cz, "mix": sz + cz}[pri]
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
    return dict(final=float(nav.iloc[-1]), maxdd=float(dd.min()))


# panel (blend at signal date)
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
    _parts.append(g[["symbol", "date"] + BLEND])
PANEL = pd.concat(_parts, ignore_index=True)
for col in BLEND:
    PANEL[col + "_z"] = (PANEL[col] - PANEL[col].mean()) / (PANEL[col].std() + 1e-9)
PANEL["blend"] = PANEL[[c + "_z" for c in BLEND]].mean(axis=1)

con = psycopg2.connect(**PG)
VARS = [
    ("score/eq", "score", 0.0),
    ("blend/eq", "blend", 0.0),
    ("mix/eq", "mix", 0.0),
    ("mix/size", "mix", 1.0),
    ("score/size", "score", 1.0),
]
res = {nm: {} for nm, _, _ in VARS}
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
        PANEL[["symbol", "date", "blend"]],
        left_on=["symbol", "sigd"],
        right_on=["symbol", "date"],
        how="left",
    )
    m["ek"] = m["symbol"] + "|" + m["entry_date"].astype(str).str[:10]
    bld = dict(zip(m["ek"], m["blend"].fillna(0.0)))
    slook = {
        x.symbol + "|" + str(x.entry_date)[:10]: (x.score if pd.notna(x.score) else 0.0)
        for x in tr.itertuples()
    }
    csv = WORK / f"_cp_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk():
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0)
            t["conv"] = bld.get(ek, 0.0)
        return sim

    for nm, pri, kc in VARS:
        res[nm][sd] = run_v(mk(), K, pri, kc)["final"]
    print("seed%d: " % sd + " ".join(f"{nm}={res[nm][sd]:.2f}" for nm, _, _ in VARS), flush=True)
con.close()

print(
    "\n=== conviction-priority (isolate at eq, then stack size) vs score/eq baseline, sign 3/3 ==="
)
base = res["score/eq"]
for nm, _, _ in VARS:
    navs = [res[nm][s] for s in SEEDS]
    d = [res[nm][s] - base[s] for s in SEEDS]
    signs = "".join("+" if x > 0 else "-" for x in d)
    print(
        f"{nm:11s}: mean={sum(navs) / 3:.3f} navs={[f'{n:.2f}' for n in navs]} | Δvs-score/eq={[f'{x:+.2f}' for x in d]} {signs}"
    )
print("CONVPRI_DONE")
