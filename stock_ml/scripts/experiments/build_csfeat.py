"""Extend the cross-sectional magnitude signal. cs_blend is only 4 features (dist20low, dist_ma20,
rsi14, ret20 = extension+momentum). Test adding orthogonal magnitude dimensions via cs-RANK (scale-free):
breakout-proximity (dist63high), trend-slope (ma20_slope), volume-confirm (volz). Each cs-rank blend
scored at the prosecution-clean config (mix priority + sizing, k=1.5, K16). Sign-consistency 3/3 gate.
WIN = an expanded blend beats cs4 3/3 (real orthogonal magnitude info); else cs4 stays (learned-sizing
overfit lesson: more features often dilute).
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
KCONV = 1.5
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
# candidate cs-rank blends (each a list of feature-rank columns to average)
BLENDS = {
    "cs4": CS4,
    "cs4+hi63": CS4 + ["dist63high"],
    "cs4+slope": CS4 + ["ma20_slope"],
    "cs4+volz": CS4 + ["volz"],
    "cs7_all": CS4 + ["dist63high", "ma20_slope", "volz"],
}
ALLF = sorted(set(sum(BLENDS.values(), [])))


def run_v(sim, K, convkey, k_conv=KCONV, roundtrip=0.006, advance_fee=0.0008):
    """mix priority (score_z + conv_z) + sizing by conv (fair mean-1). conv = sim trade[convkey]."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    cv = [t.get(convkey, 0.5) for t in sim.trades]
    mu = statistics.mean(cv)
    sd = statistics.pstdev(cv) or 1.0
    sc = [t.get("score", 0.0) for t in sim.trades]
    smu = statistics.mean(sc)
    ssd = statistics.pstdev(sc) or 1.0
    raw = []
    for t in sim.trades:
        z = (t.get(convkey, 0.5) - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8)
        raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
        t["_pri"] = (t.get("score", 0.0) - smu) / ssd + (t.get(convkey, 0.5) - mu) / sd
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
    return float(nav.iloc[-1])


_cx = duckdb.connect(MARKET, read_only=True)
_px = _cx.execute(
    "SELECT symbol,date,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
    "ORDER BY symbol,date"
).fetchdf()
_cx.close()
_px["date"] = pd.to_datetime(_px["date"])
_parts = []
for s, g in _px.groupby("symbol"):
    g = g.sort_values("date").copy()
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    ma20 = c.rolling(20).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1
    g["dist_ma20"] = c / ma20 - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret20"] = c / c.shift(20) - 1
    g["dist63high"] = c / h.rolling(63).max() - 1
    g["ma20_slope"] = ma20 / ma20.shift(10) - 1
    g["volz"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    _parts.append(g[["symbol", "date"] + ALLF])
PANEL = pd.concat(_parts, ignore_index=True)
for col in ALLF:
    PANEL[col + "_r"] = PANEL.groupby("date")[col].rank(pct=True)
for nm, cols in BLENDS.items():
    PANEL[nm] = PANEL[[c + "_r" for c in cols]].mean(axis=1)

con = psycopg2.connect(**PG)
res = {nm: {} for nm in BLENDS}
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
        PANEL[["symbol", "date"] + list(BLENDS)],
        left_on=["symbol", "sigd"],
        right_on=["symbol", "date"],
        how="left",
    )
    m["ek"] = m["symbol"] + "|" + m["entry_date"].astype(str).str[:10]
    lut = {nm: dict(zip(m["ek"], m[nm].fillna(0.5))) for nm in BLENDS}
    slook = {
        x.symbol + "|" + str(x.entry_date)[:10]: (x.score if pd.notna(x.score) else 0.0)
        for x in tr.itertuples()
    }
    csv = WORK / f"_cf_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk(nm):
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0)
            t[nm] = lut[nm].get(ek, 0.5)
        return sim

    for nm in BLENDS:
        res[nm][sd] = run_v(mk(nm), K, nm)
    print("seed%d: " % sd + " ".join(f"{nm}={res[nm][sd]:.2f}" for nm in BLENDS), flush=True)
con.close()

print("\n=== cs-rank blend feature-expansion vs cs4 (mix/size k1.5 K16), sign 3/3 ===")
base = res["cs4"]
for nm in BLENDS:
    navs = [res[nm][s] for s in SEEDS]
    d = [res[nm][s] - base[s] for s in SEEDS]
    signs = "".join("+" if x > 0 else "-" for x in d)
    tag = " (baseline)" if nm == "cs4" else ""
    print(
        f"{nm:11s}: mean={sum(navs) / 3:.3f} navs={[f'{n:.2f}' for n in navs]} | Δvs-cs4={[f'{x:+.2f}' for x in d]} {signs}{tag}"
    )
print("CSFEAT_DONE")
