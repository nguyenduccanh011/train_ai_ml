"""#3 Optimize + DECOMPOSE the new best (cs mix/size). Sweep k_conv and split the priority vs sizing
channels (gz vs cs in each) to locate WHERE the cross-sectional gain lives and find the operating point.
Offline (3 base runs + variants), K16, 3-seed, walk-forward-clean.
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


def zof(sim, key):
    vals = [t.get(key, 0.0) for t in sim.trades]
    mu = statistics.mean(vals); sd = statistics.pstdev(vals) or 1.0
    return mu, sd


def run_v(sim, K, pri_key, size_key, k_conv, roundtrip=0.006, advance_fee=0.0008):
    """pri_key in {None,'gz','cs'}: None=score-only priority, else mix score_z + that conv z.
       size_key in {'gz','cs'}: sizing tilt driver (only when k_conv>0)."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    smu, ssd = zof(sim, "score")
    gmu, gsd = zof(sim, "gz"); cmu, csd = zof(sim, "cs")
    zk = {"gz": (gmu, gsd), "cs": (cmu, csd)}
    sm, ss = zk[size_key]
    raw = []
    for t in sim.trades:
        z = (t.get(size_key, 0.0) - sm) / ss
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    for t in sim.trades:
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
    ns = pd.DataFrame(navs, columns=["date", "nav"]); nav = ns["nav"]; dd = nav / nav.cummax() - 1
    return dict(final=float(nav.iloc[-1]), maxdd=float(dd.min()))


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

# (name, pri_key, size_key, k_conv)
VARS = [("equal", None, "cs", 0.0), ("gz_mix_k1", "gz", "gz", 1.0),
        ("cs_mix_k05", "cs", "cs", 0.5), ("cs_mix_k1", "cs", "cs", 1.0),
        ("cs_mix_k15", "cs", "cs", 1.5), ("cs_mix_k2", "cs", "cs", 2.0),
        ("csPri_gzSize", "cs", "gz", 1.0), ("gzPri_csSize", "gz", "cs", 1.0)]
con = psycopg2.connect(**PG)
res = {nm: {} for nm, *_ in VARS}
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
    csv = WORK / f"_co_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk():
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0); t["gz"] = gzk.get(ek, 0.0); t["cs"] = csk.get(ek, 0.5)
        return sim
    for nm, pk, sk, kc in VARS:
        res[nm][sd] = run_v(mk(), K, pk, sk, kc)["final"]
    print("seed%d: " % sd + " ".join(f"{nm}={res[nm][sd]:.1f}" for nm, *_ in VARS), flush=True)
con.close()

print("\n=== cs-config optimize + channel decompose (K16), sign 3/3 vs equal ===")
base = res["equal"]
for nm, *_ in VARS:
    navs = [res[nm][s] for s in SEEDS]; d = [res[nm][s] - base[s] for s in SEEDS]
    signs = "".join("+" if x > 0 else "-" for x in d)
    print(f"{nm:13s}: mean={sum(navs)/3:.3f} navs={[f'{n:.2f}' for n in navs]} | Δvs-eq={[f'{x:+.2f}' for x in d]} {signs}")
print("CSOPT_DONE")
