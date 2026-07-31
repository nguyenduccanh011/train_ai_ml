"""New orthogonal lever: DEPLOY IDLE CASH (cash-drag ~35%). The book averages ~65% exposure because many
days have <16 signals -> cash sits idle. Monetize it by sizing each position off a smaller denominator
K_size (<16) while KEEPING slot-cap 16 (diversification). Bigger per-position size -> higher deployment,
CAPPED by available cash (never leverage: fair total<=1). On thin days each name gets more; on full days
cash runs out at ~K_size names. Sweep K_size, report NAV + realized exposure + DD. cs_mix k1.5 config,
signal-date cs4 (causal-clean), 3-seed. WIN = higher NAV 3/3 (check DD/exposure trade-off).
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
KSLOT = 16
KCONV = 1.5
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
KSIZES = [16, 14, 12, 10]  # sizing denominator; <16 => deploy idle cash (slot cap stays 16)


def run_v(sim, kslot, ksize, k_conv, roundtrip=0.006, advance_fee=0.0008):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    cv = [t.get("conv", 0.5) for t in sim.trades]
    mu = statistics.mean(cv)
    sd = statistics.pstdev(cv) or 1.0
    sc = [t.get("score", 0.0) for t in sim.trades]
    smu = statistics.mean(sc)
    ssd = statistics.pstdev(sc) or 1.0
    raw = []
    for t in sim.trades:
        z = (t.get("conv", 0.5) - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8)
        raw.append(t["_w"])
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
    exps = []
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
            if nopen >= kslot:
                break
            size = (nav_now / ksize) * t[
                "w"
            ]  # <-- sizing off ksize (<=kslot); cash caps => no leverage
            if cash + 1e-12 >= size:
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
                    exit_date=t["exit_date"],
                )
                cash -= size
                legs.append(leg)
                exits[t["exit_date"]].append(leg)
                nopen += 1
        pos = sum(lv(l, dt) for l in legs)
        nav = cash + pt + pos
        navs.append((dt, nav))
        exps.append(pos / nav if nav > 0 else 0.0)
    ns = pd.DataFrame(navs, columns=["date", "nav"])
    nav = ns["nav"]
    dd = nav / nav.cummax() - 1
    return float(nav.iloc[-1]), float(dd.min()), float(np.mean(exps))


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

con = psycopg2.connect(**PG)
res = {ks: {} for ks in KSIZES}
for sd in SEEDS:
    rid = run_template_experiment(template_id=FRONTIER, seed=sd)["run_id"]
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
        PANEL[["symbol", "date", "cs4"]],
        left_on=["symbol", "sigd"],
        right_on=["symbol", "date"],
        how="left",
    )
    m["ek"] = m["symbol"] + "|" + m["entry_date"].astype(str).str[:10]
    csk = dict(zip(m["ek"], m["cs4"].fillna(0.5)))
    slook = {
        x.symbol + "|" + str(x.entry_date)[:10]: (x.score if pd.notna(x.score) else 0.0)
        for x in tr.itertuples()
    }
    csv = WORK / f"_ex_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)

    def mk():
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0)
            t["conv"] = csk.get(ek, 0.5)
        return sim

    for ks in KSIZES:
        res[ks][sd] = run_v(mk(), KSLOT, ks, KCONV)
    print(
        "seed%d: " % sd
        + " ".join(
            f"Ks{ks}={res[ks][sd][0]:.1f}(dd{res[ks][sd][1] * 100:.0f},ex{res[ks][sd][2] * 100:.0f})"
            for ks in KSIZES
        ),
        flush=True,
    )
con.close()

print("\n=== deploy-idle-cash: sizing denom K_size (slot cap 16) vs K_size=16 base, sign 3/3 ===")
base = res[16]
for ks in KSIZES:
    navs = [res[ks][s][0] for s in SEEDS]
    d = [res[ks][s][0] - base[s][0] for s in SEEDS]
    dd = statistics.mean([res[ks][s][1] for s in SEEDS]) * 100
    ex = statistics.mean([res[ks][s][2] for s in SEEDS]) * 100
    signs = "".join("+" if x > 0 else "-" for x in d)
    tag = " (base)" if ks == 16 else ""
    print(
        f"K_size={ks:2d}: mean={sum(navs) / 3:.3f} dd={dd:.1f}% exp={ex:.0f}% | Δ={[f'{x:+.2f}' for x in d]} {signs}{tag}"
    )
print("EXPOSURE_DONE")
