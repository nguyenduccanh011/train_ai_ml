"""WALK-FORWARD LEARNED conviction-sizing (rigorous OOS validation + new direction). Train LightGBM to
predict per-trade pnl_pct from extension-family features, WALK-FORWARD (train entries < year Y, predict
Y) so the sizing signal is strictly OOS. Size by OOS-predicted pnl (fair mean-1 tilt, exposure-neutral,
priority-fill by score, K16). WIN (OOS) = beats equal 3/3 -> sizing breakthrough is REAL not in-sample.
"""
from __future__ import annotations
import sys, statistics
from pathlib import Path
sys.path.insert(0, "F:/PROJECTS/hb2943_work")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from collections import defaultdict
import numpy as np, pandas as pd, duckdb, psycopg2, lightgbm as lgb
from nh_nav2 import NavSim2, FEE
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
WORK = Path("F:/PROJECTS/hb2943_work/navboard"); WORK.mkdir(exist_ok=True)
FRONTIER = 3185; SEEDS = [42, 7, 99]; K = 16
FCOLS = ["dist_ma10", "dist_ma20", "dist_ma50", "dist_ma100", "rsi14", "ret5", "ret20",
         "rvol20", "dist20low", "dist63high", "ma20_slope"]


def run_sized(sim, K, k_conv, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    """fair mean-1 tilt by t['conv'], select/priority-fill by t['score']. k_conv=0 -> equal-weight."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    convs = [t.get("conv", 0.0) for t in sim.trades]
    mu = statistics.mean(convs); sd = statistics.pstdev(convs) or 1.0
    raw = []
    for t in sim.trades:
        z = (t.get("conv", 0.0) - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: -t.get("score", 0.0))
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pending = defaultdict(float); pt = 0.0; legs = []; exits = defaultdict(list); navs = []
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            if advance_fee is not None:
                cash += proceeds * (1.0 - advance_fee)
            else:
                ri = di + settle_lag
                (pending.__setitem__(cal[ri], pending[cal[ri]] + proceeds) if ri < len(cal) else None)
            legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos; nopen = len(legs)
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size and nopen < K:
                s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; ee = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=ee / c1, last_val=size, entry_date=dt, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg); nopen += 1
        pos = sum(lv(l, dt) for l in legs); navs.append((dt, cash + pt + pos))
    ns = pd.DataFrame(navs, columns=["date", "nav"]); nav = ns["nav"]
    yrs = (pd.to_datetime(ns["date"].iloc[-1]) - pd.to_datetime(ns["date"].iloc[0])).days / 365.25
    dd = nav / nav.cummax() - 1
    return dict(final=float(nav.iloc[-1]), maxdd=float(dd.min()))


# feature panel once
_cx = duckdb.connect(MARKET, read_only=True)
_px = _cx.execute("SELECT symbol,date,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                  "ORDER BY symbol,date").fetchdf(); _cx.close()
_px["date"] = pd.to_datetime(_px["date"]); _parts = []
for s, g in _px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, h, l = g["close"], g["high"], g["low"]
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    ma10, ma20, ma50, ma100 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(100).mean()
    g["dist_ma10"] = c / ma10 - 1; g["dist_ma20"] = c / ma20 - 1; g["dist_ma50"] = c / ma50 - 1
    g["dist_ma100"] = c / ma100 - 1; g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret5"] = c / c.shift(5) - 1; g["ret20"] = c / c.shift(20) - 1
    g["rvol20"] = c.pct_change().rolling(20).std(); g["dist20low"] = c / l.rolling(20).min() - 1
    g["dist63high"] = c / h.rolling(63).max() - 1; g["ma20_slope"] = ma20 / ma20.shift(10) - 1
    _parts.append(g[["symbol", "date"] + FCOLS])
PANEL = pd.concat(_parts, ignore_index=True)

con = psycopg2.connect(**PG)
res = {"equal": {}, "wf": {}, "dma20": {}}
for sd in SEEDS:
    r = run_template_experiment(template_id=FRONTIER, seed=sd); rid = r.get("run_id")
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,pnl_pct "
                     "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    tr["sigd"] = pd.to_datetime(tr["entry_signal_date"]); sg["dd"] = pd.to_datetime(sg["date"])
    tr = tr.merge(sg[["symbol", "dd", "score"]], left_on=["symbol", "sigd"], right_on=["symbol", "dd"], how="left")
    # features at entry_signal_date; OOS pnl model walk-forward
    m = tr.merge(PANEL, left_on=["symbol", "sigd"], right_on=["symbol", "date"], how="left").dropna(subset=FCOLS)
    m["yr"] = m["sigd"].dt.year
    m["ek"] = m["symbol"] + "|" + m["entry_date"].astype(str).str[:10]
    oos = {}
    for Y in range(2021, 2027):
        A = m[m.yr < Y]; B = m[m.yr == Y]
        if len(A) < 200 or len(B) < 20:
            continue
        mdl = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        mdl.fit(A[FCOLS], A["pnl_pct"])
        for ek, p in zip(B["ek"].values, mdl.predict(B[FCOLS])):
            oos[ek] = p
    dma = {r2.symbol + "|" + str(r2.entry_date)[:10]: r2.dist_ma20 for r2 in m.itertuples()}
    slook = {x.symbol + "|" + str(x.entry_date)[:10]: (x.score if pd.notna(x.score) else 0.0) for x in tr.itertuples()}
    csv = WORK / f"_wf_{sd}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].dropna().to_csv(csv, index=False)

    def score_variant(convmap):
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            ek = t["symbol"] + "|" + str(pd.to_datetime(t["entry_date"]).date())
            t["score"] = slook.get(ek, 0.0); t["conv"] = convmap.get(ek, 0.0)
        return sim
    res["equal"][sd] = run_sized(score_variant({}), K, 0.0)["final"]
    res["wf"][sd] = run_sized(score_variant(oos), K, 1.0)["final"]
    res["dma20"][sd] = run_sized(score_variant(dma), K, 1.0)["final"]
    print(f"seed{sd}: equal={res['equal'][sd]:.2f}  wf_learned={res['wf'][sd]:.2f}  dma20={res['dma20'][sd]:.2f}  (oos_n={len(oos)})", flush=True)
con.close()

print("\n=== WALK-FORWARD OOS learned sizing vs equal (K16) — sign 3/3 ===")
for nm in ("wf", "dma20"):
    navs = [res[nm][s] for s in SEEDS]; d = [res[nm][s] - res["equal"][s] for s in SEEDS]
    signs = "".join("+" if x > 0 else "-" for x in d)
    print(f"{nm:10s}: mean={sum(navs)/3:.3f}  navs={[f'{n:.2f}' for n in navs]} | Δ={[f'{x:+.2f}' for x in d]} signs={signs}")
print(f"equal:      mean={sum(res['equal'].values())/3:.3f}")
print("WF_SIZING_DONE")
