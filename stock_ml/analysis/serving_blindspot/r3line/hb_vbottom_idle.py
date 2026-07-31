# -*- coding: utf-8 -*-
"""V-BOTTOM idle deployment (user, STOCK-ONLY): champion sits in cash during/after drawdowns and its
momentum-pullback entry is slow to re-enter a V-recovery (which rips without pulling back). Deploy idle
cash into a bounce basket ONLY when a causal V-confirmation fires (deep drawdown -> reclaim short MA =
follow-through), long stocks only (no futures). Basket = equal-weight 61-univ composite. Compare vs
champion and vs naive always-deploy (which was refuted). WIN = adds CAGR without wrecking DD."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
COMBO_RID = "template/x2_struct_to_k16preempt_cssize-69338138"

con = psycopg2.connect(**PG)
e = pd.read_sql("SELECT date,nav,exposure FROM run_equity WHERE run_id=%s ORDER BY date", con, params=(COMBO_RID,))
uni = pd.read_sql("SELECT DISTINCT symbol FROM run_signals WHERE run_id=%s", con, params=(COMBO_RID,)); con.close()
e["date"] = pd.to_datetime(e["date"]); e = e[e["date"] >= "2020-01-01"].set_index("date")
SYMS = list(uni["symbol"])

cx = duckdb.connect(MARKET, read_only=True)
syms_sql = ",".join(repr(s) for s in SYMS)
px = cx.execute(f"SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({syms_sql}) "
                "AND date>='2018-06-01' ORDER BY date,symbol").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
C = px.pivot(index="date", columns="symbol", values="close").sort_index()
# equal-weight composite (rebased): daily mean of member returns
comp_ret = C.pct_change().mean(axis=1)
comp = (1 + comp_ret.fillna(0)).cumprod()
ma10 = comp.rolling(10).mean(); ma20 = comp.rolling(20).mean(); ma50 = comp.rolling(50).mean()
roll60max = comp.rolling(60).max()
dd60 = comp / roll60max - 1                                  # composite drawdown from 60d high
breadth = (C > C.rolling(20).mean()).mean(axis=1)           # % stocks above MA20

# align to champion calendar
idx = e.index
comp_ret = comp_ret.reindex(idx).fillna(0.0)
comp_r = comp.reindex(idx); ma10i = ma10.reindex(idx); ma20i = ma20.reindex(idx); ma50i = ma50.reindex(idx)
dd60i = dd60.reindex(idx); breadthi = breadth.reindex(idx)
idle = (1 - e["exposure"]).clip(lower=0)
champ_ret = e["nav"].pct_change().fillna(0.0)


def bounce_window(dd_th, use_breadth):
    """Causal V-confirm: was in drawdown < dd_th, then composite reclaims MA10 (follow-through).
    Window active until composite>MA50 (recovered) or new-low stop. Returns bool Series (deploy today)."""
    active = pd.Series(False, index=idx); armed = False; in_win = False; entry_lvl = None
    for k, dt in enumerate(idx):
        d = dd60i.iloc[k]; c = comp_r.iloc[k]; m10 = ma10i.iloc[k]; m50 = ma50i.iloc[k]
        if np.isnan(d) or np.isnan(m10):
            continue
        if d < dd_th:
            armed = True
        thrust = (c > m10) and (not use_breadth or breadthi.iloc[k] > 0.45)
        if armed and thrust and not in_win:
            in_win = True; armed = False; entry_lvl = c
        if in_win:
            active.iloc[k] = True
            if (not np.isnan(m50) and c > m50) or (c < entry_lvl * 0.94):   # recovered OR stop
                in_win = False
    return active


def overlay(active, f):
    dep = f * idle.shift(1).fillna(0.0) * active.shift(1).fillna(False).astype(float)
    r = champ_ret + dep * comp_ret
    nav = (1 + r).cumprod(); yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    py = []
    for y in range(2020, 2027):
        gg = nav[nav.index.year == y]
        py.append(f"{100*(gg.iloc[-1]/gg.iloc[0]-1):+4.0f}" if len(gg) > 2 else "  .")
    return cagr, dd, " ".join(py), int(active.sum())


champ_cg = e["nav"].iloc[-1] ** (1 / ((idx[-1] - idx[0]).days / 365.25)) - 1
champ_dd = float((e["nav"] / e["nav"].cummax() - 1).min())
print(f"champion: CAGR {100*champ_cg:.1f}%  DD {100*champ_dd:.1f}%\n", flush=True)

# naive control: deploy idle into composite ALWAYS (no V-gate) — expect refuted
print("=== control: deploy idle into composite ALWAYS (no V-timing) ===", flush=True)
allon = pd.Series(True, index=idx)
for f in (0.5, 1.0):
    cg, dd, py, _ = overlay(allon, f)
    print(f"  always f={f}: CAGR {100*cg:5.1f}%  DD {100*dd:5.1f}% | {py}", flush=True)

print("\n=== V-BOTTOM gated idle deploy (deploy only in causal bounce window) ===", flush=True)
print(f"{'variant':22s} | {'CAGR':>6s} | {'DD':>6s} | days | per-year 2020..2026", flush=True)
for name, dd_th, br in [("dd12_reclaim", -0.12, False), ("dd12_reclaim_breadth", -0.12, True),
                        ("dd08_reclaim", -0.08, False), ("dd15_reclaim", -0.15, False)]:
    act = bounce_window(dd_th, br)
    for f in (0.5, 1.0):
        cg, dd, py, nd = overlay(act, f)
        print(f"{name+' f'+str(f):22s} | {100*cg:5.1f}% | {100*dd:5.1f}% | {nd:4d} | {py}", flush=True)
print("\n(WIN = beats champion 104%/−14.9% on CAGR-per-DD; V-gate should beat always-on control)")
print("VBOTTOM_IDLE_DONE", flush=True)
