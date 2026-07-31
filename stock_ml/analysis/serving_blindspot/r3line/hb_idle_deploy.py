# -*- coding: utf-8 -*-
"""IDLE-CAPITAL deployment (user): champion holds ~31% idle cash. (A) Park idle in a beta position (VNINDEX)
to earn market return instead of 0%? (B) Is idle DEFENSIVE (occurs in bad markets -> parking it adds risk)?
First: WHEN is cash idle (corr idle vs market state). Then: beta-overlay blend (park f*idle in VN30F),
measure CAGR/DD. Exposure-neutral (idle only, no leverage: overlay <= idle fraction)."""
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
e = pd.read_sql("SELECT date,nav,exposure,n_positions FROM run_equity WHERE run_id=%s ORDER BY date", con, params=(COMBO_RID,)); con.close()
e["date"] = pd.to_datetime(e["date"]); e = e[e["date"] >= "2020-01-01"].reset_index(drop=True)

cx = duckdb.connect(MARKET, read_only=True)
vni = cx.execute("SELECT date,close FROM ohlcv WHERE timeframe='1D' AND symbol='VN30F1M' ORDER BY date").fetchdf(); cx.close()
vni["date"] = pd.to_datetime(vni["date"]); vni = vni.set_index("date")["close"]
vni_ret = vni.pct_change()
vni_dd = vni / vni.cummax() - 1
vni_ma50 = vni / vni.rolling(50).mean() - 1

e = e.set_index("date")
e["idle"] = (1 - e["exposure"]).clip(lower=0)
e["champ_ret"] = e["nav"].pct_change()
e["vni_ret"] = vni_ret.reindex(e.index)
e["vni_dd"] = vni_dd.reindex(e.index)
e["vni_ma50"] = vni_ma50.reindex(e.index)
e["vni_fwd20"] = (vni.reindex(e.index).shift(-20) / vni.reindex(e.index) - 1)

print("=== WHEN is cash idle? (is idle defensive?) ===", flush=True)
print(f"avg idle = {100*e.idle.mean():.1f}% NAV", flush=True)
# split days by market state
for tag, mask in [("mkt UP (>MA50)", e.vni_ma50 > 0), ("mkt DOWN (<MA50)", e.vni_ma50 <= 0),
                  ("mkt drawdown <−10%", e.vni_dd < -0.10), ("mkt near-high >−3%", e.vni_dd > -0.03)]:
    g = e[mask]
    print(f"  {tag:22s}: idle={100*g.idle.mean():4.1f}%  | n_pos={g.n_positions.mean():4.1f}  | vni_fwd20={100*g.vni_fwd20.mean():+5.1f}%", flush=True)
print(f"\n  corr(idle, vni_dd) = {e.idle.corr(e.vni_dd):+.2f}   (idle high when dd deep => cash is defensive)", flush=True)
print(f"  corr(idle, vni_fwd20) = {e.idle.corr(e.vni_fwd20):+.2f}   (>0 => idle precedes market gains = wasted; <0 => idle precedes drops = smart)", flush=True)

# ---- (A) beta-overlay: park f*idle in VN30F ----
def blend(f):
    r = e["champ_ret"].fillna(0.0) + (f * e["idle"].shift(1).fillna(0.0)) * e["vni_ret"].fillna(0.0)
    nav = (1 + r).cumprod()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    py = []
    for y in range(2020, 2027):
        gg = nav[nav.index.year == y]
        py.append(f"{100*(gg.iloc[-1]/gg.iloc[0]-1):+4.0f}" if len(gg) > 2 else "  .")
    return cagr, dd, " ".join(py)

print("\n=== (A) park f×idle in VN30F beta (f=0 is champion) ===", flush=True)
print(f"{'f':>5s} | {'CAGR':>6s} | {'DD':>6s} | per-year 2020..2026", flush=True)
for f in (0.0, 0.5, 1.0):
    cg, dd, py = blend(f)
    print(f"{f:5.1f} | {100*cg:5.1f}% | {100*dd:5.1f}% | {py}", flush=True)

# smart overlay: only park idle when market is in uptrend (>MA50)
def blend_gated(f):
    gate = (e["vni_ma50"] > 0).astype(float)
    r = e["champ_ret"].fillna(0.0) + (f * e["idle"].shift(1).fillna(0.0) * gate.shift(1).fillna(0.0)) * e["vni_ret"].fillna(0.0)
    nav = (1 + r).cumprod()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return nav.iloc[-1] ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())

print("\n=== (A') gated: park idle in VN30F ONLY when mkt>MA50 (skip downtrend) ===", flush=True)
for f in (0.5, 1.0):
    cg, dd = blend_gated(f)
    print(f"  f={f}: CAGR {100*cg:5.1f}%  DD {100*dd:5.1f}%", flush=True)

# ---- (B) idle is DEFENSIVE (precedes drops) -> park idle SHORT VN30F as a hedge ----
def blend_short(f, gate_down=False):
    idl = f * e["idle"].shift(1).fillna(0.0)
    if gate_down:
        idl = idl * (e["vni_ma50"] < 0).astype(float).shift(1).fillna(0.0)
    r = e["champ_ret"].fillna(0.0) - idl * e["vni_ret"].fillna(0.0)     # SHORT beta with idle
    nav = (1 + r).cumprod()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    py = []
    for y in range(2020, 2027):
        gg = nav[nav.index.year == y]
        py.append(f"{100*(gg.iloc[-1]/gg.iloc[0]-1):+4.0f}" if len(gg) > 2 else "  .")
    return cagr, dd, " ".join(py)

print("\n=== (B) park f×idle SHORT VN30F (hedge, since idle precedes drops) ===", flush=True)
print(f"{'variant':16s} | {'CAGR':>6s} | {'DD':>6s} | per-year 2020..2026", flush=True)
for f in (0.5, 1.0):
    cg, dd, py = blend_short(f, False)
    print(f"{'short f='+str(f):16s} | {100*cg:5.1f}% | {100*dd:5.1f}% | {py}", flush=True)
for f in (0.5, 1.0):
    cg, dd, py = blend_short(f, True)
    print(f"{'short-gated f='+str(f):16s} | {100*cg:5.1f}% | {100*dd:5.1f}% | {py}", flush=True)
# ---- CONTROL: is the win from idle-TIMING or just "any static short helped in 2022"? ----
def blend_static_short(sz):
    """Short a CONSTANT sz of VN30F every day (no idle timing). If this matches idle-scaled, timing adds nothing."""
    r = e["champ_ret"].fillna(0.0) - sz * e["vni_ret"].fillna(0.0)
    nav = (1 + r).cumprod(); yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    py = []
    for y in range(2020, 2027):
        gg = nav[nav.index.year == y]
        py.append(f"{100*(gg.iloc[-1]/gg.iloc[0]-1):+4.0f}" if len(gg) > 2 else "  .")
    return cagr, dd, " ".join(py)

print("\n=== CONTROL: STATIC short (constant, no idle-timing) — same avg short size ~0.31 ===", flush=True)
print(f"avg idle (=avg idle-scaled short size) = {100*e.idle.mean():.1f}%", flush=True)
print(f"{'variant':16s} | {'CAGR':>6s} | {'DD':>6s} | per-year 2020..2026", flush=True)
for sz in (0.15, 0.31):
    cg, dd, py = blend_static_short(sz)
    print(f"{'static s='+str(sz):16s} | {100*cg:5.1f}% | {100*dd:5.1f}% | {py}", flush=True)
print("(if static ~= idle-scaled -> timing worthless; if idle-scaled >> static -> champion exposure IS a market-timing signal)")
print("IDLE_DEPLOY_DONE", flush=True)
