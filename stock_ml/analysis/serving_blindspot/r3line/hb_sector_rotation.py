# -*- coding: utf-8 -*-
"""SECTOR-ROTATION sleeve (user pick). 61-univ manual ICB sector map. Weekly rebalance into top-N sectors
by sector momentum (or contrarian: bottom-N laggards). Make-or-break = is it DECORRELATED from the champion
(low corr) and does it add in dead years — OR is it just coarse momentum (high corr, dies in flat years like
prior momentum sleeves)? Standalone weight-based NAV, turnover cost 0.3%/side. Report CAGR/DD/corr/per-year."""
from __future__ import annotations
import sys, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
COMBO_RID = "template/x2_struct_to_k16preempt_cssize-69338138"
ONE_WAY = 0.003

SECTOR = {
    "Bank": "ACB BID CTG EIB HDB LPB MBB OCB SHB STB TCB TPB VCB VPB".split(),
    "Securities": "AAS HCM SSI VCI VDS VND".split(),
    "Insurance": "BVH".split(),
    "RealEstate": "AAV BCG BCM DIG HDG KBC KDH NLG NVL PDR VHM VIC".split(),
    "Steel": "HPG HSG NKG".split(),
    "OilGas": "BSR GAS PLX PVD PVS".split(),
    "Chemicals": "DCM DGC DPM".split(),
    "Power": "NT2 POW PC1".split(),
    "FoodBev": "MSN VNM SAB SBT".split(),
    "Retail": "MWG FRT PNJ".split(),
    "TechTel": "FPT VTP".split(),
    "Transport": "ACV VJC GMD".split(),
    "Industrial": "REE GEX".split(),
}
SYM2SEC = {s: sec for sec, ss in SECTOR.items() for s in ss}
SYMS = list(SYM2SEC)

cx = duckdb.connect(MARKET, read_only=True)
syms_sql = ",".join(repr(s) for s in SYMS)
px = cx.execute(f"SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({syms_sql}) "
                "AND date>='2018-06-01' ORDER BY date,symbol").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
C = px.pivot(index="date", columns="symbol", values="close").sort_index()
RET = C.pct_change()
CAL = C.index

con = psycopg2.connect(**PG)
eq = pd.read_sql("SELECT date,nav FROM run_equity WHERE run_id=%s ORDER BY date", con, params=(COMBO_RID,)); con.close()
eq["date"] = pd.to_datetime(eq["date"]); eq = eq[eq["date"] >= "2020-01-01"].sort_values("date")
champ = eq.set_index("date")["nav"]; champ = champ / champ.iloc[0]
champ_ret = champ.pct_change()


def run_rotation(lb, n_sec, contrarian, reb=5):
    """Weekly rebalance: rank sectors by mean member LB-return, hold members of top(or bottom) n_sec sectors
    equal-weight. Returns daily NAV series (>=2020)."""
    w = pd.Series(0.0, index=C.columns); nav = 1.0; navs = []; prev_w = w.copy()
    for k, dt in enumerate(CAL):
        if k > 0:
            r = RET.loc[dt].fillna(0.0)
            nav *= (1.0 + float((w * r).sum()))
        if k >= lb and k % reb == 0:
            secmom = {}
            for sec, mem in SECTOR.items():
                mem = [m for m in mem if m in C.columns]
                past = C[mem].iloc[k] / C[mem].iloc[k - lb] - 1.0
                secmom[sec] = past.mean()
            order = sorted(secmom, key=lambda s: secmom[s], reverse=not contrarian)
            picks = order[:n_sec]
            members = [m for s in picks for m in SECTOR[s] if m in C.columns]
            # equal weight, only members with valid price
            valid = [m for m in members if not np.isnan(C.loc[dt, m])]
            neww = pd.Series(0.0, index=C.columns)
            if valid:
                neww[valid] = 1.0 / len(valid)
            turn = (neww - w).abs().sum()
            nav *= (1.0 - turn * ONE_WAY)
            w = neww
        navs.append((dt, nav))
    d = pd.DataFrame(navs, columns=["date", "nav"]); d = d[d["date"] >= "2020-01-01"]
    return d.set_index("date")["nav"] / d[d["date"] >= "2020-01-01"]["nav"].iloc[0]


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    corr = nav.pct_change().reindex(champ_ret.index).corr(champ_ret)
    py = []
    for y in range(2020, 2027):
        g = nav[nav.index.year == y]
        py.append(f"{100*(g.iloc[-1]/g.iloc[0]-1):+4.0f}" if len(g) > 2 else "   .")
    return cagr, dd, corr, " ".join(py)


print(f"champion: CAGR {100*(champ.iloc[-1]**(1/((champ.index[-1]-champ.index[0]).days/365.25))-1):.1f}%  "
      f"DD {100*float((champ/champ.cummax()-1).min()):.1f}%\n", flush=True)
print("=== SECTOR-ROTATION variants (61-univ, weekly reb) ===", flush=True)
print(f"{'variant':22s} | {'CAGR':>6s} | {'DD':>6s} | corr | per-year 2020..2026", flush=True)
CONF = [
    ("mom_lb60_top3", 60, 3, False), ("mom_lb60_top5", 60, 5, False),
    ("mom_lb120_top3", 120, 3, False), ("mom_lb20_top3", 20, 3, False),
    ("contrarian_lb60_bot3", 60, 3, True), ("contrarian_lb120_bot3", 120, 3, True),
]
best = None
for name, lb, ns, con_ in CONF:
    nav = run_rotation(lb, ns, con_)
    cg, dd, corr, py = stats(nav)
    print(f"{name:22s} | {100*cg:5.1f}% | {100*dd:5.1f}% | {corr:+.2f} | {py}", flush=True)
    if best is None or (cg / abs(dd)) > best[1]:
        best = (name, cg / abs(dd), nav, corr, cg)

print("\n=== BLEND champion + best-Calmar sector variant (exposure-neutral) ===", flush=True)
bname, _, bnav, bcorr, bcg = best
sl = bnav.reindex(champ.index, method="ffill").fillna(1.0); sl = sl / sl.iloc[0]
print(f"  (best = {bname}, standalone CAGR {100*bcg:.1f}%, corr {bcorr:+.2f})", flush=True)
for w in (1.0, 0.85, 0.70):
    bl = w * champ + (1 - w) * sl
    byrs = (bl.index[-1] - bl.index[0]).days / 365.25
    bc = bl.iloc[-1] ** (1 / byrs) - 1; bd = float((bl / bl.cummax() - 1).min())
    tag = "champ only" if w == 1.0 else f"{int(w*100)}/{int((1-w)*100)}"
    print(f"  {tag:12s}: CAGR {100*bc:5.1f}%  DD {100*bd:5.1f}%", flush=True)
print("(WIN = low corr + positive dead-years 2024/2026 + blend raises CAGR-per-DD)")
print("SECTOR_ROTATION_DONE", flush=True)
