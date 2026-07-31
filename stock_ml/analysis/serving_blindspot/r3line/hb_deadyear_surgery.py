# -*- coding: utf-8 -*-
"""DEAD-YEAR surgery on the operating model (K10/cs5_ma50 + ret7, T+2). Weak years (2024/2026) are the
persistent floor. Partition the cause per YEAR with 3 orthogonal diagnostics on FILLED legs:
 H1 entry/source: MFE (max favorable excursion) low  -> picks don't run.
 H2 exit/retention: MFE ok but capture=net/MFE low   -> picks run but give back.
 H3 selection: corr(meta-prio, realized net) low      -> picking the wrong candidates.
Also: fwd20 of picks vs universe-available (did we leave better picks on the table?). 3-seed pooled."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, TPLUS, R7THR = 10, 0.005, 2.0, 2, 0.02


def prun(sim, pm, cm, r7map, rec, tplus=TPLUS):
    """record each FILLED leg: symbol, net, prio, entry/exit date, fill price."""
    s_new = (0.006 - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if tplus and (bx - be) < tplus:
            nb = min(be + tplus, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < SKIP:
            continue
        v = r7map.get((t["symbol"], t["entry_date"]), np.nan)
        if not np.isnan(v) and v < R7THR:
            continue
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def mk(t, size, di):
        rec.append(dict(symbol=t["symbol"], net=t["net"], prio=t["prio"], entry_date=t["entry_date"], exit_date=t["exit_date"], p0=t["p0"]))
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di)

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list)
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - 0.0008); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= tplus]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - 0.0008); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
        for l in legs:
            lv(l, dt)


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
CLO = {}; HI = {}; LO = {}; DIDX = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values; HI[s] = h.values; LO[s] = l.values
    DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["mw7"] = c / c.shift(7) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "mw7"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
MW7 = {(r.symbol, str(r.date.date())): (r.mw7 if pd.notna(r.mw7) else np.nan) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_dy_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

rec = []
for sd in SEEDS:
    prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], rec)
R = pd.DataFrame(rec)
R["yr"] = pd.to_datetime(R.entry_date).dt.year
# MFE / MAE over holding window, by DATE lookup (index-safe), relative to actual fill price p0, HIGH/LOW paths
mfe, mae, fwd = [], [], []
for r in R.itertuples():
    di = DIDX[r.symbol]; e_i = di.get(r.entry_date); x_i = di.get(r.exit_date)
    if e_i is None or x_i is None or x_i < e_i:
        mfe.append(np.nan); mae.append(np.nan); fwd.append(np.nan); continue
    e = r.p0; hseg = HI[r.symbol][e_i:x_i + 1]; lseg = LO[r.symbol][e_i:x_i + 1]
    mfe.append(float(hseg.max() / e - 1.0)); mae.append(float(lseg.min() / e - 1.0))
    c = CLO[r.symbol]; fwd.append(float(c[e_i + 20] / c[e_i] - 1.0) if e_i + 20 < len(c) else np.nan)
R["mfe"] = mfe; R["mae"] = mae; R["fwd20"] = fwd
R["cap"] = R.net / R.mfe.replace(0, np.nan)

print("=== DEAD-YEAR surgery: per-year decomposition (operating model, 3-seed pooled) ===", flush=True)
print(f"  {'yr':4s} | n   | net TB | MFE TB | MAE TB | capture | corr(prio,net) | win%", flush=True)
for y, g in R.groupby("yr"):
    cap = (g.net.sum() / g.mfe.sum()) if g.mfe.sum() > 0 else float("nan")
    cr = g.prio.corr(g.net)
    flag = "  <== YẾU" if y in (2024, 2026) else ""
    print(f"  {y} | {len(g):4d}| {100*g.net.mean():+5.2f}% | {100*g.mfe.mean():5.1f}% | {100*g.mae.mean():+5.1f}% | {100*cap:5.1f}% | {cr:+.3f}         | {100*(g.net>0).mean():3.0f}%{flag}", flush=True)

# universe forward-20 available on the same signal dates (did better picks exist we didn't take?)
print("\n  Picks fwd20 vs UNIVERSE-median fwd20 (cùng ngày vào, cùng năm):", flush=True)
uni_by_date = {}
for r in P.itertuples():
    pass
UF = defaultdict(dict)  # date-> {symbol: fwd20}
for s in CLO:
    c = CLO[s]
    for ds, i in DIDX[s].items():
        if i + 20 < len(c):
            UF[ds][s] = c[i + 20] / c[i] - 1.0
for y, g in R.groupby("yr"):
    umed = []
    for r in g.itertuples():
        vals = list(UF.get(r.entry_date, {}).values())
        if vals:
            umed.append(np.median(vals))
    print(f"    {y}: picks fwd20 {100*np.nanmean(g.fwd20):+5.1f}% | universe-median fwd20 {100*np.nanmean(umed):+5.1f}% | edge {100*(np.nanmean(g.fwd20)-np.nanmean(umed)):+5.1f}pp", flush=True)

# ---- T+0 vs T+2: is the capture-loss caused by T+2 forced-hold? ----
def build_R(tplus):
    rc = []
    for sd in SEEDS:
        prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], rc, tplus=tplus)
    d = pd.DataFrame(rc); d["yr"] = pd.to_datetime(d.entry_date).dt.year
    mf, nt = [], []
    for r in d.itertuples():
        di = DIDX[r.symbol]; e_i = di.get(r.entry_date); x_i = di.get(r.exit_date)
        if e_i is None or x_i is None or x_i < e_i:
            mf.append(np.nan)
        else:
            mf.append(HI[r.symbol][e_i:x_i + 1].max() / r.p0 - 1.0)
    d["mfe"] = mf; return d


print("\n  CAPTURE T+0 vs T+2 theo năm (T+2 forced-hold có làm mất capture?):", flush=True)
R0 = build_R(0); R2 = build_R(2)
print(f"  {'yr':4s} | net T+0 | net T+2 | cap T+0 | cap T+2 | Δcap", flush=True)
for y in sorted(R0.yr.unique()):
    g0 = R0[R0.yr == y]; g2 = R2[R2.yr == y]
    c0 = g0.net.sum() / g0.mfe.sum() if g0.mfe.sum() > 0 else np.nan
    c2 = g2.net.sum() / g2.mfe.sum() if g2.mfe.sum() > 0 else np.nan
    print(f"  {y} | {100*g0.net.mean():+6.2f}% | {100*g2.net.mean():+6.2f}% | {100*c0:5.1f}% | {100*c2:5.1f}% | {100*(c2-c0):+5.1f}pp", flush=True)

# ---- exit-override probe: regime-conditional take-profit (bank in chop, ride in trend) ----
# EW market index + MA100 regime (causal)
piv = px.pivot(index="date", columns="symbol", values="close").sort_index()
idx = piv.div(piv.bfill().iloc[0]).mean(axis=1); idx_ma = idx.rolling(100).mean()
CHOP = {d.strftime("%Y-%m-%d"): bool(idx.loc[d] < idx_ma.loc[d]) if pd.notna(idx_ma.loc[d]) else False for d in idx.index}
FEE_RT = 0.006

print("\n  EXIT-OVERRIDE: TP điều kiện (bank khi CHOP idx<MA100, thả khi TREND) — per-trade net actual vs override:", flush=True)
print(f"  {'TP':5s} | {'yr':4s} | net actual | net override | Δ", flush=True)
for TP in (0.04, 0.06, 0.08):
    ov = []
    for r in R.itertuples():
        di = DIDX[r.symbol]; e_i = di.get(r.entry_date); x_i = di.get(r.exit_date)
        if e_i is None or x_i is None or x_i <= e_i:
            ov.append(r.net); continue
        if not CHOP.get(r.entry_date, False):
            ov.append(r.net); continue                       # TREND: keep engine exit (ride)
        hit = None
        for k in range(e_i + 1, x_i + 1):
            if HI[r.symbol][k] >= r.p0 * (1 + TP):
                hit = k; break
        ov.append((r.p0 * (1 + TP)) / r.p0 - 1.0 - FEE_RT if hit is not None else r.net)  # bank at TP else engine
    R["_ov"] = ov
    for y in sorted(R.yr.unique()):
        g = R[R.yr == y]; na = g.net.mean(); no = g._ov.mean()
        tag = "  <==" if (y in (2024, 2026) and no > na + 0.002) else ""
        print(f"  {TP:.2f} | {y} | {100*na:+6.2f}% | {100*no:+7.2f}% | {100*(no-na):+5.2f}pp{tag}", flush=True)
    print("  " + "-" * 44, flush=True)
print("DEADYEAR_DONE", flush=True)
