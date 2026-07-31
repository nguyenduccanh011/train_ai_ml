# -*- coding: utf-8 -*-
"""SIZING forensic (user): are big winners UNDER-weighted and heavy weights on losers/small-gainers?
Operating model K10/cs5_ma50 + ret7 gate, T+2. Record each FILLED leg (weight w, conviction, realized net).
Analyze: corr(w,net) & corr(conv,net); weight-quintile pnl/winrate; misallocation cases (big-win low-w,
loss high-w); per-trade weighted E_w[net] vs equal E[net]; NAV conviction vs equal(kconv0) vs inverse(kconv<0).
If conviction<=equal or corr<=0 -> sizing misallocated."""
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
K, MARGIN, TPLUS, R7THR = 10, 0.005, 2, 0.02


def prun(sim, pm, cm, r7map, kconv, rec=None, zpk=None, taper=0.0, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if (bx - be) < TPLUS:
            nb = min(be + TPLUS, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        ze = z if (zpk is None or z <= zpk) else (zpk - taper * (z - zpk))   # hump: down-taper extreme conviction
        t["_w"] = min(max(1.0 + kconv * ze, 0.4), 1.8); raw.append(t["_w"])
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
        if rec is not None:
            rec.append(dict(symbol=t["symbol"], entry_date=t["entry_date"], w=t["w"], conv=t["conv"], net=t["net"]))
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di)

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= TPLUS]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
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
    cvf = HERE / f"_sf_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

# ---- record filled legs (conviction kconv=2.0), seed42 directional ----
rec = []
prun(NavSim2(cv_c[42], date_lo="2020-01-01"), pm[42], CMS[42], MW[42], 2.0, rec=rec)
R = pd.DataFrame(rec)
print(f"=== SIZING forensic: {len(R)} FILLED legs (K10/cs5_ma50+ret7, seed42) ===", flush=True)
print(f"  corr(weight, net_pnl) = {R.w.corr(R.net):+.3f}   corr(conviction, net_pnl) = {R.conv.corr(R.net):+.3f}", flush=True)
print(f"  (dương = nặng cân vào lệnh lời; âm/0 = SIZING LỆCH)", flush=True)
R["wq"] = pd.qcut(R.w.rank(method="first"), 5, labels=["Q1 nhẹ", "Q2", "Q3", "Q4", "Q5 nặng"])
print("\n  Bucket theo TRỌNG SỐ:", flush=True)
print(f"    {'bucket':9s} | n   | w TB | net TB  | win% | net*w TỔNG", flush=True)
for q, g in R.groupby("wq", observed=True):
    print(f"    {q:9s} | {len(g):4d}| {g.w.mean():.2f} | {100*g.net.mean():+5.2f}% | {100*(g.net>0).mean():3.0f}% | {100*(g.net*g.w).sum():+6.1f}", flush=True)
# weighted vs equal per-trade expected return
ew = (R.net * R.w).sum() / R.w.sum(); eq = R.net.mean()
print(f"\n  E_w[net] (conviction-weighted) = {100*ew:+.3f}%  vs  E[net] (equal) = {100*eq:+.3f}%  -> sizing {'GIÚP' if ew>eq else 'HẠI'} {100*(ew-eq):+.3f}pp/lệnh", flush=True)
# misallocation cases
p90 = R.net.quantile(0.90); p10 = R.net.quantile(0.10); wlo = R.w.quantile(0.33); whi = R.w.quantile(0.67)
bigwin_loww = R[(R.net >= p90) & (R.w <= wlo)]; loss_highw = R[(R.net <= p10) & (R.w >= whi)]
print(f"\n  LỆCH #1 lời-lớn-nhẹ-cân (net≥p90 & w≤p33): {len(bigwin_loww)} lệnh, net TB {100*bigwin_loww.net.mean():+.1f}%, w TB {bigwin_loww.w.mean():.2f}", flush=True)
print(f"  LỆCH #2 lỗ/kém-nặng-cân (net≤p10 & w≥p67): {len(loss_highw)} lệnh, net TB {100*loss_highw.net.mean():+.1f}%, w TB {loss_highw.w.mean():.2f}", flush=True)
print(f"  top5 lời-lớn-nhẹ-cân: {[(r.symbol, round(100*r.net), round(r.w,2)) for r in bigwin_loww.nlargest(5,'net').itertuples()]}", flush=True)
print(f"  top5 lỗ-nặng-cân:     {[(r.symbol, round(100*r.net), round(r.w,2)) for r in loss_highw.nsmallest(5,'net').itertuples()]}", flush=True)

# ---- 3-seed robustness of the Q4>Q5 (top-conviction underperforms) pattern ----
print("\n  Độ bền 3-seed: net TB Q4 vs Q5 (nhóm conviction cao nhất):", flush=True)
for sd in SEEDS:
    rr = []
    prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], 2.0, rec=rr)
    RR = pd.DataFrame(rr); RR["wq"] = pd.qcut(RR.w.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    q4 = RR[RR.wq == 4].net.mean(); q5 = RR[RR.wq == 5].net.mean()
    print(f"    seed{sd}: Q4 {100*q4:+.2f}% | Q5 {100*q5:+.2f}% | Q5-Q4 {100*(q5-q4):+.2f}pp {'(Q5 tệ hơn ✓)' if q5<q4 else ''}", flush=True)

# ---- NAV: conviction vs equal vs inverse vs HUMPED (down-weight extreme) (3-seed T+2) ----
print("\n  NAV (3-seed T+2): sizing variants", flush=True)
base3 = None
for lab, kc, zpk, tp in [("conviction kconv+2.0 (base)", 2.0, None, 0.0), ("equal kconv 0", 0.0, None, 0.0),
                          ("inverse kconv-2.0", -2.0, None, 0.0),
                          ("hump zpk1.0 taper0.5", 2.0, 1.0, 0.5), ("hump zpk1.0 taper1.0", 2.0, 1.0, 1.0),
                          ("hump zpk1.5 taper0.5", 2.0, 1.5, 0.5), ("hump zpk1.5 taper1.0", 2.0, 1.5, 1.0),
                          ("hump zpk0.8 taper1.0", 2.0, 0.8, 1.0)]:
    cgv = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], kc, zpk=zpk, taper=tp)[0] for sd in SEEDS]
    cg = statistics.mean(cgv)
    if base3 is None:
        base3 = cgv
    sgn = sum(1 for i in range(3) if cgv[i] > base3[i]); mk = "*" if (sgn == 3 and cg > statistics.mean(base3)) else ("+" if sgn == 3 else " ")
    print(f"    {lab:26s} CAGR {100*cg:5.1f}%{mk} ({sgn}/3 vs base)", flush=True)
print("SIZING_FORENSIC_DONE", flush=True)
