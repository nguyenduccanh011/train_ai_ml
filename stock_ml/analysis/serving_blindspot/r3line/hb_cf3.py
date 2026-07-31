# -*- coding: utf-8 -*-
"""Causal same-day signal khả thi: MARKET OPEN-GAP ngày khớp. Limit khớp intraday => giá MỞ CỬA đã biết
TRƯỚC khi khớp. Nếu thị trường gap-down mạnh đầu phiên => phiên đang sập => skip fill. Causal (open known
before intraday limit triggers). So base + same-day-close (look-ahead ceiling) + lag1 (causal fail cũ).
Sweep + per-year. Base K10/cs5_ma50+ret7, T+2, seeds[42,21,123]. Register nếu 3/3 vượt slot family."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
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


def prun(sim, pm, cm, r7map, fmap=None, fthr=None, fabove=True, advance_fee=0.0008, roundtrip=0.006, want_nav=False):
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
        if fmap is not None:
            fv = fmap.get((t["symbol"], t["entry_date"]))
            if fv is not None and ((fv < fthr) if fabove else (fv > fthr)):
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
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"], ratio0=t["p0"] / c0,
                    ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di)

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
    cg = float(nav.iloc[-1]) ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    if want_nav:
        return cg, dd, d
    return cg, dd


# ---- data ----
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,open,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c = g["close"]
    dd_ = c.diff(); up = dd_.clip(lower=0).rolling(14).mean(); dn = (-dd_.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / c.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    g["atrpct"] = (c - c.shift()).abs().rolling(14).mean() / c
    g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["mw7"] = c / c.shift(7) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "mw7"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
MW7 = {(r.symbol, str(r.date.date())): (r.mw7 if pd.notna(r.mw7) else np.nan) for r in P.itertuples()}

# ---- market features: OPEN-GAP (causal), close-ret same-day (look-ahead ref), lag1 (causal-fail ref) ----
piv_c = px.pivot_table(index="date", columns="symbol", values="close").sort_index()
piv_o = px.pivot_table(index="date", columns="symbol", values="open").sort_index()
gap_mat = (piv_o / piv_c.shift(1) - 1.0).clip(-0.15, 0.15)     # mỗi mã: open gap vs close hôm qua
mkt_gap = gap_mat.median(axis=1)                                # median open-gap toàn thị trường / ngày
ewd = piv_c.pct_change().clip(-0.15, 0.15).median(axis=1)       # close-to-close daily (same-day = look-ahead)
GAP = {d.strftime("%Y-%m-%d"): (float(v) if pd.notna(v) else np.nan) for d, v in mkt_gap.items()}
SAME = {d.strftime("%Y-%m-%d"): (float(v) if pd.notna(v) else np.nan) for d, v in ewd.items()}
LAG1 = {d.strftime("%Y-%m-%d"): (float(v) if pd.notna(v) else np.nan) for d, v in ewd.shift(1).items()}
# intraday floor: open-gap NHƯNG chỉ dùng phần đã xảy ra tới open (bằng GAP). Thêm gap+lag1 (mở cửa đỏ SAU khi hôm qua đỏ)

# ---- trades + meta ----
con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["ed"] = cvtr["entry_date"].astype(str); cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_cf3_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
GAPm = {sd: {(sym, ed): GAP.get(ed, np.nan) for sym, ed, _ in key_c[sd]} for sd in SEEDS}
SAMEm = {sd: {(sym, ed): SAME.get(ed, np.nan) for sym, ed, _ in key_c[sd]} for sd in SEEDS}
LAG1m = {sd: {(sym, ed): LAG1.get(ed, np.nan) for sym, ed, _ in key_c[sd]} for sd in SEEDS}


def ev(fmap_sd=None, fthr=None, fabove=True, want_nav=False):
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd],
               fmap=(fmap_sd[sd] if fmap_sd is not None else None), fthr=fthr, fabove=fabove, want_nav=want_nav) for sd in SEEDS]
    if want_nav:
        return rr
    return statistics.mean([c for c, _ in rr]), statistics.mean([d for _, d in rr]), [c for c, _ in rr]


bcg, bdd, bs = ev()
print(f"=== base {100*bcg:.1f}%/DD{100*bdd:.1f}% ===", flush=True)


def row(lab, cg, dd, sv):
    sgn = sum(1 for i in range(3) if sv[i] > bs[i]); mk = "*" if (sgn == 3 and cg > bcg) else ("+" if sgn == 3 else " ")
    print(f"  {lab:44s} | {100*cg:6.1f} {100*dd:6.1f} | ΔCAGR {100*(cg-bcg):+5.1f}{mk} | {sgn}/3 | {[round(100*x) for x in sv]}", flush=True)


# how often does market gap-down at open?
allgap = np.array([v for sd in SEEDS for v in GAPm[sd].values()], float)
for p in (-0.005, -0.01, -0.015, -0.02):
    print(f"  [info] %fill với mkt_open_gap<{p:+.3f}: {100*np.nanmean(allgap < p):.1f}%", flush=True)
print("--- CAUSAL: skip fill khi thị trường GAP-DOWN mở cửa ---", flush=True)
for thr in (-0.005, -0.008, -0.010, -0.012, -0.015, -0.020):
    cg, dd, sv = ev(GAPm, fthr=thr); row(f"skip mkt_open_gap < {thr:+.3f}", cg, dd, sv)
print("--- ref: same-day close (LOOK-AHEAD ceiling) & lag1 (causal-fail) ---", flush=True)
for thr in (-0.02, -0.025):
    cg, dd, sv = ev(SAMEm, fthr=thr); row(f"[LA] skip mkt_close(ngày) < {thr:+.3f}", cg, dd, sv)
cg, dd, sv = ev(LAG1m, fthr=-0.02); row("[causal-fail] skip mkt(hôm qua) < -0.020", cg, dd, sv)

# per-year cho open-gap tốt nhất (chọn ngưỡng có 3/3 & ΔCAGR>0 lớn nhất; mặc định -0.010)
print("\n=== PER-YEAR: base vs open_gap<-0.010 vs same<-0.02(LA) ===", flush=True)


def peryear(rr):
    yd = defaultdict(list)
    for cg, dd, d in rr:
        d = d.copy(); d["y"] = d["date"].dt.year
        for y, g in d.groupby("y"):
            yd[y].append(g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0)
    return {y: statistics.mean(v) for y, v in yd.items()}


yb = peryear(ev(want_nav=True))
yg = peryear(ev(GAPm, fthr=-0.010, want_nav=True))
ys = peryear(ev(SAMEm, fthr=-0.02, want_nav=True))
print(f"  {'year':6s} {'base':>8s} {'gap':>8s} {'same(LA)':>9s}", flush=True)
for y in sorted(yb):
    print(f"  {y:<6d} {100*yb[y]:+7.1f}% {100*yg.get(y,0):+7.1f}% {100*ys.get(y,0):+8.1f}%", flush=True)
print("CF3_DONE", flush=True)
