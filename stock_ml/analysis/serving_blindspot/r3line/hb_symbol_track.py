# -*- coding: utf-8 -*-
"""Does a symbol's PAST trade quality (trailing win-rate / avg-net of the model's prior trades on that symbol)
predict the next trade's outcome -> use for SELECTION (skip) or SIZING? Causal: only prior trades whose
exit_date < this trade's entry_date. Operating model K10/cs5_ma50+ret7, T+2. Diagnostic (corr/AUC/bucket)
+ NAV test (filter low-trailing-winrate; sizing tilt). Register if 3/3-beats 127.8%."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from bisect import bisect_left
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
MINP = 3  # min prior trades for a valid trailing stat


def trailing_map(cvtr, win_days=None):
    """(symbol,entry_date_str) -> (n_prior, winrate, avgnet) using prior same-symbol trades exited before entry."""
    df = cvtr.copy(); df["ent"] = pd.to_datetime(df["entry_date"]); df["ext"] = pd.to_datetime(df["exit_date"])
    df["ret"] = df["exit_price"] / df["entry_price"] - 1.0
    out = {}
    for s, g in df.groupby("symbol"):
        g = g.sort_values("ext"); exts = g["ext"].values; wins = (g["ret"].values > 0).astype(float); rets = g["ret"].values
        for r in g.itertuples():
            ent = np.datetime64(r.ent); lo = 0
            if win_days is not None:
                lo = bisect_left(exts, ent - np.timedelta64(win_days, "D"))
            hi = bisect_left(exts, ent)  # prior trades with exit strictly before entry
            if hi - lo >= MINP:
                w = wins[lo:hi]; rr = rets[lo:hi]
                out[(s, str(r.ent.date()))] = (hi - lo, float(w.mean()), float(rr.mean()))
    return out


def prun(sim, pm, cm, r7map, tw=None, wr_min=None, size_k=0.0, advance_fee=0.0008, roundtrip=0.006):
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
        z = (t["conv"] - mu) / sd; w = 1.0 + KCONV * z
        if size_k and tw is not None:  # sizing tilt by trailing winrate
            st = tw.get((t["symbol"], t["entry_date"]))
            if st is not None:
                w += size_k * (st[1] - 0.5)
        t["_w"] = min(max(w, 0.4), 1.8); raw.append(t["_w"])
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
        if wr_min is not None and tw is not None:  # selection filter by trailing winrate
            st = tw.get((t["symbol"], t["entry_date"]))
            if st is not None and st[1] < wr_min:
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

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c, TW, TWr = {}, {}, {}, {}, {}
cvtr42 = None
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_st_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    TW[sd] = trailing_map(cvtr, win_days=None)       # all-history
    TWr[sd] = trailing_map(cvtr, win_days=365)       # trailing 1yr
    if sd == 42:
        cvtr42 = cvtr
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

# ---- diagnostic (seed42, base trades) ----
tw = TW[42]; rows = []
for r in cvtr42.itertuples():
    st = tw.get((r.symbol, r.ed))
    if st is not None:
        rows.append((st[1], st[2], r.exit_price / r.entry_price - 1.0))
D = pd.DataFrame(rows, columns=["twin", "tavg", "net"])
from sklearn.metrics import roc_auc_score
print(f"=== SYMBOL-TRACK (thành tích quá khứ mã dự báo lệnh kế?) — {len(D)}/{len(cvtr42)} lệnh có ≥{MINP} lệnh trước ===", flush=True)
print(f"  corr(trailing-winrate, net) = {D.twin.corr(D.net):+.3f} | corr(trailing-avgnet, net) = {D.tavg.corr(D.net):+.3f}", flush=True)
y = (D.net > 0).astype(int).values
print(f"  AUC trailing-winrate -> lệnh này thắng: {roc_auc_score(y, D.twin.values):.3f} (0.5=vô dụng)", flush=True)
print("  bucket theo trailing-winrate:", flush=True)
D["q"] = pd.qcut(D.twin.rank(method="first"), 4, labels=["Q1 thấp", "Q2", "Q3", "Q4 cao"])
for q, g in D.groupby("q", observed=True):
    print(f"    {q:9s}: n={len(g):4d} twin TB {g.twin.mean():.2f} | net kế TB {100*g.net.mean():+.2f}% | thắng {100*(g.net>0).mean():3.0f}%", flush=True)

# ---- NAV test ----
def ev(**kw):
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], **kw) for sd in SEEDS]
    return statistics.mean([c for c, d in rr]), statistics.mean([d for c, d in rr]), [c for c, d in rr]


bcg, bdd, bs = ev()
print(f"\n=== NAV (T+2, K10/cs5_ma50+ret7) — base {100*bcg:.1f}%/DD{100*bdd:.1f} ===", flush=True)
def row(lab, cg, dd, sv):
    sgn = sum(1 for i in range(3) if sv[i] > bs[i]); mk = "*" if (sgn == 3 and cg > bcg) else ("+" if sgn == 3 else " ")
    print(f"  {lab:36s} | {100*cg:5.1f} {100*dd:5.1f} | {100*(cg-bcg):+5.1f}{mk} | {sgn}/3", flush=True)
for wr in (0.3, 0.4, 0.5):
    cg, dd, sv = ev(tw=TW, wr_min=wr); row(f"[filter] skip twin<{wr} (all-hist)", cg, dd, sv)
for wr in (0.4, 0.5):
    cg, dd, sv = ev(tw=TWr, wr_min=wr); row(f"[filter] skip twin<{wr} (1yr)", cg, dd, sv)
for sk in (0.5, 1.0, -0.5):
    cg, dd, sv = ev(tw=TW, size_k=sk); row(f"[sizing] w+={sk}*(twin-0.5)", cg, dd, sv)
print("SYMTRACK_DONE", flush=True)
