# -*- coding: utf-8 -*-
"""CHURN forensic: entries that die in 0-1 sessions waste a slot + roundtrip fee. Question (user): can we
detect AT THE BUY-SIGNAL SESSION (causal, no lookahead) that a SELL signal is present/imminent, and skip
those entries? run_signals stores BOTH score AND exit_score per (symbol,date) -> exit_score AT the buy
signal date IS the model's sell-strength, fully causal. Also test price-extension features (overext/rsi).
Measure: (A) churn economics, (B) causal predictors AUC, (C) NAV impact of skipping, T+0 and T+2, 3-seed."""
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
K, MARGIN, KCONV, SIG = 10, 0.005, 2.0, "cs5_ma50"   # top operating point


def prun(sim, pm, cm, tplus, skipmap=None, skip_thr=None, skip_above=True, advance_fee=0.0008, roundtrip=0.006):
    """skipmap/skip_thr: skip entries whose causal feature at signal date is >thr (skip_above) or <thr."""
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
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
        if skipmap is not None:
            v = skipmap.get((t["symbol"], t["entry_date"]), np.nan)
            if not np.isnan(v) and ((v > skip_thr) if skip_above else (v < skip_thr)):
                continue                                         # causal skip
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
                cand = [l for l in legs if (di - l["be_di"]) >= tplus]
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


# ---- causal price features at signal date (extension / overbought) ----
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    g["ext20h"] = c / h.rolling(20).max() - 1; g["ret5"] = c / c.shift(5) - 1     # extension / recent runup
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "ext20h", "ret5"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
FEATd = {(r.symbol, str(r.date.date())): r for r in P.itertuples()}   # for AUC feature lookup by signal date

SKIPFEATS = ["dist_ma20", "rsi14", "ext20h", "ret5"]
con = psycopg2.connect(**PG); feat = None
cv_c, cm_c, pm, es_c, diag_rows = {}, {}, {}, {}, []
fmap_c = {f: {} for f in SKIPFEATS}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,exit_reason from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    sig = pd.read_sql("select symbol,date,exit_score from run_signals where run_id=%s and signal=1", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    sig["date"] = pd.to_datetime(sig["date"])
    # exit_score at the BUY signal date (causal). map keyed by (symbol, entry_date_fill)
    esd = {(r.symbol, str(r.date.date())): (float(r.exit_score) if pd.notna(r.exit_score) else np.nan) for r in sig.itertuples()}
    cm_c[sd] = {(r.symbol, r.ed): CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in cvtr.itertuples()}
    es_c[sd] = {(r.symbol, r.ed): esd.get((r.symbol, str(r.sigd.date())), np.nan) for r in cvtr.itertuples()}
    for r in cvtr.itertuples():
        ft = FEATd.get((r.symbol, str(r.sigd.date())))
        for f in SKIPFEATS:
            fmap_c[f].setdefault(sd, {})[(r.symbol, r.ed)] = getattr(ft, f, np.nan) if ft else np.nan
    cvf = HERE / f"_ch_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    # sessions held from NavSim (i1-i0)
    sim0 = NavSim2(str(cvf), date_lo="2020-01-01")
    sess = {(t["symbol"], t["entry_date"]): t["i1"] - t["i0"] for t in sim0.trades}
    net = {(t["symbol"], t["entry_date"]): (t["x_raw"] * (1 - (0.006 - FEE) / 2) / (t["e_raw"] * (1 + (0.006 - FEE) / 2)) - 1 - FEE) for t in sim0.trades}
    for r in cvtr.itertuples():
        k = (r.symbol, r.ed); ssd = str(r.sigd.date())
        if k not in sess:
            continue
        ft = FEATd.get((r.symbol, ssd))
        diag_rows.append(dict(seed=sd, symbol=r.symbol, sessions=sess[k], net=net.get(k, np.nan),
                              exit_score=es_c[sd][k], exit_reason=r.exit_reason,
                              dist_ma20=getattr(ft, "dist_ma20", np.nan) if ft else np.nan,
                              rsi14=getattr(ft, "rsi14", np.nan) if ft else np.nan,
                              ext20h=getattr(ft, "ext20h", np.nan) if ft else np.nan,
                              ret5=getattr(ft, "ret5", np.nan) if ft else np.nan))
con.close()
D = pd.DataFrame(diag_rows)

# ============ (A) churn economics ============
print("=== (A) HOLDING-SESSION distribution + economics (3-seed pooled base trades) ===", flush=True)
print(f"  total trades={len(D)}", flush=True)
for lo, hi, lab in [(0, 0, "0 phiên (cùng phiên)"), (1, 1, "1 phiên"), (2, 2, "2 phiên"), (3, 5, "3-5 phiên"), (6, 999, "6+ phiên")]:
    m = (D.sessions >= lo) & (D.sessions <= hi); n = int(m.sum())
    if not n:
        continue
    sub = D[m]; feedrag = n * 0.006
    print(f"  {lab:20s} n={n:4d} ({100*n/len(D):4.1f}%) | net TB {100*sub.net.mean():+5.2f}% | net TỔNG {100*sub.net.sum():+6.1f}% | phí ~{100*feedrag:.0f}%", flush=True)
# "churn" thật = giữ NGẮN & lỗ: nhóm 3-5 phiên net âm. Label = sessions<=5
churn = D.sessions <= 5
print(f"\n  SHORT-HOLD (<=5 phiên): {int(churn.sum())} lệnh ({100*churn.mean():.1f}%) | net TB {100*D[churn].net.mean():+.2f}% | net TỔNG {100*D[churn].net.sum():+.1f}%", flush=True)
print(f"  exit_reason nhóm short-hold: {D[churn].exit_reason.value_counts().head(8).to_dict()}", flush=True)

# ============ (B) causal predictors of short-hold (AUC) ============
from sklearn.metrics import roc_auc_score
print("\n=== (B) CAUSAL predictors of SHORT-HOLD-loss (AUC, tại NGÀY tín hiệu mua) ===", flush=True)
y = churn.astype(int).values
for col in ["exit_score", "dist_ma20", "rsi14", "ext20h", "ret5"]:
    v = D[col].values.astype(float); ok = ~np.isnan(v)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        print(f"  {col:12s} — n/a", flush=True); continue
    a = roc_auc_score(y[ok], v[ok]); a = max(a, 1 - a); dirn = "cao=>ngắn" if roc_auc_score(y[ok], v[ok]) > 0.5 else "thấp=>ngắn"
    print(f"  {col:12s} AUC={a:.3f}  ({dirn})", flush=True)

# ============ (C) NAV impact of causal skip (exit_score + best price feats) ============
def ev(tplus, smap_by_seed=None, thr=None, above=True):
    cg, dd = [], []
    for sd in SEEDS:
        c, d = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cm_c[sd], tplus,
                    skipmap=(smap_by_seed[sd] if smap_by_seed else None), skip_thr=thr, skip_above=above)
        cg.append(c); dd.append(d)
    return statistics.mean(cg), statistics.mean(dd)


print("\n=== (C) NAV khi lọc entry theo feature causal (top model K10/cs5_ma50, 3-seed) ===", flush=True)
CANDS = [("rsi14 THẤP", fmap_c["rsi14"], "rsi14", False), ("dist_ma20 THẤP", fmap_c["dist_ma20"], "dist_ma20", False),
         ("ext20h THẤP", fmap_c["ext20h"], "ext20h", False), ("ret5 THẤP", fmap_c["ret5"], "ret5", False)]
for tplus, tl in [(0, "T+0"), (2, "T+2")]:
    b_cg, b_dd = ev(tplus)
    print(f"  [{tl}] baseline: CAGR {100*b_cg:.1f}% / DD {100*b_dd:.1f}%", flush=True)
    for lab, smap, col, above in CANDS:
        v = D[col].dropna().values
        for p in (10, 20):
            thr = float(np.nanpercentile(v, 100 - p if above else p))
            cg, dd = ev(tplus, smap, thr, above)
            print(f"    skip {lab:16s} top{p}% (>{thr:+.3f}): CAGR {100*cg:.1f}% / DD {100*dd:.1f}%  ({100*(cg-b_cg):+.1f}pp)", flush=True)

# ============ (D) characterize ret5-skip under T+2 (CAGR+DD+% loại) ============
print("\n=== (D) ret5<thr dưới T+2 (VN thực tế) — CAGR/DD per-seed + % lệnh loại ===", flush=True)
frac = {thr: float((D.ret5 < thr).mean()) for thr in (0.0, 0.02, 0.03, 0.05)}
base = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cm_c[sd], 2) for sd in SEEDS]
print(f"  baseline: CAGR {['%.0f' % (100*c) for c, d in base]} / DD {['%.0f' % (100*d) for c, d in base]}", flush=True)
for thr in (0.0, 0.02, 0.03, 0.05):
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cm_c[sd], 2, skipmap=fmap_c["ret5"][sd], skip_thr=thr, skip_above=False) for sd in SEEDS]
    dcg = [rr[i][0] - base[i][0] for i in range(3)]; sgn = sum(1 for x in dcg if x > 0)
    print(f"  ret5<{thr:.2f} (loại {100*frac[thr]:.0f}% book): CAGR {['%.0f' % (100*c) for c, d in rr]} (Δ {['%+.0f' % (100*x) for x in dcg]}, {sgn}/3) / DD {['%.0f' % (100*d) for c, d in rr]}", flush=True)
print("CHURN_FORENSIC_DONE", flush=True)
