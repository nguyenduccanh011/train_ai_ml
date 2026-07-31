# -*- coding: utf-8 -*-
"""RUNAWAY-PREDICTABILITY (user idea): are runaways preceded by tight accumulation / sharp shakeout, then
breakout volume + strong candle? And can a model, at SIGNAL time (pre-signal features only, causal),
predict runaway-vs-pullback OR — the make-or-break — separate breakout-that-RUNS from breakout-that-COLLAPSES?
The causal OCO test showed entering ALL breakouts is net-negative because collapses dominate. If pre-signal
features separate the winners AUC>>0.5 robustly, routing becomes possible. If AUC~0.5, direction is closed.

Signal universe = run_pending (symbol, signal_date, limit_price). Features from OHLCV+volume up to bS only.
Label: OCO outcome (breakout-first@+CONF vs dip-first@limit within WIN); for breakout, normal-exit return.
Diagnostics: (1) cohort mix, (2) user's hypothesis features runaway-win vs runaway-loss mean+AUC,
(3) walk-forward LGBM OOS AUC + decile lift on breakout-return. Read-only, seed-independent."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd
import lightgbm as lgb

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
COMBO_RID = "template/x2_struct_to_k16preempt_cssize-69338138"
ROUNDTRIP = 0.006; WIN = 40; CONF = 0.05
MH, ACT, GB, OVX = 14, 0.27, 0.08, 0.12

con = psycopg2.connect(**PG)
pend = pd.read_sql("SELECT DISTINCT symbol,signal_date,limit_price FROM run_pending WHERE run_id=%s", con, params=(COMBO_RID,))
con.close()
pend["signal_date"] = pd.to_datetime(pend["signal_date"])

cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,open,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND date>='2017-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])

# ---- per-symbol causal features (all use data up to and INCLUDING each bar) ----
FCOLS = ["ret5", "ret20", "ret60", "dist_ma20", "dist_ma50", "dist_ma100", "dist20low", "dist63high",
         "rsi14", "bbwidth", "atrpct", "atr_contract", "donch_w", "volz20", "vol_ratio", "body",
         "close_pos", "updays5", "lowwick5", "shake10"]
CA = {}; HI = {}; LO = {}; BO = {}; DT = {}; FEA = {}
frames = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); rng = (h - l).replace(0, np.nan)
    g2 = pd.DataFrame({"symbol": s, "date": g["date"]})
    g2["ret5"] = c.pct_change(5); g2["ret20"] = c.pct_change(20); g2["ret60"] = c.pct_change(60)
    g2["dist_ma20"] = c / c.rolling(20).mean() - 1; g2["dist_ma50"] = c / c.rolling(50).mean() - 1
    g2["dist_ma100"] = c / c.rolling(100).mean() - 1
    g2["dist20low"] = c / l.rolling(20).min() - 1; g2["dist63high"] = c / h.rolling(63).max() - 1
    g2["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g2["bbwidth"] = c.rolling(20).std() / c.rolling(20).mean()          # tightness (low=tight accumulation)
    g2["atrpct"] = atr / c
    g2["atr_contract"] = atr / atr.shift(20)                            # <1 = volatility contraction (coiling)
    g2["donch_w"] = (h.rolling(20).max() - l.rolling(20).min()) / c     # range compression
    vma = v.rolling(20).mean(); vsd = v.rolling(20).std()
    g2["volz20"] = (v - vma) / (vsd + 1e-9)                             # breakout volume at signal
    g2["vol_ratio"] = v / (vma + 1e-9)
    g2["body"] = (c - o).abs() / rng                                    # candle body strength
    g2["close_pos"] = (c - l) / rng                                     # close near high = strong
    g2["updays5"] = (dd > 0).rolling(5).sum()
    lw = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    g2["lowwick5"] = lw.rolling(5).mean()                               # shakeout lower-wick
    g2["shake10"] = c / l.rolling(10).min() - 1                         # recovery from recent 10-low
    frames.append(g2)
    CA[s] = c.to_numpy(); HI[s] = h.to_numpy(); LO[s] = l.to_numpy(); DT[s] = g["date"].to_numpy()
    BO[s] = {d: i for i, d in enumerate(g["date"])}
F = pd.concat(frames, ignore_index=True)
# cross-sectional RS rank of ret20 per date (causal: same-day ranks)
F["csrank_ret20"] = F.groupby("date")["ret20"].rank(pct=True)
FCOLS = FCOLS + ["csrank_ret20"]
FMAP = {(r.symbol, r.date): r for r in F.itertuples()}


def normal_exit(sym, be):
    """Trailing (gb after peak>=+ACT) + max_hold. Returns gross return."""
    ca, hi = CA[sym], HI[sym]; ep = ca[be]; peak = ep; n = len(ca)
    for i in range(be + 1, min(be + MH, n - 1) + 1):
        peak = max(peak, hi[i])
        if peak >= ep * (1 + ACT) and ca[i] <= peak * (1 - GB):
            return ca[i] / ep - 1
    j = min(be + MH, n - 1)
    return ca[j] / ep - 1


rows = []
for r in pend.itertuples():
    sym = r.symbol; bS = BO.get(sym, {}).get(r.signal_date)
    if bS is None or bS + 1 >= len(CA[sym]):
        continue
    ca, lo = CA[sym], LO[sym]; cS = ca[bS]
    if cS <= 0:
        continue
    thr = cS * (1 + CONF); outcome = "neither"; bc = None
    for i in range(bS + 1, min(bS + WIN, len(ca) - 1) + 1):
        if lo[i] <= r.limit_price:
            outcome = "pullback"; break
        if ca[i] >= thr:
            outcome = "runaway"; bc = i; break
    feat = FMAP.get((sym, pd.Timestamp(DT[sym][bS])))
    if feat is None:
        continue
    ret = np.nan
    if outcome == "runaway" and bc + 1 < len(ca):
        ret = (1 + normal_exit(sym, bc + 1)) * (1 - ROUNDTRIP) - 1
    d = {"symbol": sym, "year": pd.Timestamp(r.signal_date).year, "outcome": outcome, "ret": ret}
    for fc in FCOLS:
        d[fc] = getattr(feat, fc)
    rows.append(d)
D = pd.DataFrame(rows)
print(f"signals: {len(D)}  | runaway={ (D.outcome=='runaway').sum() }  pullback={ (D.outcome=='pullback').sum() }  neither={ (D.outcome=='neither').sum() }", flush=True)
rw = D[D.outcome == "runaway"].dropna(subset=["ret"]).copy()
rw["win"] = (rw.ret > 0).astype(int)
print(f"runaway cohort: n={len(rw)}  win%={100*rw.win.mean():.1f}  avg_ret={100*rw.ret.mean():+.2f}%  "
      f"(win avg {100*rw[rw.win==1].ret.mean():+.1f}% / loss avg {100*rw[rw.win==0].ret.mean():+.1f}%)\n", flush=True)


def auc(a, b):
    a = a.dropna(); b = b.dropna()
    if len(a) < 30 or len(b) < 30:
        return np.nan
    allv = pd.concat([a, b]); rk = allv.rank(); n1 = len(a)
    return (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(b))


print("=== user hypothesis: runaway-WIN vs runaway-LOSS (pre-signal features) ===", flush=True)
print(f"{'feature':14s} | win-mean | loss-mean | |AUC-.5|", flush=True)
res = []
for fc in FCOLS:
    wm = rw[rw.win == 1][fc].mean(); lm = rw[rw.win == 0][fc].mean()
    a = auc(rw[rw.win == 1][fc], rw[rw.win == 0][fc])
    res.append((fc, wm, lm, abs(a - 0.5) if a == a else np.nan))
for fc, wm, lm, sep in sorted(res, key=lambda x: -(x[3] if x[3] == x[3] else -1)):
    print(f"{fc:14s} | {wm:+8.3f} | {lm:+8.3f} | {sep:.3f}", flush=True)

# also runaway vs pullback (does user's shape separate the two paths at all?)
pb = D[D.outcome == "pullback"]
print("\n=== runaway vs pullback path (top separators) ===", flush=True)
rp = sorted([(fc, abs(auc(rw[fc], pb[fc]) - 0.5)) for fc in FCOLS], key=lambda x: -(x[1] if x[1] == x[1] else -1))[:6]
for fc, sep in rp:
    print(f"  {fc:14s} |AUC-.5|={sep:.3f}  (runaway {rw[fc].mean():+.3f} vs pullback {pb[fc].mean():+.3f})", flush=True)

# ---- walk-forward LGBM: predict runaway-WIN from pre-signal features, OOS AUC + decile lift ----
print("\n=== walk-forward LGBM predict runaway-WIN (OOS), decile lift on breakout-return ===", flush=True)
rw = rw.sort_values("year").reset_index(drop=True); oof = np.full(len(rw), np.nan)
for ty in range(2021, 2027):
    tr_idx = rw.index[rw.year < ty]; te_idx = rw.index[rw.year == ty]
    if len(tr_idx) < 200 or len(te_idx) < 20:
        continue
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, min_child_samples=50, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8, verbose=-1)
    m.fit(rw.loc[tr_idx, FCOLS], rw.loc[tr_idx, "win"])
    oof[te_idx] = m.predict_proba(rw.loc[te_idx, FCOLS])[:, 1]
rw["pred"] = oof; ok = rw.dropna(subset=["pred"])
A = auc(ok[ok.win == 1]["pred"], ok[ok.win == 0]["pred"])
print(f"  OOS AUC (win prediction) = {A:.3f}   (n_oos={len(ok)})", flush=True)
ok = ok.copy(); ok["dec"] = pd.qcut(ok["pred"], 10, labels=False, duplicates="drop")
g = ok.groupby("dec").agg(n=("ret", "size"), avg_ret=("ret", "mean"), win=("win", "mean"))
print("  decile | n | avg breakout-return | win%", flush=True)
for dec, r in g.iterrows():
    print(f"    d{int(dec):02d} | {int(r.n):4d} | {100*r.avg_ret:+6.2f}% | {100*r.win:4.0f}%", flush=True)
print(f"\n  top-decile avg_ret {100*g.iloc[-1].avg_ret:+.2f}% vs bottom {100*g.iloc[0].avg_ret:+.2f}%  "
      f"spread {100*(g.iloc[-1].avg_ret-g.iloc[0].avg_ret):+.2f}%", flush=True)
print("(AUC>~0.58 robust + monotone decile lift = runaway predictable -> routing viable; ~0.5 = closed)")
print("RUNAWAY_PREDICT_DONE", flush=True)
