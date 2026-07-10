# -*- coding: utf-8 -*-
"""hc_10_separator: separator ex-ante tai bar ENTRY cho 1.027 lenh capped mh16.

Label: hold_win40 (giu tiep den tran 40 THANG control chain recycle — tu hc_01).
Feature: chi dung du lieu <= bar entry (OHLCV db; khong co VNINDEX -> market
state = breadth/median universe cua chinh cohort symbols).

Do: AUC per-feature (overall + fold theo nam >=2022), logistic combo
leave-one-year-out, decile lift, null-check permutation (n=200).
Doi chieu an runaway-separator (null).
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line/holdclass"
DB = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
S0 = 0.001
rng = np.random.default_rng(42)

ext = pd.read_csv(f"{BASE}/hc_capped_ext.csv")
tr = pd.read_csv("f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line/r3_mh16_s42_trades.csv")
tr["entry_date"] = tr.entry_date.astype(str).str[:10]
ext = ext.merge(tr[["symbol", "entry_date", "entry_price"]].drop_duplicates(
    ["symbol", "entry_date"]), on=["symbol", "entry_date"], how="left")

syms = sorted(set(ext.symbol))
con = sqlite3.connect(DB)
px = pd.read_sql_query(
    "SELECT symbol,date,high,low,close,volume FROM ohlcv WHERE symbol IN (%s) "
    "AND date>='2018-06-01'" % ",".join("?" * len(syms)), con, params=syms)
con.close()
px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
g = px.groupby("symbol")
px["ret1"] = g["close"].pct_change()
px["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
px["ma60"] = g["close"].transform(lambda x: x.rolling(60).mean())
px["ru5"] = g["close"].pct_change(5)
px["ru20"] = g["close"].pct_change(20)
px["ru60"] = g["close"].pct_change(60)
px["vol20"] = g["ret1"].transform(lambda x: x.rolling(20).std())
px["v5"] = g["volume"].transform(lambda x: x.rolling(5).mean())
px["v20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
px["hi252"] = g["close"].transform(lambda x: x.rolling(252, min_periods=60).max())
px["lo20"] = g["close"].transform(lambda x: x.rolling(20).min())
px["hi20"] = g["close"].transform(lambda x: x.rolling(20).max())
px["above20"] = (px.close > px.ma20).astype(float)

mkt = px.groupby("date").agg(breadth=("above20", "mean"),
                             mkt_ru20=("ru20", "median"),
                             mkt_vol20=("vol20", "median")).reset_index()
px = px.merge(mkt, on="date", how="left")

feat = px.set_index(["symbol", "date"])
rows = []
for r in ext.itertuples():
    try:
        f = feat.loc[(r.symbol, r.entry_date)]
    except KeyError:
        continue
    rows.append(dict(
        tid=r.tid, year=int(str(r.entry_date)[:4]), label=bool(r.hold_win40),
        edge40=r.edge40,
        ru5=f.ru5, ru20=f.ru20, ru60=f.ru60,
        dma20=f.close / f.ma20 - 1, dma60=f.close / f.ma60 - 1,
        vol20=f.vol20,
        snr60=f.ru60 / (f.vol20 * np.sqrt(60)) if f.vol20 > 0 else 0.0,
        volr=f.v5 / f.v20 if f.v20 > 0 else 1.0,
        rngpos20=(f.close - f.lo20) / (f.hi20 - f.lo20) if f.hi20 > f.lo20 else 0.5,
        dd252=f.close / f.hi252 - 1,
        fillgap=r.entry_price / (1 + S0) / f.close - 1,
        breadth=f.breadth, mkt_ru20=f.mkt_ru20, mkt_vol20=f.mkt_vol20))
X = pd.DataFrame(rows).dropna()
FEATS = ["ru5", "ru20", "ru60", "dma20", "dma60", "vol20", "snr60", "volr",
         "rngpos20", "dd252", "fillgap", "breadth", "mkt_ru20", "mkt_vol20"]
y = X.label.values
print(f"n={len(X)} (mat {len(ext)-len(X)} do thieu feature) | base rate {y.mean()*100:.1f}%")

print("\n=== AUC per-feature (flip ve >=0.5) ===")
years = sorted(X.year.unique())
yr_test = [yy for yy in years if yy >= 2022]
hdr = "feature      overall " + " ".join(f"{yy}" for yy in yr_test)
print(hdr)
for f in FEATS:
    a = roc_auc_score(y, X[f])
    a = max(a, 1 - a)
    per = []
    for yy in yr_test:
        m = X.year == yy
        if m.sum() > 10 and 0 < y[m].mean() < 1:
            ay = roc_auc_score(y[m], X.loc[m, f])
            per.append(f"{max(ay,1-ay):.3f}")
        else:
            per.append("  -  ")
    print(f"{f:12s} {a:.3f}   " + " ".join(per))

print("\n=== Logistic combo — leave-one-year-out ===")
aucs = {}
oof = np.full(len(X), np.nan)
for yy in years:
    m_te = (X.year == yy).values
    m_tr = ~m_te
    if m_te.sum() < 15 or 0 in (y[m_te].mean(), 1 - y[m_te].mean()):
        continue
    sc = StandardScaler().fit(X.loc[m_tr, FEATS])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(X.loc[m_tr, FEATS]), y[m_tr])
    p = clf.predict_proba(sc.transform(X.loc[m_te, FEATS]))[:, 1]
    oof[m_te] = p
    aucs[yy] = roc_auc_score(y[m_te], p)
for yy, a in aucs.items():
    n = (X.year == yy).sum()
    print(f"  {yy}: AUC {a:.3f} (n={n})")
m_all = ~np.isnan(oof)
auc_oof = roc_auc_score(y[m_all], oof[m_all])
m22 = m_all & (X.year >= 2022).values
auc_22 = roc_auc_score(y[m22], oof[m22])
print(f"OOF pooled: {auc_oof:.3f} | pooled >=2022: {auc_22:.3f}")

# null-check permutation: xao label trong tung nam, giu pipeline LOYO
print("\n=== Null permutation (n=200, LOYO pooled AUC) ===")
null_aucs = []
Xf = X[FEATS].values
for it in range(200):
    yp = y.copy()
    for yy in years:
        m = (X.year == yy).values
        yp[m] = rng.permutation(yp[m])
    oofp = np.full(len(X), np.nan)
    for yy in aucs:
        m_te = (X.year == yy).values
        m_tr = ~m_te
        sc = StandardScaler().fit(Xf[m_tr])
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(sc.transform(Xf[m_tr]), yp[m_tr])
        oofp[m_te] = clf.predict_proba(sc.transform(Xf[m_te]))[:, 1]
    mm = ~np.isnan(oofp)
    null_aucs.append(roc_auc_score(yp[mm], oofp[mm]))
null_aucs = np.array(null_aucs)
pval = (null_aucs >= auc_oof).mean()
print(f"null mean {null_aucs.mean():.3f} p95 {np.percentile(null_aucs,95):.3f} "
      f"| observed {auc_oof:.3f} | p={pval:.3f}")

# decile lift theo OOF score: mean edge40 per decile
print("\n=== Decile lift (OOF score -> mean edge40, win rate) ===")
D = X.loc[m_all].copy()
D["score"] = oof[m_all]
D["dec"] = pd.qcut(D.score, 10, labels=False, duplicates="drop")
print(D.groupby("dec").agg(n=("label", "size"), win=("label", "mean"),
                           edge40=("edge40", "mean")).round(4).to_string())
X.assign(oof=oof).to_csv(f"{BASE}/hc_sep_features.csv", index=False)
print("\nsaved hc_sep_features.csv")
