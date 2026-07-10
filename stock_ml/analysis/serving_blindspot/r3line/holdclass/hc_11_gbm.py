# -*- coding: utf-8 -*-
"""hc_11_gbm: kiem tra phi tuyen — HistGradientBoosting LOYO tren cung feature
set hc_10; + do on dinh top-quintile theo nam. Khep an separator."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASE = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line/holdclass"
X = pd.read_csv(f"{BASE}/hc_sep_features.csv")
FEATS = ["ru5", "ru20", "ru60", "dma20", "dma60", "vol20", "snr60", "volr",
         "rngpos20", "dd252", "fillgap", "breadth", "mkt_ru20", "mkt_vol20"]
y = X.label.values.astype(int)
years = sorted(X.year.unique())

oof = np.full(len(X), np.nan)
print("=== GBM LOYO ===")
for yy in years:
    m_te = (X.year == yy).values
    m_tr = ~m_te
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                         learning_rate=0.05, random_state=42)
    clf.fit(X.loc[m_tr, FEATS], y[m_tr])
    oof[m_te] = clf.predict_proba(X.loc[m_te, FEATS])[:, 1]
    print(f"  {yy}: AUC {roc_auc_score(y[m_te], oof[m_te]):.3f} (n={m_te.sum()})")
m = ~np.isnan(oof)
print(f"OOF pooled: {roc_auc_score(y[m], oof[m]):.3f} | "
      f">=2022: {roc_auc_score(y[m & (X.year>=2022)], oof[m & (X.year>=2022)]):.3f}")

# top-quintile (theo score OOF, cat trong nam) co on dinh edge duong khong?
D = X.loc[m].copy()
D["score"] = oof[m]
D["q_in_year"] = D.groupby("year")["score"].transform(
    lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
top = D[D.q_in_year == D.groupby("year")["q_in_year"].transform("max")]
print("\n=== Top-quintile trong nam (GBM score) — edge40 mean theo nam ===")
rows = []
for yy, g in D.groupby("year"):
    mt = g.q_in_year == g.q_in_year.max()
    rows.append(dict(year=yy, n_top=int(mt.sum()),
                     edge_top=g.loc[mt, "edge40"].mean(),
                     edge_all=g.edge40.mean(),
                     win_top=g.loc[mt, "label"].mean()))
print(pd.DataFrame(rows).set_index("year").round(4).to_string())
