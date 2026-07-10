# -*- coding: utf-8 -*-
"""SCORE AUDIT buoc 2 — cau truc tuong quan.

  a) Ma tran corr (Pearson, panel symbol-ngay) giua z cua 6 chuoi: pooled + tung nam.
  b) 4 ensemble co thoai hoa thanh 1 tin hieu? -> corr trung binh cap (z2..z5) theo nam
     + corr cua chi bao QUYET DINH (buy2..buy5, phi Pearson tren nhi phan).
  c) Entry vs exit head: corr(zE, zX) theo nam.
  d) Score vs force-gate state: corr z voi f_dl12, bma20, nonbull, mkt_drop_z, breadth.
Out: sa02_corr_pooled.csv, sa02_pairmean.csv + bang in.
"""
import os

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\scoreaudit"
Z = ["z_score", "z_score2", "z_score3", "z_score4", "z_score5", "z_exit_score"]
ENS = ["z_score2", "z_score3", "z_score4", "z_score5"]
BUYS = ["buy2", "buy3", "buy4", "buy5"]
STATE = ["f_dl12", "bma20", "nonbull", "mkt_drop_z", "breadth"]

ph = pd.read_parquet(os.path.join(OUT, "sa_scores.parquet"))
ph["year"] = ph.date.str[:4]
ph.loc[ph.date >= "2026-01-01", "year"] = "2026H1"
for c in ["f_dl12", "bma20", "nonbull"]:
    ph[c] = ph[c].astype(float)

pd.set_option("display.width", 250)
cm = ph[Z].corr()
cm.to_csv(os.path.join(OUT, "sa02_corr_pooled.csv"))
print("==== corr z pooled ====")
print(cm.round(3).to_string())

rows = []
for y, g in ph.groupby("year"):
    c = g[Z].corr()
    # trung binh corr cap giua 4 ensemble
    vals = [c.loc[a, b] for i, a in enumerate(ENS) for b in ENS[i + 1:]]
    bc = g[BUYS].astype(float).corr()
    bvals = [bc.loc[a, b] for i, a in enumerate(BUYS) for b in BUYS[i + 1:]]
    # % ngay-buy cua union den tu >=2 ensemble cung luc (do trung lap quyet dinh)
    nb = g[BUYS].astype(int).sum(axis=1)
    anyb = nb > 0
    multi = (nb >= 2)[anyb].mean() if anyb.any() else np.nan
    rows.append(dict(
        year=y, ens_pair_mean=np.mean(vals), ens_pair_max=np.max(vals),
        buy_pair_phi_mean=np.mean(bvals), pct_multibuy=multi * 100,
        zE_zX=c.loc["z_score", "z_exit_score"],
        z2_z5=c.loc["z_score2", "z_score5"], z3_z4=c.loc["z_score3", "z_score4"],
        z4_z5=c.loc["z_score4", "z_score5"]))
pm = pd.DataFrame(rows)
pm.to_csv(os.path.join(OUT, "sa02_pairmean.csv"), index=False)
print("\n==== do thoai hoa ensemble + entry-vs-exit theo nam ====")
print(pm.round(3).to_string(index=False))

print("\n==== corr z vs force-gate state (pooled) ====")
sc = ph[Z + STATE].corr().loc[Z, STATE]
print(sc.round(3).to_string())

print("\n==== corr z_exit_score vs state theo nam ====")
rows = []
for y, g in ph.groupby("year"):
    c = g[Z + STATE].corr()
    rows.append(dict(year=y, **{s: c.loc["z_exit_score", s] for s in STATE},
                     zE_dl12=c.loc["z_score", "f_dl12"]))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# de bo sung EXIT_ATTRIBUTION: khi sell_ml bat, force da bat bao nhieu %?
print("\n==== sell_ml vs sell_force overlap theo nam ====")
for y, g in ph.groupby("year"):
    sm = g.sell_ml
    if sm.sum() == 0:
        print(" %s sell_ml=0" % y)
        continue
    print(" %s n_sell_ml=%d | force cung bar %.1f%% | force-truoc-hoac-cung (trong 3 bar) %.1f%%" % (
        y, sm.sum(), 100 * g.sell_force[sm].mean(),
        100 * (g.sell_force.rolling(4, min_periods=1).max())[sm].mean()))
