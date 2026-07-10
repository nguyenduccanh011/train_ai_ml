# -*- coding: utf-8 -*-
"""Robustness check cho 3 ung vien T3 borderline: pooled 2024+ IC + null,
va correlation ho feature (residualized) tai exit bars."""
import numpy as np
import pandas as pd
from dx04_deriv_screen import load, build_features, build_market, resid, winz, RNG

N_NULL = 500

fut, st, tr, uni = load()
X, _ = build_features(fut)
for c in X.columns:
    X[c] = winz(X[c])
piv, mret, lvl, C = build_market(st, uni)
X = X.reindex(lvl.index)
C = C.reindex(lvl.index)
Xr = pd.DataFrame({f: resid(X[f], C) for f in X.columns})

ex = tr[tr.exit_date.notna() & tr.rallied.notna()].copy()
fdates = X.dropna(how="all").index
ex["dec_date"] = [fdates[max(int(np.searchsorted(fdates, d)) - 1, 0)] for d in ex.exit_date]
ex["yr"] = ex.exit_date.dt.year

V = Xr.reindex(ex.dec_date).reset_index(drop=True)
V["rallied"] = ex.rallied.astype(float).to_numpy()
V["yr"] = ex.yr.to_numpy()

print("=== corr (Spearman) giua cac feature residualized tai exit bar ===")
fam = ["gap_abs5", "volz20", "co5", "range_c", "rv10", "f1_ret5"]
print(V[fam].corr(method="spearman").round(2).to_string())

for lo, tag in ((2022, "pooled >=2022"), (2024, "pooled 2024+")):
    sub = V[V.yr >= lo]
    print(f"\n=== T3 {tag} (n={len(sub)}, rallied rate {sub.rallied.mean():.2f}) ===")
    for f in X.columns:
        m = sub[f].notna()
        ic = float(sub.loc[m, f].rank().corr(sub.loc[m, "rallied"].rank()))
        stats = []
        lab = sub["rallied"].to_numpy(float)
        yrs = sub["yr"].to_numpy()
        vals = sub[f].to_numpy(float)
        mm = ~np.isnan(vals)
        for _ in range(N_NULL):
            lp = lab.copy()
            for y in set(yrs):
                my = yrs == y
                lp[my] = RNG.permutation(lp[my])
            stats.append(float(pd.Series(vals[mm]).rank().corr(pd.Series(lp[mm]).rank())))
        nlo, nhi = np.percentile(stats, [2.5, 97.5])
        flag = "  <-- ngoai null" if (ic < nlo or ic > nhi) else ""
        print(f"{f:15s} IC {ic:+.4f}  null ({nlo:+.4f},{nhi:+.4f}){flag}")
