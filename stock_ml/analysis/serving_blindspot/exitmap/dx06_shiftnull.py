# -*- coding: utf-8 -*-
"""Null circular-shift theo NGAY cho T3 (dung cho feature persistent):
shift chuoi feature k ngay (k random >=30) roi sample lai tai decision bars.
Giu nguyen autocorrelation cua feature VA clustering cua label rallied."""
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

for lo, tag in ((2022, ">=2022"), (2024, "2024+")):
    sub = ex[ex.yr >= lo]
    print(f"\n=== T3 shift-null {tag} (n={len(sub)}) ===")
    lab = sub.rallied.astype(float).to_numpy()
    for f in ["slope", "backwd_streak", "gap_abs5", "volz20", "co5", "range_c"]:
        xs = Xr[f]
        vals = xs.reindex(sub.dec_date).to_numpy(float)
        m = ~np.isnan(vals)
        ic = float(pd.Series(vals[m]).rank().corr(pd.Series(lab[m]).rank()))
        arr = xs.to_numpy(float)
        n = len(arr)
        stats = []
        for _ in range(N_NULL):
            k = int(RNG.integers(30, n - 30))
            xsh = pd.Series(np.roll(arr, k), index=xs.index)
            vp = xsh.reindex(sub.dec_date).to_numpy(float)
            mp = ~np.isnan(vp)
            if mp.sum() < 100:
                continue
            stats.append(float(pd.Series(vp[mp]).rank().corr(pd.Series(lab[mp]).rank())))
        nlo, nhi = np.percentile(stats, [2.5, 97.5])
        flag = "  <-- ngoai null" if (ic < nlo or ic > nhi) else ""
        print(f"{f:15s} IC {ic:+.4f}  shift-null ({nlo:+.4f},{nhi:+.4f}){flag}")
