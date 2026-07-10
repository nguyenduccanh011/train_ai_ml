# -*- coding: utf-8 -*-
"""P3 Phase C — IC screen v2: feature INTRADAY THAT (p3b) vs target dau exit.

Tai dung nguyen harness dx04_deriv_screen (load/build_market/build_targets/resid/
fold_ics/null_band — time-series Spearman theo fold nam, residualize tren 8 control
daily he DA thay, null circular-shift) + dx06 shift-null cho T3 event-level.

Khac dx04: X = feature intraday that tu p3b_intraday_features.parquet.
Lo 2023-03..09: feature NaN -> fold 2023 chi con ~113 ngay co du lieu (>=30 obs
van tinh IC, ghi ro n); null band tinh tren cap (x,y) da dropna nhu dx04.
N_NULL shift = 200 (theo de bai; dx04 dung 300 — giu 300 cua null_band goc,
T3 dung 200).
"""
import sys
import numpy as np
import pandas as pd

EM = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"
P3 = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml\p3_probe"
sys.path.insert(0, EM)

from dx04_deriv_screen import (load, build_market, build_targets, resid, winz,
                               fold_ics, null_band, FOLDS, RNG)

N_NULL_T3 = 200
FEATS = ["rv_park", "rv_5m", "mom_30", "mom_60", "mom_pm", "vol_imb_pm",
         "gap_open", "range_c_id", "range_comp", "chop5", "maxdd_15m",
         "vwap_dev", "atc_volshare", "slope_id", "backwd_id"]


def main():
    fut, st, tr, uni = load()
    X = pd.read_parquet(P3 + r"\p3b_intraday_features.parquet")[FEATS]
    for c in X.columns:
        X[c] = winz(X[c])
    piv, mret, lvl, C = build_market(st, uni)
    cal = lvl.index
    X = X.reindex(cal)
    C = C.reindex(cal)
    Y = build_targets(piv[sorted(set(tr.symbol) & set(piv.columns))], lvl, tr)
    print(f"features {X.shape}, targets {Y.shape}, controls {C.shape}")
    print("feature coverage theo fold (rv_park):",
          {n: int(X.loc[(X.index >= ts) & (X.index < te), 'rv_park'].notna().sum())
           for n, ts, te in FOLDS}, flush=True)

    rows = []
    for feat in X.columns:
        x = X[feat]
        xr = resid(x, C)
        for tgt in Y.columns:
            y = Y[tgt]
            raw = fold_ics(x, y)
            ort = fold_ics(xr, y)
            nb_raw = null_band(x, y)
            nb_ort = null_band(xr, y)
            nb24 = null_band(xr, y, lo="2024-01-01")
            rows.append({"feature": feat, "target": tgt,
                         **{f"raw_{k}": v for k, v in raw.items()},
                         **{f"ort_{k}": v for k, v in ort.items()},
                         "raw_null": nb_raw, "ort_null": nb_ort, "ort_null24": nb24})
            print(f"{feat:14s} {tgt:14s} raw {raw['pooled']:+.3f} "
                  f"ort {ort['pooled']:+.3f} ort24 {ort['pool24']:+.3f} "
                  f"null {nb_ort} null24 {nb24}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(P3 + r"\p3c_ic_v2.csv", index=False)

    # ---------------- T3 rallied: event-level + shift-null (khuon dx06) ----------
    Xr = pd.DataFrame({f: resid(X[f], C) for f in X.columns})
    ex = tr[tr.exit_date.notna() & tr.rallied.notna()].copy()
    fdates = X.dropna(how="all").index
    ex["dec_date"] = [fdates[max(int(np.searchsorted(fdates, d)) - 1, 0)]
                      for d in ex.exit_date]
    ex["yr"] = ex.exit_date.dt.year
    t3rows = []
    for lo, tag in ((2022, ">=2022"), (2024, "2024+")):
        sub = ex[ex.yr >= lo]
        lab = sub.rallied.astype(float).to_numpy()
        print(f"\n=== T3 shift-null {tag} (n={len(sub)}) ===", flush=True)
        for f in FEATS:
            xs = Xr[f]
            vals = xs.reindex(sub.dec_date).to_numpy(float)
            m = ~np.isnan(vals)
            if m.sum() < 100:
                continue
            ic = float(pd.Series(vals[m]).rank().corr(pd.Series(lab[m]).rank()))
            arr = xs.to_numpy(float)
            n = len(arr)
            stats = []
            for _ in range(N_NULL_T3):
                k = int(RNG.integers(30, n - 30))
                xsh = pd.Series(np.roll(arr, k), index=xs.index)
                vp = xsh.reindex(sub.dec_date).to_numpy(float)
                mp = ~np.isnan(vp)
                if mp.sum() < 100:
                    continue
                stats.append(float(pd.Series(vp[mp]).rank()
                                   .corr(pd.Series(lab[mp]).rank())))
            nlo, nhi = np.percentile(stats, [2.5, 97.5])
            out = ic < nlo or ic > nhi
            t3rows.append({"slice": tag, "feature": f, "n": int(m.sum()),
                           "ic": round(ic, 4), "null_lo": round(float(nlo), 4),
                           "null_hi": round(float(nhi), 4), "outside": out})
            print(f"{f:14s} IC {ic:+.4f} (n={m.sum()}) shift-null "
                  f"({nlo:+.4f},{nhi:+.4f}){'  <-- NGOAI NULL' if out else ''}",
                  flush=True)
    pd.DataFrame(t3rows).to_csv(P3 + r"\p3c_t3_rallied_v2.csv", index=False)
    print("\nsaved p3c_ic_v2.csv / p3c_t3_rallied_v2.csv", flush=True)


if __name__ == "__main__":
    main()
