# -*- coding: utf-8 -*-
"""HY manh B: zX (reward_risk h10, bundle n2_pbfill seed42 WF) lam CONG-GIU cho gb_x08?

Cau hoi: tai decision bar (bar truoc exit_date) cua cac lenh gb_x08 bi dong,
zX co TACH duoc "dong dung" khoi "dong non" (rallied = post_max_c>=5%/20bar)?
Hypothesis defer: zX THAP = song con khoe = dang le nen GIU -> rallied.
Do: AUC(zX -> rallied) per year_exit + decile lift; cohort: (a) 648 force_downleg12,
(b) toan bo 1378, (c) 663 rallied vs 715 khong.
"""
import os

import numpy as np
import pandas as pd

BASE = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
BUNDLE = r"f:/PROJECTS/train_ai_ml/bundles/bundle_n2_pbfill_p0p03_w25_2026-01-01_wf/prediction_history.parquet"
W, MP = 252, 60

tr = pd.read_csv(os.path.join(BASE, "exitmap", "gbx08_enriched2.csv"),
                 parse_dates=["entry_date", "exit_date"])
tr = tr[tr.exit_reason != "open"].copy()
print("trades:", len(tr), "| rallied:", int(tr.rallied.sum()),
      "| dl12:", int((tr.label == "force_downleg12").sum()))

ph = pd.read_parquet(BUNDLE, columns=["symbol", "date", "exit_score"])
ph["date"] = pd.to_datetime(ph["date"])
ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)


def causal_z(s, g):
    def _cz(x):
        m = x.rolling(W, min_periods=MP).mean()
        sd = x.rolling(W, min_periods=MP).std()
        return (x - m) / (sd + 1e-12)
    return s.groupby(g, sort=False).transform(_cz)


ph["zx"] = causal_z(ph["exit_score"], ph["symbol"])
CAL = {s: g.set_index("date")["zx"] for s, g in ph.groupby("symbol")}

zx_dec = np.full(len(tr), np.nan)
for k, (sym, xd) in enumerate(zip(tr.symbol.to_numpy(), tr.exit_date.to_numpy())):
    ser = CAL.get(sym)
    if ser is None:
        continue
    idx = ser.index.searchsorted(xd)
    # decision bar = bar truoc exit_date tren lich cua ma
    if idx >= 1:
        zx_dec[k] = ser.iloc[idx - 1]
tr["zx_dec"] = zx_dec
tr = tr.dropna(subset=["zx_dec"]).copy()
print("co zX tai decision bar:", len(tr))


def auc(score, label):
    """AUC cua score du bao label=1 (rank-based)."""
    s = np.asarray(score, float); y = np.asarray(label, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(df, name):
    print(f"\n=== {name} (n={len(df)}, rallied {df.rallied.mean()*100:.0f}%) ===")
    rows = []
    for label, sub in [("ALL", df)] + [(str(y), g) for y, g in df.groupby("year_exit")]:
        a = auc(-sub.zx_dec, sub.rallied)  # zX thap -> rallied (defer direction)
        rows.append((label, len(sub), int(sub.rallied.sum()), a,
                     sub.loc[sub.rallied, "zx_dec"].median(),
                     sub.loc[~sub.rallied, "zx_dec"].median()))
    print(f"{'slice':>6} {'n':>5} {'n_ral':>5} {'AUC(-zX)':>9} {'zX_med_ral':>11} {'zX_med_ok':>10}")
    for r in rows:
        print(f"{r[0]:>6} {r[1]:>5} {r[2]:>5} {r[3]:>9.3f} {r[4]:>11.2f} {r[5]:>10.2f}")
    # slice >=2022 / >=2024 pooled
    for cut in (2022, 2024):
        sub = df[df.year_exit >= cut]
        print(f"  pooled >={cut}: n={len(sub)} AUC(-zX)={auc(-sub.zx_dec, sub.rallied):.3f}")
    # decile lift (pooled): P(rallied) + mean post_max_c theo decile zX
    d = df.copy()
    d["dec"] = pd.qcut(d.zx_dec, 10, labels=False, duplicates="drop")
    piv = d.groupby("dec").agg(n=("rallied", "size"), p_ral=("rallied", "mean"),
                               post_max=("post_max_c", "mean"),
                               zx_lo=("zx_dec", "min"), zx_hi=("zx_dec", "max")).round(3)
    print("  decile zX (0=thap nhat):")
    print(piv.to_string())


report(tr[tr.label == "force_downleg12"], "COHORT dl12-closed (648)")
report(tr, "COHORT toan bo trades")

# decile theo nam nhom lon (kiem calibration dao theo nam)
d = tr.copy()
d["era"] = np.where(d.year_exit <= 2021, "2020-21",
                    np.where(d.year_exit <= 2023, "2022-23", "2024-26"))
for era, g in d.groupby("era"):
    g = g.copy()
    g["dec"] = pd.qcut(g.zx_dec, 5, labels=False, duplicates="drop")
    piv = g.groupby("dec").agg(n=("rallied", "size"), p_ral=("rallied", "mean"),
                               post_max=("post_max_c", "mean")).round(3)
    print(f"\n  quintile zX @ {era}: AUC(-zX)={auc(-g.zx_dec, g.rallied):.3f}")
    print(piv.to_string())

tr[["symbol", "entry_date", "exit_date", "year_exit", "label", "rallied",
    "post_max_c", "pnl_pct", "zx_dec"]].to_csv(
    os.path.join(BASE, "pureml", "hy_03_zxgate.csv"), index=False)
print("\nsaved hy_03_zxgate.csv")
