# -*- coding: utf-8 -*-
"""SCORE AUDIT buoc 3 — calibration curve tung nam + oracle threshold (chi do TRAN).

ENTRY (5 kenh): tren cac bar hop le (entry: sau gate neu kenh chinh):
  - decile z theo nam -> mean fwd20 (%).
  - diem van hanh: percentile cua threshold trong phan phoi z nam do; mean fwd20 excess
    cua tap chon; oracle t* = argmax tong-excess-fwd20 (don vi %-point x luot chon).
EXIT (z_exit_score): chi tren bar dang-mo (tu gbx08_s42_trades) va truoc gate cons2_w20
  + sau gate — outcome = -fwd10 (ban tot = fwd10 thap). Oracle t* = argmax tong loss-avoided.
Out: sa03_entry_calib.csv, sa03_entry_oracle.csv, sa03_exit_calib.csv, sa03_exit_oracle.csv
"""
import os

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\scoreaudit"
EM = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"
CH = {"z_score": (-1.9, "gate_entry"), "z_score2": (0.9, None), "z_score3": (0.7, None),
      "z_score4": (0.7, None), "z_score5": (0.7, None)}

ph = pd.read_parquet(os.path.join(OUT, "sa_scores.parquet"))
ph["year"] = ph.date.str[:4]
ph.loc[ph.date >= "2026-01-01", "year"] = "2026H1"
g = ph.groupby("symbol", sort=False)
ph["fwd20"] = g["close"].shift(-20) / ph["close"] - 1.0
ph["fwd10"] = g["close"].shift(-10) / ph["close"] - 1.0

# ---------------- ENTRY ----------------
cal_rows, orc_rows = [], []
for zc, (thr, gate) in CH.items():
    base = ph.dropna(subset=[zc, "fwd20"])
    if gate:
        base = base[base[gate]]
    for y, gy in base.groupby("year"):
        if len(gy) < 500:
            continue
        d = pd.qcut(gy[zc], 10, labels=False, duplicates="drop")
        dec = gy.groupby(d)["fwd20"].mean() * 100
        cal_rows.append(dict(score=zc, year=y, **{"d%d" % i: dec.get(i, np.nan) for i in range(10)}))
        # oracle threshold: grid quantile
        zv = gy[zc].to_numpy()
        fw = gy["fwd20"].to_numpy()
        mu = fw.mean()
        grid = np.quantile(zv, np.linspace(0.02, 0.98, 49))
        vals = [((fw[zv > t] - mu).sum() if (zv > t).any() else 0.0) for t in grid]
        i_star = int(np.argmax(vals))
        cur_sel = zv > thr
        cur_val = (fw[cur_sel] - mu).sum() if cur_sel.any() else 0.0
        orc_rows.append(dict(
            score=zc, year=y, thr_cur=thr,
            thr_pctile=float((zv <= thr).mean()) * 100,
            n_cur=int(cur_sel.sum()), excess_cur=cur_val * 100,
            thr_oracle=float(grid[i_star]), excess_oracle=vals[i_star] * 100,
            n_oracle=int((zv > grid[i_star]).sum()),
            mean_fwd20_cur=(fw[cur_sel].mean() * 100 if cur_sel.any() else np.nan),
            mean_fwd20_all=mu * 100))
pd.DataFrame(cal_rows).round(2).to_csv(os.path.join(OUT, "sa03_entry_calib.csv"), index=False)
eo = pd.DataFrame(orc_rows)
eo.to_csv(os.path.join(OUT, "sa03_entry_oracle.csv"), index=False)
pd.set_option("display.width", 250)
print("==== ENTRY oracle (excess = tong %-point fwd20 so voi mean nam, tren symbol-ngay) ====")
print(eo.round(2).to_string(index=False))

# ---------------- EXIT ----------------
tr = pd.read_csv(os.path.join(EM, "gbx08_s42_trades.csv"),
                 usecols=["symbol", "entry_date", "exit_date"])
held = set()
by_sym = {s: gph[["date"]].reset_index(drop=True) for s, gph in ph.groupby("symbol")}
for r in tr.itertuples():
    d = by_sym.get(r.symbol)
    if d is None:
        continue
    m = (d.date >= r.entry_date) & (d.date <= r.exit_date)
    for dt in d.date[m]:
        held.add((r.symbol, dt))
ph["held"] = [((s, d) in held) for s, d in zip(ph.symbol, ph.date)]
hp = ph[ph.held & ph.fwd10.notna() & ph.z_exit_score.notna()]
print("\nheld bars:", len(hp))

cal_rows, orc_rows = [], []
for label, sub in [("pre-gate", hp), ("post-gate(cons2_w20)", hp[hp.gate_exit])]:
    for y, gy in sub.groupby("year"):
        if len(gy) < 200:
            continue
        d = pd.qcut(gy.z_exit_score, 10, labels=False, duplicates="drop")
        dec = gy.groupby(d)["fwd10"].mean() * 100
        cal_rows.append(dict(set=label, year=y,
                             **{"d%d" % i: dec.get(i, np.nan) for i in range(10)}))
        zv = gy.z_exit_score.to_numpy()
        fw = gy.fwd10.to_numpy()
        mu = fw.mean()
        grid = np.quantile(zv, np.linspace(0.30, 0.99, 47))
        vals = [(((mu - fw[zv > t]).sum()) if (zv > t).any() else 0.0) for t in grid]
        i_star = int(np.argmax(vals))
        cur_sel = zv > 2.0
        cur_val = (mu - fw[cur_sel]).sum() if cur_sel.any() else 0.0
        orc_rows.append(dict(
            set=label, year=y, thr_pctile=float((zv <= 2.0).mean()) * 100,
            n_cur=int(cur_sel.sum()), avoided_cur=cur_val * 100,
            thr_oracle=float(grid[i_star]), avoided_oracle=vals[i_star] * 100,
            n_oracle=int((zv > grid[i_star]).sum()),
            mean_fwd10_cur=(fw[cur_sel].mean() * 100 if cur_sel.any() else np.nan),
            mean_fwd10_held=mu * 100))
pd.DataFrame(cal_rows).round(2).to_csv(os.path.join(OUT, "sa03_exit_calib.csv"), index=False)
xo = pd.DataFrame(orc_rows)
xo.to_csv(os.path.join(OUT, "sa03_exit_oracle.csv"), index=False)
print("\n==== EXIT oracle (avoided = tong %-point fwd10 tranh duoc vs mean held-bar) ====")
print(xo.round(2).to_string(index=False))

print("\n==== EXIT decile calib (post-gate, fwd10 % theo decile zX) ====")
xc = pd.DataFrame(cal_rows)
print(xc[xc["set"] == "post-gate(cons2_w20)"].to_string(index=False))
