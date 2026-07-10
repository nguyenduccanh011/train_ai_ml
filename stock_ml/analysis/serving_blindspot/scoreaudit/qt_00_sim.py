# -*- coding: utf-8 -*-
"""qt_00: SIM OFFLINE nguong tu-hieu-chinh (rolling quantile) cho exit head gb_x08.

Y tuong: thay sell_ml = zX>2.0 (tinh) bang exit_score >= Q_q(rolling 252, min_periods 60,
per-symbol, TRAILING chua bar hien tai — dung convention _causal_zscore_by_symbol).
Trigger tinh trong RAW-space (quantile cua cua so raw; z la affine theo t nen rank-safe).

Do 3 thu, thuan offline tu sa_scores.parquet + run_trades DB:
 (1) sell-bar/nam tai moi q (raw trigger, post-gate cons2_w20, additive sau force)
 (2) outcome: fwd10 cua bar ban (open-bars cua trades gb_x08) vs mean nam — decile,
     % ban-nham-winner (fwd10 > mean nam), avoided = sum(mean_nam - fwd10)
 (3) dpnl first-order tren trades gb_x08 (template/gb_x08-32a8dfee):
     A: quantile ban SOM hon exit that -> dpnl = (1+pnl)*(C[t_new]/C[t_exit]) - 1 - pnl
     B: exit that la sell_ml-thuan (khong force cung bar) ma quantile KHONG ban truoc do
        -> trade keo dai den bar (force | quantile-sell) ke tiep -> dpnl tuong tu.
Caveat first-order: bo qua exit_snr defer cua 2783 va occupancy re-entry; chi la dau + do lon.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.join(HERE, "sa_scores.parquet")
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
GB_RUN = "template/gb_x08-32a8dfee"
W, MP = 252, 60
QS = [0.97, 0.98, 0.985, 0.99, 0.995]

df = pd.read_parquet(SA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df["year"] = df["date"].dt.year.astype(str)
df.loc[df["year"] == "2026", "year"] = "2026H1"

g = df.groupby("symbol", sort=False)
df["fwd10"] = g["close"].transform(lambda s: s.shift(-10) / s - 1.0)

# rolling quantile per-symbol, trailing (chua bar hien tai) — dong bo _causal_zscore_by_symbol
for q in QS:
    col = f"rq{int(q*1000)}"
    df[col] = g["exit_score"].transform(lambda s, q=q: s.rolling(W, min_periods=MP).quantile(q))
    df[f"trig{int(q*1000)}"] = df["exit_score"] >= df[col]
    df[f"sell{int(q*1000)}"] = df[f"trig{int(q*1000)}"] & df["gate_exit"]

# ---- trades gb_x08 ----
con = psycopg2.connect(**PG)
tr = pd.read_sql(
    "SELECT symbol, entry_date, exit_date, pnl_pct, exit_reason FROM run_trades "
    "WHERE run_id=%s", con, params=(GB_RUN,))
con.close()
tr["entry_date"] = pd.to_datetime(tr["entry_date"])
tr["exit_date"] = pd.to_datetime(tr["exit_date"])
tr["year"] = tr["exit_date"].dt.year.astype(str)
tr.loc[tr["year"] == "2026", "year"] = "2026H1"
print(f"trades gb_x08: {len(tr)}  pnl_sum={tr.pnl_pct.sum():.4f}")

# open-bar mask (entry < date <= exit) — nhu sa03
df["open_bar"] = False
idx = df.set_index(["symbol", "date"]).index
pos = pd.Series(np.arange(len(df)), index=idx)
sym_dates = {s: sd["date"].to_numpy() for s, sd in df.groupby("symbol")}
sym_base = {s: sd.index[0] for s, sd in df.groupby("symbol")}
open_mask = np.zeros(len(df), dtype=bool)
for t in tr.itertuples():
    dates = sym_dates.get(t.symbol)
    if dates is None:
        continue
    b = sym_base[t.symbol]
    i0 = np.searchsorted(dates, np.datetime64(t.entry_date), side="right")
    i1 = np.searchsorted(dates, np.datetime64(t.exit_date), side="right")
    open_mask[b + i0: b + i1] = True
df["open_bar"] = open_mask
print(f"open bars: {open_mask.sum()}")

# ================= (1) sell-bar / nam =================
rows = []
for y, gy in df.groupby("year"):
    r = {"year": y, "bars": len(gy), "sell_ml_cu": int(gy.sell_ml.sum())}
    for q in QS:
        k = int(q * 1000)
        s = gy[f"sell{k}"]
        r[f"q{k}"] = int(s.sum())
        r[f"q{k}_add"] = int((s & ~gy.sell_force).sum())
    rows.append(r)
t1 = pd.DataFrame(rows).set_index("year")
print("\n===== (1) sell-bar/nam (post-gate cons2_w20; _add = khong trung force cung bar) =====")
print(t1.to_string())

# ================= (2) outcome tren open-bars =================
print("\n===== (2) outcome fwd10 tren open-bars (trades gb_x08) =====")
ob = df[df.open_bar & df.fwd10.notna()].copy()
for y, gy in ob.groupby("year"):
    mean_y = gy.fwd10.mean()
    try:
        gy = gy.assign(dec=pd.qcut(gy.fwd10, 10, labels=False, duplicates="drop"))
    except ValueError:
        gy = gy.assign(dec=np.nan)
    line = [f"{y} openbars={len(gy)} meanF10={mean_y*100:+.2f}%"]
    for q in QS:
        k = int(q * 1000)
        f = gy[gy[f"sell{k}"]]
        if len(f) == 0:
            line.append(f" q{k}: n=0")
            continue
        winner = (f.fwd10 > mean_y).mean() * 100
        avoided = ((mean_y - f.fwd10).sum()) * 100
        dec_mean = f.dec.mean()
        line.append(
            f" q{k}: n={len(f)} meanF10={f.fwd10.mean()*100:+.2f}% winner%={winner:.0f}"
            f" avoided={avoided:+.0f} decTB={dec_mean:.1f}")
    print("\n".join(line))

# ================= (3) dpnl first-order tren trades =================
print("\n===== (3) dpnl first-order (A: ban som; B: keo dai khi sell_ml cu bien mat) =====")
close_map = {s: sd.set_index("date")["close"] for s, sd in df.groupby("symbol")}
# bar-level series per symbol de tra cuu nhanh
sym_df = {s: sd.reset_index(drop=True) for s, sd in df.groupby("symbol")}

summary = {}
for q in QS:
    k = int(q * 1000)
    recs = []
    for t in tr.itertuples():
        sd = sym_df.get(t.symbol)
        if sd is None:
            continue
        dts = sd["date"].to_numpy()
        i0 = np.searchsorted(dts, np.datetime64(t.entry_date), side="right")
        ie = np.searchsorted(dts, np.datetime64(t.exit_date), side="left")
        if ie >= len(dts) or dts[ie] != np.datetime64(t.exit_date):
            continue  # exit bar khong co trong panel (hiem)
        c_exit = sd["close"].iloc[ie]
        sell_new = sd[f"sell{k}"].to_numpy()
        force = sd["sell_force"].to_numpy()
        # A: quantile ban som hon (bar dau tien trong (entry, exit) co sell_new)
        win = np.arange(i0, ie)
        fire = win[sell_new[win]] if len(win) else np.array([], dtype=int)
        if len(fire) > 0:
            tn = fire[0]
            c_new = sd["close"].iloc[tn]
            dp = (1 + t.pnl_pct) * (c_new / c_exit) - 1 - t.pnl_pct
            recs.append((t.year, "A_som", dp, int(ie - tn)))
            continue
        # B: exit that la sell_ml thuan (khong force cung bar) va quantile im -> keo dai
        if bool(sd["sell_ml"].iloc[ie]) and not bool(force[ie]) and not bool(sell_new[ie]):
            ext = np.arange(ie + 1, len(dts))
            nxt = ext[(sell_new[ext] | force[ext])] if len(ext) else np.array([], dtype=int)
            if len(nxt) > 0:
                tx = nxt[0]
                c_new = sd["close"].iloc[tx]
                dp = (1 + t.pnl_pct) * (c_new / c_exit) - 1 - t.pnl_pct
                recs.append((t.year, "B_keodai", dp, int(tx - ie)))
    rd = pd.DataFrame(recs, columns=["year", "kind", "dpnl", "dbar"])
    summary[k] = rd
    agg = rd.groupby(["year", "kind"]).agg(n=("dpnl", "size"), dpnl=("dpnl", "sum"),
                                           dbar_tb=("dbar", "mean"))
    tot = rd.groupby("year").dpnl.sum()
    print(f"\n-- q={q} --  tong dpnl={rd.dpnl.sum():+.3f}  (n_changed={len(rd)}/{len(tr)})")
    print(agg.round(3).to_string())
    print("tong theo nam: " + "  ".join(f"{y}:{v:+.2f}" for y, v in tot.items()))

print("\n===== TONG KET dpnl theo q =====")
for q in QS:
    k = int(q * 1000)
    rd = summary[k]
    by = rd.groupby("year").dpnl.sum()
    print(f"q={q}: TONG={rd.dpnl.sum():+.3f} | " +
          " ".join(f"{y}={by.get(y, 0):+.2f}" for y in
                   ["2020", "2021", "2022", "2023", "2024", "2025", "2026H1"]))
