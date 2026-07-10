# -*- coding: utf-8 -*-
"""na_05: giai phau episode MaxDD 2025-04 (tariff crash 2025-04-03..09).
- c2 K22 vs gb K25: peak/trough/depth quanh 2025-03..06, exposure truoc crash,
  so vi the mo xuyen crash, pnl cac lenh entry/exit trong cua so.
- Kiem "may man cau truc so": c2 co thoat truoc crash nho hold ngan khong?
"""
import sys

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from na_navlib import run_sim

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
C2 = f"{R2}/r2_c2_pb40snr_s42_trades.csv"
WIN = ("2025-03-01", "2025-06-30")

for name, csv, k in [("c2 K22", C2, 22), ("gb K25", GB, 25)]:
    m = run_sim(csv, K=k, record_legs_window=WIN)
    ns = m["ns"].set_index("date")
    w = ns.loc["2025-01-01":"2025-08-31"]
    peak_d = w.nav.loc[:"2025-04-30"].idxmax()
    trough_d = w.nav.loc["2025-03-15":"2025-05-15"].idxmin()
    peak, trough = w.nav[peak_d], w.nav[trough_d]
    rec = w.nav.loc[trough_d:][w.nav.loc[trough_d:] >= peak]
    rec_d = rec.index[0].date() if len(rec) else "chua"
    print(f"\n=== {name} === episode: peak {peak_d.date()} nav={peak:.2f} -> trough {trough_d.date()}"
          f" nav={trough:.2f} depth={(trough/peak-1)*100:.2f}% phuc hoi {rec_d}")
    snap = {d: (legs, cashp) for d, legs, cashp in m["legs_snap"]}
    for d in ["2025-04-02", "2025-04-03", "2025-04-09"]:
        legs, cashp = snap[d]
        inv = sum(v for *_ , v in legs)
        nav_d = float(ns.nav[pd.Timestamp(d)])
        print(f"  {d}: {len(legs)} vi the mo, exposure={(inv/nav_d)*100:.0f}% NAV, cash={cashp/nav_d*100:.0f}%")
    legs0402 = snap["2025-04-02"][0]
    df = pd.read_csv(csv)
    df["ed"] = df.entry_date.astype(str).str[:10]
    df["xd"] = df.exit_date.astype(str).str[:10]
    held = df[(df.ed <= "2025-04-02") & (df.xd >= "2025-04-03")]
    print(f"  lenh om xuyen crash (entry<=04-02, exit>=04-03): {len(held)}"
          f" | pnl mean {held.pnl_pct.mean()*100:+.1f}% | thua: {(held.pnl_pct<0).sum()}")
    print("   exit_reason:", held.exit_reason.value_counts().to_dict() if "exit_reason" in held else "n/a")
    pre_exit = df[(df.xd >= "2025-03-20") & (df.xd <= "2025-04-02")]
    crash_entry = df[(df.ed >= "2025-04-03") & (df.ed <= "2025-04-15")]
    print(f"  thoat 03-20..04-02 (truoc crash): {len(pre_exit)} lenh, pnl mean {pre_exit.pnl_pct.mean()*100:+.1f}%")
    print(f"  vao 04-03..04-15 (bat day): {len(crash_entry)} lenh, pnl mean {crash_entry.pnl_pct.mean()*100:+.1f}%"
          f" | win% {(crash_entry.pnl_pct>0).mean()*100:.0f}")
    # DD cua he trong dung cua so crash (peak 2025-03->trough 04-09), doc lap MaxDD toan cuc
    seg = ns.nav.loc["2025-03-01":"2025-05-15"]
    segdd = (seg / seg.cummax() - 1).min()
    print(f"  DD noi bo cua so 03-01..05-15: {segdd*100:.2f}%")
