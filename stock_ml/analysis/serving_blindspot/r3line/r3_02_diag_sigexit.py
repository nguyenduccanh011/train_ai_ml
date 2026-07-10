# -*- coding: utf-8 -*-
"""r3_02: chan doan kenh signal-exit cua t2903 (gem maxhold20) — attribution theo
exit_reason: tong pnl(u), so lenh, wr, hold, phan bo theo nam; so voi t2429 base."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
R2 = HERE.parent / "r2line" / "navscan"
FILES = {
    "t2903_mh20": R2 / "nv_nv_2429_maxhold20_s42_trades.csv",
    "t2429_base": R2 / "nv_n2_consw20_conv04_vg_combo_hb_nbpbw_s42_trades.csv",
}
pd.set_option("display.width", 200)
for name, f in FILES.items():
    df = pd.read_csv(f)
    df["year"] = df.entry_date.astype(str).str[:4]
    print(f"\n===== {name} ({len(df)} trades, tong pnl {df.pnl_pct.sum():.1f}u) =====")
    g = df.groupby("exit_reason").agg(
        n=("pnl_pct", "size"), pnl_u=("pnl_pct", "sum"), avg=("pnl_pct", "mean"),
        wr=("pnl_pct", lambda s: (s > 0).mean()), hold=("holding_days", "median"))
    print(g.sort_values("pnl_u").round(3))
    sig = df[df.exit_reason.astype(str).str.contains("signal", case=False, na=False)]
    if len(sig):
        print("-- signal-exit theo nam:")
        gy = sig.groupby("year").agg(n=("pnl_pct", "size"), pnl_u=("pnl_pct", "sum"),
                                     wr=("pnl_pct", lambda s: (s > 0).mean()),
                                     hold=("holding_days", "median"))
        print(gy.round(3))
        print("-- signal-exit theo hold bucket:")
        sig = sig.copy()
        sig["hb"] = pd.cut(sig.holding_days, [0, 3, 6, 10, 15, 20, 999],
                           labels=["1-3", "4-6", "7-10", "11-15", "16-20", ">20"])
        gh = sig.groupby("hb", observed=True).agg(n=("pnl_pct", "size"), pnl_u=("pnl_pct", "sum"),
                                                  wr=("pnl_pct", lambda s: (s > 0).mean()))
        print(gh.round(3))
