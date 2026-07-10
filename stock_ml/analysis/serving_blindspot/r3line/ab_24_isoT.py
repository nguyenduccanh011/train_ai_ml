# -*- coding: utf-8 -*-
"""ab_24: co lap co che cum T — matched trades ab_noT (t2936) vs r3_gb_mh16 (t2910).
Khac biet duy nhat: trailing_struct_apply_overext + trailing_struct_donch_win."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
gb = pd.read_csv(HERE / "r3_gb_mh16_s42_trades.csv")   # co T
nt = pd.read_csv(HERE / "ab_noT_s42_trades.csv")        # bo T
gb["k"] = gb.symbol + "|" + gb.entry_date.astype(str).str[:10]
nt["k"] = nt.symbol + "|" + nt.entry_date.astype(str).str[:10]
m = gb.merge(nt, on="k", suffixes=("_gb", "_nt"))
print(f"matched {len(m)} / gb {len(gb)} / noT {len(nt)}")
diff = m[m.exit_date_gb != m.exit_date_nt].copy()
diff["dp"] = (diff.pnl_pct_nt - diff.pnl_pct_gb) * 100
print(f"exit khac: {len(diff)}; delta pnl (noT - gb) = {diff.dp.sum()/100:.1f}u")
g = diff.groupby(["exit_reason_gb", "exit_reason_nt"]).agg(
    n=("dp", "size"), hold_gb=("holding_days_gb", "mean"), hold_nt=("holding_days_nt", "mean"),
    pnl_gb=("pnl_pct_gb", "mean"), pnl_nt=("pnl_pct_nt", "mean"),
    delta_u=("dp", lambda x: x.sum() / 100))
g["pnl_gb"] *= 100
g["pnl_nt"] *= 100
print(g.round(2).sort_values("delta_u").to_string())
# so ngay-giu giai phong & lech vong quay
gbo = gb[~gb.k.isin(nt.k)]
nto = nt[~nt.k.isin(gb.k)]
print(f"\ngb-only {len(gbo)} pnl={gbo.pnl_pct.sum():.1f}u | noT-only {len(nto)} "
      f"pnl={nto.pnl_pct.sum():.1f}u (entry moi nho slot giai phong som)")
print(f"tong ngay-giu: gb={gb.holding_days.sum():.0f} noT={nt.holding_days.sum():.0f} "
      f"(giai phong {gb.holding_days.sum()-nt.holding_days.sum():.0f} symbol-ngay)")
print("AB24_DONE")
