# -*- coding: utf-8 -*-
"""P2-E1 step 2: oracle ceiling cho cancel-policy (perfect foresight).

Oracle = hủy đúng các limit sẽ-fill-thành-loser. Slot giải phóng = 0 (bảo thủ).
Kill bar (NEW_FAMILY_OPTIONS P2): ceiling cancel-worst-decile >=2022 < +3u -> đóng.

Hai biến thể:
  A. ALL fills — kể cả wait=1 (về cơ chế wait=1 KHÔNG cancel được: không có bar
     thông tin mới nào giữa lúc đặt và lúc fill -> đây là trần "gate-contaminated").
  B. wait>=2 fills — trần THẬT của cơ chế cancel (có >=1 bar mới để quyết định).
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
orders = pd.read_parquet(os.path.join(OUT, "p2_orders.parquet"))

fills = orders[orders.filled & (orders.exit_reason != "end_of_data")].copy()
fills["year"] = fills.year_end  # năm fill

print(f"fills đóng: {len(fills)}  sum u = {fills.pnl_pct.sum():.2f}")


def oracle(df, label):
    print(f"\n===== ORACLE {label} =====")
    sub = df[df.year >= 2022]
    print(f"n fills >=2022: {len(sub)}, sum u = {sub.pnl_pct.sum():.2f}")
    # worst decile theo pnl thực, per-year và pooled
    rows = []
    for yr, g in sub.groupby("year"):
        k = max(1, int(np.floor(len(g) * 0.10)))
        worst = g.nsmallest(k, "pnl_pct")
        rows.append({"year": yr, "n": len(g), "n_cancel": k,
                     "saved_u": -worst.pnl_pct.sum(),
                     "mean_pnl_cancel": worst.pnl_pct.mean(),
                     "n_bigwin_cancel": int((worst.pnl_pct >= 0.15).sum())})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print(f"TOTAL saved >=2022 (per-year decile): {t.saved_u.sum():+.2f}u")
    # pooled decile
    k = max(1, int(np.floor(len(sub) * 0.10)))
    worst = sub.nsmallest(k, "pnl_pct")
    print(f"pooled decile: cancel {k} lệnh, saved {-worst.pnl_pct.sum():+.2f}u, "
          f"pnl range [{worst.pnl_pct.min():.3f}, {worst.pnl_pct.max():.3f}]")
    # tham chiếu: hủy MỌI lệnh lỗ (trần tuyệt đối)
    neg = sub[sub.pnl_pct < 0]
    print(f"trần tuyệt đối (hủy mọi fill pnl<0): {-neg.pnl_pct.sum():+.2f}u "
          f"({len(neg)} lệnh = {len(neg)/len(sub)*100:.1f}%)")
    knife = sub[sub.pnl_pct <= -0.15]
    print(f"knife <=-15%: {len(knife)} lệnh, {knife.pnl_pct.sum():.2f}u")


oracle(fills, "A - ALL fills (gate-contaminated, wait=1 không cancel được)")
oracle(fills[fills.wait_bars >= 2], "B - wait>=2 (trần thật của cơ chế cancel)")

# đóng góp của wait=1 vào worst decile
sub = fills[fills.year >= 2022]
k = max(1, int(np.floor(len(sub) * 0.10)))
worst = sub.nsmallest(k, "pnl_pct")
w1 = worst[worst.wait_bars <= 1]
print(f"\nwait=1 trong pooled worst-decile >=2022: {len(w1)}/{len(worst)} lệnh, "
      f"{-w1.pnl_pct.sum():+.2f}u trong saved")

# per-year u của fills wait=1 (để hiểu phần gate-equivalent)
print("\nfills wait=1 theo năm (n / sum u):")
print(fills[fills.wait_bars <= 1].groupby("year").pnl_pct.agg(["count", "sum"]).to_string())
