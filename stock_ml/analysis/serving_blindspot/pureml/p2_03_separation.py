# -*- coding: utf-8 -*-
"""P2-E1 step 3: separation ex-ante tại DECISION BAR (bar cuối trước fill).

Frame A: fills wait>=2, decision bar = age (wait-1) — cơ hội cancel cuối cùng,
mọi feature chỉ dùng dữ liệu <= close bar đó. So sánh phân phối
fill-thành-loser (pnl<=-5%) vs fill-thành-big_win (pnl>=+15%) vs còn lại.
AUC per-feature (loser vs non-loser; loser vs big_win). Kiểm tra trực diện
giả thuyết "rơi NHANH về limit = dao" (speed, age, ret1/3/5, gap, m_z20...).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = os.path.dirname(os.path.abspath(__file__))
orders = pd.read_parquet(os.path.join(OUT, "p2_orders.parquet"))
bars = pd.read_parquet(os.path.join(OUT, "p2_bars.parquet"))

fills = orders[orders.filled & (orders.exit_reason != "end_of_data")
               & (orders.wait_bars >= 2) & (~orders.spans_bad_adjustment)].copy()
# decision bar = bar (wait-1) của order
key = fills.set_index("order_id").wait_bars - 1
bars = bars.merge(key.rename("dec_age"), left_on="order_id", right_index=True, how="inner")
dec = bars[bars.age == bars.dec_age].set_index("order_id")

df = fills.set_index("order_id").join(
    dec[["dist_close", "dist_low", "dist_minlow", "drop_from_sig", "speed",
         "ret1", "ret3", "ret5", "gap_open", "red_streak", "atr_dist",
         "vol_shock", "m_ret1", "m_ret5", "m_z20", "m_breadth", "age"]],
    how="inner")
df["year"] = df.year_end
df["bucket3"] = np.where(df.pnl_pct <= -0.05, "loser",
                         np.where(df.pnl_pct >= 0.15, "big_win", "mid"))

FEATS = ["age", "dist_close", "dist_low", "dist_minlow", "drop_from_sig", "speed",
         "ret1", "ret3", "ret5", "gap_open", "red_streak", "atr_dist", "vol_shock",
         "m_ret1", "m_ret5", "m_z20", "m_breadth",
         "pre_ret20", "depth", "dist_ma20_sig", "sig_score"]

print(f"frame A: {len(df)} fills wait>=2, loser {int((df.bucket3=='loser').sum())}, "
      f"big_win {int((df.bucket3=='big_win').sum())}")


def auc_table(d, label):
    print(f"\n===== {label} (n={len(d)}) =====")
    print(f"{'feature':<15}{'med loser':>11}{'med big_win':>13}{'med mid':>10}"
          f"{'AUC l|rest':>12}{'AUC l|bw':>10}")
    y_l = (d.bucket3 == "loser").astype(int)
    lb = d[d.bucket3.isin(["loser", "big_win"])]
    y_lb = (lb.bucket3 == "loser").astype(int)
    for f in FEATS:
        x = d[f]
        m = x.notna()
        med = d.groupby("bucket3")[f].median()
        try:
            a1 = roc_auc_score(y_l[m], x[m]) if y_l[m].nunique() > 1 else np.nan
        except ValueError:
            a1 = np.nan
        xm = lb[f].notna()
        try:
            a2 = roc_auc_score(y_lb[xm], lb[f][xm]) if y_lb[xm].nunique() > 1 else np.nan
        except ValueError:
            a2 = np.nan
        print(f"{f:<15}{med.get('loser', np.nan):>11.4f}{med.get('big_win', np.nan):>13.4f}"
              f"{med.get('mid', np.nan):>10.4f}{a1:>12.3f}{a2:>10.3f}")


auc_table(df, "FULL 2020-2026")
auc_table(df[df.year >= 2022], ">=2022")

# ---- giả thuyết "rơi nhanh = dao": age-to-fill x outcome (>=2022) ----
sub = df[df.year >= 2022].copy()
sub["age_b"] = pd.cut(sub.wait_bars, [1, 2, 3, 5, 10, 20, 40],
                      labels=["2", "3", "4-5", "6-10", "11-20", "21-40"])
t = sub.groupby("age_b", observed=True).agg(
    n=("pnl_pct", "size"), mean_pnl=("pnl_pct", "mean"),
    loser_rate=("loser", "mean"), bigwin_rate=("big_win", "mean"),
    sum_u=("pnl_pct", "sum"))
print("\n===== wait-to-fill vs outcome (fills >=2022, wait>=2) =====")
print(t.to_string(float_format=lambda x: f"{x: .4f}"))

sub["speed_q"] = pd.qcut(sub.speed, 5, labels=False, duplicates="drop")
t2 = sub.groupby("speed_q").agg(
    n=("pnl_pct", "size"), speed_med=("speed", "median"),
    mean_pnl=("pnl_pct", "mean"), loser_rate=("loser", "mean"),
    bigwin_rate=("big_win", "mean"), sum_u=("pnl_pct", "sum"))
print("\n===== quintile SPEED (drop/bar về limit, âm = rơi nhanh) >=2022 =====")
print(t2.to_string(float_format=lambda x: f"{x: .4f}"))

# index đang sập?
sub["mz_q"] = pd.qcut(sub.m_z20, 5, labels=False, duplicates="drop")
t3 = sub.groupby("mz_q").agg(
    n=("pnl_pct", "size"), mz_med=("m_z20", "median"),
    mean_pnl=("pnl_pct", "mean"), loser_rate=("loser", "mean"),
    bigwin_rate=("big_win", "mean"), sum_u=("pnl_pct", "sum"))
print("\n===== quintile m_z20 (index z tại decision bar) >=2022 =====")
print(t3.to_string(float_format=lambda x: f"{x: .4f}"))

# gap-down qua limit tại decision bar? (open ngày fill mới là gap thật - future;
# đây là gap của bar TRƯỚC fill)
sub["gapdown"] = sub.gap_open <= -0.02
t4 = sub.groupby("gapdown").agg(n=("pnl_pct", "size"), mean_pnl=("pnl_pct", "mean"),
                                loser_rate=("loser", "mean"), bigwin_rate=("big_win", "mean"))
print("\n===== gap_open <= -2% tại decision bar, >=2022 =====")
print(t4.to_string(float_format=lambda x: f"{x: .4f}"))
