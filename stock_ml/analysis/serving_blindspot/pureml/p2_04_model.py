# -*- coding: utf-8 -*-
"""P2-E1 step 4: LGBM walk-forward trên decision bar (bar cuối trước fill).

Frame A là frame LẠC QUAN cho cancel-policy: nó điều-kiện-hóa trên việc biết
fill sắp xảy ra (thông tin policy per-bar thật KHÔNG có, vì policy thật phải
bắn trên mọi bar treo kể cả của 7,582 lệnh không bao giờ fill). Nếu model
chết ở frame này thì policy per-bar chết a-fortiori.

Đo: (1) PnL thật của decile dự-đoán-xấu per-year >=2022 (OOF walk-forward);
    (2) precision + u cứu tại ràng buộc recall big_win >= 99%.
Purge: train chỉ dùng fills có exit_date < 01-01 năm test.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

OUT = os.path.dirname(os.path.abspath(__file__))
BS = os.path.dirname(OUT)
orders = pd.read_parquet(os.path.join(OUT, "p2_orders.parquet"))
bars = pd.read_parquet(os.path.join(OUT, "p2_bars.parquet"))
trades = pd.read_csv(os.path.join(BS, "trades_raw.csv"))

# exit_date join (purge)
trades["okey"] = (trades.symbol + "|" + trades.entry_signal_date.astype(str)
                  + "|" + trades.entry_date.astype(str))
orders["okey"] = orders.symbol + "|" + orders.sig_date + "|" + orders.end_date
orders = orders.merge(trades[["okey", "exit_date"]], on="okey", how="left")

fills = orders[orders.filled & (orders.exit_reason != "end_of_data")
               & (orders.wait_bars >= 2) & (~orders.spans_bad_adjustment)].copy()
key = fills.set_index("order_id").wait_bars - 1
b = bars.merge(key.rename("dec_age"), left_on="order_id", right_index=True, how="inner")
dec = b[b.age == b.dec_age].set_index("order_id")

BAR_FEATS = ["age", "bars_left", "dist_close", "dist_low", "dist_minlow",
             "drop_from_sig", "speed", "ret1", "ret3", "ret5", "gap_open",
             "red_streak", "atr_dist", "vol_shock",
             "m_ret1", "m_ret5", "m_z20", "m_breadth"]
ORD_FEATS = ["pre_ret20", "depth", "dist_ma20_sig", "sig_score"]
FEATS = BAR_FEATS + ORD_FEATS

df = fills.set_index("order_id").join(dec[BAR_FEATS], how="inner").reset_index()
df["year"] = df.year_end
print(f"model frame: {len(df)} fills wait>=2 | >=2022: {(df.year>=2022).sum()}")

params = dict(objective="regression", learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1)

preds = []
for yr in range(2022, 2027):
    tr = df[(df.exit_date.astype(str) < f"{yr}-01-01")]
    te = df[df.year == yr]
    if len(tr) < 300 or len(te) == 0:
        continue
    p_seeds = np.zeros(len(te))
    for seed in (42, 7, 99):
        m = lgb.train({**params, "seed": seed},
                      lgb.Dataset(tr[FEATS], label=tr.pnl_pct.astype(float)),
                      num_boost_round=300)
        p_seeds += m.predict(te[FEATS])
    te = te.copy()
    te["pred"] = p_seeds / 3.0
    preds.append(te)
    print(f"  fold {yr}: train {len(tr)} test {len(te)}")

oof = pd.concat(preds, ignore_index=True)

# (1) decile dự-đoán-xấu per-year
print("\n===== PnL thật của decile dự-đoán-XẤU (pred pnl thấp nhất 10%/năm) =====")
rows = []
for yr, g in oof.groupby("year"):
    k = max(1, int(np.floor(len(g) * 0.10)))
    worst = g.nsmallest(k, "pred")
    rows.append({"year": yr, "n": len(g), "n_cancel": k,
                 "saved_u": -worst.pnl_pct.sum(),
                 "mean_pnl": worst.pnl_pct.mean(),
                 "loser_rate": worst.loser.mean(),
                 "n_bigwin_cancel": int(worst.big_win.sum())})
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
print(f"TOTAL saved >=2022: {t.saved_u.sum():+.2f}u | big_win bị hủy: {t.n_bigwin_cancel.sum()}"
      f"/{int(oof.big_win.sum())}")

# spearman rank-IC pred vs pnl
from scipy.stats import spearmanr
ic = oof.groupby("year").apply(
    lambda g: spearmanr(g["pred"], g["pnl_pct"]).statistic)
print("\nrank-IC (pred, pnl) per year:")
print(ic.to_string(float_format=lambda x: f"{x: .4f}"))

# (2) ràng buộc recall big_win >= 99% (pooled >=2022)
nbw = int(oof.big_win.sum())
allowed = int(np.floor(0.01 * nbw))
s = oof.sort_values("pred").reset_index(drop=True)
cum_bw = s.big_win.cumsum()
mask = cum_bw <= allowed
if mask.any():
    n_cancel = int(mask[::-1].idxmax()) + 1 if mask.iloc[0] else 0
    # số dòng liên tục từ đầu thỏa ràng buộc
    n_cancel = int((cum_bw <= allowed).sum())  # prefix vì cum_bw đơn điệu
    C = s.iloc[:n_cancel]
    print(f"\n===== policy @ recall big_win >= 99% (pooled >=2022) =====")
    print(f"big_win total {nbw}, cho phép hủy nhầm {allowed}")
    print(f"cancel {len(C)} lệnh ({len(C)/len(s)*100:.1f}% fills) | "
          f"u cứu = {-C.pnl_pct.sum():+.2f}u | precision(loser<=-5%) = {C.loser.mean():.3f} | "
          f"mean pnl lệnh hủy = {C.pnl_pct.mean():+.4f}")
    print(f"big_win bị hủy: {int(C.big_win.sum())}")
else:
    print("\nkhông tồn tại ngưỡng nào thỏa recall >= 99%")

# tham chiếu: sweep độ sâu cancel không ràng buộc
print("\nsweep cancel k% dự-đoán-xấu (pooled >=2022): k, u cứu, bigwin hủy, precision")
for k in (0.02, 0.05, 0.10, 0.20, 0.30):
    kk = int(len(s) * k)
    C = s.iloc[:kk]
    print(f"  {k:.0%}: {-C.pnl_pct.sum():+7.2f}u  bw {int(C.big_win.sum()):3d}  "
          f"prec {C.loser.mean():.3f}")

# feature importance fold cuối
imp = pd.Series(m.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
print("\ntop feature gain (fold 2026):")
print(imp.head(8).to_string(float_format=lambda x: f"{x:.0f}"))
