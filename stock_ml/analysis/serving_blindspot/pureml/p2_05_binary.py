# -*- coding: utf-8 -*-
"""P2-E1 step 5 (công tố): thay regression bằng classifier nhị phân nhắm thẳng
loser (<=-5%) và knife (<=-15%) — kiểm tra kết luận âm có phải do objective.
Cùng frame/features/walk-forward với p2_04.
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
df["knife"] = df.pnl_pct <= -0.15

params = dict(objective="binary", learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, is_unbalance=True, verbose=-1)

from sklearn.metrics import roc_auc_score

for target in ("loser", "knife"):
    preds = []
    for yr in range(2022, 2027):
        tr = df[df.exit_date.astype(str) < f"{yr}-01-01"]
        te = df[df.year == yr]
        if len(tr) < 300 or te.empty or tr[target].sum() < 10:
            continue
        p = np.zeros(len(te))
        for seed in (42, 7, 99):
            m = lgb.train({**params, "seed": seed},
                          lgb.Dataset(tr[FEATS], label=tr[target].astype(int)),
                          num_boost_round=300)
            p += m.predict(te[FEATS])
        te = te.copy()
        te["p"] = p / 3
        preds.append(te)
    oof = pd.concat(preds, ignore_index=True)
    auc = roc_auc_score(oof[target].astype(int), oof.p)
    print(f"\n===== target={target} | OOF AUC >=2022 = {auc:.3f} =====")
    rows = []
    for yr, g in oof.groupby("year"):
        k = max(1, int(np.floor(len(g) * 0.10)))
        worst = g.nlargest(k, "p")
        rows.append({"year": yr, "n_cancel": k, "saved_u": -worst.pnl_pct.sum(),
                     "hit_rate": worst[target].mean(),
                     "n_bigwin": int(worst.big_win.sum())})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"decile TOTAL saved >=2022: {t.saved_u.sum():+.2f}u | bigwin hủy {t.n_bigwin.sum()}")
    # recall99 pooled
    nbw = int(oof.big_win.sum())
    allowed = int(np.floor(0.01 * nbw))
    s = oof.sort_values("p", ascending=False).reset_index(drop=True)
    ncan = int((s.big_win.cumsum() <= allowed).sum())
    C = s.iloc[:ncan]
    print(f"@recall_bigwin>=99%: cancel {len(C)} | u cứu {-C.pnl_pct.sum():+.2f}u | "
          f"precision({target}) {C[target].mean() if len(C) else float('nan'):.3f} | "
          f"{target}-rate nền {oof[target].mean():.3f}")
