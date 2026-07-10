# -*- coding: utf-8 -*-
"""pr3_30: tội 3 — forensic overext_trail trên OHLC thật (sqlite).
Engine: trigger khi lows[i] <= peak_high*(1-0.04) (peak CẬP NHẬT bằng high CÙNG BAR
trước khi so low -> giả định high-trước-low trong bar); fill = close phiên i+1 (close_next).
Kiểm: (a) bao nhiêu trigger xảy ra trên bar vừa lập peak mới (intrabar ambiguity);
(b) fill close[i+1] vs trail level — gap/slippage thực tế; (c) nếu bar low-trước-high
thì trigger muộn hơn 1 bar, fill đổi bao nhiêu (bound tác động)."""
import sqlite3

import pandas as pd

DB = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
CSV = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/r2c_oxt04_p42_s42_trades.csv"
S0 = 0.001
TRAIL = 0.04

ta = pd.read_csv(CSV)
ox = ta[ta.exit_reason == "overext_trail"].copy()
print(f"overext_trail trades: {len(ox)} / {len(ta)}, sum_pnl {ox.pnl_pct.sum():+.2f} "
      f"(toàn sổ {ta.pnl_pct.sum():+.2f})")

con = sqlite3.connect(DB)
syms = sorted(set(ox.symbol))
px = pd.read_sql_query(
    "SELECT symbol, date, open, high, low, close FROM ohlcv WHERE symbol IN (%s)"
    % ",".join("?" * len(syms)), con, params=syms)
con.close()
bars = {s: g.sort_values("date").reset_index(drop=True) for s, g in px.groupby("symbol")}
idx = {s: {d: i for i, d in enumerate(g.date)} for s, g in bars.items()}

n_ok = n_price_mismatch = n_same_bar_peak = n_gap_through = 0
slips, ride_alt = [], []
n_trig_notfound = 0
for r in ox.itertuples():
    s = r.symbol
    g, ix = bars[s], idx[s]
    ed, xd = str(r.entry_date)[:10], str(r.exit_date)[:10]
    if ed not in ix or xd not in ix:
        n_price_mismatch += 1
        continue
    i_e, i_x = ix[ed], ix[xd]
    # scale check: CSV entry_price (adj*1.001) vs DB close — nếu lệch >2% là khác basis (bỏ)
    scale = (r.entry_price / (1 + S0)) / g.close[i_e]
    if abs(scale - 1) > 0.02:
        n_price_mismatch += 1
        continue
    i_t = i_x - 1  # trigger bar (fill next-bar close)
    if i_t <= i_e:
        n_trig_notfound += 1
        continue
    peak = g.high[i_e:i_t + 1].max()          # peak đến hết bar trigger (như engine)
    peak_prev = g.high[i_e:i_t].max()          # peak KHÔNG tính bar trigger
    trail_lv = peak * (1 - TRAIL)
    trig_ok = g.low[i_t] <= trail_lv + 1e-9
    if not trig_ok:
        n_trig_notfound += 1
        continue
    n_ok += 1
    # (a) trigger trên bar lập peak mới? (high[i_t] > peak_prev) -> intrabar ambiguity
    same_bar = g.high[i_t] > peak_prev
    n_same_bar_peak += same_bar
    if same_bar:
        # nếu low-trước-high: trail theo peak_prev có trigger không?
        if g.low[i_t] > peak_prev * (1 - TRAIL):
            # KHÔNG trigger bar này nếu low xảy ra trước high -> engine bán sớm 1+ bar
            # fill thật (nếu trigger bar sau): xem close bar i_x+1 nếu có
            pass
    # (b) fill close[i_x] vs trail level
    slips.append(g.close[i_x] / trail_lv - 1)
    n_gap_through += g.open[i_x] < trail_lv * 0.98  # gap qua trail >2%
sl = pd.Series(slips)
print(f"\nforensic được {n_ok} trades (mismatch/skip {n_price_mismatch}, "
      f"trigger-not-found {n_trig_notfound})")
print(f"(a) trigger trên bar VỪA lập peak mới (intrabar ambiguity): {n_same_bar_peak}/{n_ok}")
print(f"(b) fill(close T+1) vs trail level: mean {sl.mean()*100:+.2f}% "
      f"med {sl.median()*100:+.2f}% p10 {sl.quantile(.1)*100:+.2f}% p90 {sl.quantile(.9)*100:+.2f}%")
print(f"    fill DƯỚI trail level (bán rẻ hơn mức trail): {(sl<0).mean()*100:.0f}% số lệnh")
print(f"    gap-through >2% tại open T+1: {n_gap_through}/{n_ok}")
print("DONE")
