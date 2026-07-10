# -*- coding: utf-8 -*-
"""P2-E1 step 1: dựng dataset lệnh-treo (mọi limit đã đặt) cho cancel-policy.

Nguồn: events.csv (filled 3,813 + missed 7,582 = 11,395 limit đã đặt, khớp
NEW_FAMILY_OPTIONS P2), trades_raw.csv (outcome fills), ohlcv store serving
(RAW prices, cùng nguồn 02_enrich.py), sieutinhieu_missed_adjustments.csv
(flag fake gaps), signals.csv (score tại signal bar - tham chiếu).

Out (pureml/):
  p2_orders.parquet — 1 dòng / limit đã đặt: đặc điểm bar đặt + outcome.
  p2_bars.parquet   — 1 dòng / (order, bar đang treo t) với t < bar fill/expiry:
                      feature ex-ante tính từ dữ liệu ≤ close bar t
                      (quyết định cancel thi hành trước open bar t+1).
Quy ước:
  age = t - sig_idx. age=0 = close chính bar signal (KHÔNG có thông tin mới
  so với lúc đặt lệnh → cancel tại age=0 tương đương entry-gate; flag riêng).
  Lệnh fill wait=1 bar chỉ có dòng age=0 → về cơ chế là KHÔNG cancel được.
"""
import os
import sys

SERVING = r"C:\Users\DUC CANH PC\Desktop\stock-serving"
BS = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = os.path.join(BS, "pureml")
SLIP = 0.0015
WINDOW = 40

os.chdir(SERVING)
sys.path.insert(0, SERVING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from serving.ohlcv_store import OhlcvStore  # noqa: E402

# ---------- load ----------
events = pd.read_csv(os.path.join(BS, "events.csv"))
events = events[events["kind"].isin(["filled", "missed"])].reset_index(drop=True)
trades = pd.read_csv(os.path.join(BS, "trades_raw.csv"))
signals = pd.read_csv(os.path.join(BS, "signals.csv"))

ohlcv = OhlcvStore("data/ohlcv.db").load()
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date.astype(str)

badj = pd.read_csv(os.path.join(SERVING, "sieutinhieu_missed_adjustments.csv"))
bad_map: dict[str, list[str]] = {}
for _, r in badj.iterrows():
    bad_map.setdefault(str(r["symbol"]), []).append(str(r["ex_date"]))

# ---------- per-symbol arrays ----------
sym_arrays = {}
for sym, g in ohlcv.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"].to_numpy(float)
    h = g["high"].to_numpy(float)
    lo = g["low"].to_numpy(float)
    o = g["open"].to_numpy(float)
    v = g["volume"].to_numpy(float)
    n = len(c)
    # ATR14 (SMA of TR), vol_ma20 — dùng làm chuẩn hóa tại bar signal
    tr = np.empty(n)
    tr[0] = h[0] - lo[0]
    tr[1:] = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])])
    atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().to_numpy()
    vma20 = pd.Series(v).rolling(20, min_periods=5).mean().to_numpy()
    ma20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
    sym_arrays[sym] = {
        "dates": g["date"].to_numpy(),
        "idx": {d: k for k, d in enumerate(g["date"])},
        "o": o, "h": h, "l": lo, "c": c, "v": v,
        "atr14": atr14, "vma20": vma20, "ma20": ma20,
    }

# ---------- market frame (index proxy = universe store, eqw) ----------
piv = ohlcv.pivot_table(index="date", columns="symbol", values="close").sort_index()
ret1 = piv.pct_change()
m_ret1 = ret1.median(axis=1)
m_idx = (1.0 + m_ret1.fillna(0)).cumprod()
m_ret5 = m_idx.pct_change(5)
m_z20 = (m_idx - m_idx.rolling(60).mean()) / m_idx.rolling(60).std()
ma20_piv = piv.rolling(20, min_periods=10).mean()
breadth = (piv > ma20_piv).sum(axis=1) / piv.notna().sum(axis=1)
market = pd.DataFrame({
    "m_ret1": m_ret1, "m_ret5": m_ret5, "m_z20": m_z20, "m_breadth": breadth,
})
market_idx = {d: k for k, d in enumerate(market.index)}
M = market.to_numpy(float)

# ---------- outcome join: fills ----------
trades = trades.copy()
trades["key"] = trades["symbol"] + "|" + trades["entry_signal_date"].astype(str) + "|" + trades["entry_date"].astype(str)
tr_map = trades.set_index("key")[["pnl_pct", "exit_reason", "exit_date", "holding_days"]].to_dict("index")

# score tại signal bar (tham chiếu; prior: FLAT)
signals["skey"] = signals["symbol"] + "|" + signals["date"].astype(str)
score_col = "score" if "score" in signals.columns else None
sc_map = signals.set_index("skey")[score_col].to_dict() if score_col else {}

order_rows = []
bar_rows = []
skipped = 0

for oid, ev in enumerate(events.itertuples(index=False)):
    sym = ev.symbol
    A = sym_arrays.get(sym)
    if A is None:
        skipped += 1
        continue
    sig_date = str(ev.date)
    sidx = A["idx"].get(sig_date)
    eidx = A["idx"].get(str(ev.aux_date))  # fill bar hoặc bar hết window
    if sidx is None or eidx is None:
        skipped += 1
        continue
    limit = float(ev.limit_price)
    filled = ev.kind == "filled"
    c, h, lo, o, v = A["c"], A["h"], A["l"], A["o"], A["v"]
    sig_close = c[sidx]
    atr = A["atr14"][sidx]
    vma = A["vma20"][sidx]
    ma20s = A["ma20"][sidx]

    pnl = np.nan
    exit_reason = ""
    if filled:
        rec = tr_map.get(sym + "|" + sig_date + "|" + str(ev.aux_date))
        if rec is not None:
            pnl = float(rec["pnl_pct"])
            exit_reason = str(rec["exit_reason"])

    spans_bad = any(sig_date < xd <= str(ev.aux_date) for xd in bad_map.get(sym, []))
    pre20 = c[sidx] / c[sidx - 20] - 1.0 if sidx >= 20 else np.nan
    sig_score = sc_map.get(sym + "|" + sig_date, np.nan)
    wait = eidx - sidx  # với fill: số bar từ signal đến fill; missed: tuổi hết hạn

    order_rows.append({
        "order_id": oid, "symbol": sym, "sig_date": sig_date,
        "end_date": str(ev.aux_date), "filled": filled, "wait_bars": wait,
        "limit": limit, "sig_close": sig_close,
        "depth": limit / sig_close - 1.0,
        "pre_ret20": pre20, "sig_score": sig_score,
        "dist_ma20_sig": sig_close / ma20s - 1.0 if np.isfinite(ma20s) else np.nan,
        "pnl_pct": pnl, "exit_reason": exit_reason,
        "spans_bad_adjustment": spans_bad,
        "year_end": int(str(ev.aux_date)[:4]),
    })

    # ----- bar-level: t = sidx .. eidx-1 (trước bar fill/expiry) -----
    lows_since = np.inf
    red_streak = 0
    for t in range(sidx, eidx):
        age = t - sidx
        if age > 0:
            lows_since = min(lows_since, lo[t])
            red_streak = red_streak + 1 if c[t] < c[t - 1] else 0
        ret1b = c[t] / c[t - 1] - 1.0 if t >= 1 else np.nan
        ret3b = c[t] / c[t - 3] - 1.0 if t >= 3 else np.nan
        ret5b = c[t] / c[t - 5] - 1.0 if t >= 5 else np.nan
        gap = o[t] / c[t - 1] - 1.0 if t >= 1 else np.nan
        mi = market_idx.get(A["dates"][t])
        mrow = M[mi] if mi is not None else [np.nan] * 4
        drop = c[t] / sig_close - 1.0
        bar_rows.append({
            "order_id": oid, "t_date": A["dates"][t], "age": age,
            "bars_left": WINDOW - age,
            "dist_close": c[t] / limit - 1.0,
            "dist_low": lo[t] / limit - 1.0,
            "dist_minlow": (lows_since / limit - 1.0) if np.isfinite(lows_since) else np.nan,
            "drop_from_sig": drop,
            "speed": drop / age if age >= 1 else np.nan,
            "ret1": ret1b, "ret3": ret3b, "ret5": ret5b,
            "gap_open": gap,
            "red_streak": red_streak,
            "atr_dist": (c[t] - limit) / atr if np.isfinite(atr) and atr > 0 else np.nan,
            "vol_shock": v[t] / vma if np.isfinite(vma) and vma > 0 else np.nan,
            "m_ret1": mrow[0], "m_ret5": mrow[1],
            "m_z20": mrow[2], "m_breadth": mrow[3],
        })

orders = pd.DataFrame(order_rows)
bars = pd.DataFrame(bar_rows)

# nhãn bucket + big_win (chuẩn 02_enrich: big_win = pnl >= 0.15)
orders["big_win"] = orders["filled"] & (orders["pnl_pct"] >= 0.15)
orders["loser"] = orders["filled"] & (orders["pnl_pct"] <= -0.05)
orders["year_sig"] = orders["sig_date"].str[:4].astype(int)

orders.to_parquet(os.path.join(OUT, "p2_orders.parquet"), index=False)
bars.to_parquet(os.path.join(OUT, "p2_bars.parquet"), index=False)

print("orders:", len(orders), "filled:", int(orders.filled.sum()),
      "missed:", int((~orders.filled).sum()), "skipped:", skipped)
print("fills có pnl:", int(orders.pnl_pct.notna().sum()),
      "big_win:", int(orders.big_win.sum()), "loser<=-5%:", int(orders.loser.sum()))
print("bars:", len(bars))
print("wait=1 fills (không cancel được):",
      int((orders.filled & (orders.wait_bars <= 1)).sum()))
print("spans_bad:", int(orders.spans_bad_adjustment.sum()))
