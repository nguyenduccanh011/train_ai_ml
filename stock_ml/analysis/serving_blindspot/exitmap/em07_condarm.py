"""EXIT MAP gb_x08 — conditional tight-arm counterfactuals (existing engine knobs, config-only):
  A) vol_spike arm: ATR14/close z(252,min60) >= 2.0 & peak_gain >= 0.10 -> trail 4% from peak-close
  B) dist arm: ad_balance_20 <= -2 & peak_gain >= 0.08 -> trail 5%
  C) trend_break_lock: peak_gain >= 0.10 & close < MA50 -> exit now
First-order per-trade counterfactual on OHLCV (no knock-on/slot effects), engine formulas replicated.
"""
import json
import duckdb
import numpy as np
import pandas as pd
import psycopg2

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
E = pd.read_csv(EM + "/gbx08_enriched2.csv", parse_dates=["entry_date", "exit_date"])
pd.set_option("display.width", 220)

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, open, high, low, close, volume FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])

SYM = {}
for sym, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g["close"].to_numpy(float); h = g["high"].to_numpy(float); lo = g["low"].to_numpy(float)
    v = g["volume"].to_numpy(float)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    ar = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy() / np.where(c == 0, 1e-9, c)
    s = pd.Series(ar)
    volz = ((s - s.rolling(252, min_periods=60).mean()) / (s.rolling(252, min_periods=60).std() + 1e-9)).to_numpy()
    rng = np.where((h - lo) <= 0, 1e-9, h - lo)
    clpos = (c - lo) / rng
    vavg = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    volr = v / np.where((vavg <= 0) | np.isnan(vavg), np.nan, vavg)
    dist_bar = ((volr > 1.2) & (clpos < 0.45)).astype(float)
    acc_bar = ((volr > 1.2) & (clpos > 0.55)).astype(float)
    adbal = (pd.Series(acc_bar).rolling(20, min_periods=10).sum()
             - pd.Series(dist_bar).rolling(20, min_periods=10).sum()).to_numpy()
    ma50 = pd.Series(c).rolling(50, min_periods=50).mean().to_numpy()
    SYM[sym] = dict(dates=pd.DatetimeIndex(g["date"]), c=c, volz=volz, adbal=adbal, ma50=ma50)


def cf_arm(t, cond_fn, trail, min_gain, lock=False):
    s = SYM.get(t.symbol)
    if s is None:
        return np.nan
    di = s["dates"]
    ei = di.get_indexer([t.entry_date])[0]
    xi = di.get_indexer([t.exit_date])[0]
    if ei < 0 or xi < 0:
        return np.nan
    c = s["c"]; ep = t.entry_price
    peak = c[ei]
    armed = False
    for j in range(ei + 1, xi):  # decision bars strictly before actual fill bar
        peak = max(peak, c[j])
        if not armed and peak / ep - 1.0 >= min_gain and cond_fn(s, j):
            armed = True
        if armed:
            if lock or c[j] <= peak * (1 - trail):
                fill = c[min(j + 1, len(c) - 1)] * 0.9985
                return (fill / ep - 1.0 - 0.004) - t.pnl_pct
    return np.nan  # never fired earlier than actual exit


scenarios = {
    "A_volspike_z2_t4_g10": lambda t: cf_arm(t, lambda s, j: s["volz"][j] >= 2.0, 0.04, 0.10),
    "B_distarm_ad-2_t5_g8": lambda t: cf_arm(t, lambda s, j: not np.isnan(s["adbal"][j]) and s["adbal"][j] <= -2, 0.05, 0.08),
    "C_trendbreak_ma50_g10": lambda t: cf_arm(t, lambda s, j: not np.isnan(s["ma50"][j]) and s["c"][j] < s["ma50"][j], 0.0, 0.10, lock=True),
}
for name, fn in scenarios.items():
    d = E.apply(fn, axis=1)
    m = d.notna()
    tot = d[m].sum()
    by_year = d[m].groupby(E.year_exit[m]).sum().round(1).to_dict()
    ge22 = d[m & (E.year_exit >= 2022)].sum()
    print(f"{name}: fired n={m.sum():4d} delta={tot:+.1f}u | >=2022 {ge22:+.1f}u | by year {by_year}")
