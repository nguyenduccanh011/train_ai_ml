"""EXIT MAP gb_x08 seed-42: enrich per-trade with rule attribution + MFE/MAE from OHLCV.

Conventions verified in engine.py (close_next fills):
  - exit decision at bar d, fill at close of bar d+1 = exit_date. decision bar = exit_date - 1 bar.
  - entry_price / exit_price in run_trades already include slippage fills; pnl_pct is net of
    round-trip cost. MFE/MAE computed against entry_price (entry-fill units, same as ax_join.py).
Force-gate reconstruction (experiment.py):
  - downleg12: per-symbol causal zigzag (_causal_leg_dir, pct=0.12) == -1  [backstop, all regimes]
  - belowma20p2 & nonbull: close<MA20 for 2 consecutive bars AND EW-universe proxy below MA35
    for 2 consecutive bars (nonbull_ma_win=35, persist=2)
  - lowbreadth downleg6: zigzag6 == -1 AND full-duckdb pct_above_ma50 < 0.25 (level)
  - SNR(20) universe (engine formula, ax_join parity): rolling20 sum of mean rets / xsec std of
    rolling20 sums; snr_extend defers signal-exit when snr>=0.8 & peak_gain>=0.27 & giveback>=0.08
"""
import json
import duckdb
import numpy as np
import pandas as pd
import psycopg2

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
POST_W = 20

tr = pd.read_csv(EM + "/gbx08_s42_trades.csv", parse_dates=["entry_date", "exit_date", "entry_signal_date"])
print("trades:", len(tr), "pnl", round(tr.pnl_pct.sum(), 3))

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
print("universe:", len(uni_syms))

duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, open, high, low, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
# full-universe closes for breadth
alls = duck.execute("SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' ORDER BY symbol, date").df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
alls["date"] = pd.to_datetime(alls["date"])

# ---------- market series ----------
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
W = 20
snr_series = rets.mean(axis=1).rolling(W).sum() / (rets.rolling(W).sum().std(axis=1) + 1e-9)

# EW proxy level for nonbull (same construction as _market_nonbull_mask on the traded universe)
mret = rets.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
lvl = (1.0 + mret.fillna(0.0)).cumprod()
ma35 = lvl.rolling(35, min_periods=35).mean()
below35 = (lvl < ma35)
nonbull = (below35.rolling(2, min_periods=2).sum() >= 2).where(ma35.notna(), True)

# full-universe breadth pct_above_ma50, level < 0.25
pall = alls.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
ma50a = pall.rolling(50, min_periods=50).mean()
ind = (pall > ma50a)
breadth = ind.sum(axis=1) / ind.notna().sum(axis=1).clip(lower=1)
lowbreadth = breadth < 0.25

# exit market-drop gate (zscore mode, window 5, lb 60, thr -1.75) — suppresses signal exit
drop5 = mret.rolling(5).sum()
mu = drop5.rolling(252, min_periods=60).mean()
sd = drop5.rolling(252, min_periods=60).std()
mkt_drop = ((drop5 - mu) / sd.replace(0, np.nan)) < -1.75
mkt_drop = mkt_drop.fillna(False)


def causal_leg(close, pct):
    n = len(close)
    leg = np.zeros(n, dtype=np.int8)
    if n == 0:
        return leg
    direction, ext = 0, close[0]
    for i in range(1, n):
        p = close[i]
        if direction >= 0 and p > ext:
            ext = p; direction = 1
        elif direction <= 0 and p < ext:
            ext = p; direction = -1
        elif direction == 1 and p <= ext * (1.0 - pct):
            direction = -1; ext = p
        elif direction == -1 and p >= ext * (1.0 + pct):
            direction = 1; ext = p
        leg[i] = direction
    return leg


# ---------- per-symbol precompute ----------
SYM = {}
for sym, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g["close"].to_numpy(float)
    cs = pd.Series(c)
    ma20 = cs.rolling(20, min_periods=20).mean().to_numpy()
    below20 = c < ma20
    below20[np.isnan(ma20)] = False
    bma20p2 = (pd.Series(below20).rolling(2, min_periods=2).sum().to_numpy() >= 2)
    SYM[sym] = dict(
        dates=pd.DatetimeIndex(g["date"]), close=c,
        high=g["high"].to_numpy(float), low=g["low"].to_numpy(float),
        leg12=causal_leg(c, 0.12), leg6=causal_leg(c, 0.06), bma20p2=bma20p2,
    )
print("symbols precomputed:", len(SYM))

snr_d = snr_series  # date-indexed
rows = []
for _, t in tr.iterrows():
    s = SYM.get(t.symbol)
    if s is None:
        rows.append({}); continue
    di = s["dates"]
    ei = di.get_indexer([t.entry_date])[0]
    xi = di.get_indexer([t.exit_date])[0]
    if ei < 0 or xi < 0:
        rows.append({}); continue
    d = xi - 1 if t.exit_reason != "open" else xi  # decision bar
    ep = t.entry_price  # entry fill (incl slip)
    seg_h = s["high"][ei:xi + 1]
    seg_l = s["low"][ei:xi + 1]
    seg_c = s["close"][ei:xi + 1]
    mfe = seg_h.max() / ep - 1.0
    mfe_c = seg_c.max() / ep - 1.0
    mae = seg_l.min() / ep - 1.0
    peak_pos_c = int(np.argmax(seg_c))          # bar index (rel entry) of highest close
    bars_after_peak = (xi - ei) - peak_pos_c    # exit fill bar lag vs peak-close bar
    # decision-bar state
    close_d = s["close"][d]
    gain_d = close_d / ep - 1.0
    peak_d = s["high"][ei:d + 1].max() / ep - 1.0 if d >= ei else np.nan
    giveback_d = peak_d - gain_d
    dt_d = di[d]
    snr_at = snr_d.asof(dt_d)
    nonbull_at = bool(nonbull.asof(dt_d)) if not pd.isna(nonbull.asof(dt_d)) else True
    lowb_at = bool(lowbreadth.asof(dt_d)) if not pd.isna(lowbreadth.asof(dt_d)) else False
    drop_at = bool(mkt_drop.asof(dt_d)) if not pd.isna(mkt_drop.asof(dt_d)) else False
    f_dl12 = s["leg12"][d] == -1
    f_nb = bool(s["bma20p2"][d]) and nonbull_at
    f_lb = (s["leg6"][d] == -1) and lowb_at
    # snr-defer trace: was there a bar in (entry+min_hold .. d-1) where defer condition held?
    deferred_before = False
    first_defer_dt = None
    if xi - ei >= 3:
        hh = np.maximum.accumulate(s["high"][ei:xi + 1])
        for j in range(ei + 2, d):
            pk = hh[j - ei] / ep - 1.0
            if pk < 0.27:
                continue
            cj = s["close"][j]
            gb = (hh[j - ei] * ep / ep - 0) if False else (pk - (cj / ep - 1.0))
            if gb < 0.08:
                continue
            sv = snr_d.asof(di[j])
            if not pd.isna(sv) and sv >= 0.8:
                deferred_before = True
                first_defer_dt = di[j]
                break
    defer_eligible_d = (peak_d >= 0.27) and (giveback_d >= 0.08)
    # post-exit forward window (from exit fill bar xi, exit fill price = exit_price)
    pe_h = s["high"][xi + 1: xi + 1 + POST_W]
    pe_c = s["close"][xi + 1: xi + 1 + POST_W]
    xp = t.exit_price
    post_max_h = pe_h.max() / xp - 1.0 if len(pe_h) else np.nan
    post_max_c = pe_c.max() / xp - 1.0 if len(pe_c) else np.nan
    post_end_c = pe_c[-1] / xp - 1.0 if len(pe_c) else np.nan
    post_min_c = pe_c.min() / xp - 1.0 if len(pe_c) else np.nan
    # primary attribution label
    if t.exit_reason != "signal":
        label = t.exit_reason
    elif f_dl12:
        label = "force_downleg12"
    elif f_nb:
        label = "force_nonbull_bma20p2"
    elif f_lb:
        label = "force_lowbreadth_dl6"
    else:
        label = "head_signal"
    rows.append(dict(
        label=label, f_dl12=f_dl12, f_nb=f_nb, f_lb=f_lb, mkt_drop_at=drop_at,
        snr_at=snr_at, gain_d=gain_d, peak_d=peak_d, giveback_d=giveback_d,
        mfe=mfe, mfe_c=mfe_c, mae=mae, bars_after_peak=bars_after_peak,
        deferred_before=deferred_before, first_defer_dt=first_defer_dt,
        defer_eligible_d=defer_eligible_d,
        post_max_h=post_max_h, post_max_c=post_max_c, post_end_c=post_end_c,
        post_min_c=post_min_c, n_post=len(pe_c),
    ))

E = pd.concat([tr.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
E["year_exit"] = E.exit_date.dt.year
E["year_entry"] = E.entry_date.dt.year
E["eff"] = np.where(E.mfe > 0.005, E.pnl_pct / E.mfe, np.nan)
E["giveback_u"] = E.mfe - E.pnl_pct
E.to_csv(EM + "/gbx08_enriched.csv", index=False)
print("saved enriched:", len(E))
print("\nlabel counts:\n", E.label.value_counts())
print("\nsanity: mfe>=pnl fraction:", float((E.mfe >= E.pnl_pct - 1e-9).mean()).__round__(4))
print("mean eff:", round(E.eff.mean(), 4), " total giveback_u:", round(E.giveback_u.sum(), 2))
