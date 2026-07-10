# -*- coding: utf-8 -*-
"""PDR forensic: truy vet per-bar co che nuot signal-exit cua gb_x08 (template 2783 seed-42).

Chay tu repo root (f:/PROJECTS/train_ai_ml) de _load_vnindex path dung.
Part A: trace PDR 2024-03-20..2024-04-30 (moi bar: gia, gain/peak/giveback, trend MA10,
        don_low80, force gates, mkt_drop, SNR, protect band + release, hold rule, snr-defer).
Part B: quet toan bo trades gb_x08 — lenh nao co force-sell bi co che nuot -> exit sau te hon.
Parity nguon: engine.py (_market_*_dates, trend_up MA10 minp1 slope3, don_low rolling80 shift1,
hold MA50 ext/ATR14, protect release ATR20*2.5, snr defer 0.8/0.27/0.08), em02_enrich.py
(universe id8 v2, force gates, nonbull/lowbreadth).
Han che ghi ro: khong co score model per-bar (head sell + s3z) -> sell proxy = force gates
(dl12/nonbull/lowbreadth); dieu kien s3z>=0.5 cua hold rule va veto s3z>=1.6 khong kiem duoc.
"""
import json
import sys

import duckdb
import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, r"f:/PROJECTS/train_ai_ml/stock_ml")
from src.backtest.engine import _causal_leg_age, _load_vnindex  # noqa: E402

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"

# ---- engine params (template 2783) ----
SLIP = 0.0015
RT_COST = 2 * 0.0015 + 0.001
MIN_HOLD = 2
TRAIL_ACT = 0.27
DONCH = 80
SNR_W, SNR_THR = 20, 0.8
SNR_MIN_GAIN, SNR_MIN_GB = 0.27, 0.08
PROT_LO, PROT_HI = 0.08, 0.99
REL_K, REL_W = 2.5, 20
HOLD_K0, HOLD_MA = 1.3, 50
RS_SCALE, LEGAGE_SCALE, LEGAGE_PCT, LEGAMP_SCALE = 8.0, 0.5, 0.06, 0.2
DROP_W, DROP_THR, DROP_LB = 5, -1.75, 60

tr = pd.read_csv(EM + "/gbx08_s42_trades.csv",
                 parse_dates=["entry_date", "exit_date", "entry_signal_date"])
print("trades:", len(tr), "pnl", round(tr.pnl_pct.sum(), 3))

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                      user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()

duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, open, high, low, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
alls = duck.execute("SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' "
                    "ORDER BY symbol, date").df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
alls["date"] = pd.to_datetime(alls["date"])

# ---------- market series (engine formulas) ----------
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
# SNR (_market_snr_dates)
snr = rets.mean(axis=1).rolling(SNR_W).sum() / (rets.rolling(SNR_W).sum().std(axis=1) + 1e-9)
# market drop (_market_drop_dates, zscore lb60)
mret = rets.mean(axis=1)
roll5 = mret.rolling(DROP_W).sum()
drop_z = (roll5 - roll5.rolling(DROP_LB).mean()) / (roll5.rolling(DROP_LB).std() + 1e-9)
mkt_drop = drop_z <= DROP_THR
# nonbull (em02 parity)
mret_c = rets.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
lvl = (1.0 + mret_c.fillna(0.0)).cumprod()
ma35 = lvl.rolling(35, min_periods=35).mean()
nonbull = ((lvl < ma35).rolling(2, min_periods=2).sum() >= 2).where(ma35.notna(), True)
# lowbreadth full universe
pall = alls.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
ind = (pall > pall.rolling(50, min_periods=50).mean())
breadth = ind.sum(axis=1) / ind.notna().sum(axis=1).clip(lower=1)
lowbreadth = breadth < 0.25

vni = _load_vnindex()


def causal_leg(close, pct):
    n = len(close)
    leg = np.zeros(n, dtype=np.int8)
    direction, ext = 0, close[0] if n else 0.0
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


SYM = {}
for sym, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g["close"].to_numpy(float)
    h = g["high"].to_numpy(float)
    lo = g["low"].to_numpy(float)
    cs = pd.Series(c)
    n = len(c)
    # trend_up MA10 (trailing_skip_above_ma, minp1, slope lb3)
    sma10 = cs.rolling(10, min_periods=1).mean()
    trend_up = (c > sma10.to_numpy()) & ((sma10 - sma10.shift(3)).to_numpy() > 0)
    # don_low 80 (shift1)
    don_low = pd.Series(lo).rolling(DONCH, min_periods=DONCH).min().shift(1).to_numpy()
    # ATR14 / ATR20 ratios
    pc = np.concatenate([[c[0]], c[:-1]])
    trr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    atr14r = pd.Series(trr).rolling(14, min_periods=1).mean().to_numpy() / np.where(c == 0, 1e-9, c)
    atr20r = pd.Series(trr).rolling(REL_W, min_periods=1).mean().to_numpy() / np.where(c == 0, 1e-9, c)
    # hold rule (MA50 minp50, slope5)
    ma50 = cs.rolling(HOLD_MA, min_periods=HOLD_MA).mean()
    ma50v = ma50.to_numpy()
    ext50 = c / np.where(np.isnan(ma50v) | (ma50v == 0), np.nan, ma50v) - 1.0
    ext_atr_hold = ext50 / np.where((atr14r <= 0) | np.isnan(atr14r), np.nan, atr14r)
    hold_trend_ok = (c > ma50v) & ((ma50 - ma50.shift(5)).to_numpy() > 0)
    # rs_vsma
    if vni is not None:
        vd = pd.to_datetime(g["date"]).dt.normalize()
        vni_a = vd.map(vni).to_numpy(dtype=float)
        rsl = c / np.where((vni_a <= 0) | np.isnan(vni_a), np.nan, vni_a)
        rma = pd.Series(rsl).rolling(50, min_periods=20).mean().to_numpy()
        rs_vsma = np.nan_to_num(rsl / np.where(np.isnan(rma) | (rma == 0), np.nan, rma) - 1.0, nan=0.0)
    else:
        rs_vsma = np.zeros(n)
    legage, legamp = _causal_leg_age(c, LEGAGE_PCT)
    hold_k = (HOLD_K0
              * np.clip(1.0 + RS_SCALE * rs_vsma, 0.5, 2.0)
              * np.clip(1.0 - LEGAGE_SCALE * legage, 0.5, 2.0)
              * np.clip(1.0 + LEGAMP_SCALE * legamp, 0.5, 2.0))
    # force gates
    below20 = c < cs.rolling(20, min_periods=20).mean().to_numpy()
    below20[np.isnan(cs.rolling(20, min_periods=20).mean().to_numpy())] = False
    bma20p2 = pd.Series(below20).rolling(2, min_periods=2).sum().to_numpy() >= 2
    SYM[sym] = dict(dates=pd.DatetimeIndex(g["date"]), c=c, h=h, lo=lo,
                    trend_up=trend_up, don_low=don_low, atr20r=atr20r,
                    ext_atr_hold=ext_atr_hold, hold_trend_ok=hold_trend_ok, hold_k=hold_k,
                    leg12=causal_leg(c, 0.12), leg6=causal_leg(c, 0.06), bma20p2=bma20p2)
print("symbols precomputed:", len(SYM))


def date_flag(series, dt):
    v = series.asof(dt)
    return bool(v) if not pd.isna(v) else False


def mech_at(s, j, peak, entry_fill, dt):
    """Trang thai tung suppressor tai bar j (engine order). Tra ve (blocker, dict chi tiet)."""
    c = s["c"][j]
    pg = peak / entry_fill - 1.0
    gb_e = (peak - c) / entry_fill
    drop_pk = 1.0 - c / peak if peak > 0 else 0.0
    det = {}
    det["mkt_drop"] = date_flag(mkt_drop, dt)
    hold_conds = (c > entry_fill * (1.0 - 0.03)
                  and not np.isnan(s["ext_atr_hold"][j]) and s["ext_atr_hold"][j] < s["hold_k"][j]
                  and bool(s["hold_trend_ok"][j]))
    det["hold_maybe"] = bool(hold_conds)  # can s3z>=0.5 (khong kiem duoc)
    prot = PROT_LO <= pg < PROT_HI and bool(s["trend_up"][j])
    released = drop_pk >= REL_K * s["atr20r"][j]
    det["protect"] = bool(prot and not released)
    det["protect_release"] = bool(prot and released)
    snr_v = snr.asof(dt)
    det["snr"] = float(snr_v) if not pd.isna(snr_v) else np.nan
    det["snr_defer"] = bool(not pd.isna(snr_v) and snr_v >= SNR_THR
                            and pg >= SNR_MIN_GAIN and gb_e >= SNR_MIN_GB)
    if det["mkt_drop"]:
        return "mkt_drop", det
    if det["hold_maybe"]:
        return "hold_extatr", det
    if det["protect"]:
        return "protect_band", det
    if det["snr_defer"]:
        return "snr_defer", det
    return None, det


# ================= PART A: PDR trace =================
print("\n" + "=" * 100)
print("PART A: PDR entry 2023-12-19 fill 21.3871 — trace 2024-03-20 .. 2024-04-30")
s = SYM["PDR"]
di = s["dates"]
ENTRY_FILL = 21.387117540762127
ei = di.get_indexer([pd.Timestamp("2023-12-19")])[0]
xi = di.get_indexer([pd.Timestamp("2024-04-24")])[0]
assert ei >= 0 and xi >= 0
rows = []
peak = float(s["h"][ei])
for j in range(ei, xi + 1):
    peak = max(peak, float(s["h"][j]))
    dt = di[j]
    if dt < pd.Timestamp("2024-03-20"):
        continue
    c = s["c"][j]
    pg = peak / ENTRY_FILL - 1.0
    blocker, det = mech_at(s, j, peak, ENTRY_FILL, dt)
    f1 = s["leg12"][j] == -1
    f2 = bool(s["bma20p2"][j]) and date_flag(nonbull, dt)
    f3 = (s["leg6"][j] == -1) and date_flag(lowbreadth, dt)
    armed = pg >= TRAIL_ACT and not bool(s["trend_up"][j])
    struct_break = (not np.isnan(s["don_low"][j])) and c < s["don_low"][j]
    rows.append(dict(
        date=dt.date(), close=round(c, 2),
        gain=round(c / ENTRY_FILL - 1, 4), peak_gain=round(pg, 4),
        giveback_e=round((peak - c) / ENTRY_FILL, 4),
        trend_up_ma10=bool(s["trend_up"][j]), trail_armed=armed,
        don_low80=round(s["don_low"][j], 2), struct_break=struct_break,
        force_dl12=bool(f1), force_nonbull=bool(f2), force_lowbreadth=bool(f3),
        sell_force=bool(f1 or f2 or f3),
        mkt_drop=det["mkt_drop"], snr=round(det["snr"], 2),
        snr_defer=det["snr_defer"],
        hold_extatr=round(float(s["ext_atr_hold"][j]), 2) if not np.isnan(s["ext_atr_hold"][j]) else np.nan,
        hold_k=round(float(s["hold_k"][j]), 2), hold_maybe=det["hold_maybe"],
        protect=det["protect"], prot_release=det["protect_release"],
        blocker=blocker if (f1 or f2 or f3) else "",
    ))
A = pd.DataFrame(rows)
A.to_csv(EM + "/pdr_trace.csv", index=False)
print(A.to_string(index=False))

# counterfactual: ban tai bar dau tien sell_force & khong blocker=None... in chi tiet
first_sell = A[A.sell_force]
if len(first_sell):
    print("\nfirst force-sell bar:", first_sell.iloc[0].date, "blocker:", first_sell.iloc[0].blocker)
for _, r in first_sell.iterrows():
    jj = di.get_indexer([pd.Timestamp(r.date)])[0]
    if jj + 1 <= xi:
        cf_fill = s["c"][jj + 1] * (1 - SLIP)
        cf_pnl = cf_fill / ENTRY_FILL - 1 - RT_COST
        print(f"  {r.date} blocker={r.blocker or 'NONE->engine_would_exit'} "
              f"cf_exit_next_close={s['c'][jj+1]:.2f} cf_pnl={cf_pnl:+.4f}")

# ================= PART B: full scan =================
print("\n" + "=" * 100)
print("PART B: quet toan bo trades — force-sell bi nuot -> exit thuc te te hon?")
out = []
for _, t in tr.iterrows():
    if t.exit_reason == "open" or t.symbol not in SYM:
        continue
    s = SYM[t.symbol]
    di = s["dates"]
    ei = di.get_indexer([t.entry_date])[0]
    xi = di.get_indexer([t.exit_date])[0]
    if ei < 0 or xi < 0:
        continue
    d_bar = xi - 1  # decision bar cua exit thuc te
    ep = float(t.entry_price)
    peak = float(s["h"][ei])
    first = None
    for j in range(ei, d_bar):  # truoc decision bar
        peak = max(peak, float(s["h"][j]))
        if j - ei < MIN_HOLD:
            continue
        dt = di[j]
        f1 = s["leg12"][j] == -1
        f2 = bool(s["bma20p2"][j]) and date_flag(nonbull, dt)
        f3 = (s["leg6"][j] == -1) and date_flag(lowbreadth, dt)
        if not (f1 or f2 or f3):
            continue
        blocker, det = mech_at(s, j, peak, ep, dt)
        if blocker is None:
            # engine le ra exit o day (theo recon) — mismatch voi thuc te -> ghi rieng
            blocker = "recon_mismatch"
        if first is None:
            cf_fill = s["c"][j + 1] * (1 - SLIP)
            cf_pnl = cf_fill / ep - 1 - RT_COST
            first = dict(sw_date=dt.date(), blocker=blocker, cf_pnl=cf_pnl,
                         sw_gain=s["c"][j] / ep - 1, sw_peak=peak / ep - 1)
            break
    if first is None:
        continue
    delta = float(t.pnl_pct) - first["cf_pnl"]  # <0: tri hoan lam te hon
    out.append(dict(symbol=t.symbol, entry_date=t.entry_date.date(), exit_date=t.exit_date.date(),
                    exit_reason=t.exit_reason, pnl=float(t.pnl_pct),
                    year_entry=t.entry_date.year, **first, delta=delta))
B = pd.DataFrame(out)
B.to_csv(EM + "/pdr_forensic_scan.csv", index=False)
print("trades co force-sell bi nuot (truoc decision bar):", len(B), "/", len(tr))
for lab, gseg in [("ALL", B), (">=2022", B[B.year_entry >= 2022])]:
    print(f"\n--- {lab} ---")
    agg = gseg.groupby("blocker").agg(
        n=("delta", "size"), net_u=("delta", "sum"), mean=("delta", "mean"),
        worse5=("delta", lambda x: (x <= -0.05).sum()),
        helped5=("delta", lambda x: (x >= 0.05).sum()))
    print(agg.round(3).to_string())
print("\ntheo blocker x nam entry (net delta, >=2022):")
pv = B[B.year_entry >= 2022].pivot_table(index="year_entry", columns="blocker",
                                         values="delta", aggfunc="sum")
print(pv.round(2).to_string())
pv2 = B[B.year_entry >= 2022].pivot_table(index="year_entry", columns="blocker",
                                          values="delta", aggfunc="size")
print("\nn theo blocker x nam entry (>=2022):")
print(pv2.to_string())
