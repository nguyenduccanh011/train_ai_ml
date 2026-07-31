"""PHASE 2a — FEASIBILITY GATE: can causal daily TA features SEE a sub-top out-of-sample?

The fair bench (fb_01) proved the value of sub-swinging is DISCRIMINATION (fire only on
true sub-tops, suppress wiggles), and that it tolerates a few bars of lag. Before building
any policy/backtest, the cheapest honest gate is: train a classifier to detect the sub-top
bar from PAST-ONLY TA features, walk-forward, and check it beats a label-shuffle null and is
STABLE across years (the old exit head decayed / flipped sign — the disqualifier to avoid).

Setup (matches fb_01 bench):
  * Oracle up-legs: zigzag(pct=0.15, min_leg_bars=5) bottom->peak. We are "in position" the
    whole leg (the HOLD baseline).
  * Label per in-position bar b in [leg_bottom, leg_peak): 1 if b is an inner sub-PEAK (a
    1->0 transition of the hindsight inner-swing mask at inner_pct=s) — i.e. a TRUE place to
    sell-and-rebuy-lower; 0 otherwise. Positives are rare (≈1 bar / sub-swing) -> headline
    metric = Average Precision (PR-AUC), not ROC-AUC.
  * Features: causal daily OHLCV TA only (the user's shakeout-vs-top vocabulary) + position
    context vs the (oracle) entry. All shifted/rolling past-only.
  * Walk-forward: test year Y in {2022..2025}; train = bars strictly before Y with a 30-day
    EMBARGO at the boundary (drop train bars whose forward-derived label could peek into Y).
  * Null: retrain on row-shuffled labels -> AP should collapse to the base rate.

PASS = OOS AP >> base-rate AND ROC-AUC materially >0.5 in EVERY test year (stable sign).
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

DB = "market_data/market.duckdb"
START, END = "2020-01-02", "2025-12-31"
UNIVERSE = ['AAS','AAV','ACB','ACV','BCG','BCM','BID','BSR','BVH','CTG','DCM','DGC','DIG',
    'DPM','EIB','FPT','FRT','GAS','GEX','GMD','HCM','HDB','HDG','HPG','HSG','KBC','KDH',
    'LPB','MBB','MSN','MWG','NKG','NLG','NT2','NVL','OCB','PC1','PDR','PLX','PNJ','POW',
    'PVD','PVS','REE','SAB','SBT','SHB','SSI','STB','TCB','TPB','VCB','VCI','VDS','VHM',
    'VIC','VJC','VND','VNM','VPB','VTP']
PCT, MIN_LEG_BARS, INNER_S = 0.15, 5, 0.08


def _zigzag_pivots(close, pct, min_leg_bars=0):
    n = len(close); bottoms, peaks = [], []
    if n == 0: return bottoms, peaks
    direction, ext_idx, ext_price, last = 0, 0, close[0], 0
    for i in range(1, n):
        price = close[i]
        if direction >= 0 and price > ext_price:
            ext_price, ext_idx, direction = price, i, 1
        elif direction <= 0 and price < ext_price:
            ext_price, ext_idx, direction = price, i, -1
        elif direction == 1 and price <= ext_price * (1.0 - pct):
            if ext_idx - last >= min_leg_bars: peaks.append(ext_idx); last = ext_idx
            direction, ext_price, ext_idx = -1, price, i
        elif direction == -1 and price >= ext_price * (1.0 + pct):
            if ext_idx - last >= min_leg_bars: bottoms.append(ext_idx); last = ext_idx
            direction, ext_price, ext_idx = 1, price, i
    return bottoms, peaks


def _inner_subtrade_pos(close_sub, inner_pct):
    n = len(close_sub); pos = np.ones(n, dtype=np.int8)
    if n < 3 or inner_pct <= 0: return pos
    sb, sp = _zigzag_pivots(close_sub, inner_pct)
    piv = sorted([(i, 'b') for i in sb] + [(i, 'p') for i in sp])
    state, prev = 1, 0
    for pi, typ in piv:
        if typ == 'p': pos[prev:pi + 1] = state; state, prev = 0, pi + 1
        else: pos[prev:pi] = state; pos[pi] = 1; state, prev = 1, pi + 1
    pos[prev:] = state
    return pos


def features(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = g['open'], g['high'], g['low'], g['close'], g['volume']
    rng = (h - l).replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f = pd.DataFrame(index=g.index)
    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    f['dist_ma20'] = c / ma20 - 1
    f['dist_ma50'] = c / ma50 - 1
    f['ma20_slope'] = ma20.pct_change(5)
    # RSI14 (Wilder)
    d = c.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / dn.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    f['rsi14'] = 100 - 100 / (1 + rs)
    f['atr_pct'] = tr.rolling(14).mean() / c
    # candle anatomy
    f['upper_wick'] = (h - np.maximum(o, c)) / rng
    f['lower_wick'] = (np.minimum(o, c) - l) / rng
    f['body'] = (c - o).abs() / rng
    f['clv'] = ((c - l) - (h - c)) / rng
    f['ret1'] = c.pct_change()
    f['gap'] = o / c.shift() - 1
    # consecutive up days
    up_day = (c > c.shift()).astype(int)
    grp = (up_day == 0).cumsum()
    f['up_streak'] = up_day.groupby(grp).cumsum()
    # distance below recent high / above recent low
    f['dist_hi20'] = c / h.rolling(20).max() - 1
    f['dist_lo20'] = c / l.rolling(20).min() - 1
    # volume / supply
    relvol = v / v.rolling(20).mean()
    f['relvol'] = relvol
    f['vol_z'] = (np.log(v.replace(0, np.nan)) - np.log(v.replace(0, np.nan)).rolling(20).mean()) \
                 / np.log(v.replace(0, np.nan)).rolling(20).std()
    # accumulation/distribution balance over 20 bars: strong-close high-vol (+) vs weak-close high-vol (-)
    strong = ((f['clv'] > 0.3) & (relvol > 1.2)).astype(int)
    weak = ((f['clv'] < -0.3) & (relvol > 1.2)).astype(int)
    f['ad_bal20'] = (strong - weak).rolling(20).sum()
    # range contraction (squeeze): recent TR vs 20d TR (<1 = contracting)
    f['range_contr'] = tr.rolling(5).mean() / tr.rolling(20).mean()
    f['nr7'] = (tr <= tr.rolling(7).min()).astype(int)
    # shakeout pair: yesterday red & today reclaims yesterday high on volume
    f['shakeout'] = (((c.shift() < c.shift(2)) & (c > h.shift()) & (relvol > 1.0)).astype(int))
    # blow-off: today's range >> atr (climax)
    f['range_atr'] = (h - l) / tr.rolling(14).mean()
    return f


def build_panel():
    con = duckdb.connect(DB, read_only=True)
    q = f"""select symbol, date, open, high, low, close, volume from ohlcv
            where timeframe='1D' and symbol in ({','.join("'"+s+"'" for s in UNIVERSE)})
              and date >= '{START}' and date <= '{END}' order by symbol, date"""
    df = con.execute(q).df(); con.close()
    df['date'] = pd.to_datetime(df['date'])
    out = []
    for sym, g in df.groupby('symbol'):
        g = g.reset_index(drop=True)
        close = g['close'].to_numpy(np.float64)
        feats = features(g)
        bottoms, peaks = _zigzag_pivots(close, PCT, MIN_LEG_BARS)
        piv = sorted([(i, 'b') for i in bottoms] + [(i, 'p') for i in peaks])
        inpos = np.zeros(len(close), dtype=np.int8)
        label = np.zeros(len(close), dtype=np.int8)
        for k in range(len(piv) - 1):
            (b, t), (p, t2) = piv[k], piv[k + 1]
            if not (t == 'b' and t2 == 'p' and p > b): continue
            sub = close[b:p + 1]
            pos = _inner_subtrade_pos(sub, INNER_S)
            inpos[b:p] = 1  # in position over [b, p) under the HOLD baseline
            for i in range(1, len(sub) - 1):  # exclude the forced peak (leg boundary)
                if pos[i - 1] == 1 and pos[i] == 0:
                    label[b + i] = 1  # a true sub-top: sell here, rebuy lower
        rec = feats.copy()
        rec['symbol'] = sym; rec['date'] = g['date'].values
        rec['inpos'] = inpos; rec['y'] = label
        out.append(rec[inpos == 1])
    panel = pd.concat(out, ignore_index=True)
    panel['year'] = panel['date'].dt.year
    return panel


FEATS = ['dist_ma20','dist_ma50','ma20_slope','rsi14','atr_pct','upper_wick','lower_wick',
    'body','clv','ret1','gap','up_streak','dist_hi20','dist_lo20','relvol','vol_z','ad_bal20',
    'range_contr','nr7','shakeout','range_atr']


def main():
    panel = build_panel()
    panel = panel.replace([np.inf, -np.inf], np.nan)
    print(f"in-position bars: {len(panel)}, sub-top positives: {int(panel['y'].sum())} "
          f"(base rate {panel['y'].mean():.4f})\n")

    rng = np.random.RandomState(0)
    print(f"{'year':>5} {'n_test':>7} {'pos':>5} {'base':>7} {'AP':>7} {'AUC':>7} {'AP_null':>8} {'lift':>6}")
    aps, aucs = [], []
    for Y in (2022, 2023, 2024, 2025):
        embargo = pd.Timestamp(f'{Y}-01-01') - pd.Timedelta(days=30)
        tr = panel[panel['date'] < embargo]
        te = panel[panel['year'] == Y]
        if te['y'].sum() < 5 or tr['y'].sum() < 20:
            continue
        Xtr, ytr = tr[FEATS], tr['y']
        Xte, yte = te[FEATS], te['y']
        spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
        params = dict(objective='binary', n_estimators=300, learning_rate=0.03,
                      num_leaves=31, min_child_samples=80, subsample=0.8,
                      colsample_bytree=0.7, scale_pos_weight=spw, verbose=-1)
        m = lgb.LGBMClassifier(**params).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        ap = average_precision_score(yte, p); auc = roc_auc_score(yte, p)
        # shuffle null
        yshuf = ytr.sample(frac=1.0, random_state=1).values
        mn = lgb.LGBMClassifier(**params).fit(Xtr, yshuf)
        pn = mn.predict_proba(Xte)[:, 1]
        apn = average_precision_score(yte, pn)
        base = yte.mean()
        aps.append(ap); aucs.append(auc)
        print(f"{Y:>5} {len(te):>7} {int(yte.sum()):>5} {base:>7.4f} {ap:>7.4f} {auc:>7.4f} "
              f"{apn:>8.4f} {ap/base:>6.2f}")

    print(f"\npooled >=2022: mean AP={np.mean(aps):.4f}  mean AUC={np.mean(aucs):.4f}  "
          f"min-year AUC={min(aucs):.4f}")

    # feature importance (last model)
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\ntop features (last fold, gain-split importance):")
    print(imp.head(12).to_string())

    print("\n===== GATE VERDICT =====")
    stable = min(aucs) > 0.55
    lift = np.mean([a for a in aps]) / panel['y'].mean()
    print(f"min-year AUC {min(aucs):.3f} ({'>' if stable else '<='} 0.55 stable-sign gate); "
          f"mean AP lift vs base = {lift:.2f}x")
    print("PASS -> build capture-ratio policy (fb_03)" if stable and lift > 1.5
          else "WEAK/FAIL -> daily TA can't cleanly see sub-tops OOS; reconsider target")


if __name__ == "__main__":
    main()
