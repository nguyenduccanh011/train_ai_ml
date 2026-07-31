"""PHASE 2b — CAPTURE-RATIO on the fair bench (the decision-relevant metric).

fb_02 showed daily TA sees the sub-top with a STABLE sign OOS (AUC 0.91-0.94 every year),
but AUC is flattered by easy negatives. The honest question: plugging the classifier's
P(sub-top) into the swing policy on the oracle-entry bench, does it convert to PROFIT over
HOLD, net of cost, >=2022? Metric = capture-ratio = (policy - hold) / (ceiling - hold).
  reactive rule (fb_01) = NEGATIVE; ceiling = 1.0; lag-oracle L1/L2 = the "good predictor" ref.

Walk-forward: a leg is scored by the classifier trained on data strictly before its ENTRY
year (30d embargo). Only legs entering >=2022 are evaluated (clean OOS + the regime window).
Two rebuy modes bracket the result:
  * oracle-rebuy  : rebuy at the true inner sub-trough -> isolates SELL-side value (upper bound).
  * reactive-rebuy: rebuy on a >=s rise from the running low -> realistic dumb rebuy (lower bound).
"""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fb_02_subtop_clf import (_zigzag_pivots, _inner_subtrade_pos, features, FEATS,
    UNIVERSE, DB, START, END, PCT, MIN_LEG_BARS, INNER_S)

COST = 0.004


def build():
    con = duckdb.connect(DB, read_only=True)
    q = f"""select symbol, date, open, high, low, close, volume from ohlcv
            where timeframe='1D' and symbol in ({','.join("'"+s+"'" for s in UNIVERSE)})
              and date >= '{START}' and date <= '{END}' order by symbol, date"""
    df = con.execute(q).df(); con.close()
    df['date'] = pd.to_datetime(df['date'])
    rows = []          # per in-position bar: features + leg_id + loc + y + entry_year + date
    legmeta = {}       # leg_id -> dict(close, entry_year, subtrough_locs)
    lid = 0
    for sym, g in df.groupby('symbol'):
        g = g.reset_index(drop=True)
        close = g['close'].to_numpy(np.float64)
        feats = features(g).replace([np.inf, -np.inf], np.nan)
        bottoms, peaks = _zigzag_pivots(close, PCT, MIN_LEG_BARS)
        piv = sorted([(i, 'b') for i in bottoms] + [(i, 'p') for i in peaks])
        for k in range(len(piv) - 1):
            (b, t), (p, t2) = piv[k], piv[k + 1]
            if not (t == 'b' and t2 == 'p' and p > b):
                continue
            sub = close[b:p + 1]
            pos = _inner_subtrade_pos(sub, INNER_S)
            entry_year = int(pd.Timestamp(g['date'].iloc[b]).year)
            subtrough = {i for i in range(1, len(sub)) if pos[i - 1] == 0 and pos[i] == 1}
            legmeta[lid] = dict(close=sub, entry_year=entry_year, subtrough=subtrough)
            for i in range(b, p):  # in-position bars [b, p)
                loc = i - b
                is_top = int(loc >= 1 and loc < len(sub) - 1 and pos[loc - 1] == 1 and pos[loc] == 0)
                fr = feats.iloc[i].to_dict()
                fr.update(dict(leg_id=lid, loc=loc, y=is_top, entry_year=entry_year,
                               date=g['date'].iloc[i]))
                rows.append(fr)
            lid += 1
    panel = pd.DataFrame(rows)
    return panel, legmeta


def score_walkforward(panel):
    """Return dict leg_id -> {loc: P}. Each leg scored by a model trained before its entry year."""
    panel = panel.copy()
    P_by_leg = {}
    for Y in (2022, 2023, 2024, 2025):
        embargo = pd.Timestamp(f'{Y}-01-01') - pd.Timedelta(days=30)
        tr = panel[panel['date'] < embargo]
        te = panel[panel['entry_year'] == Y]
        if len(te) == 0 or tr['y'].sum() < 20:
            continue
        spw = (tr['y'] == 0).sum() / max(1, (tr['y'] == 1).sum())
        m = lgb.LGBMClassifier(objective='binary', n_estimators=300, learning_rate=0.03,
            num_leaves=31, min_child_samples=80, subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=spw, verbose=-1).fit(tr[FEATS], tr['y'])
        p = m.predict_proba(te[FEATS])[:, 1]
        for (lg, loc), pv in zip(zip(te['leg_id'], te['loc']), p):
            P_by_leg.setdefault(lg, {})[loc] = pv
    return P_by_leg


def hold_ret(close):
    return close[-1] / close[0] * (1.0 - COST) - 1.0


def ceil_ret(close):
    pos = _inner_subtrade_pos(close, INNER_S)
    mult = 1.0
    for i in range(1, len(close)):
        if pos[i - 1] == 1:
            mult *= close[i] / close[i - 1]
        if pos[i - 1] == 1 and pos[i] == 0:
            mult *= (1.0 - COST)
    if pos[-1] == 1:
        mult *= (1.0 - COST)
    return mult - 1.0


def policy_ret(close, P, thr, rebuy, subtrough):
    n = len(close)
    in_pos, entry_px, run_hi, run_lo = True, close[0], close[0], None
    mult, nsell = 1.0, 0
    for loc in range(1, n):
        px = close[loc]
        if in_pos:
            if px > run_hi:
                run_hi = px
            if loc < n - 1 and P.get(loc, 0.0) >= thr:   # classifier sell (never on the peak)
                mult *= (px / entry_px) * (1.0 - COST); in_pos = False; run_lo = px; nsell += 1
        else:
            if px < run_lo:
                run_lo = px
            if rebuy == 'oracle':
                if loc in subtrough:
                    entry_px, in_pos, run_hi = px, True, px
            else:  # reactive
                if px >= run_lo * (1.0 + INNER_S):
                    entry_px, in_pos, run_hi = px, True, px
    if in_pos:
        mult *= (close[-1] / entry_px) * (1.0 - COST)
    return mult - 1.0, nsell


def main():
    panel, legmeta = build()
    print(f"legs total {len(legmeta)}, >=2022 legs "
          f"{sum(1 for m in legmeta.values() if m['entry_year']>=2022)}; "
          f"in-pos bars {len(panel)}\n")
    P_by_leg = score_walkforward(panel)

    legs22 = [(lid, m) for lid, m in legmeta.items() if m['entry_year'] >= 2022]
    hold_sum = sum(hold_ret(m['close']) for _, m in legs22) * 100
    ceil_sum = sum(ceil_ret(m['close']) for _, m in legs22) * 100

    # reactive-only baseline (no classifier): sell on >=s drop from run high
    def reactive_only(close):
        n = len(close); in_pos, entry_px, run_hi, run_lo, mult = True, close[0], close[0], None, 1.0
        for loc in range(1, n):
            px = close[loc]
            if in_pos:
                run_hi = max(run_hi, px)
                if loc < n - 1 and px <= run_hi * (1 - INNER_S):
                    mult *= (px/entry_px)*(1-COST); in_pos=False; run_lo=px
            else:
                run_lo = min(run_lo, px)
                if px >= run_lo*(1+INNER_S):
                    entry_px, in_pos, run_hi = px, True, px
        if in_pos: mult *= (close[-1]/entry_px)*(1-COST)
        return mult-1
    react_sum = sum(reactive_only(m['close']) for _, m in legs22) * 100

    print(f">=2022 (sum per-leg return %, net cost {COST}):  HOLD={hold_sum:.0f}  "
          f"CEILING={ceil_sum:.0f}  (room={ceil_sum-hold_sum:.0f})  reactive-only={react_sum:.0f} "
          f"(cap={ (react_sum-hold_sum)/(ceil_sum-hold_sum):+.2f})\n")

    # precision/recall of the sell trigger among >=2022 in-pos bars (why it can't convert)
    pr = [(P_by_leg.get(r.leg_id, {}).get(r.loc, 0.0), r.y)
          for r in panel[panel.entry_year >= 2022].itertuples()]
    Parr = np.array([x[0] for x in pr]); Yarr = np.array([x[1] for x in pr])
    tot_pos = Yarr.sum()
    print(f"sell-trigger precision/recall (>=2022, {len(Yarr)} bars, {int(tot_pos)} true sub-tops):")
    print(f"  {'thr':>5} {'fired':>6} {'prec':>6} {'recall':>7}")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        fired = Parr >= thr
        prec = Yarr[fired].mean() if fired.sum() else 0.0
        rec = Yarr[fired].sum() / tot_pos
        print(f"  {thr:>5.2f} {int(fired.sum()):>6} {prec:>6.3f} {rec:>7.3f}")
    print()

    print(f"{'rebuy':>9} {'thr':>5} {'sells':>6} {'policy':>8} {'pol-hold':>9} {'capture':>8}")
    for rebuy in ('oracle', 'reactive'):
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            psum, nsell = 0.0, 0
            for lid, m in legs22:
                r, ns = policy_ret(m['close'], P_by_leg.get(lid, {}), thr, rebuy, m['subtrough'])
                psum += r; nsell += ns
            psum *= 100
            cap = (psum - hold_sum) / (ceil_sum - hold_sum)
            print(f"{rebuy:>9} {thr:>5.2f} {nsell:>6} {psum:>8.0f} {psum-hold_sum:>+9.0f} {cap:>+8.2f}")

    print("\n===== VERDICT =====")
    print("capture > 0 net of cost = the classifier ADDS profit over HOLD inside winners "
          "(seam realizable). oracle-rebuy = sell-side ceiling; reactive-rebuy = with a dumb rebuy.")


if __name__ == "__main__":
    main()
