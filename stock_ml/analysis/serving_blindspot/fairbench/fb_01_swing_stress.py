"""FAIR-BENCH stress test (kill-or-keep the swing-exit thesis).

Question (user, 2026-07-10): the champion HOLDS a winner leg; the oracle sub-swing
inside that leg is worth +371u (HYBRID_VERDICT mảnh C). Prior work concluded "chop
loses to hold" — but it compared HOLD to the CHAMPION's trend-exit rules acting as a
swing timer (wrong tool), on CHAMPION entries (entry noise). This re-runs the decisive
experiment on a FAIR bench:

  * ENTRY is fixed to the ORACLE leg-bottom (min-amplitude zigzag) for BOTH policies.
  * FINAL exit is the ORACLE leg-peak for BOTH policies.
  * The ONLY difference is what happens INSIDE the up-leg:
       HOLD          — sit through to the peak (1 round-trip).
       SWING-causal  — a real-time reversal trader: sell when close falls >= s from the
                       running high since entry; rebuy when close rises >= s from the
                       running low while flat; force-close at the peak. Causal (reacts to
                       confirmed moves on close, fills at that close). Pays a round-trip
                       PER sub-trade (the honest churn cost + rebuy-miss risk).
       ORACLE-inner  — hindsight ceiling: sells every sub-peak, buys every sub-trough
                       (zigzag at inner_pct=s). The best case for the SAME s.

If SWING-causal <= HOLD even with a PERFECT entry and PERFECT final exit, net of cost,
the swing thesis is dead regardless of entry quality -> KILL. If it beats HOLD at some
(pct, s) net of realistic cost, especially >=2022 -> the +371u is reachable -> KEEP and
build the shakeout-vs-top classifier.

Data: market_data/market.duckdb (ohlcv, 1D, back-adjusted), 61-symbol champion universe,
2020-01-02..2025-12-31. Self-contained; verbatim copy of the repo _zigzag_pivots so the
legs match action_oracle exactly.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

DB = "market_data/market.duckdb"
START, END = "2020-01-02", "2025-12-31"
UNIVERSE = ['AAS','AAV','ACB','ACV','BCG','BCM','BID','BSR','BVH','CTG','DCM','DGC','DIG',
    'DPM','EIB','FPT','FRT','GAS','GEX','GMD','HCM','HDB','HDG','HPG','HSG','KBC','KDH',
    'LPB','MBB','MSN','MWG','NKG','NLG','NT2','NVL','OCB','PC1','PDR','PLX','PNJ','POW',
    'PVD','PVS','REE','SAB','SBT','SHB','SSI','STB','TCB','TPB','VCB','VCI','VDS','VHM',
    'VIC','VJC','VND','VNM','VPB','VTP']


def _zigzag_pivots(close: np.ndarray, pct: float, min_leg_bars: int = 0):
    """Verbatim copy of stock_ml.src.targets.zigzag._zigzag_pivots (deterministic)."""
    n = len(close)
    bottoms: list[int] = []
    peaks: list[int] = []
    if n == 0:
        return bottoms, peaks
    direction = 0
    ext_idx = 0
    ext_price = close[0]
    last_confirmed_idx = 0
    for i in range(1, n):
        price = close[i]
        if direction >= 0 and price > ext_price:
            ext_price = price; ext_idx = i; direction = 1
        elif direction <= 0 and price < ext_price:
            ext_price = price; ext_idx = i; direction = -1
        elif direction == 1 and price <= ext_price * (1.0 - pct):
            if ext_idx - last_confirmed_idx >= min_leg_bars:
                peaks.append(ext_idx); last_confirmed_idx = ext_idx
            direction = -1; ext_price = price; ext_idx = i
        elif direction == -1 and price >= ext_price * (1.0 + pct):
            if ext_idx - last_confirmed_idx >= min_leg_bars:
                bottoms.append(ext_idx); last_confirmed_idx = ext_idx
            direction = 1; ext_price = price; ext_idx = i
    return bottoms, peaks


def up_legs(close: np.ndarray, pct: float, min_leg_bars: int):
    """List of (b, p) index pairs: confirmed bottom -> the next confirmed peak."""
    bottoms, peaks = _zigzag_pivots(close, pct, min_leg_bars)
    piv = sorted([(i, 'b') for i in bottoms] + [(i, 'p') for i in peaks])
    legs = []
    for k in range(len(piv) - 1):
        (i, t), (j, t2) = piv[k], piv[k + 1]
        if t == 'b' and t2 == 'p' and j > i:
            legs.append((i, j))
    return legs


def hold_ret(close, b, p, cost):
    return close[p] / close[b] * (1.0 - cost) - 1.0


def swing_causal_ret(close, b, p, s, cost):
    """Real-time reversal trader inside [b, p]. Long at b. Sell on >=s drop from running
    high; rebuy on >=s rise from running low; force-close at p. Returns (net_ret, n_rt)."""
    mult = 1.0
    n_rt = 0
    in_pos = True
    entry_px = close[b]
    run_hi = close[b]      # running high since entry (for sell trigger)
    run_lo = None          # running low since going flat (for rebuy trigger)
    for i in range(b + 1, p + 1):
        px = close[i]
        if in_pos:
            if px > run_hi:
                run_hi = px
            # sell on a confirmed >=s reversal from the running high (but not on the last
            # bar — the last bar is the forced peak close handled below)
            if i < p and px <= run_hi * (1.0 - s):
                mult *= (px / entry_px) * (1.0 - cost)
                n_rt += 1
                in_pos = False
                run_lo = px
        else:
            if px < run_lo:
                run_lo = px
            if px >= run_lo * (1.0 + s):
                entry_px = px          # rebuy at this close
                in_pos = True
                run_hi = px
    # force-close at the peak if still long (the oracle final exit, shared with HOLD)
    if in_pos:
        mult *= (close[p] / entry_px) * (1.0 - cost)
        n_rt += 1
    return mult - 1.0, n_rt


def _inner_subtrade_pos(close_sub, inner_pct):
    """Verbatim from action_oracle: in-position(1)/flat(0) per bar over a rising leg,
    EXIT at each hindsight sub-peak, RE-ENTER at the following sub-trough. Long at idx 0."""
    n = len(close_sub)
    pos = np.ones(n, dtype=np.int8)
    if n < 3 or inner_pct <= 0:
        return pos
    sb, sp = _zigzag_pivots(close_sub, inner_pct)
    piv = sorted([(i, 'b') for i in sb] + [(i, 'p') for i in sp])
    state, prev = 1, 0
    for pi, typ in piv:
        if typ == 'p':
            pos[prev:pi + 1] = state; state, prev = 0, pi + 1
        else:
            pos[prev:pi] = state; pos[pi] = 1; state, prev = 1, pi + 1
    pos[prev:] = state
    return pos


def oracle_inner_ret(close, b, p, s, cost):
    """Hindsight ceiling over the FULL leg: perfect sub-peak sell / sub-trough rebuy.
    Long on every rising sub-segment, flat on every falling one; cost per sell."""
    sub = close[b:p + 1]
    if len(sub) < 3:
        return hold_ret(close, b, p, cost)
    pos = _inner_subtrade_pos(sub, s)
    mult = 1.0
    for i in range(1, len(sub)):
        if pos[i - 1] == 1:
            mult *= sub[i] / sub[i - 1]
        # a sell = position goes 1 -> 0 at this bar -> pay a round-trip cost
        if pos[i - 1] == 1 and pos[i] == 0:
            mult *= (1.0 - cost)
    if pos[-1] == 1:  # forced close at the peak
        mult *= (1.0 - cost)
    return mult - 1.0


def oracle_lagged_ret(close, b, p, s, cost, lag):
    """Perfect hindsight sub-swing mask, but EXECUTED `lag` bars LATE (a predictor that
    sees the turn `lag` bars after it happens). lag=0 = ceiling. Quantifies how precise a
    real classifier must be: where does (swing - hold) cross zero as lag grows?"""
    sub = close[b:p + 1]
    if len(sub) < 3:
        return hold_ret(close, b, p, cost)
    pos = _inner_subtrade_pos(sub, s)
    if lag > 0:
        shifted = np.ones(len(pos), dtype=np.int8)  # forced long at entry
        shifted[lag:] = pos[:-lag]
        pos = shifted
    mult = 1.0
    for i in range(1, len(sub)):
        if pos[i - 1] == 1:
            mult *= sub[i] / sub[i - 1]
        if pos[i - 1] == 1 and pos[i] == 0:
            mult *= (1.0 - cost)
    if pos[-1] == 1:
        mult *= (1.0 - cost)
    return mult - 1.0


def load_prices():
    con = duckdb.connect(DB, read_only=True)
    q = f"""select symbol, date, close from ohlcv
            where timeframe='1D' and symbol in ({','.join("'"+s+"'" for s in UNIVERSE)})
              and date >= '{START}' and date <= '{END}'
            order by symbol, date"""
    df = con.execute(q).df()
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def main():
    df = load_prices()
    print(f"loaded {df['symbol'].nunique()} symbols, {len(df)} rows, "
          f"{df['date'].min().date()}..{df['date'].max().date()}\n")

    PCTS = [0.10, 0.15, 0.20]
    SS = [0.05, 0.08, 0.10, 0.15]
    COSTS = [0.0, 0.004]
    MIN_LEG_BARS = 5

    rows = []
    # cache legs per (symbol, pct)
    for pct in PCTS:
        # collect all legs across universe
        legs_all = []  # (sym, close_arr, b, p, year)
        for sym, g in df.groupby('symbol'):
            close = g['close'].to_numpy(np.float64)
            dates = g['date'].to_numpy()
            for (b, p) in up_legs(close, pct, MIN_LEG_BARS):
                yr = pd.Timestamp(dates[b]).year
                legs_all.append((close, b, p, yr))
        for s in SS:
            if s >= pct:
                continue
            for cost in COSTS:
                agg = {}
                for scope in ('all', 'ge2022'):
                    agg[scope] = dict(n=0, hold=0.0, swing=0.0, orc=0.0, win=0, rt=0)
                for (close, b, p, yr) in legs_all:
                    h = hold_ret(close, b, p, cost)
                    sw, nrt = swing_causal_ret(close, b, p, s, cost)
                    oc = oracle_inner_ret(close, b, p, s, cost)
                    for scope in ('all', 'ge2022'):
                        if scope == 'ge2022' and yr < 2022:
                            continue
                        a = agg[scope]
                        a['n'] += 1; a['hold'] += h; a['swing'] += sw; a['orc'] += oc
                        a['win'] += int(sw > h); a['rt'] += nrt
                for scope in ('all', 'ge2022'):
                    a = agg[scope]
                    if a['n'] == 0:
                        continue
                    rows.append(dict(pct=pct, s=s, cost=cost, scope=scope, n=a['n'],
                        hold=a['hold']*100, swing=a['swing']*100, orc=a['orc']*100,
                        winrate=a['win']/a['n'], rt=a['rt']/a['n']))

    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200, 'display.max_rows', 200)
    for scope in ('all', 'ge2022'):
        print(f"\n===== SCOPE = {scope} (sum of per-leg returns, in % / 'u') =====")
        sub = res[res.scope == scope].copy()
        sub = sub[['pct','s','cost','n','hold','swing','orc','winrate','rt']]
        sub['sw-hold'] = sub['swing'] - sub['hold']
        sub['ceil-hold'] = sub['orc'] - sub['hold']
        sub = sub.round({'pct':2,'s':2,'cost':3,'hold':0,'swing':0,'orc':0,
                         'winrate':3,'rt':1,'sw-hold':0,'ceil-hold':0})
        print(sub.to_string(index=False))

    # LAG SWEEP: how precise must a predictor be? (perfect sub-turn mask, executed L bars late)
    print("\n===== LAG SWEEP (perfect sub-swing mask executed L bars late, cost=0.004) =====")
    print("L=0 is the ceiling; the break-even L is the max lag a classifier can afford.")
    for pct, s in [(0.15, 0.08), (0.15, 0.10), (0.20, 0.10)]:
        legs_ls = []
        for sym, g in df.groupby('symbol'):
            close = g['close'].to_numpy(np.float64)
            dates = g['date'].to_numpy()
            for (b, p) in up_legs(close, pct, MIN_LEG_BARS):
                legs_ls.append((close, b, p, pd.Timestamp(dates[b]).year))
        print(f"\n  pct={pct} s={s}  (n={len(legs_ls)} legs)")
        for scope in ('all', 'ge2022'):
            base_hold = sum(hold_ret(c, b, p, 0.004) for (c, b, p, y) in legs_ls
                            if scope == 'all' or y >= 2022) * 100
            line = f"    [{scope:6s}] hold={base_hold:7.0f} | "
            for L in (0, 1, 2, 3):
                sw = sum(oracle_lagged_ret(c, b, p, s, 0.004, L) for (c, b, p, y) in legs_ls
                         if scope == 'all' or y >= 2022) * 100
                line += f"L{L}:{sw - base_hold:+7.0f}  "
            print(line)

    # verdict
    print("\n===== VERDICT =====")
    real = res[(res.cost == 0.004)]
    for scope in ('all', 'ge2022'):
        sc = real[real.scope == scope]
        wins = sc[sc.swing > sc.hold]
        best = sc.loc[(sc.swing - sc.hold).idxmax()]
        print(f"[{scope}] net-of-cost: SWING beats HOLD in {len(wins)}/{len(sc)} (pct,s) cells. "
              f"best cell pct={best.pct} s={best.s}: swing={best.swing:.0f} vs hold={best.hold:.0f} "
              f"(Δ={best.swing-best.hold:+.0f}), ceiling={best.orc:.0f}")


if __name__ == "__main__":
    main()
