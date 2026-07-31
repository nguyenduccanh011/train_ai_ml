"""EXIT-DEFER-MIRROR — gate 1: does the champion leave money on the table after SOFT exits >=2022?

The killed sub-top SELL was counter-trend (negative-EV). Its MIRROR is pro-trend: DEFER an exit
when the trend is still strong -> capture more continuation (base-rate favors it for a momentum
system). But the champion already has max_hold/overext_trail/giveback-guard (harvested +5.3) and
EXIT_ATTRIBUTION put remaining exit-timing room ~0-1u. So gate 1 (cheap, decisive): after a SOFT
exit (signal / overext_trail — a model decision, not a hard stop), what is the POST-EXIT forward
return >=2022? If ~0/negative, exits are well-timed -> thread dead. If materially positive and
SEPARABLE by momentum features at the exit bar, a deferral classifier has room.

Champion trades: hb_2943_s42 (current lineage). Post-exit path from market.duckdb close.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

WORK = "../hb2943_work"


def load_prices():
    con = duckdb.connect("market_data/market.duckdb", read_only=True)
    df = con.execute("select symbol, date, close, high, low, volume from ohlcv where timeframe='1D'").df()
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    return {s: g.sort_values('date').reset_index(drop=True) for s, g in df.groupby('symbol')}


def main():
    price = load_prices()
    d = pd.read_csv(f"{WORK}/hb_2943_s42_trades.csv", parse_dates=['entry_date', 'exit_date'])
    d['year'] = d.exit_date.dt.year

    rows = []
    for r in d.itertuples():
        g = price.get(r.symbol)
        if g is None:
            continue
        idx = g['date'].searchsorted(r.exit_date)
        if idx >= len(g) or g['date'].iloc[idx] != r.exit_date:
            # exit date not a trading day in market db; take nearest prior
            idx = g['date'].searchsorted(r.exit_date, side='right') - 1
            if idx < 0:
                continue
        c0 = g['close'].iloc[idx]
        if c0 <= 0:
            continue
        f10 = g['close'].iloc[min(idx + 10, len(g) - 1)] / c0 - 1
        f20 = g['close'].iloc[min(idx + 20, len(g) - 1)] / c0 - 1
        window = g['close'].iloc[idx + 1: idx + 21]
        fmax = (window.max() / c0 - 1) if len(window) else 0.0
        fmin = (window.min() / c0 - 1) if len(window) else 0.0
        # momentum-continuation features AT the exit bar (causal)
        cl = g['close']
        ma20 = cl.rolling(20).mean().iloc[idx] if idx >= 20 else np.nan
        dist_ma20 = c0 / ma20 - 1 if ma20 == ma20 else np.nan
        up_streak = 0
        j = idx
        while j > 0 and cl.iloc[j] > cl.iloc[j - 1]:
            up_streak += 1; j -= 1
        rows.append(dict(sym=r.symbol, year=r.year, reason=r.exit_reason, pnl=r.pnl_pct,
                         f10=f10, f20=f20, fmax=fmax, fmin=fmin,
                         dist_ma20=dist_ma20, up_streak=up_streak))
    e = pd.DataFrame(rows)
    soft = e[e.reason.isin(['signal', 'overext_trail'])]

    print("POST-EXIT forward return by exit_reason (ALL years):")
    for reason, g in e.groupby('reason'):
        print(f"  {reason:16s} n={len(g):4d}  f10 {g.f10.mean():+.3f}  f20 {g.f20.mean():+.3f}  "
              f"fmax20 {g.fmax.mean():+.3f}  %f20>0 {(g.f20>0).mean():.0%}")

    print("\nSOFT exits (signal+overext_trail) post-exit f20, by year:")
    for y, g in soft.groupby('year'):
        print(f"  {y}: n={len(g):4d}  f20 {g.f20.mean():+.4f}  fmax20 {g.fmax.mean():+.4f}  "
              f"%f20>0 {(g.f20>0).mean():.0%}  (fmin20 {g.fmin.mean():+.3f})")

    s22 = soft[soft.year >= 2022]
    print(f"\n>=2022 SOFT exits: n={len(s22)}  mean f20 {s22.f20.mean():+.4f}  "
          f"fmax20(perfect-defer ceiling) {s22.fmax.mean():+.4f}")
    # is post-exit continuation SEPARABLE by momentum at exit? (defer only strong ones)
    from scipy.stats import spearmanr
    for c in ['dist_ma20', 'up_streak']:
        ok = s22.dropna(subset=[c])
        rho = spearmanr(ok[c], ok['f20']).correlation
        print(f"  IC({c} @exit, post-exit f20) = {rho:+.4f}")
    # top-momentum tercile: do strong-at-exit names continue up more?
    ok = s22.dropna(subset=['dist_ma20'])
    q = ok['dist_ma20'].quantile([.33, .67]).values
    hi = ok[ok.dist_ma20 >= q[1]]; loo = ok[ok.dist_ma20 <= q[0]]
    print(f"  strong-at-exit (hi dist_ma20) f20 {hi.f20.mean():+.4f} vs weak {loo.f20.mean():+.4f} "
          f"(spread {hi.f20.mean()-loo.f20.mean():+.4f})")

    print("\n===== GATE =====")
    print("mean f20 >> 0 AND positive IC = champion exits too early on strong names -> deferral has "
          "room. ~0 = exits well-timed -> thread dead (confirms EXIT_ATTRIBUTION).")


if __name__ == "__main__":
    main()
