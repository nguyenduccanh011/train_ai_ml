"""COMPOSITE-vs-NAV MASK TEST — does the leaderboard (composite) hide a better-NAV model?

The champion is picked by COMPOSITE (unit-weight, unlimited slots). But real money runs a K-slot
NAV book. If a lower-composite menu has HIGHER NAV >=2022 at equal/better MDD, the composite referee
is masking a deployable improvement (no sizing/multipos/external — just a different exit-hold menu).

Menus (real champion-lineage trade files, hb2943_work): t2943 (composite champ), mh16/mh20 (shorter
max_hold), gb_t2783 (older gb_x08: fewer, higher-quality trades). All marked close-to-close on
market.duckdb over each menu's own (entry,exit) windows -> consistent. K-slot equal-weight, baseline
allocation (random tiebreak, avg seeds), net cost 0.4%.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

COST = 0.004
WORK = "../hb2943_work"
MENUS = {
    't2943(composite champ)': 'hb_2943_s42_trades.csv',
    'mh16': 'hb_2943_mh16_s42_trades.csv',
    'mh20': 'hb_2943_mh20_s42_trades.csv',
    'gb_t2783(gb_x08)': 'gb_t2783_s42_trades.csv',
}


def load_prices():
    con = duckdb.connect("market_data/market.duckdb", read_only=True)
    df = con.execute("select symbol, date, close from ohlcv where timeframe='1D'").df()
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    return {s: g.set_index('date')['close'].sort_index() for s, g in df.groupby('symbol')}


def load_menu(fn):
    d = pd.read_csv(f"{WORK}/{fn}", parse_dates=['entry_date', 'exit_date'])
    return d[['symbol', 'entry_date', 'exit_date']].rename(
        columns={'entry_date': 'entry', 'exit_date': 'exit'}).sort_values('entry').reset_index(drop=True)


def simulate(menu, price, dates, K, seed):
    rng = np.random.RandomState(seed)
    ebd = {}
    for r in menu.itertuples():
        ebd.setdefault(r.entry, []).append(r)
    open_pos, cash, eq = {}, 1.0, []

    def px(sym, d):
        s = price.get(sym)
        if s is None:
            return None
        i = s.index.searchsorted(d, side='right') - 1
        return float(s.iloc[i]) if i >= 0 else None

    for d in dates:
        for sym in list(open_pos):
            if open_pos[sym]['xt'] <= d:
                cur = px(sym, d) or open_pos[sym]['ep']
                cash += open_pos[sym]['b'] * (cur / open_pos[sym]['ep']) * (1 - COST)
                del open_pos[sym]
        fresh = [r for r in ebd.get(d, []) if r.symbol not in open_pos]
        rng.shuffle(fresh)
        for r in fresh:
            if len(open_pos) >= K:
                break
            ep = px(r.symbol, d)
            if ep is None:
                continue
            eqn = cash + sum(open_pos[s]['b'] * ((px(s, d) or open_pos[s]['ep']) / open_pos[s]['ep'])
                             for s in open_pos)
            bet = eqn / K
            if cash >= bet:
                cash -= bet
                open_pos[r.symbol] = dict(b=bet * (1 - COST), ep=ep, xt=r.exit)
        val = sum(open_pos[s]['b'] * ((px(s, d) or open_pos[s]['ep']) / open_pos[s]['ep']) for s in open_pos)
        eq.append((d, cash + val))
    return pd.Series({d: e for d, e in eq})


def stats(eqs):
    r = eqs.iloc[-1] / eqs.iloc[0]
    mdd = (eqs / eqs.cummax() - 1).min()
    e = eqs[eqs.index >= pd.Timestamp('2022-01-01')]
    r22 = e.iloc[-1] / e.iloc[0]
    y22 = (e.index[-1] - e.index[0]).days / 365.25
    c22 = r22 ** (1 / y22) - 1
    mdd22 = (e / e.cummax() - 1).min()
    return r, mdd, r22, c22, mdd22


def main():
    price = load_prices()
    menus = {k: load_menu(v) for k, v in MENUS.items()}
    lo = min(m['entry'].min() for m in menus.values())
    hi = max(m['exit'].max() for m in menus.values())
    dates = pd.date_range(lo, hi, freq='B')

    print(f"{'menu':>24} {'K':>3} {'navx':>6} {'mdd':>7} | {'nav22':>6} {'cagr22':>7} {'mdd22':>7}")
    for K in (15, 20, 25):
        for name, menu in menus.items():
            S = np.mean([stats(simulate(menu, price, dates, K, s)) for s in range(6)], axis=0)
            print(f"{name:>24} {K:>3} {S[0]:>6.2f} {S[1]:>+7.2f} | {S[2]:>6.2f} {S[3]:>+7.2f} {S[4]:>+7.2f}")
        print()

    print("===== READ =====")
    print("If gb_t2783 (or mh) nav22/cagr22 > t2943 at equal/better mdd22, the COMPOSITE referee is")
    print("masking a better-NAV deployable model (no sizing/multipos/external).")


if __name__ == "__main__":
    main()
