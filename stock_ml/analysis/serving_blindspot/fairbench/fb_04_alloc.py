"""BREAKTHROUGH PROBE v2 — cross-sectional SLOT ALLOCATION on the champion's REAL trade menu.

Uses the champion-lineage REAL trades (enriched_trades t1844: 1702 trades, real fills/exits,
mean pnl +6.8%, max 54 concurrent -> a K=25 book JAMS: occupancy mean 21.9, p90 41). Every prior
NAV attempt monetized via SIZING and failed; NOBODY tried ALLOCATION of the K equal-weight slots.

Ranker = score3 (the continuation head): the ONLY entry head with a cross-sectional within-day IC
that survives >=2022 (+0.055 pooled; per-year +0.02/-0.07/+0.15/+0.12; plain `score` is useless at
-0.03). No sizing, no extra slots, no external data.

Menu = enriched trades (symbol, entry, exit, score3). Returns marked close-to-close on
market.duckdb (consistent across policies). K-slot equal-weight. Policies:
  P0 baseline  : same-day competing entries funded in RANDOM order (avg over seeds) up to free slots.
  P1 rankfill  : fund highest score3 first.
  P2 switch    : rankfill + drop the lowest-score3 holding for a fresh entry whose score3 exceeds it
                 by >= margin (opportunity-cost exit). Causal (entry score3 for both).
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

COST = 0.004
UNIVERSE_DB = "market_data/market.duckdb"


def load_prices():
    con = duckdb.connect(UNIVERSE_DB, read_only=True)
    df = con.execute("select symbol, date, close from ohlcv where timeframe='1D'").df()
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    return {s: g.set_index('date')['close'].sort_index() for s, g in df.groupby('symbol')}


def load_menu():
    d = pd.read_csv("results/_forensic_t1844/enriched_trades.csv",
                    parse_dates=['entry_date', 'exit_date'])
    d = d[['symbol', 'entry_date', 'exit_date', 'score3', 'pnl']].copy()
    d = d.rename(columns={'entry_date': 'entry', 'exit_date': 'exit'})
    return d.sort_values('entry').reset_index(drop=True)


def simulate(menu, price, dates, K, policy, margin=0.0, seed=0):
    rng = np.random.RandomState(seed)
    entries_by_day = {}
    for r in menu.itertuples():
        entries_by_day.setdefault(r.entry, []).append(r)
    open_pos = {}   # sym -> dict(basis, entry_price, planned_exit, rank)
    cash = 1.0
    eq_hist = []

    def px(sym, d):
        s = price.get(sym)
        if s is None:
            return None
        idx = s.index.searchsorted(d, side='right') - 1
        return float(s.iloc[idx]) if idx >= 0 else None

    for d in dates:
        # planned exits
        for sym in list(open_pos.keys()):
            if open_pos[sym]['planned_exit'] <= d:
                cur = px(sym, d) or open_pos[sym]['entry_price']
                cash += open_pos[sym]['basis'] * (cur / open_pos[sym]['entry_price']) * (1 - COST)
                del open_pos[sym]

        fresh = [r for r in entries_by_day.get(d, []) if r.symbol not in open_pos]
        if policy == 'baseline':
            rng.shuffle(fresh)
        else:
            fresh.sort(key=lambda r: (r.score3 if r.score3 == r.score3 else -9), reverse=True)

        # opportunity switch
        if policy == 'switch' and fresh:
            for r in fresh:
                if len(open_pos) < K:
                    break
                weak = min(open_pos.keys(), key=lambda s: open_pos[s]['rank'])
                if (r.score3 if r.score3 == r.score3 else -9) > open_pos[weak]['rank'] + margin:
                    cur = px(weak, d) or open_pos[weak]['entry_price']
                    cash += open_pos[weak]['basis'] * (cur / open_pos[weak]['entry_price']) * (1 - COST)
                    del open_pos[weak]
                else:
                    break

        # fund into free slots
        for r in fresh:
            if len(open_pos) >= K:
                break
            ep = px(r.symbol, d)
            if ep is None or r.symbol in open_pos:
                continue
            equity_now = cash + sum(open_pos[s]['basis'] * ((px(s, d) or open_pos[s]['entry_price'])
                         / open_pos[s]['entry_price']) for s in open_pos)
            bet = equity_now / K
            if cash >= bet:
                cash -= bet
                open_pos[r.symbol] = dict(basis=bet * (1 - COST), entry_price=ep,
                                          planned_exit=r.exit,
                                          rank=(r.score3 if r.score3 == r.score3 else -9))

        val = sum(open_pos[s]['basis'] * ((px(s, d) or open_pos[s]['entry_price'])
                  / open_pos[s]['entry_price']) for s in open_pos)
        eq_hist.append((d, cash + val))

    return pd.Series({d: e for d, e in eq_hist})


def stats(eqs):
    ret = eqs.iloc[-1] / eqs.iloc[0]
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    cagr = ret ** (1 / yrs) - 1
    mdd = (eqs / eqs.cummax() - 1).min()
    e22 = eqs[eqs.index >= pd.Timestamp('2022-01-01')]
    r22 = e22.iloc[-1] / e22.iloc[0]
    y22 = (e22.index[-1] - e22.index[0]).days / 365.25
    c22 = r22 ** (1 / y22) - 1
    mdd22 = (e22 / e22.cummax() - 1).min()
    return ret, cagr, mdd, r22, c22, mdd22


def main():
    price = load_prices()
    menu = load_menu()
    dates = pd.date_range(menu['entry'].min(), menu['exit'].max(), freq='B')
    print(f"menu {len(menu)} real trades, {menu['entry'].min().date()}..{menu['exit'].max().date()}; "
          f"ranker=score3 (>=2022 within-day IC +0.055)\n")

    print(f"{'policy':>16} {'K':>3} {'navx':>6} {'cagr':>6} {'mdd':>7} | {'nav22':>6} {'cagr22':>7} {'mdd22':>7}")
    for K in (10, 15, 20, 25):
        # baseline averaged over seeds
        bs = [stats(simulate(menu, price, dates, K, 'baseline', seed=s)) for s in range(6)]
        b = np.mean(bs, axis=0)
        print(f"{'baseline(avg6)':>16} {K:>3} {b[0]:>6.2f} {b[1]:>+6.2f} {b[2]:>+7.2f} | "
              f"{b[3]:>6.2f} {b[4]:>+7.2f} {b[5]:>+7.2f}")
        for policy, marg in [('rankfill', 0), ('switch', 0.05), ('switch', 0.15)]:
            r = stats(simulate(menu, price, dates, K, policy, marg))
            tag = policy + (f"-m{marg}" if policy == 'switch' else "")
            d22 = r[3] - b[3]
            print(f"{tag:>16} {K:>3} {r[0]:>6.2f} {r[1]:>+6.2f} {r[2]:>+7.2f} | "
                  f"{r[3]:>6.2f} {r[4]:>+7.2f} {r[5]:>+7.2f}  (nav22 {d22:+.2f} vs base)")
        print()

    print("===== READ =====")
    print("rankfill/switch nav22 > baseline nav22 WITHOUT worse mdd22 = allocation harvests the")
    print("cross-sectional score3 edge to recover jammed-book pnl (no sizing/multipos/external).")


if __name__ == "__main__":
    main()
