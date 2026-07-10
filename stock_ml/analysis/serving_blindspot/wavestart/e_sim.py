"""Final offline test: alternative B-only exit stacks for confirmation-breakout entries.

Cost model replicated from engine (verified vs trades_bch_*.csv):
  entry fill = close(entry_date) * 1.0015 ; exit fill = close(exit_date) * 0.9985
  pnl = exit_fill/entry_fill - 1 - 0.004  (flat round-trip fee)
No lookahead: all exit conditions evaluated on close/low of bar t, fill at close of bar t+1.
Horizon exits fill AT the horizon bar close (decision known in advance).
"""
import sqlite3, json
import pandas as pd, numpy as np

DB = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
BASE = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/wavestart"
SLIP_IN, SLIP_OUT, FEE = 1.0015, 0.9985, 0.004
SPLIT_GAP = 0.18  # |close/close-1| beyond exchange limits => corporate action in raw db

con = sqlite3.connect(DB)

def load_px(symbols):
    q = f"select symbol,date,open,high,low,close from ohlcv where symbol in ({','.join(['?']*len(symbols))}) order by symbol,date"
    px = pd.read_sql(q, con, params=list(symbols))
    return {s: g.reset_index(drop=True) for s, g in px.groupby('symbol')}

def net(entry_close, exit_close):
    return (exit_close*SLIP_OUT)/(entry_close*SLIP_IN) - 1 - FEE

def sim_trade(g, si, ei, scheme):
    """g: per-symbol OHLCV df; si: signal bar idx; ei: entry bar idx. Returns dict or None."""
    n = len(g)
    close, low, high = g['close'].values, g['low'].values, g['high'].values
    ec = close[ei]
    # structural stop level: min low of 10 bars before signal
    stop = low[max(0, si-10):si].min() if si > 0 else np.nan
    ma20 = pd.Series(close).rolling(20).mean().values
    donch = pd.Series(low).rolling(20).min().shift(1).values  # min low of prior 20 bars, excl current

    horizon = {'E1': 60, 'E2': 21, 'E3': 60, 'E4': 42}[scheme]
    peak = ec
    armed = False
    was_above_ma = False
    exit_i, reason = None, None
    for t in range(ei+1, n):
        bars = t - ei
        peak = max(peak, close[t])
        if scheme == 'E1':
            if not armed and peak >= ec*1.10: armed = True
            trig = (low[t] < stop) or (armed and not np.isnan(donch[t]) and close[t] < donch[t])
            reason_t = 'stop' if low[t] < stop else 'donch'
        elif scheme == 'E2':
            trig, reason_t = False, ''
        elif scheme == 'E3':
            trig, reason_t = close[t] < peak*0.92, 'trail8'
        elif scheme == 'E4':
            if not armed and peak >= ec*1.08: armed = True
            if not np.isnan(ma20[t]) and close[t] > ma20[t]: was_above_ma = True
            stop_hit = low[t] < stop
            ma_fail = armed and was_above_ma and not np.isnan(ma20[t]) and close[t] < ma20[t]
            dc_fail = armed and not np.isnan(donch[t]) and close[t] < donch[t]
            trig = stop_hit or ma_fail or dc_fail
            reason_t = 'stop' if stop_hit else ('ma20fail' if ma_fail else 'donch')
        if trig:
            fi = min(t+1, n-1)  # fill next bar close (last bar if data ends)
            exit_i, reason = fi, reason_t
            break
        if bars >= horizon:
            exit_i, reason = t, 'horizon'
            break
    open_flag = False
    if exit_i is None:  # data ran out
        exit_i, reason, open_flag = n-1, 'open', True
    xc = close[exit_i]
    hold = exit_i - ei
    mae = low[ei:exit_i+1].min()/(ec*SLIP_IN) - 1 if exit_i > ei else low[ei]/(ec*SLIP_IN) - 1
    # split guard: raw close-to-close gap beyond limit inside window
    seg = close[ei:exit_i+1]
    gaps = np.abs(np.diff(seg)/seg[:-1])
    split = bool((gaps > SPLIT_GAP).any()) if len(gaps) else False
    return dict(pnl=net(ec, xc), hold=hold, mae=mae, reason=reason,
                exit_date=g['date'].iloc[exit_i], open=open_flag, split_flag=split)

def run_config(name):
    be = pd.read_csv(f"{BASE}/runs/{name}/b_entries.csv")
    tr = pd.read_csv(f"{BASE}/runs/{name}/trades_{name}.csv")
    m = be.merge(tr, left_on=['symbol','signal_date','entry_date'],
                 right_on=['symbol','entry_signal_date','entry_date'], how='left')
    assert m['pnl_pct'].notna().all(), "unmatched b_entries"
    pxs = load_px(m['symbol'].unique())
    rows = []
    for _, r in m.iterrows():
        g = pxs[r['symbol']]
        didx = {d: i for i, d in enumerate(g['date'])}
        si, ei = didx.get(r['signal_date']), didx.get(r['entry_date'])
        if si is None or ei is None:
            rows.append(dict(symbol=r['symbol'], entry_date=r['entry_date'], missing=True)); continue
        base = dict(symbol=r['symbol'], entry_date=r['entry_date'], year=int(r['entry_date'][:4]),
                    e0_pnl=r['pnl_pct'], e0_hold=r['holding_days'], e0_reason=r['exit_reason'], missing=False)
        for sc in ['E1','E2','E3','E4']:
            res = sim_trade(g, si, ei, sc)
            for k, v in res.items(): base[f'{sc}_{k}'] = v
        rows.append(base)
    df = pd.DataFrame(rows)
    df.to_csv(f"{BASE}/runs/{name}/exit_sim.csv", index=False)
    return df

def summarize(df, name):
    out = [f"=== {name} (n={len(df)}) ==="]
    if df.get('missing') is not None and df['missing'].any():
        out.append(f"MISSING in db: {df['missing'].sum()}")
        df = df[~df['missing']]
    schemes = {'E0': ('e0_pnl', 'e0_hold', None),
               'E1': ('E1_pnl','E1_hold','E1_mae'), 'E2': ('E2_pnl','E2_hold','E2_mae'),
               'E3': ('E3_pnl','E3_hold','E3_mae'), 'E4': ('E4_pnl','E4_hold','E4_mae')}
    hdr = f"{'sch':<4}{'net_u':>8}{'ex20_u':>8}{'WR':>6}{'mean%':>7}{'med%':>7}{'hold':>6}{'maeP10':>8}{'MDDu':>7}{'open':>5}{'splitflag':>10}"
    out.append(hdr)
    for sc, (pc, hc, mc) in schemes.items():
        p = df[pc]; ex = df.loc[df['year'] != 2020, pc]
        wr = (p > 0).mean()
        maep10 = df[mc].quantile(0.10) if mc else np.nan
        # cohort cumsum MDD, trades ordered by exit then entry date
        ed = df[f'{sc}_exit_date'] if sc != 'E0' else df['entry_date']
        cs = p.loc[ed.sort_values().index].cumsum()
        mdd = (cs - cs.cummax()).min()
        nopen = int(df[f'{sc}_open'].sum()) if sc != 'E0' else 0
        nsplit = int(df[f'{sc}_split_flag'].sum()) if sc != 'E0' else 0
        out.append(f"{sc:<4}{p.sum():>8.2f}{ex.sum():>8.2f}{wr:>6.2f}{p.mean()*100:>7.2f}{p.median()*100:>7.2f}"
                   f"{df[hc].median():>6.0f}{(maep10*100 if mc else float('nan')):>8.1f}{mdd:>7.2f}{nopen:>5}{nsplit:>10}")
    # per-year
    out.append("\nper-year net (u) / count:")
    yt = df.groupby('year').agg(n=('e0_pnl','size'), E0=('e0_pnl','sum'), E1=('E1_pnl','sum'),
                                E2=('E2_pnl','sum'), E3=('E3_pnl','sum'), E4=('E4_pnl','sum')).round(2)
    out.append(yt.to_string())
    return "\n".join(out), df

for cfg in ['bch_z09', 'bch_z12_lb8']:
    df = run_config(cfg)
    txt, dfc = summarize(df, cfg)
    print(txt)
    # worst 10 MAE for best-looking scheme printed later from CSV
    print()
