# -*- coding: utf-8 -*-
"""hb_94: MEAN-REV SLEEVE lap slot trong (fresh clue hb_93: 2024 = han han momentum, book util ~60%,
40% cash idle). Sinh mean-rev trades RULE THO (mua dip oversold TRONG uptrend = decorrelated voi
momentum mua suc manh), MERGE vao so momentum (struct_to 3185 s42), NavSim2 @K25 uu tien momentum
(same-day momentum fill truoc). So momentum-only vs +sleeve THEO NAM. Neu combined > momentum, nhat la
dead-year 2024/26 -> thesis dung (sleeve khong can multi-lot vi book chua day). Rule tho chi de PROVE;
neu duong -> dau tu ML mean-rev target."""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, duckdb, pandas as pd, numpy as np
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; HERE = Path(__file__).parent
MOM_RID = "template/x2_struct_to-69338138"


def gen_meanrev(syms, dip=-0.08, rsi_lo=15.0, max_hold=12, stop=-0.08, weak_only=False, weak_ma=50):
    """Buy oversold dip in an uptrend; exit on reclaim of MA20 / max_hold / stop. Causal.
    weak_only: only enter when equal-weight market index < its MA(weak_ma) (momentum starved = slots idle)."""
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,open,high,low,close from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date'])
    # causal equal-weight market regime (weak = index < MA)
    piv = px.pivot_table(index='date', columns='symbol', values='close').sort_index()
    idx = (1.0 + piv.pct_change().mean(axis=1)).cumprod()
    weak = (idx < idx.rolling(weak_ma, min_periods=weak_ma // 2).mean())
    weak_map = weak.to_dict()
    rows = []
    for s, g in px.groupby('symbol'):
        g = g.sort_values('date').reset_index(drop=True)
        c = g['close']; ma20 = c.rolling(20, min_periods=20).mean(); ma100 = c.rolling(100, min_periods=100).mean()
        # RSI(2) Wilder
        delta = c.diff(); up = delta.clip(lower=0); dn = -delta.clip(upper=0)
        rs = up.rolling(2).mean() / (dn.rolling(2).mean() + 1e-9); rsi2 = 100 - 100 / (1 + rs)
        uptrend = c > ma100                          # long-term up
        dipcond = (c / ma20 - 1.0 < dip) & (rsi2 < rsi_lo)  # short-term oversold dip
        cond = (uptrend & dipcond).to_numpy()
        dates_s = g['date'].tolist()
        n = len(g); i = 0
        while i < n - 1:
            gate = (not weak_only) or bool(weak_map.get(dates_s[i], False))
            if cond[i] and gate and not np.isnan(ma100.iloc[i]):
                e = i + 1                            # enter next bar open (causal)
                if e >= n: break
                ep = g['open'].iloc[e]; xj = None
                for j in range(e + 1, min(e + 1 + max_hold, n)):
                    ret = g['close'].iloc[j] / ep - 1.0
                    if (g['close'].iloc[j] > ma20.iloc[j]) or ret <= stop:  # reclaim MA20 or stop
                        xj = j; break
                if xj is None: xj = min(e + max_hold, n - 1)
                rows.append(dict(symbol=s, entry_date=g['date'].iloc[e], exit_date=g['date'].iloc[xj],
                                 entry_price=ep, exit_price=g['close'].iloc[xj]))
                i = xj + 1
            else:
                i += 1
    return pd.DataFrame(rows)


def nav_years(csv):
    out = {}
    for lo, lab in [("2020-01-01", "full")] + [(f"{y}-01-01", str(y)) for y in range(2020, 2027)]:
        try:
            out[lab] = shuffle_stats(NavSim2(str(csv), date_lo=lo), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=12)["mean"]
        except Exception:
            out[lab] = float('nan')
    return out


def main():
    con = psycopg2.connect(**PG)
    mom = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                      "where run_id=%s and exit_date is not null", con, params=(MOM_RID,))
    con.close()
    mom['entry_date'] = pd.to_datetime(mom['entry_date']); mom['exit_date'] = pd.to_datetime(mom['exit_date'])
    syms = mom.symbol.unique().tolist()
    mom['_src'] = 0
    cmom = HERE / "_k94_mom.csv"
    mom[['symbol','entry_date','exit_date','entry_price','exit_price']].to_csv(cmom, index=False)
    nm = nav_years(cmom)
    print("\n  book       " + "  ".join(f"{k:>7s}" for k in nm), flush=True)
    print("  momentum   " + "  ".join(f"{nm[k]:7.2f}" for k in nm), flush=True)
    for lab, kw in [("+mr_all", dict(weak_only=False)), ("+mr_weak50", dict(weak_only=True, weak_ma=50)),
                    ("+mr_weak100", dict(weak_only=True, weak_ma=100))]:
        mr = gen_meanrev(syms, **kw); mr = mr[mr.exit_date >= "2019-06-01"].copy()
        mr['_src'] = 1
        both = pd.concat([mom, mr], ignore_index=True).sort_values(['entry_date', '_src'])
        cb = HERE / f"_k94_{lab}.csv"
        both[['symbol','entry_date','exit_date','entry_price','exit_price']].to_csv(cb, index=False)
        nb = nav_years(cb)
        print(f"  {lab:10s} " + "  ".join(f"{nb[k]:7.2f}" for k in nb) + f"   (mr n={len(mr)})", flush=True)
        print(f"   delta%    " + "  ".join(f"{(nb[k]/nm[k]-1)*100:+6.1f}%" for k in nm), flush=True)
    print("HB_94_DONE", flush=True)


if __name__ == "__main__":
    main()
