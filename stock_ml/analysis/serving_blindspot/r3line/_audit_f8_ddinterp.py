# -*- coding: utf-8 -*-
"""AUDIT finding #8: NAV leg marks = linear interp entry-ratio -> exit-ratio (future exit price
injected into intermediate marks). Reproduce: replay gtrail registered portfolio, compare stored
interp marks vs true close-MTM (shares*close anchored at entry-day close), recompute DD (close & intraday-low)."""
import psycopg2, duckdb, pandas as pd, numpy as np

RID = "template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138"
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
eq = pd.read_sql("select date, nav, cash from run_equity where run_id=%s order by date", con, params=(RID,))
hd = pd.read_sql("""select date, symbol, weight, unreal_pnl, entry_date, days_held, is_exit
                    from run_portfolio_daily where run_id=%s order by date""", con, params=(RID,))
con.close()
eq["date"] = eq["date"].astype(str); hd["date"] = hd["date"].astype(str); hd["entry_date"] = hd["entry_date"].astype(str)
hd = hd[~hd.is_exit].copy()
navmap = dict(zip(eq.date, eq.nav)); cashmap = dict(zip(eq.date, eq.cash))

syms = sorted(hd.symbol.unique())
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("select symbol, cast(date as date) d, close, low from ohlcv where timeframe='1D' and symbol in (%s) and d>='2019-06-01'"
                % ",".join("'%s'" % s for s in syms)).fetchdf()
cx.close()
px["d"] = px["d"].astype(str)
CLO = {(r.symbol, r.d): r.close for r in px.itertuples()}
LOW = {(r.symbol, r.d): r.low for r in px.itertuples()}

# per-holding-day: stored interp value vs true close-MTM (invested*close[d]/close[entry_date])
rows, miss = [], 0
for r in hd.itertuples():
    nav = navmap[r.date]
    val_stored = r.weight * nav
    invested = val_stored / (1.0 + r.unreal_pnl) if (1.0 + r.unreal_pnl) != 0 else np.nan
    c0 = CLO.get((r.symbol, r.entry_date)); cj = CLO.get((r.symbol, r.date)); lj = LOW.get((r.symbol, r.date))
    if c0 is None or cj is None or not np.isfinite(invested):
        miss += 1; continue
    val_true = invested * cj / c0
    val_low = invested * (lj if lj is not None else cj) / c0
    rows.append((r.date, r.symbol, r.entry_date, int(r.days_held), invested, val_stored, val_true, val_low))
df = pd.DataFrame(rows, columns=["date","symbol","entry_date","days_held","invested","val_stored","val_true","val_low"])
df["diff_pct"] = (df.val_true - df.val_stored) / df.invested
print("holding-day marks: n=%d (miss=%d)" % (len(df), miss))
mid = df[df.days_held > 0]  # entry-day marks are anchored identical by construction
print("mark diff (true - stored)/invested over days_held>0: n=%d mean %+.2f%% p5 %+.2f%% p95 %+.2f%% min %+.2f%% max %+.2f%%"
      % (len(mid), 100*mid.diff_pct.mean(), 100*mid.diff_pct.quantile(.05), 100*mid.diff_pct.quantile(.95),
         100*mid.diff_pct.min(), 100*mid.diff_pct.max()))
print("|diff|>2%%: %d rows (%.1f%%)  |diff|>5%%: %d rows (%.1f%%)"
      % ((mid.diff_pct.abs()>.02).sum(), 100*(mid.diff_pct.abs()>.02).mean(),
         (mid.diff_pct.abs()>.05).sum(), 100*(mid.diff_pct.abs()>.05).mean()))

# worst single smoothed mark: show the future-exit injection concretely
w = mid.loc[mid.diff_pct.abs().idxmax()]
print("\nworst mark: %s %s (entered %s, day %d): stored %.4f vs true-close %.4f -> diff %+.2f%% of invested"
      % (w.symbol, w.date, w.entry_date, w.days_held, w.val_stored, w.val_true, 100*w.diff_pct))

# NAV path rebuild: same cash, marks replaced by true MTM
agg_s = df.groupby("date").val_stored.sum(); agg_t = df.groupby("date").val_true.sum(); agg_l = df.groupby("date").val_low.sum()
eqi = eq.set_index("date")
nav_st = eqi.cash + agg_s.reindex(eqi.index).fillna(0.0)   # sanity: should ~= stored nav
nav_tr = eqi.cash + agg_t.reindex(eqi.index).fillna(0.0)
nav_lo = eqi.cash + agg_l.reindex(eqi.index).fillna(0.0)
recon_err = (nav_st - eqi.nav).abs().max()
print("\nsanity max|cash+sum(stored marks) - stored nav| = %.2e" % recon_err)

def dd(s): return float((s / s.cummax() - 1).min()), str(s.index[(s / s.cummax() - 1).idxmin() == s.index][0]) if False else None
def ddm(s):
    d = s / s.cummax() - 1
    return float(d.min()), d.idxmin()
dd_reg, dt_reg = ddm(eqi.nav)
dd_tru, dt_tru = ddm(nav_tr)
d_intr = nav_lo / nav_tr.cummax() - 1
dd_int, dt_int = float(d_intr.min()), d_intr.idxmin()
print("registered equity : final x%.2f  maxDD(close) %.2f%% on %s" % (eqi.nav.iloc[-1], 100*dd_reg, dt_reg))
print("true close-MTM    : final x%.2f  maxDD(close) %.2f%% on %s" % (nav_tr.iloc[-1], 100*dd_tru, dt_tru))
print("true intraday-low : maxDD %.2f%% on %s" % (100*dd_int, dt_int))
