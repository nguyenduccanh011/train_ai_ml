"""NICHE LOSS MAP — trade-level offline analysis (khong chay backtest).
So gb_x08 (top-1) voi cac ghost superseded (dsb60, sx_w60, sx_w40) + pyramid + champion.
Outputs: per-year pnl, oracle ceiling, niche map per ghost (year x month x hold x symbol),
market-context tu duckdb (breadth/floor/limit-up/value-shock), intraday data depth.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
SQ = HERE.parent
BS = SQ.parent  # serving_blindspot

FILES = {
    "gb_x08":   HERE / "gb_x08_s42_trades.csv",
    "champ":    SQ / "st_champ2646_s42_trades.csv",
    "dsb60":    SQ / "vx_dsb60_trades.csv",
    "sx_w60":   SQ / "autopsy" / "sx_w60_s42_trades.csv",
    "sx_w40":   SQ / "autopsy" / "sx_w40_s42_trades.csv",
    "pyramid":  BS / "week0" / "best_trades" / "trades_w0_pyr_u10_r02.csv",
}

T = {}
for k, f in FILES.items():
    df = pd.read_csv(f, parse_dates=["entry_date", "exit_date"])
    df["y"] = df.entry_date.dt.year
    df["ym"] = df.entry_date.dt.to_period("M")
    T[k] = df
    print(f"loaded {k:8s} n={len(df):4d} pnl={df.pnl_pct.sum():8.3f}")

# ---------------- 1. per-entry-year pnl ----------------
years = sorted(set().union(*[set(d.y) for d in T.values()]))
tab = pd.DataFrame({k: d.groupby("y").pnl_pct.sum() for k, d in T.items()}).reindex(years).fillna(0)
print("\n== PNL theo NAM ENTRY (u) ==")
print(tab.round(2).to_string())

# ---------------- 2. oracle ceiling ----------------
POOL = ["gb_x08", "dsb60", "sx_w60", "champ"]
oracle = tab[POOL].max(axis=1)
pick = tab[POOL].idxmax(axis=1)
print("\n== ORACLE (moi nam chon model tot nhat trong 4) ==")
for y in years:
    print(f"  {y}: pick={pick[y]:7s} {oracle[y]:7.2f}  (gb_x08 {tab.loc[y,'gb_x08']:7.2f}, d={oracle[y]-tab.loc[y,'gb_x08']:+6.2f})")
print(f"  ORACLE total = {oracle.sum():.2f}  vs gb_x08 {tab['gb_x08'].sum():.2f}  -> tran regime-switching = {oracle.sum()-tab['gb_x08'].sum():+.2f}u")
POOL5 = POOL + ["pyramid"]
oracle5 = tab[POOL5].max(axis=1)
print(f"  (them pyramid vao pool: oracle = {oracle5.sum():.2f}, d vs gb_x08 = {oracle5.sum()-tab['gb_x08'].sum():+.2f}u; picks: "
      + ", ".join(f"{y}={tab.loc[y, POOL5].idxmax()}" for y in years) + ")")

# ---------------- 3. niche map per ghost vs gb_x08 ----------------
gb = T["gb_x08"]
for g in ["dsb60", "sx_w60", "sx_w40", "pyramid"]:
    v = T[g]
    m = gb.merge(v, on=["symbol", "entry_date"], how="outer", suffixes=("_g", "_v"), indicator=True)
    both = m[m._merge == "both"].copy()
    gonly = m[m._merge == "left_only"]
    vonly = m[m._merge == "right_only"]
    both["d"] = both.pnl_pct_v - both.pnl_pct_g
    dtot = both.d.sum() + vonly.pnl_pct_v.sum() - gonly.pnl_pct_g.sum()
    print(f"\n==== {g} vs gb_x08 ====  (matched {len(both)}, {g}-only {len(vonly)} [{vonly.pnl_pct_v.sum():+.2f}], gb-only {len(gonly)} [{gonly.pnl_pct_g.sum():+.2f}], d_total={dtot:+.2f}u)")
    both["y"] = both.entry_date.dt.year
    ytab = pd.DataFrame({
        "d_matched": both.groupby("y").d.sum(),
        "vonly": vonly.groupby(vonly.entry_date.dt.year).pnl_pct_v.sum(),
        "gonly_lost": -gonly.groupby(gonly.entry_date.dt.year).pnl_pct_g.sum(),
    }).fillna(0)
    ytab["d_year"] = ytab.sum(axis=1)
    print(ytab.round(2).to_string())
    # months where ghost wins the most (matched delta + vonly - gonly)
    both["ym"] = both.entry_date.dt.to_period("M")
    dm = (both.groupby("ym").d.sum()
          .add(vonly.groupby(vonly.entry_date.dt.to_period("M")).pnl_pct_v.sum(), fill_value=0)
          .add(-gonly.groupby(gonly.entry_date.dt.to_period("M")).pnl_pct_g.sum(), fill_value=0))
    top = dm.sort_values(ascending=False)
    print(f"  top thang ghost THANG: " + ", ".join(f"{i}={v:+.2f}" for i, v in top.head(8).items()))
    print(f"  top thang ghost THUA:  " + ", ".join(f"{i}={v:+.2f}" for i, v in top.tail(5).items()))
    # trade traits of ghost-advantage matched pairs
    adv = both[both.d > 0.05]
    dis = both[both.d < -0.05]
    if len(adv):
        print(f"  matched adv (d>+5pp): n={len(adv)} sum={adv.d.sum():+.2f} | hold ghost med {adv.holding_days_v.median():.0f} vs gb {adv.holding_days_g.median():.0f} | top mã: "
              + ", ".join(f"{s}{r:+.2f}" for s, r in adv.groupby('symbol').d.sum().sort_values(ascending=False).head(6).items()))
    if len(dis):
        print(f"  matched dis (d<-5pp): n={len(dis)} sum={dis.d.sum():+.2f} | top mã: "
              + ", ".join(f"{s}{r:+.2f}" for s, r in dis.groupby('symbol').d.sum().sort_values().head(6).items()))

# ---------------- 4. market context (duckdb) ----------------
import duckdb
con = duckdb.connect(str(HERE.parents[4] / "market_data" / "market.duckdb"), read_only=True)
daily = con.execute("""
    WITH d AS (
      SELECT symbol, date, close,
             close / lag(close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS ret,
             traded_value
      FROM ohlcv WHERE timeframe='1D' AND date >= '2019-06-01'
    )
    SELECT date,
           avg(ret) AS mean_ret,
           avg(CASE WHEN ret <= -0.065 THEN 1 ELSE 0 END) AS pct_floor,
           avg(CASE WHEN ret >= 0.065 THEN 1 ELSE 0 END) AS pct_ceil,
           sum(traded_value) AS tval,
           count(*) AS n
    FROM d WHERE ret IS NOT NULL GROUP BY date ORDER BY date
""").df()
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.set_index("date")
daily["tval_z"] = (daily.tval - daily.tval.rolling(60).mean()) / daily.tval.rolling(60).std()
mon = daily.resample("M").agg(mean_ret=("mean_ret", "sum"), pct_floor=("pct_floor", "mean"),
                               pct_ceil=("pct_ceil", "mean"), tval_z=("tval_z", "mean"))
mon.index = mon.index.to_period("M")

# ghost-win months (dsb60 + sx_w60 union of top months)
print("\n== BOI CANH THI TRUONG cac thang ghost thang manh nhat (universe breadth tu duckdb) ==")
gg = T["sx_w60"]
m2 = gb.merge(gg, on=["symbol", "entry_date"], how="outer", suffixes=("_g", "_v"), indicator=True)
b2 = m2[m2._merge == "both"].copy(); b2["d"] = b2.pnl_pct_v - b2.pnl_pct_g
b2["ym"] = b2.entry_date.dt.to_period("M")
dmw = b2.groupby("ym").d.sum()
sel = dmw.sort_values(ascending=False).head(6).index.tolist() + dmw.sort_values().head(3).index.tolist()
for p in sel:
    if p in mon.index:
        r = mon.loc[p]
        print(f"  {p}: sx_w60 d={dmw[p]:+.2f} | uni_ret_thang={r.mean_ret*100:6.1f}% pct_ceil={r.pct_ceil*100:4.1f}% pct_floor={r.pct_floor*100:4.1f}% tval_z={r.tval_z:+.2f}")

# ---------------- 5. fast regime signal probe (lead >= 1 bar) ----------------
# Cau hoi: tin hieu NHANH nao (limit-up count, value shock, breadth thrust 5d) bat duoc
# melt-up 2020-21 SOM hon MA60, va co tat kip truoc 2022 khong?
daily["ceil5"] = daily.pct_ceil.rolling(5).mean()
daily["thrust"] = (daily.mean_ret.rolling(5).sum())
sig = (daily.ceil5 >= 0.02) & (daily.tval_z > 0.5)   # >=2% ma tran + value dang no
sig_on = sig.groupby(daily.index.to_period("M")).mean()
print("\n== FAST-SIGNAL PROBE: %ngay/thang co (ceil5>=2% & tval_z>0.5) ==")
for yy in range(2020, 2026):
    row = sig_on[sig_on.index.year == yy]
    print(f"  {yy}: " + " ".join(f"{p.month:02d}={v*100:3.0f}%" for p, v in row.items()))

# intraday depth
try:
    print("\nintraday 5m depth:", con.execute("SELECT min(date), max(date), count(DISTINCT symbol) FROM ohlcv WHERE timeframe='5m'").fetchall())
    print("intraday 1m depth:", con.execute("SELECT min(date), max(date), count(DISTINCT symbol) FROM ohlcv WHERE timeframe='1m'").fetchall())
except Exception as e:
    print("intraday ERR", e)
con.close()
print("\nDONE")
