"""Forensic on the NEW frontier x2_struct_to (double-RS + structure-trail) to locate the biggest
remaining edge — where is P&L still left on the table, and what characterizes it. Reveals the next
breakthrough niche empirically (vs blind sweeping).
"""
import numpy as np, pandas as pd, duckdb, sqlalchemy as sa

ROOT = "f:/PROJECTS/train_ai_ml"
MARKET = f"{ROOT}/market_data/market.duckdb"
PG = "postgresql+psycopg2://stockml:stockml_dev@localhost:5433/stockml"
RUN = "template/x2_struct_to-69338138"
COST = 0.006


def load():
    e = sa.create_engine(PG)
    with e.connect() as c:
        tr = pd.read_sql(sa.text(
            "select symbol,entry_date,exit_date,entry_price,exit_price,holding_days,pnl_pct,"
            "exit_reason,entry_signal_date from run_trades where run_id=:r"), c, params={"r": RUN})
    for col in ("entry_date", "exit_date", "entry_signal_date"):
        tr[col] = pd.to_datetime(tr[col])
    syms = sorted(tr["symbol"].unique())
    con = duckdb.connect(MARKET, read_only=True)
    inl = ",".join(repr(s) for s in syms)
    px = con.execute(f"select symbol,date,open,high,low,close from ohlcv where timeframe='1D' "
                     f"and symbol in ({inl}) and date>='2018-06-01' order by symbol,date").fetchdf()
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    # cross-sectional momentum rank (RS proxy) per day
    piv = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    ret20 = piv / piv.shift(20) - 1.0
    rs_rank = ret20.rank(axis=1, pct=True)  # [0,1] cross-sectional
    rs_long = rs_rank.stack().rename("rs_rank").reset_index()
    rs_long.columns = ["date", "symbol", "rs_rank"]
    # MFE/MAE per trade
    by = {s: g.set_index("date")[["high", "low"]] for s, g in px.groupby("symbol")}
    mfe = np.full(len(tr), np.nan); mae = np.full(len(tr), np.nan)
    for k, r in enumerate(tr.itertuples(index=False)):
        pl = by.get(r.symbol)
        if pl is None or not r.entry_price:
            continue
        seg = pl.loc[(pl.index >= r.entry_date) & (pl.index <= r.exit_date)]
        if len(seg):
            mfe[k] = seg["high"].max() / r.entry_price - 1.0
            mae[k] = seg["low"].min() / r.entry_price - 1.0
    tr["mfe"] = mfe; tr["mae"] = mae
    tr["give_back"] = tr["mfe"] - tr["pnl_pct"]
    tr["capture"] = np.where(tr["mfe"] > 1e-6, tr["pnl_pct"] / tr["mfe"], np.nan)
    tr = tr.merge(rs_long, left_on=["symbol", "entry_signal_date"], right_on=["symbol", "date"], how="left").drop(columns="date")
    tr["entry_year"] = tr["entry_date"].dt.year
    return tr


if __name__ == "__main__":
    tr = load()
    print(f"frontier trades: {len(tr)}  total pnl {tr['pnl_pct'].sum():.1f}u  avg {tr['pnl_pct'].mean():.4f}")
    print("\n=== per entry-year ===")
    g = tr.groupby("entry_year").agg(n=("pnl_pct", "count"), tot=("pnl_pct", "sum"), avg=("pnl_pct", "mean"),
                                     win=("pnl_pct", lambda x: (x > 0).mean()), mfe=("mfe", "mean"),
                                     gb=("give_back", "mean"), cap=("capture", "median"))
    print(g.round(4).to_string())
    print("\n=== exit_reason distribution (where P&L comes from / leaks) ===")
    er = tr.groupby("exit_reason").agg(n=("pnl_pct", "count"), tot=("pnl_pct", "sum"), avg=("pnl_pct", "mean"),
                                       gb=("give_back", "mean")).sort_values("tot")
    print(er.round(4).to_string())
    print("\n=== RS-rank tercile at entry vs pnl (does the frontier still mis-select by RS?) ===")
    tr["rs_bucket"] = pd.cut(tr["rs_rank"], [0, 0.33, 0.66, 1.0], labels=["lowRS", "midRS", "hiRS"])
    print(tr.groupby("rs_bucket").agg(n=("pnl_pct", "count"), avg=("pnl_pct", "mean"),
                                      win=("pnl_pct", lambda x: (x > 0).mean()), mfe=("mfe", "mean")).round(4).to_string())
    print("\n=== biggest losers: what are they? (bottom 10% pnl) ===")
    lo = tr[tr["pnl_pct"] <= tr["pnl_pct"].quantile(0.10)]
    print(f"n={len(lo)} totpnl={lo['pnl_pct'].sum():.1f}u avg={lo['pnl_pct'].mean():.3f} "
          f"mean_rs={lo['rs_rank'].mean():.3f} mean_hold={lo['holding_days'].mean():.1f} "
          f"mean_mfe={lo['mfe'].mean():.3f} (reached +{lo['mfe'].mean()*100:.1f}% then lost)")
    print("loser exit_reasons:", lo["exit_reason"].value_counts().head(5).to_dict())
