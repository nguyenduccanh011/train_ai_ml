"""Shared trade-level forensic harness for the exit-geometry / dead-year campaign.

Loads the champion gb_x08's ACTUAL trades and enriches each with:
  - MFE/MAE during the hold (peak favorable / worst adverse excursion from entry fill)
  - give_back = MFE - realized pnl (how much of the peak was handed back)
  - capture  = realized / MFE (fraction of the peak captured)
  - entry head score (entry ensemble) on the signal date
  - observable regime state on the entry day: cross-sectional dispersion tercile of dist_ma20,
    market EW>MA50, VNINDEX-proxy trend

Every diagnostic imports load_trades() so all agents share ONE validated build. gb_x08 is the
composite champion WITHOUT the rgskip exit — the clean base to see the raw round-trip problem the
exit-geometry idea targets.

Usage:
    from trade_forensics import load_trades
    tr = load_trades()   # one row per trade, ~1378 rows
"""
import numpy as np
import pandas as pd
import duckdb
import sqlalchemy as sa

ROOT = "f:/PROJECTS/train_ai_ml"
MARKET = f"{ROOT}/market_data/market.duckdb"
PG = "postgresql+psycopg2://stockml:stockml_dev@localhost:5433/stockml"
CHAMP_RUN = "template/gb_x08-32a8dfee"


def _price_panel(symbols):
    con = duckdb.connect(MARKET, read_only=True)
    inl = ",".join(repr(s) for s in symbols)
    px = con.execute(
        f"""select symbol,date,open,high,low,close from ohlcv
            where timeframe='1D' and symbol in ({inl}) and date>='2018-06-01'
            order by symbol,date"""
    ).fetchdf()
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    return px.sort_values(["symbol", "date"]).reset_index(drop=True)


def _regime_series(px):
    """Causal observable regime state per date (universe-level)."""
    piv = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    ma20 = piv.rolling(20).mean()
    dist = piv / ma20 - 1.0
    disp = dist.std(axis=1).rename("disp_dm20")              # cross-sectional dispersion (D9 metric)
    # trailing-tercile label of dispersion (causal expanding quantiles)
    q33 = disp.rolling(252, min_periods=60).quantile(0.33)
    q66 = disp.rolling(252, min_periods=60).quantile(0.66)
    disp_bucket = pd.Series("mid", index=disp.index)
    disp_bucket[disp <= q33] = "low"
    disp_bucket[disp >= q66] = "high"
    ret1 = piv.pct_change()
    ew = (1.0 + ret1.mean(axis=1).fillna(0)).cumprod()
    ew_ma50 = ew.rolling(50).mean()
    mkt_bull = (ew > ew_ma50).astype(float).rename("mkt_bull")   # EW index > MA50
    breadth = (piv > piv.rolling(50).mean()).mean(axis=1).rename("breadth")
    reg = pd.concat([disp, disp_bucket.rename("disp_bucket"), mkt_bull, breadth], axis=1)
    reg.index.name = "date"
    return reg.reset_index()


def load_trades():
    e = sa.create_engine(PG)
    with e.connect() as c:
        tr = pd.read_sql(sa.text(
            "select symbol,entry_date,exit_date,entry_price,exit_price,holding_days,pnl_pct,"
            "exit_reason,entry_signal_date from run_trades where run_id=:r"),
            c, params={"r": CHAMP_RUN})
        sig = pd.read_sql(sa.text(
            "select symbol,date,score,exit_score from run_signals where run_id=:r"),
            c, params={"r": CHAMP_RUN})
    for col in ("entry_date", "exit_date", "entry_signal_date"):
        tr[col] = pd.to_datetime(tr[col])
    sig["date"] = pd.to_datetime(sig["date"])

    px = _price_panel(sorted(tr["symbol"].unique()))
    # per-symbol date->(high,low) lookup for MFE/MAE over the hold
    by_sym = {s: g.set_index("date")[["high", "low"]] for s, g in px.groupby("symbol")}

    mfe = np.full(len(tr), np.nan)
    mae = np.full(len(tr), np.nan)
    for k, row in enumerate(tr.itertuples(index=False)):
        pl = by_sym.get(row.symbol)
        if pl is None:
            continue
        seg = pl.loc[(pl.index >= row.entry_date) & (pl.index <= row.exit_date)]
        if len(seg) == 0 or not row.entry_price:
            continue
        mfe[k] = seg["high"].max() / row.entry_price - 1.0
        mae[k] = seg["low"].min() / row.entry_price - 1.0
    tr["mfe"] = mfe
    tr["mae"] = mae
    tr["give_back"] = tr["mfe"] - tr["pnl_pct"]                 # peak handed back
    tr["capture"] = np.where(tr["mfe"] > 1e-6, tr["pnl_pct"] / tr["mfe"], np.nan)

    # entry head score on the signal date (fallback: entry_date)
    smap = {(r.symbol, r.date): r.score for r in sig.itertuples(index=False)}
    tr["score"] = [smap.get((s, d), smap.get((s, ed)))
                   for s, d, ed in zip(tr["symbol"], tr["entry_signal_date"], tr["entry_date"])]

    reg = _regime_series(px)
    tr = tr.merge(reg, left_on="entry_signal_date", right_on="date", how="left").drop(columns=["date"])
    tr["entry_year"] = tr["entry_date"].dt.year
    tr["exit_year"] = tr["exit_date"].dt.year
    return tr


if __name__ == "__main__":
    t = load_trades()
    print("trades", len(t), "cols", list(t.columns))
    print("MFE/MAE non-null:", t["mfe"].notna().sum(), "| score non-null:", t["score"].notna().sum(),
          "| disp_bucket non-null:", t["disp_bucket"].notna().sum())
    print("\nmean give_back by entry_year:")
    print(t.groupby("entry_year")[["pnl_pct", "mfe", "give_back", "capture"]].mean().round(4).to_string())
