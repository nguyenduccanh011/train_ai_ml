"""Build a broad CORE-SIGNAL feature battery (cross-sectional ranks of many signal families) +
forward returns on the 61-symbol universe, saved to parquet. Shared by the core-signal research
probes. Goal: find the NEXT regime-robust feature (how RS-momentum was found) — features whose
forward IC stays positive in the dead years (2024/2026) where the champion flips.

All features are causal cross-sectional ranks [0,1] within-day (survive the recombine z-scoring per
[[cross-sectional-survives-zscoring]]) or per-symbol causal stats. No external data.
"""
import numpy as np, pandas as pd, duckdb

ROOT = "f:/PROJECTS/train_ai_ml"
MARKET = f"{ROOT}/market_data/market.duckdb"
OUT = f"{ROOT}/stock_ml/analysis/regime_predict/core_features.parquet"


def build():
    # champion universe
    syms = sorted(pd.read_parquet(
        f"{ROOT}/bundles/bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf/prediction_history.parquet"
    )["symbol"].unique())
    con = duckdb.connect(MARKET, read_only=True)
    inl = ",".join(repr(s) for s in syms)
    px = con.execute(f"select symbol,date,open,high,low,close,volume from ohlcv where timeframe='1D' "
                     f"and symbol in ({inl}) and date>='2018-06-01' order by symbol,date").fetchdf()
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    close = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    high = px.pivot_table(index="date", columns="symbol", values="high", aggfunc="last").sort_index()
    low = px.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
    vol = px.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last").sort_index()

    def csr(df):  # cross-sectional rank [0,1] within each day
        return df.rank(axis=1, pct=True)

    feats = {}
    # --- MOMENTUM ranks (RS family, various horizons) ---
    for h in (5, 10, 20, 60, 126):
        feats[f"mom{h}_rank"] = csr(close / close.shift(h) - 1.0)
    # momentum acceleration / rotation
    feats["mom_accel_rank"] = csr((close / close.shift(20) - 1.0) - (close.shift(20) / close.shift(40) - 1.0))
    feats["mom_rot_rank"] = csr(csr(close / close.shift(20) - 1.0) - csr(close.shift(10) / close.shift(30) - 1.0))
    # --- VOLUME / LIQUIDITY ranks ---
    dollar = close * vol
    feats["dvol20_rank"] = csr(dollar.rolling(20).mean())          # liquidity
    feats["volsurge_rank"] = csr(vol / (vol.rolling(20).mean() + 1e-9))  # volume surge
    feats["dvol_trend_rank"] = csr(dollar.rolling(5).mean() / (dollar.rolling(60).mean() + 1e-9))
    # accumulation: up-volume vs down-volume (per-symbol OBV-like), ranked
    ret1 = close.pct_change()
    upvol = (vol * (ret1 > 0)).rolling(20).sum()
    dnvol = (vol * (ret1 < 0)).rolling(20).sum()
    feats["accum_rank"] = csr(upvol / (upvol + dnvol + 1e-9))
    # --- VOLATILITY ranks ---
    feats["rvol20_rank"] = csr(ret1.rolling(20).std())
    feats["lowvol_rank"] = csr(-ret1.rolling(60).std())            # low-vol preference
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()]).groupby(level=[0, 1]).max() if False else (high - low)
    feats["atr_contract_rank"] = csr((high - low).rolling(5).mean() / ((high - low).rolling(60).mean() + 1e-9))
    # --- TREND / DISTANCE ranks ---
    for m in (20, 50, 100, 200):
        feats[f"dist_ma{m}_rank"] = csr(close / close.rolling(m).mean() - 1.0)
    feats["ma_slope50_rank"] = csr(close.rolling(50).mean() / close.rolling(50).mean().shift(10) - 1.0)
    # --- BREAKOUT / NEW-HIGH ranks ---
    feats["dist_252high_rank"] = csr(close / high.rolling(252).max() - 1.0)
    feats["dist_63high_rank"] = csr(close / high.rolling(63).max() - 1.0)
    feats["nearhigh20_rank"] = csr(close / high.rolling(20).max() - 1.0)
    # --- MEAN-REVERSION / OVERSOLD ranks ---
    d = close.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / (dn + 1e-9))
    feats["rsi_rank"] = csr(rsi)
    feats["oversold_rank"] = csr(-(close / close.rolling(20).mean() - 1.0))   # most-below-MA
    feats["dist_20low_rank"] = csr(close / low.rolling(20).min() - 1.0)
    # --- consolidation / range ---
    feats["consol_rank"] = csr(-(high.rolling(20).max() / low.rolling(20).min() - 1.0))  # tight range
    feats["rangepos_rank"] = csr((close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-9))

    # --- forward returns (targets) ---
    fwd = {}
    for h in (5, 10, 20):
        fwd[f"fwd{h}"] = close.shift(-h) / close - 1.0
        fwd[f"fwd{h}_xs"] = fwd[f"fwd{h}"] - fwd[f"fwd{h}"].mean(axis=1).values.reshape(-1, 1)  # cross-sectional demean
        fwd[f"fwd{h}_rank"] = csr(fwd[f"fwd{h}"])  # cross-sectional rank of forward return (cs target)

    # stack to long
    out = None
    def melt(df, name):
        m = df.stack().rename(name).reset_index()
        m.columns = ["date", "symbol", name]
        return m
    base = melt(close, "close")[["date", "symbol"]]
    frames = [base.set_index(["date", "symbol"])]
    for nm, df in {**feats, **fwd}.items():
        frames.append(melt(df, nm).set_index(["date", "symbol"])[nm])
    out = pd.concat(frames, axis=1).reset_index()
    out["year"] = out["date"].dt.year
    out.to_parquet(OUT)
    print(f"saved {OUT}: {len(out)} rows, {len([c for c in out.columns if c.endswith('_rank') and not c.startswith('fwd')])} features")
    return out


if __name__ == "__main__":
    build()
