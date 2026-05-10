from __future__ import annotations

import numpy as np
import pandas as pd


class RelativeStrengthBlock:
    """Cross-sectional relative strength. Must be applied AFTER per-symbol compute."""

    name = "relative_strength"
    requires = ["close", "symbol", "timestamp"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-sectional RS. Expects pooled multi-symbol DataFrame."""
        df = df.copy()

        for period in [20, 60]:
            col = f"_ret_{period}d"
            df[col] = df.groupby("symbol")["close"].pct_change(period)

        df["rs_vs_market_20d"] = np.nan
        df["rs_vs_market_60d"] = np.nan
        df["rs_rank_20d"] = np.nan
        df["rs_divergence"] = np.nan

        for ts in df["timestamp"].unique():
            mask = df["timestamp"] == ts
            ret20 = df.loc[mask, "_ret_20d"]
            ret60 = df.loc[mask, "_ret_60d"]
            market_mean_20 = ret20.mean()
            market_mean_60 = ret60.mean()
            df.loc[mask, "rs_vs_market_20d"] = ret20 - market_mean_20
            df.loc[mask, "rs_vs_market_60d"] = ret60 - market_mean_60
            df.loc[mask, "rs_rank_20d"] = ret20.rank(pct=True)

        close_chg20 = df.groupby("symbol")["close"].pct_change(20)
        price_flat = (close_chg20.abs() < 0.03).astype(float)
        rs_rising = (df["rs_vs_market_20d"] > 0).astype(float)
        df["rs_divergence"] = price_flat * rs_rising

        df.drop(columns=["_ret_20d", "_ret_60d"], inplace=True)
        return df

    def get_feature_names(self) -> list[str]:
        return ["rs_vs_market_20d", "rs_vs_market_60d", "rs_rank_20d", "rs_divergence"]
