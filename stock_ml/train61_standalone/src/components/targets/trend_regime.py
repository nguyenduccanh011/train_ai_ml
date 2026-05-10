from __future__ import annotations

import pandas as pd


class TrendRegimeTarget:
    """Classify each bar into UPTREND (1), SIDEWAYS (0), DOWNTREND (-1).

    Mirrors legacy TargetGenerator(target_type="trend_regime").
    """

    name = "trend_regime"

    def __init__(
        self,
        trend_method: str = "dual_ma",
        short_window: int = 10,
        long_window: int = 40,
        n_classes: int = 3,
    ) -> None:
        self.trend_method = trend_method
        self.short_window = short_window
        self.long_window = long_window
        self.n_classes = n_classes

    @property
    def num_classes(self) -> int:
        return self.n_classes

    @property
    def supports_exit_labels(self) -> bool:
        return False

    def generate_entry_labels(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        if self.trend_method == "dual_ma":
            df = self._dual_ma(df)
        elif self.trend_method == "hhll":
            df = self._hhll(df)
        else:
            raise ValueError(f"Unknown trend_method: {self.trend_method!r}")
        return df["target"]

    def generate_exit_labels(
        self,
        df: pd.DataFrame,
        forward_window: int,
        loss_threshold: float,
    ) -> pd.Series | None:
        return None

    def _dual_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        sma_short = df["close"].rolling(self.short_window).mean()
        sma_long = df["close"].rolling(self.long_window).mean()

        cond_up = (sma_short > sma_long) & (df["close"] > sma_short)
        cond_down = (sma_short < sma_long) & (df["close"] < sma_short)

        if self.n_classes == 3:
            target = pd.Series(0, index=df.index, dtype=float)
            target[cond_up] = 1.0
            target[cond_down] = -1.0
        else:
            target = pd.Series(0, index=df.index, dtype=float)
            target[cond_up] = 1.0

        # Predict tomorrow's regime
        df["target"] = target.shift(-1)
        return df

    def _hhll(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.short_window
        roll_high = df["high"].rolling(w).max()
        roll_low = df["low"].rolling(w).min()
        prev_high = roll_high.shift(w)
        prev_low = roll_low.shift(w)

        hh = roll_high > prev_high
        hl = roll_low > prev_low
        lh = roll_high < prev_high
        ll = roll_low < prev_low

        target = pd.Series(0, index=df.index, dtype=float)
        target[hh & hl] = 1.0
        target[lh & ll] = -1.0
        df["target"] = target.shift(-1)
        return df
