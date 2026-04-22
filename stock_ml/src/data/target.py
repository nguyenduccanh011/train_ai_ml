"""
Target variable generation for stock prediction.
Supports trend regime classification, return-based targets, etc.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class TargetGenerator:
    """
    Generate target labels for ML training.

    Main target: trend_regime (UPTREND=1, SIDEWAYS=0, DOWNTREND=-1)
    This aligns with the goal of catching big waves and avoiding drawdowns.
    """

    def __init__(
        self,
        target_type: str = "trend_regime",
        trend_method: str = "dual_ma",
        short_window: int = 10,
        long_window: int = 40,
        n_classes: int = 3,
        forward_window: int = 10,
        gain_threshold: float = 0.05,
        loss_threshold: float = 0.03,
        rr_threshold: float = 2.0,
    ):
        self.target_type = target_type
        self.trend_method = trend_method
        self.short_window = short_window
        self.long_window = long_window
        self.n_classes = n_classes
        self.forward_window = forward_window
        self.gain_threshold = gain_threshold
        self.loss_threshold = loss_threshold
        self.rr_threshold = rr_threshold

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target column(s) to DataFrame. Expects per-symbol data."""
        df = df.copy()

        if self.target_type == "trend_regime":
            df = self._trend_regime(df)
        elif self.target_type == "return_classification":
            df = self._return_classification(df)
        elif self.target_type == "return_regression":
            df = self._return_regression(df)
        elif self.target_type == "forward_risk_reward":
            df = self._forward_risk_reward(df)
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

        return df

    def _trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each day into UPTREND (1), SIDEWAYS (0), DOWNTREND (-1).

        dual_ma method:
          - UPTREND: short MA > long MA and price > short MA
          - DOWNTREND: short MA < long MA and price < short MA
          - SIDEWAYS: everything else
        """
        if self.trend_method == "dual_ma":
            df["_sma_short"] = df["close"].rolling(self.short_window).mean()
            df["_sma_long"] = df["close"].rolling(self.long_window).mean()

            conditions_up = (
                (df["_sma_short"] > df["_sma_long"])
                & (df["close"] > df["_sma_short"])
            )
            conditions_down = (
                (df["_sma_short"] < df["_sma_long"])
                & (df["close"] < df["_sma_short"])
            )

            if self.n_classes == 3:
                df["target"] = 0  # SIDEWAYS
                df.loc[conditions_up, "target"] = 1  # UPTREND
                df.loc[conditions_down, "target"] = -1  # DOWNTREND
            else:
                # Binary: up vs not-up
                df["target"] = 0
                df.loc[conditions_up, "target"] = 1

            df.drop(columns=["_sma_short", "_sma_long"], inplace=True)

        elif self.trend_method == "hhll":
            # Higher-highs, higher-lows method
            df = self._hhll_regime(df)
        else:
            raise ValueError(f"Unknown trend method: {self.trend_method}")

        # Shift target by -1 so we predict TOMORROW's regime
        df["target"] = df["target"].shift(-1)

        return df

    def _hhll_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend detection via higher-highs/higher-lows pattern.
        Uses rolling window to detect swing points.
        """
        window = self.short_window

        df["_roll_high"] = df["high"].rolling(window).max()
        df["_roll_low"] = df["low"].rolling(window).min()
        df["_prev_roll_high"] = df["_roll_high"].shift(window)
        df["_prev_roll_low"] = df["_roll_low"].shift(window)

        hh = df["_roll_high"] > df["_prev_roll_high"]  # higher high
        hl = df["_roll_low"] > df["_prev_roll_low"]    # higher low
        lh = df["_roll_high"] < df["_prev_roll_high"]  # lower high
        ll = df["_roll_low"] < df["_prev_roll_low"]    # lower low

        df["target"] = 0
        df.loc[hh & hl, "target"] = 1   # UPTREND
        df.loc[lh & ll, "target"] = -1  # DOWNTREND

        df.drop(columns=[
            "_roll_high", "_roll_low", "_prev_roll_high", "_prev_roll_low"
        ], inplace=True)

        return df

    def _return_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify based on forward return threshold."""
        fwd_return = df["close"].pct_change(5).shift(-5)

        if self.n_classes == 3:
            threshold = fwd_return.std() * 0.5
            df["target"] = 0  # neutral
            df.loc[fwd_return > threshold, "target"] = 1
            df.loc[fwd_return < -threshold, "target"] = -1
        else:
            df["target"] = (fwd_return > 0).astype(int)

        return df

    def _return_regression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward return as continuous target."""
        df["target"] = df["close"].pct_change(5).shift(-5)
        return df

    def _forward_risk_reward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-looking target: Is today a good BUY POINT?
        
        Looks N days ahead and checks:
        - Max potential gain >= gain_threshold
        - Max potential loss > -loss_threshold (limited downside)
        - Risk/reward ratio >= rr_threshold
        
        BUY=1 if all conditions met, else 0.
        """
        close = df["close"].values
        n = len(close)
        fw = self.forward_window
        gain_thresh = self.gain_threshold
        loss_thresh = self.loss_threshold
        rr_thresh = self.rr_threshold
        
        targets = np.full(n, 0.0)
        
        for i in range(n - fw):
            future = close[i+1 : i+1+fw]
            if len(future) == 0:
                continue
            
            max_gain = (np.max(future) - close[i]) / close[i]
            max_loss = (np.min(future) - close[i]) / close[i]
            
            # Risk-reward ratio
            abs_loss = abs(max_loss) if max_loss < 0 else 0.001
            rr = max_gain / abs_loss
            
            if max_gain >= gain_thresh and max_loss > -loss_thresh and rr >= rr_thresh:
                targets[i] = 1
        
        df["target"] = targets
        # Mark last fw rows as NaN (no future data)
        df.loc[df.index[-fw:], "target"] = np.nan
        
        return df

    def generate_for_all_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target generation per symbol in a pooled DataFrame."""
        parts = []
        for symbol, group in df.groupby("symbol"):
            group = self.generate(group)
            parts.append(group)

        result = pd.concat(parts, ignore_index=True)
        # Drop rows with NaN target
        result = result.dropna(subset=["target"])
        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TargetGenerator":
        """Create from config dict."""
        tgt_cfg = config.get("target", config)
        return cls(
            target_type=tgt_cfg.get("type", "trend_regime"),
            trend_method=tgt_cfg.get("trend_method", "dual_ma"),
            short_window=tgt_cfg.get("short_window", 10),
            long_window=tgt_cfg.get("long_window", 40),
            n_classes=tgt_cfg.get("classes", 3),
            forward_window=tgt_cfg.get("forward_window", 10),
            gain_threshold=tgt_cfg.get("gain_threshold", 0.05),
            loss_threshold=tgt_cfg.get("loss_threshold", 0.03),
            rr_threshold=tgt_cfg.get("rr_threshold", 2.0),
        )
