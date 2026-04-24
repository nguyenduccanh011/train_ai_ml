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
        elif self.target_type == "early_wave":
            df = self._early_wave(df)
        elif self.target_type == "early_wave_v2":
            df = self._early_wave_v2(df)
        elif self.target_type == "early_wave_dual":
            df = self._early_wave(df)              # primary buy target
            df = self._early_exit_signal(df)       # adds target_sell column
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

    def _early_wave(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Early-wave label: dạy model nhận ĐẦU sóng thay vì giữa/cuối sóng.

        BUY (=1) khi đồng thời:
          - Backward N ngày: giá đi ngang / tích lũy (range nhỏ, return tuyệt đối nhỏ)
          - Forward N ngày: giá tăng mạnh (max_gain >= gain_threshold)
          - Forward N ngày: drawdown hạn chế (min_loss > -loss_threshold)

        Nếu chỉ có uptrend mạnh (no sideway trước đó) → target = 0 (model không học đu sóng).
        Nếu sideway mà không có breakout → target = 0.

        3-class mode (n_classes=3):
          +1 = early wave (BUY tốt)
           0 = trung tính
          -1 = rõ ràng downtrend/giai đoạn tránh

        Thông số:
          forward_window       : số ngày nhìn về tương lai (default 10)
          short_window         : số ngày lookback đánh giá tích lũy (default 10)
          gain_threshold       : min forward gain để gọi BUY (default 0.08)
          loss_threshold       : max forward drawdown cho phép (default 0.05)
          long_window          : số ngày đánh giá downtrend (default 20)
        """
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        n = len(close)
        fw = self.forward_window
        back = self.short_window
        down_win = self.long_window
        gain_thresh = self.gain_threshold
        loss_thresh = self.loss_threshold

        targets = np.full(n, 0.0)

        for i in range(n):
            # === forward lookahead ===
            if i + fw >= n:
                targets[i] = np.nan
                continue
            future = close[i + 1 : i + 1 + fw]
            max_gain = (np.max(future) - close[i]) / close[i] if close[i] > 0 else 0
            max_loss = (np.min(future) - close[i]) / close[i] if close[i] > 0 else 0

            # === backward accumulation check ===
            # Yêu cầu range (high-low)/close < 12% trong N ngày VÀ |return| < 8%
            # Đây là "before breakout" signature
            if i >= back:
                past_h = np.max(high[i - back : i + 1])
                past_l = np.min(low[i - back : i + 1])
                past_range = (past_h - past_l) / close[i] if close[i] > 0 else 1
                past_ret = (close[i] - close[i - back]) / close[i - back] if close[i - back] > 0 else 0
                is_accumulating = (past_range < 0.12) and (abs(past_ret) < 0.08)
            else:
                is_accumulating = False

            # === downtrend check ===
            if i >= down_win:
                long_ret = (close[i] - close[i - down_win]) / close[i - down_win] if close[i - down_win] > 0 else 0
                is_downtrend = long_ret < -0.10
            else:
                is_downtrend = False

            # === label ===
            if is_accumulating and max_gain >= gain_thresh and max_loss > -loss_thresh:
                targets[i] = 1.0  # early wave BUY
            elif self.n_classes == 3 and is_downtrend and max_gain < 0.03:
                targets[i] = -1.0  # downtrend / avoid
            else:
                targets[i] = 0.0

        df["target"] = targets
        return df

    def _early_wave_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """V35a early-wave target — widened.

        Goal: label more buy points after V-shape recoveries (fix DCM-like misses).
        Changes vs _early_wave:
          - Accumulation: relax past_ret < 8% -> past_range < 15% only (no return bound).
          - Add RULE-TRIGGER branch: MACD_hist > 0 AND close > MA20 AND close > open
            AND forward max_gain >= gain_threshold*0.7 -> target=1 (even if not accumulating).
          - Smaller default gain_threshold honored as-is (pass 0.05 from config).
        """
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        open_ = df["open"].values if "open" in df.columns else close
        n = len(close)
        fw = self.forward_window
        back = self.short_window
        down_win = self.long_window
        gain_thresh = self.gain_threshold
        loss_thresh = self.loss_threshold

        # Compute MACD_hist + MA20 once (needed for rule trigger)
        close_s = pd.Series(close)
        ema12 = close_s.ewm(span=12, adjust=False).mean()
        ema26 = close_s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal).values
        ma20 = close_s.rolling(20).mean().values

        targets = np.full(n, 0.0)

        for i in range(n):
            if i + fw >= n:
                targets[i] = np.nan
                continue
            future = close[i + 1 : i + 1 + fw]
            max_gain = (np.max(future) - close[i]) / close[i] if close[i] > 0 else 0
            max_loss = (np.min(future) - close[i]) / close[i] if close[i] > 0 else 0

            # Relaxed accumulation: only range bound
            if i >= back:
                past_h = np.max(high[i - back : i + 1])
                past_l = np.min(low[i - back : i + 1])
                past_range = (past_h - past_l) / close[i] if close[i] > 0 else 1
                is_accumulating = past_range < 0.15
            else:
                is_accumulating = False

            # Rule trigger branch: MACD_hist>0 AND close>MA20 AND close>open
            rule_trigger = (
                i >= 26
                and not np.isnan(macd_hist[i])
                and not np.isnan(ma20[i])
                and macd_hist[i] > 0
                and close[i] > ma20[i]
                and close[i] > open_[i]
            )

            # Downtrend
            if i >= down_win:
                long_ret = (close[i] - close[i - down_win]) / close[i - down_win] if close[i - down_win] > 0 else 0
                is_downtrend = long_ret < -0.10
            else:
                is_downtrend = False

            # Label
            if is_accumulating and max_gain >= gain_thresh and max_loss > -loss_thresh:
                targets[i] = 1.0
            elif rule_trigger and max_gain >= gain_thresh * 0.7 and max_loss > -loss_thresh * 1.3:
                targets[i] = 1.0
            elif self.n_classes == 3 and is_downtrend and max_gain < 0.03:
                targets[i] = -1.0
            else:
                targets[i] = 0.0

        df["target"] = targets
        return df

    def _early_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """V37b: binary 'should-exit-soon' label.

        target_sell=1 if within forward_window N bars the price drops by
        >= loss_threshold from current close (forward drawdown). Else 0.
        Used by V37b dual-head ML to give an early exit signal.
        """
        close = df["close"].values
        n = len(close)
        fw = self.forward_window
        loss_thresh = self.loss_threshold

        sell = np.full(n, 0.0)
        for i in range(n):
            if i + fw >= n:
                sell[i] = np.nan
                continue
            future = close[i + 1 : i + 1 + fw]
            if close[i] <= 0:
                continue
            max_drawdown = (np.min(future) - close[i]) / close[i]
            if max_drawdown <= -loss_thresh:
                sell[i] = 1.0
        df["target_sell"] = sell
        return df

    def generate_for_all_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target generation per symbol in a pooled DataFrame."""
        parts = []
        for symbol, group in df.groupby("symbol"):
            group = self.generate(group)
            parts.append(group)

        result = pd.concat(parts, ignore_index=True)
        # Drop rows with NaN target
        drop_cols = ["target"]
        if "target_sell" in result.columns:
            drop_cols.append("target_sell")
        result = result.dropna(subset=drop_cols)
        return result

    @classmethod
    def generate_exit_labels(
        cls,
        df: pd.DataFrame,
        forward_window: int = 15,
        loss_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Generate target_sell column independently from entry target type.

        Can be called on any DataFrame that already has a 'close' column,
        regardless of the primary target type. Does not require early_wave_dual.
        Operates per symbol and drops NaN rows from target_sell.
        """
        gen = cls(forward_window=forward_window, loss_threshold=loss_threshold)
        parts = []
        for _, group in df.groupby("symbol"):
            group = gen._early_exit_signal(group.copy())
            parts.append(group)
        result = pd.concat(parts, ignore_index=True)
        return result.dropna(subset=["target_sell"])

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
