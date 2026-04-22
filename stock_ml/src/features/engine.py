"""
Feature engineering engine.
Computes all technical indicators and features from OHLCV data.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


class FeatureEngine:
    """
    Compute features from OHLCV data.
    Organized by feature groups that can be enabled/disabled via config.
    """

    EXTRA_GROUPS = {"A", "B", "C", "D", "E", "F"}

    def __init__(self, feature_set: str = "minimal", scaling: str = "robust",
                 extra_groups: Optional[List[str]] = None):
        self.feature_set = feature_set
        self.scaling = scaling
        self.extra_groups = set(extra_groups or [])
        self._scaler = None
        self._feature_columns: List[str] = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all enabled features for a single symbol DataFrame."""
        df = df.copy()

        # Always compute price action basics
        df = self._price_action(df)
        df = self._volume_features(df)
        df = self._moving_averages(df)

        if self.feature_set in ("technical", "full"):
            df = self._momentum(df)
            df = self._trend_indicators(df)
            df = self._volatility_indicators(df)
            df = self._volume_advanced(df)

        if self.feature_set == "full":
            df = self._regime_features(df)

        if self.feature_set in ("leading", "full_v2", "leading_v2"):
            df = self._price_action(df)
            df = self._volume_features(df)
            df = self._moving_averages(df)
            df = self._momentum(df)
            df = self._trend_indicators(df)
            df = self._volatility_indicators(df)
            df = self._volume_advanced(df)
            df = self._leading_signals(df)

        if self.feature_set == "leading_v2":
            df = self._market_structure(df)
            df = self._exhaustion_signals(df)
            df = self._volatility_regime(df)
            df = self._multi_timeframe(df)

        if "A" in self.extra_groups:
            df = self._market_structure(df)
        if "B" in self.extra_groups:
            df = self._exhaustion_signals(df)
        if "C" in self.extra_groups:
            df = self._volatility_regime(df)
        if "D" in self.extra_groups:
            df = self._multi_timeframe(df)
        if "E" in self.extra_groups:
            pass  # handled at pool level in compute_for_all_symbols
        if "F" in self.extra_groups:
            df = self._liquidity_features(df)

        return df

    def compute_for_all_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature computation per symbol in a pooled DataFrame."""
        parts = []
        for symbol, group in df.groupby("symbol"):
            group = self.compute(group)
            parts.append(group)
        result = pd.concat(parts, ignore_index=True)

        if "E" in self.extra_groups:
            result = self._relative_strength(result)

        return result

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of computed feature column names (exclude metadata & target)."""
        exclude = {
            "timestamp", "symbol", "exchange", "asset_type",
            "data_provider", "timeframe", "open", "high", "low",
            "close", "volume", "traded_value", "target",
        }
        return [c for c in df.columns if c not in exclude]

    # ── Price Action ──────────────────────────────────────────────

    def _price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        df["return_1d"] = c.pct_change(1)
        df["return_5d"] = c.pct_change(5)
        df["return_10d"] = c.pct_change(10)
        df["return_20d"] = c.pct_change(20)
        df["log_return_1d"] = np.log(c / c.shift(1))
        df["volatility_10d"] = df["return_1d"].rolling(10).std()
        df["volatility_20d"] = df["return_1d"].rolling(20).std()
        df["high_low_range"] = (df["high"] - df["low"]) / c
        df["close_to_high"] = (df["high"] - c) / c
        df["close_to_low"] = (c - df["low"]) / c
        return df

    # ── Volume ────────────────────────────────────────────────────

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        v = df["volume"].replace(0, np.nan)
        df["volume_ratio_5d"] = v / v.rolling(5).mean()
        df["volume_ratio_20d"] = v / v.rolling(20).mean()
        if "traded_value" in df.columns:
            tv = df["traded_value"].replace(0, np.nan)
            df["traded_value_ratio_5d"] = tv / tv.rolling(5).mean()
        return df

    # ── Moving Averages ───────────────────────────────────────────

    def _moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        for w in [5, 10, 20, 50]:
            df[f"sma_{w}"] = c.rolling(w).mean()
        for w in [10, 20]:
            df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

        df["price_to_sma20"] = c / df["sma_20"] - 1
        df["price_to_sma50"] = c / df["sma_50"] - 1
        df["sma10_cross_sma20"] = (df["sma_10"] > df["sma_20"]).astype(int)
        return df

    # ── Momentum ──────────────────────────────────────────────────

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]

        # RSI
        for period in [7, 14]:
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Stochastic
        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Williams %R
        df["williams_r"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

        # ROC
        df["roc_10"] = c.pct_change(10) * 100
        df["roc_20"] = c.pct_change(20) * 100
        df["momentum_10"] = c - c.shift(10)

        return df

    # ── Trend ─────────────────────────────────────────────────────

    def _trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ADX (simplified)
        high, low = df["high"], df["low"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([
            high - low,
            (high - c.shift(1)).abs(),
            (low - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        df["plus_di"] = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        df["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = (
            (df["plus_di"] - df["minus_di"]).abs()
            / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
            * 100
        )
        df["adx_14"] = dx.rolling(14).mean()

        # CCI
        tp = (df["high"] + df["low"] + c) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df["cci_20"] = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)

        # Aroon
        df["aroon_up"] = df["high"].rolling(25).apply(
            lambda x: x.argmax() / 24 * 100, raw=True
        )
        df["aroon_down"] = df["low"].rolling(25).apply(
            lambda x: x.argmin() / 24 * 100, raw=True
        )

        return df

    # ── Volatility ────────────────────────────────────────────────

    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]

        # Bollinger Bands
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"] = (sma20 + 2 * std20 - c) / c
        df["bb_lower"] = (c - sma20 + 2 * std20) / c
        df["bb_width"] = (4 * std20) / sma20.replace(0, np.nan)
        df["bb_pctb"] = (c - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

        # ATR
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - c.shift(1)).abs(),
            (low - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / c

        # Keltner Channels
        ema20 = c.ewm(span=20, adjust=False).mean()
        df["keltner_upper"] = (ema20 + 2 * df["atr_14"] - c) / c
        df["keltner_lower"] = (c - ema20 + 2 * df["atr_14"]) / c

        return df

    # ── Volume Advanced ───────────────────────────────────────────

    def _volume_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        c, v = df["close"], df["volume"]

        # OBV
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_slope_10"] = obv.diff(10) / obv.rolling(10).mean().replace(0, np.nan)

        # VWAP ratio (rolling approximation)
        cum_vp = (c * v).rolling(20).sum()
        cum_v = v.rolling(20).sum()
        vwap = cum_vp / cum_v.replace(0, np.nan)
        df["vwap_ratio"] = c / vwap - 1

        # MFI
        tp = (df["high"] + df["low"] + c) / 3
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        mfr = pos_mf / neg_mf.replace(0, np.nan)
        df["mfi_14"] = 100 - (100 / (1 + mfr))

        # CMF
        clv = ((c - df["low"]) - (df["high"] - c)) / (df["high"] - df["low"]).replace(0, np.nan)
        df["cmf_20"] = (clv * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

        return df

    # ── Regime Features ───────────────────────────────────────────

    def _regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        h = df["high"]
        l = df["low"]

        # 52-week (252 trading days) high/low
        high_252 = h.rolling(252, min_periods=50).max()
        low_252 = l.rolling(252, min_periods=50).min()
        df["distance_from_52w_high"] = (high_252 - c) / c
        df["distance_from_52w_low"] = (c - low_252) / c

        # Days since 52w high/low
        df["days_since_52w_high"] = h.rolling(252, min_periods=50).apply(
            lambda x: len(x) - 1 - x.argmax(), raw=True
        )
        df["days_since_52w_low"] = l.rolling(252, min_periods=50).apply(
            lambda x: len(x) - 1 - x.argmin(), raw=True
        )

        return df

    # ── Leading Signals (Early Detection) ─────────────────────────

    def _leading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features that LEAD price moves - detect setups BEFORE breakout."""
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"].replace(0, np.nan)

        # 1. VOLUME PRECURSORS
        vol_ma5 = v.rolling(5).mean()
        vol_ma20 = v.rolling(20).mean()
        df["vol_surge_ratio"] = vol_ma5 / vol_ma20  # >1.5 = volume building
        # Price-volume divergence: volume up but price flat
        df["pv_divergence"] = df.get("volume_ratio_20d", v / vol_ma20) - abs(df.get("return_5d", c.pct_change(5)))

        # 2. VOLATILITY CONTRACTION (precedes expansion/breakout)
        if "bb_width" not in df.columns:
            sma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            bb_w = (4 * std20) / sma20.replace(0, np.nan)
        else:
            bb_w = df["bb_width"]
        df["bb_width_percentile"] = bb_w.rolling(60).rank(pct=True)  # low = squeeze
        
        atr5 = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1).rolling(5).mean()
        atr20 = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1).rolling(20).mean()
        df["atr_contraction"] = atr5 / atr20.replace(0, np.nan)  # <0.8 = squeezing

        # 3. ACCUMULATION PATTERNS
        df["close_position_in_range"] = (c - l) / (h - l).replace(0, np.nan)  # >0.7 = bullish close
        df["close_pos_ma5"] = df["close_position_in_range"].rolling(5).mean()  # sustained buying
        
        # OBV trend vs price trend divergence
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        obv_slope = obv.diff(10) / obv.rolling(10).mean().abs().replace(0, np.nan)
        price_slope = c.pct_change(10)
        df["obv_price_divergence"] = obv_slope - price_slope  # positive = accumulation

        # 4. SUPPORT/RESISTANCE PROXIMITY
        high_20 = h.rolling(20).max()
        low_20 = l.rolling(20).min()
        df["dist_to_resistance"] = (high_20 - c) / c  # small = near breakout
        df["dist_to_support"] = (c - low_20) / c  # small = near support
        df["range_position_20d"] = (c - low_20) / (high_20 - low_20).replace(0, np.nan)

        # 5. MARKET STRUCTURE
        # Higher lows count (bullish structure)
        lows = l.values
        hl_count = pd.Series(0.0, index=df.index)
        for i in range(4, len(lows)):
            cnt = 0
            for j in range(1, 5):
                if i-j >= 0 and lows[i-j+1] > lows[i-j]:
                    cnt += 1
            hl_count.iloc[i] = cnt
        df["higher_lows_count"] = hl_count  # 4 = strong bullish structure

        # Consolidation detection: days with range < 2%
        daily_range_pct = (h - l) / c
        df["consolidation_score"] = (daily_range_pct < 0.02).rolling(10).sum()  # high = tight consolidation

        # 6. MOMENTUM SHIFT DETECTION
        # RSI divergence: price making lower lows but RSI making higher lows
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"]
        else:
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
        
        df["rsi_slope_5d"] = rsi.diff(5)  # positive = momentum building
        df["price_rsi_divergence"] = rsi.diff(10) - (c.pct_change(10) * 100)  # bullish divergence

        # 7. BREAKOUT SETUP SCORE (composite)
        # Combine signals: vol building + tight range + near resistance + accumulation
        df["breakout_setup_score"] = (
            (df["vol_surge_ratio"] > 1.2).astype(float) +
            (df["bb_width_percentile"] < 0.3).astype(float) +
            (df["dist_to_resistance"] < 0.02).astype(float) +
            (df["close_pos_ma5"] > 0.6).astype(float) +
            (df["higher_lows_count"] >= 3).astype(float)
        )

        return df

    # ── Group A: Market Structure ───────────────────────────────────

    def _market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(df)

        for order in [3, 5, 7]:
            ph = np.zeros(n)
            pl = np.zeros(n)
            for i in range(order, n - order):
                if all(h[i] >= h[i - j] for j in range(1, order + 1)) and \
                   all(h[i] >= h[i + j] for j in range(1, min(order + 1, n - i))):
                    ph[i] = 1.0
                if all(l[i] <= l[i - j] for j in range(1, order + 1)) and \
                   all(l[i] <= l[i + j] for j in range(1, min(order + 1, n - i))):
                    pl[i] = 1.0
            df[f"pivot_high_{order}"] = ph
            df[f"pivot_low_{order}"] = pl

        last_swing_h = np.full(n, np.nan)
        last_swing_l = np.full(n, np.nan)
        sh_val, sl_val = np.nan, np.nan
        ph5 = df["pivot_high_5"].values
        pl5 = df["pivot_low_5"].values
        for i in range(n):
            if ph5[i] == 1.0:
                sh_val = h[i]
            if pl5[i] == 1.0:
                sl_val = l[i]
            last_swing_h[i] = sh_val
            last_swing_l[i] = sl_val
        df["dist_to_last_swing_high"] = (last_swing_h - c) / np.where(c > 0, c, 1.0)
        df["dist_to_last_swing_low"] = (c - last_swing_l) / np.where(c > 0, c, 1.0)

        bos_up = np.zeros(n)
        bos_down = np.zeros(n)
        choch = np.zeros(n)
        prev_sh = np.nan
        prev_sl = np.nan
        last_direction = 0
        for i in range(1, n):
            if ph5[i] == 1.0 and not np.isnan(prev_sh):
                if h[i] > prev_sh:
                    bos_up[i] = 1.0
                    if last_direction == -1:
                        choch[i] = 1.0
                    last_direction = 1
            if pl5[i] == 1.0 and not np.isnan(prev_sl):
                if l[i] < prev_sl:
                    bos_down[i] = 1.0
                    if last_direction == 1:
                        choch[i] = -1.0
                    last_direction = -1
            if ph5[i] == 1.0:
                prev_sh = h[i]
            if pl5[i] == 1.0:
                prev_sl = l[i]
        df["bos_up"] = bos_up
        df["bos_down"] = bos_down
        df["choch"] = choch

        for window in [20, 40]:
            regime = np.zeros(n)
            for i in range(window, n):
                seg_h = h[i - window:i + 1]
                seg_l = l[i - window:i + 1]
                hh_count = sum(1 for j in range(1, len(seg_h)) if seg_h[j] > seg_h[j - 1])
                hl_count = sum(1 for j in range(1, len(seg_l)) if seg_l[j] > seg_l[j - 1])
                lh_count = sum(1 for j in range(1, len(seg_h)) if seg_h[j] < seg_h[j - 1])
                ll_count = sum(1 for j in range(1, len(seg_l)) if seg_l[j] < seg_l[j - 1])
                total = window
                bull_score = (hh_count + hl_count) / total
                bear_score = (lh_count + ll_count) / total
                if bull_score > 0.6:
                    regime[i] = 1
                elif bear_score > 0.6:
                    regime[i] = -1
            df[f"hh_hl_regime_{window}"] = regime

        return df

    # ── Group B: Exhaustion & Failure Signals ────────────────────────

    def _exhaustion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        v = df["volume"].values.astype(float)
        n = len(df)

        vol_ma20 = pd.Series(v).rolling(20).mean().values

        upthrust = np.zeros(n)
        spring = np.zeros(n)
        for i in range(1, n):
            rng = h[i] - l[i]
            if rng <= 0:
                continue
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            prev_high = np.max(h[max(0, i - 20):i]) if i >= 1 else h[i]
            prev_low = np.min(l[max(0, i - 20):i]) if i >= 1 else l[i]
            if h[i] > prev_high and c[i] < o[i] and upper_wick > 0.5 * rng:
                upthrust[i] = 1.0
            if l[i] < prev_low and c[i] > o[i] and lower_wick > 0.5 * rng:
                spring[i] = 1.0
        df["upthrust"] = upthrust
        df["spring"] = spring

        climax_up = np.zeros(n)
        climax_down = np.zeros(n)
        for i in range(20, n):
            vol_ratio = v[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1.0
            if vol_ratio < 2.0:
                continue
            rng = h[i] - l[i]
            if rng <= 0:
                continue
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            if c[i] > c[i - 1] and upper_wick > 0.4 * rng:
                climax_up[i] = vol_ratio
            if c[i] < c[i - 1] and lower_wick > 0.4 * rng:
                climax_down[i] = vol_ratio
        df["climax_volume_up"] = climax_up
        df["climax_volume_down"] = climax_down

        gap_up = np.zeros(n)
        gap_down = np.zeros(n)
        gap_filled = np.zeros(n)
        for i in range(1, n):
            gap = (o[i] - c[i - 1]) / c[i - 1] if c[i - 1] > 0 else 0
            if gap > 0.005:
                gap_up[i] = gap
                if c[i] < o[i]:
                    gap_filled[i] = 1.0
            elif gap < -0.005:
                gap_down[i] = abs(gap)
                if c[i] > o[i]:
                    gap_filled[i] = -1.0
        df["gap_up_pct"] = gap_up
        df["gap_down_pct"] = gap_down
        df["gap_filled"] = gap_filled

        reversal = np.zeros(n)
        for i in range(3, n):
            vol_ok = v[i] > vol_ma20[i] * 1.2 if not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0 else False
            if c[i - 3] > c[i - 2] > c[i - 1] and c[i] > c[i - 1] and vol_ok:
                reversal[i] = 1.0
            elif c[i - 3] < c[i - 2] < c[i - 1] and c[i] < c[i - 1] and vol_ok:
                reversal[i] = -1.0
        df["reversal_3bar"] = reversal

        return df

    # ── Group C: Volatility Regime + Expansion Timing ────────────────

    def _volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]

        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()

        df["atr_percentile_60d"] = atr14.rolling(60, min_periods=20).rank(pct=True)
        df["true_range_percentile"] = tr.rolling(60, min_periods=20).rank(pct=True)
        df["overnight_gap_pct"] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)

        if "bb_width" in df.columns:
            bb_w = df["bb_width"]
        else:
            sma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            bb_w = (4 * std20) / sma20.replace(0, np.nan)
        bb_pct20 = bb_w.rolling(60, min_periods=20).rank(pct=True)
        squeeze = (bb_pct20 < 0.2).astype(float)
        compress = np.zeros(len(df))
        cnt = 0
        for i in range(len(df)):
            if squeeze.iloc[i] == 1.0:
                cnt += 1
            else:
                cnt = 0
            compress[i] = cnt
        df["compression_duration"] = compress

        ret1d = c.pct_change(1)
        vol5 = ret1d.rolling(5).std()
        vol20 = ret1d.rolling(20).std()
        expansion = (vol5 > vol20 * 1.5).astype(float)
        post_fail = np.zeros(len(df))
        for i in range(3, len(df)):
            if expansion.iloc[i - 3] == 1.0 or expansion.iloc[i - 2] == 1.0:
                recent_ret = (c.iloc[i] - c.iloc[i - 2]) / c.iloc[i - 2] if c.iloc[i - 2] > 0 else 0
                if abs(recent_ret) < 0.01:
                    post_fail[i] = 1.0
        df["post_expansion_failure"] = post_fail

        return df

    # ── Group D: Multi-timeframe Context ─────────────────────────────

    def _multi_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        h = df["high"]
        l = df["low"]

        df["weekly_return_1w"] = c.pct_change(5)
        df["weekly_return_4w"] = c.pct_change(20)

        wma20 = c.rolling(100).mean()  # ~20 weeks * 5 days
        wma50 = c.rolling(250).mean()  # ~50 weeks * 5 days
        df["price_vs_wma20"] = c / wma20.replace(0, np.nan) - 1
        df["price_vs_wma50"] = c / wma50.replace(0, np.nan) - 1

        sma20_daily = c.rolling(20).mean()
        daily_trend = (c > sma20_daily).astype(int)
        weekly_trend = (c > wma20).astype(int)
        df["weekly_trend_alignment"] = (daily_trend == weekly_trend).astype(float)

        high_20w = h.rolling(100, min_periods=20).max()
        low_20w = l.rolling(100, min_periods=20).min()
        df["weekly_swing_proximity"] = (c - low_20w) / (high_20w - low_20w).replace(0, np.nan)

        return df

    # ── Group E: Relative Strength (cross-sectional) ─────────────────

    def _relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional RS features. Must be called on pooled DataFrame."""
        for period in [20, 60]:
            col = f"_ret_{period}d"
            df[col] = df.groupby("symbol")["close"].pct_change(period)

        for ts, ts_group in df.groupby("timestamp"):
            pass  # just force the groupby to work

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

        close_chg5 = df.groupby("symbol")["close"].pct_change(5)
        close_chg20 = df.groupby("symbol")["close"].pct_change(20)
        price_flat = (close_chg20.abs() < 0.03).astype(float)
        rs_rising = (df["rs_vs_market_20d"] > 0).astype(float)
        df["rs_divergence"] = price_flat * rs_rising

        df.drop(columns=["_ret_20d", "_ret_60d"], inplace=True)
        return df

    # ── Group F: Liquidity/Execution Reality ─────────────────────────

    def _liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"].replace(0, np.nan)

        ret1d = c.pct_change(1).abs()
        if "traded_value" in df.columns:
            tv = df["traded_value"].replace(0, np.nan)
        else:
            tv = c * v
        df["amihud_illiquidity_20d"] = (ret1d / tv).rolling(20, min_periods=5).mean()

        turnover = v / v.rolling(252, min_periods=20).mean().replace(0, np.nan)
        df["turnover_percentile_60d"] = turnover.rolling(60, min_periods=20).rank(pct=True)

        df["avg_spread_proxy"] = (h - l) / c.replace(0, np.nan)

        vol_mean = v.rolling(20).mean()
        vol_std = v.rolling(20).std()
        df["volume_consistency"] = vol_std / vol_mean.replace(0, np.nan)

        return df

    def add_market_context(
        self, df: pd.DataFrame, context_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge market context features (VNIndex, HNXIndex, futures) into stock data.
        """
        # Prepare context features
        for ctx_name, ctx_df in context_data.items():
            prefix = ctx_name.lower().replace("index", "index")
            ctx = ctx_df[["timestamp", "close", "volume"]].copy()
            ctx[f"{prefix}_return_1d"] = ctx["close"].pct_change(1)
            ctx[f"{prefix}_return_5d"] = ctx["close"].pct_change(5)

            sma20 = ctx["close"].rolling(20).mean()
            ctx[f"{prefix}_sma20_ratio"] = ctx["close"] / sma20 - 1

            # RSI
            delta = ctx["close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            ctx[f"{prefix}_rsi_14"] = 100 - (100 / (1 + rs))

            ctx[f"{prefix}_volatility_20d"] = ctx["close"].pct_change().rolling(20).std()

            # Select only computed columns
            merge_cols = ["timestamp"] + [
                c for c in ctx.columns
                if c.startswith(prefix) and c != "timestamp"
            ]
            ctx_merge = ctx[merge_cols].copy()

            # Merge on timestamp (date)
            df = df.merge(ctx_merge, on="timestamp", how="left")

        return df

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureEngine":
        feat_cfg = config.get("features", config)
        return cls(
            feature_set=feat_cfg.get("feature_set", "minimal"),
            scaling=feat_cfg.get("scaling", "robust"),
        )
