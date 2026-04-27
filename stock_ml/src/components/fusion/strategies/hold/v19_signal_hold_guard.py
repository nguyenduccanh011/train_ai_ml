"""V19_3 hold-layer guard: gộp 4 modifiers chặn signal-exit non-forceful.

Map 1-1 với legacy backtest_v19_3 (src/strategies/legacy.py:3091-3168). Chỉ
chạy khi pending_exit_reason == "signal" (driver gắn vào trade_state). Trả
"keep_position" nếu một trong các điều kiện carry/confirm/save khớp.

Trade_state inputs (driver compute pre-bar):
  - pending_exit_reason: str   (chỉ active khi == "signal")
  - cum_ret, max_profit, hold_days, trend
  - raw_signal: 0/1
  - macd_hist_falling: bool
  - bearish_score precomputed? — KHÔNG, ta tự compute từ indicators
  - regime_cfg.exit_score_threshold, base_confirm_bars
  - mods: h, d, i

Counters output via metadata["counters"]:
  - n_v18_signal_quality_saves
  - n_v19_exit_quality_saved
  - n_confirmed_exit_blocked
  - n_time_decay_exit
  - n_trend_carry_saved
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class V19SignalHoldGuard:
    """Guard signal-driven exits with confirm/carry/quality logic."""

    name: str = "v19_signal_hold_guard"
    layer: FusionLayer = "hold"
    priority: int = 50

    EXIT_CONFIRM = 3

    def apply(self, ctx: BarContext) -> FusionResult:  # noqa: PLR0911, PLR0912, PLR0915, C901
        cfg = ctx.config
        ts: dict[str, Any] = cfg.get("trade_state") or {}
        pending = ts.get("pending_exit_reason")
        if pending != "signal":
            return FusionResult(action="pass", reason="")

        ind = cfg.get("indicators")
        if ind is None:
            return FusionResult(action="pass", reason="")
        mods: dict[str, bool] = cfg.get("mods", {})
        regime_cfg: dict[str, Any] = cfg.get("regime_cfg", {})
        i = ctx.bar_idx
        n = int(ind["n"])
        if i >= n:
            return FusionResult(action="pass", reason="")

        close = ind["close"]
        opn = ind["open"]
        sma20 = ind["sma20"]
        sma50 = ind["sma50"]
        ema8 = ind["ema8"]
        macd_hist = ind["macd_hist"]
        avg_vol20 = ind["avg_vol20"]
        volume = ind["volume"]
        ret_5d = ind["ret_5d"]
        feat_arrays = ind["feat_arrays"]

        cum_ret = float(ts.get("cum_ret", 0.0))
        max_profit = float(ts.get("max_profit", 0.0))
        hold_days = int(ts.get("hold_days", 0))
        trend = str(ts.get("trend", "weak"))
        raw_signal = int(ts.get("raw_signal", 0))
        consecutive_exit_signals = int(ts.get("consecutive_exit_signals", 0))

        rs = float(feat_arrays["rsi_slope_5d"][i]) if i < n else 0.0

        counters: dict[str, int] = {}
        state_updates: dict[str, int] = {}
        new_keep = False

        # ---- 1) ConfirmedSignalExitScoring (mod_h) ----
        if mods.get("h", True):
            below_ma20 = not np.isnan(sma20[i]) and close[i] < sma20[i]
            below_ma50 = not np.isnan(sma50[i]) and close[i] < sma50[i]
            old_bearish_confirm = (below_ma20 and macd_hist[i] < 0) or below_ma50
            heavy_vol = not np.isnan(avg_vol20[i]) and volume[i] > 1.4 * avg_vol20[i]
            bearish_candle = close[i] < opn[i]
            macd_falling = macd_hist[i] < macd_hist[i - 1] if i > 0 else False
            below_ema8 = not np.isnan(ema8[i]) and close[i] < ema8[i] * 0.997
            weak_rebound = ret_5d[i] < 0.01 and rs <= 0

            bearish_score = 0.0
            bearish_score += 2.0 if below_ma50 else 0.0
            bearish_score += 1.0 if below_ma20 else 0.0
            bearish_score += 1.0 if macd_hist[i] < -0.03 else 0.0
            bearish_score += 0.8 if (macd_hist[i] < 0 and macd_falling) else 0.0
            bearish_score += 0.8 if (bearish_candle and heavy_vol) else 0.0
            bearish_score += 0.7 if below_ema8 else 0.0
            bearish_score += 0.5 if weak_rebound else 0.0

            score_threshold = float(regime_cfg.get("exit_score_threshold", 2.0))
            if cum_ret > 0.06 and max_profit > 0.10 and trend == "strong":
                score_threshold += 0.7
            if hold_days < 7 and cum_ret > -0.02:
                score_threshold += 0.4
            if cum_ret < -0.03:
                score_threshold -= 0.4
            if hold_days > 15 and cum_ret < 0.03:
                score_threshold *= 0.60
                counters["n_time_decay_exit"] = counters.get("n_time_decay_exit", 0) + 1
            elif hold_days > 10 and cum_ret < 0.01:
                score_threshold *= 0.75

            bearish_confirm = bearish_score >= score_threshold
            if old_bearish_confirm and not bearish_confirm:
                counters["n_v18_signal_quality_saves"] = (
                    counters.get("n_v18_signal_quality_saves", 0) + 1
                )
                counters["n_v19_exit_quality_saved"] = (
                    counters.get("n_v19_exit_quality_saved", 0) + 1
                )
            if not bearish_confirm:
                new_keep = True
                counters["n_confirmed_exit_blocked"] = (
                    counters.get("n_confirmed_exit_blocked", 0) + 1
                )

        # ---- 2) ExitSignalConfirmBars ----
        if not new_keep:
            if cum_ret < 0:
                confirm_bars = 0
            elif mods.get("d", False):
                if cum_ret < 0 or (max_profit > 0 and cum_ret < max_profit * 0.6):
                    confirm_bars = 1
                else:
                    confirm_bars = self.EXIT_CONFIRM
            else:
                confirm_bars = int(regime_cfg.get("base_confirm_bars", 3))
                if cum_ret < -0.03:
                    confirm_bars = max(1, confirm_bars - 1)
            new_consec = consecutive_exit_signals + 1 if raw_signal == 0 else 0
            if new_consec < confirm_bars:
                new_keep = True
                state_updates["consecutive_exit_signals"] = new_consec
            else:
                state_updates["consecutive_exit_signals"] = 0

        # ---- 3) StrongUptrendCarry ----
        if not new_keep:
            if cum_ret > 0.03 and trend == "strong":
                new_keep = True

        # ---- 4) TrendCarryOverride (mod_i) ----
        if not new_keep and mods.get("i", True):
            still_supported = not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.99
            trend_ok = trend in ("strong", "moderate")
            if (
                cum_ret > 0.03
                and max_profit > 0.06
                and trend_ok
                and still_supported
                and macd_hist[i] > -0.02
            ):
                new_keep = True
                counters["n_trend_carry_saved"] = counters.get("n_trend_carry_saved", 0) + 1

        metadata: dict[str, Any] = {"counters": counters, **state_updates}
        if new_keep:
            return FusionResult(
                action="keep_position",
                reason="signal_hold_guard",
                metadata=metadata,
            )
        return FusionResult(action="pass", reason="", metadata=metadata)
