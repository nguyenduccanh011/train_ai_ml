"""V19_3 entry cascade: gá»™p 5 entry sources + 8 filters + sizing thÃ nh 1 strategy.

Map 1-1 vá»›i legacy backtest_v19_3 (src/strategies/legacy.py:2718-2947) Ä‘á»ƒ Ä‘áº£m báº£o
parity exact vá»›i golden 1910 trades. Logic chia sáº» state (entry_alpha_ok,
breakout_entry, vshape_entry, strong_breakout_context) khÃ³ tÃ¡ch rá»i nÃªn giá»¯
dáº¡ng monolithic â€” Ä‘á»•i láº¡i an toÃ n parity.

Äáº§u vÃ o trong ctx.config:
  - indicators: dict ndarray tá»« helpers.compute_v19_indicators
  - mods: {"a","b","c","d","e","f","g","h","i","j": bool}
  - entry_state: {"cooldown_remaining": int, "last_exit_price": float,
                  "last_exit_reason": str, "last_exit_bar": int,
                  "prev_pred": int}
  - regime_cfg: dict tá»« helpers.get_regime_adapter (Ä‘Ã£ cache theo bar)

Äáº§u ra FusionResult.metadata:
  - size: float (position_size sau khi clip)
  - entry_features: dict (entry_wp/dp/rs/vs/bs/hl/od/bb/score/profile/...)
  - flags: {"quick_reentry","breakout_entry","vshape_entry"}
  - counters: dict[str,int] (n_vshape_entries/n_secondary_breakout/
              n_v19_alpha_blocked/n_v18_relaxed_*/n_bear_blocked/n_chop_blocked/
              n_v19_overheat_entries) â€” driver cá»™ng dá»“n vÃ o counters globals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


_FEATURE_DEFAULTS: dict[str, float] = {
    "rsi_slope_5d": 0.0,
    "vol_surge_ratio": 1.0,
    "range_position_20d": 0.5,
    "dist_to_resistance": 0.05,
    "breakout_setup_score": 0.0,
    "bb_width_percentile": 0.5,
    "higher_lows_count": 0.0,
    "obv_price_divergence": 0.0,
}


def _gf(feat_arrays: dict[str, np.ndarray], name: str, idx: int, n: int) -> float:
    if idx < n:
        return float(feat_arrays[name][idx])
    return _FEATURE_DEFAULTS[name]


def _format_date(value: object) -> str:
    ts = pd.Timestamp(value)
    if ts.time() == pd.Timestamp(ts.date()).time():
        return ts.date().isoformat()
    return ts.isoformat()


class V19EntryCascade:
    """Reproduces v19_3 entry-decision pipeline exactly."""

    name: str = "v19_entry_cascade"
    layer: FusionLayer = "entry"
    priority: int = 10

    QUICK_REENTRY_WINDOW = 3

    def apply(self, ctx: BarContext) -> FusionResult:  # noqa: PLR0911, PLR0912, PLR0915, C901
        cfg = ctx.config
        ind = cfg.get("indicators")
        if ind is None:
            return FusionResult(action="pass", reason="")
        mods: dict[str, bool] = cfg.get("mods", {})
        params: dict[str, Any] = cfg.get("params", {}) or {}
        entry_state: dict[str, Any] = cfg.get("entry_state", {})
        regime_cfg: dict[str, Any] = cfg.get("regime_cfg", {})
        i = ctx.bar_idx
        n = int(ind["n"])
        if i < 1 or i >= n:
            return FusionResult(action="pass", reason="")

        feat_arrays: dict[str, np.ndarray] = ind["feat_arrays"]
        close = ind["close"]
        opn = ind["open"]
        volume = ind["volume"]
        sma20 = ind["sma20"]
        sma50 = ind["sma50"]
        ema8 = ind["ema8"]
        macd_line = ind["macd_line"]
        macd_hist = ind["macd_hist"]
        avg_vol20 = ind["avg_vol20"]
        atr14 = ind["atr14"]
        local_low_20 = ind["local_low_20"]
        ret_5d = ind["ret_5d"]
        ret_20d = ind["ret_20d"]
        ret_60d = ind["ret_60d"]
        drop_from_peak_20 = ind["drop_from_peak_20"]
        stabilized_sideways = ind["stabilized_sideways"]
        consolidation_breakout = ind["consolidation_breakout"]
        secondary_breakout = ind["secondary_breakout"]
        vshape_bypass = ind["vshape_bypass"]

        wp = _gf(feat_arrays, "range_position_20d", i, n)
        dp = _gf(feat_arrays, "dist_to_resistance", i, n)
        rs = _gf(feat_arrays, "rsi_slope_5d", i, n)
        vs = _gf(feat_arrays, "vol_surge_ratio", i, n)
        bs = _gf(feat_arrays, "breakout_setup_score", i, n)
        hl = _gf(feat_arrays, "higher_lows_count", i, n)
        od = _gf(feat_arrays, "obv_price_divergence", i, n)
        bb = _gf(feat_arrays, "bb_width_percentile", i, n)

        trend = entry_state.get("trend") or cfg.get("trend", "weak")
        dp_floor = float(regime_cfg.get("dp_floor", 0.020))
        ret5_hot = float(regime_cfg.get("ret5_hot", 0.060))
        size_mult = float(regime_cfg.get("size_mult", 1.0))
        relax_prev_pred_strong = bool(params.get("patch_relax_prev_pred_strong", False))
        relax_prev_pred_min_score = int(params.get("patch_relax_prev_pred_min_score", 3))
        disable_cooldown_filter = bool(params.get("patch_disable_cooldown_filter", False))
        disable_price_proximity_filter = bool(
            params.get("patch_disable_price_proximity_filter", False)
        )
        relax_price_proximity_strong = bool(params.get("patch_relax_price_proximity_strong", False))
        relax_price_proximity_min_score = int(
            params.get("patch_relax_price_proximity_min_score", 3)
        )
        relax_price_proximity_moderate = bool(
            params.get("patch_relax_price_proximity_moderate", False)
        )
        relax_price_proximity_moderate_min_score = int(
            params.get("patch_relax_price_proximity_moderate_min_score", 4)
        )
        relax_price_proximity_min_dp = float(params.get("patch_relax_price_proximity_min_dp", 0.0))
        relax_price_proximity_moderate_min_bs = float(
            params.get("patch_relax_price_proximity_moderate_min_bs", 0.0)
        )

        raw_signal = 1 if ctx.entry_signal == 1 else 0
        new_position = raw_signal

        cooldown_remaining = int(entry_state.get("cooldown_remaining", 0))
        last_exit_price = float(entry_state.get("last_exit_price", 0.0))
        last_exit_reason = str(entry_state.get("last_exit_reason", ""))
        last_exit_bar = int(entry_state.get("last_exit_bar", -999))
        prev_pred = int(entry_state.get("prev_pred", 0))

        quick_reentry = False
        breakout_entry = False
        vshape_entry = False
        washout_reversal_entry = False
        pullback_reclaim_entry = False
        early_pullback_start_entry = False
        near_sma_continuation_entry = False
        above_sma_continuation_entry = False
        deep_bottom_entry = False
        counters: dict[str, int] = {}

        # Quick re-entry after trailing_stop.
        if new_position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (
                bars_since_exit <= self.QUICK_REENTRY_WINDOW
                and trend in ("strong", "moderate")
                and macd_line[i] > 0
                and not np.isnan(sma20[i])
                and close[i] > sma20[i]
            ):
                new_position = 1
                quick_reentry = True

        # BO quality precondition (mod_f).
        bo_quality_ok = True
        if mods.get("f", True):
            macd_pos = macd_hist[i] > 0
            bullish = close[i] > opn[i]
            heavy_vol = not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i]
            bo_quality_ok = bool(macd_pos and bullish and heavy_vol)

        # Consolidation breakout.
        if new_position == 0 and consolidation_breakout[i] and bo_quality_ok:
            new_position = 1
            breakout_entry = True

        # Secondary breakout (mod_e).
        if mods.get("e", True) and new_position == 0 and secondary_breakout[i] and bo_quality_ok:
            new_position = 1
            breakout_entry = True
            counters["n_secondary_breakout"] = counters.get("n_secondary_breakout", 0) + 1

        # V-shape bypass (mod_a).
        if mods.get("a", True) and new_position == 0 and vshape_bypass[i]:
            if not np.isnan(ema8[i]) and close[i] >= ema8[i] * 0.99:
                new_position = 1
                vshape_entry = True
                counters["n_vshape_entries"] = counters.get("n_vshape_entries", 0) + 1

        if bool(params.get("patch_washout_reversal_entry", False)) and new_position == 0:
            washout_drop = float(params.get("patch_washout_drop20", -0.06))
            washout_dist = float(params.get("patch_washout_dist_sma20", -0.02))
            washout_max_ret5 = float(params.get("patch_washout_max_ret5", 0.01))
            washout_min_bounce = float(params.get("patch_washout_min_bounce", 0.003))
            washout_vol_floor = float(params.get("patch_washout_vol_floor", 0.65))
            bounced_from_low = not np.isnan(local_low_20[i]) and close[i] >= local_low_20[i] * (
                1 + washout_min_bounce
            )
            close_not_bearish = (
                close[i] >= opn[i] or close[i] >= close[i - 1] if i > 0 else close[i] >= opn[i]
            )
            volume_ok = np.isnan(avg_vol20[i]) or volume[i] >= avg_vol20[i] * washout_vol_floor
            washout_setup = (
                drop_from_peak_20[i] <= washout_drop
                and not np.isnan(sma20[i])
                and close[i] / sma20[i] - 1 <= washout_dist
                and ret_5d[i] <= washout_max_ret5
                and bounced_from_low
                and close_not_bearish
                and volume_ok
            )
            if washout_setup:
                new_position = 1
                washout_reversal_entry = True
                counters["n_washout_reversal_entries"] = (
                    counters.get("n_washout_reversal_entries", 0) + 1
                )

        if bool(params.get("patch_pullback_reclaim_entry", False)) and new_position == 0:
            pullback_min_drop = float(params.get("patch_pullback_min_drop20", -0.045))
            pullback_max_drop = float(params.get("patch_pullback_max_drop20", -0.015))
            pullback_min_ret5 = float(params.get("patch_pullback_min_ret5", -0.015))
            pullback_max_ret5 = float(params.get("patch_pullback_max_ret5", 0.025))
            pullback_sma_band = float(params.get("patch_pullback_sma_band", 0.015))
            pullback_vol_floor = float(params.get("patch_pullback_vol_floor", 0.7))
            near_sma20_reclaim = (
                not np.isnan(sma20[i])
                and close[i] >= sma20[i] * (1 - pullback_sma_band)
                and close[i] <= sma20[i] * (1 + pullback_sma_band)
            )
            momentum_reclaim = rs > 0 or (not np.isnan(macd_hist[i]) and macd_hist[i] > 0)
            volume_ok = np.isnan(avg_vol20[i]) or volume[i] >= avg_vol20[i] * pullback_vol_floor
            pullback_setup = (
                pullback_min_drop <= drop_from_peak_20[i] <= pullback_max_drop
                and pullback_min_ret5 <= ret_5d[i] <= pullback_max_ret5
                and near_sma20_reclaim
                and close[i] >= opn[i]
                and trend in ("strong", "moderate")
                and momentum_reclaim
                and volume_ok
            )
            if pullback_setup:
                new_position = 1
                pullback_reclaim_entry = True
                counters["n_pullback_reclaim_entries"] = (
                    counters.get("n_pullback_reclaim_entries", 0) + 1
                )

        if bool(params.get("patch_early_pullback_start_entry", False)) and new_position == 0:
            early_min_drop = float(params.get("patch_early_pullback_min_drop20", -0.085))
            early_max_drop = float(params.get("patch_early_pullback_max_drop20", -0.035))
            early_min_ret5 = float(params.get("patch_early_pullback_min_ret5", -0.035))
            early_max_ret5 = float(params.get("patch_early_pullback_max_ret5", 0.015))
            early_min_ret20 = float(params.get("patch_early_pullback_min_ret20", -0.080))
            early_max_ret20 = float(params.get("patch_early_pullback_max_ret20", 0.030))
            early_sma_low = float(params.get("patch_early_pullback_sma_low", -0.040))
            early_sma_high = float(params.get("patch_early_pullback_sma_high", 0.015))
            early_min_bounce = float(params.get("patch_early_pullback_min_bounce", 0.003))
            early_vol_floor = float(params.get("patch_early_pullback_vol_floor", 0.75))
            dist_sma20 = close[i] / sma20[i] - 1 if not np.isnan(sma20[i]) else np.nan
            bounced_from_low = not np.isnan(local_low_20[i]) and close[i] >= local_low_20[i] * (
                1 + early_min_bounce
            )
            close_not_bearish = (
                close[i] >= opn[i] or close[i] >= close[i - 1] if i > 0 else close[i] >= opn[i]
            )
            volume_ok = np.isnan(avg_vol20[i]) or volume[i] >= avg_vol20[i] * early_vol_floor
            early_setup = (
                early_min_drop <= drop_from_peak_20[i] <= early_max_drop
                and early_min_ret5 <= ret_5d[i] <= early_max_ret5
                and early_min_ret20 <= ret_20d[i] <= early_max_ret20
                and not np.isnan(dist_sma20)
                and early_sma_low <= dist_sma20 <= early_sma_high
                and trend in ("strong", "moderate", "weak")
                and bounced_from_low
                and close_not_bearish
                and volume_ok
            )
            if early_setup:
                new_position = 1
                early_pullback_start_entry = True
                counters["n_early_pullback_start_entries"] = (
                    counters.get("n_early_pullback_start_entries", 0) + 1
                )

        if bool(params.get("patch_near_sma_continuation_entry", False)) and new_position == 0:
            near_min_drop = float(params.get("patch_near_sma_min_drop20", -0.045))
            near_max_drop = float(params.get("patch_near_sma_max_drop20", -0.004))
            near_min_ret5 = float(params.get("patch_near_sma_min_ret5", -0.018))
            near_max_ret5 = float(params.get("patch_near_sma_max_ret5", 0.010))
            near_min_ret20 = float(params.get("patch_near_sma_min_ret20", -0.045))
            near_max_ret20 = float(params.get("patch_near_sma_max_ret20", 0.016))
            near_sma_low = float(params.get("patch_near_sma_low", -0.020))
            near_sma_high = float(params.get("patch_near_sma_high", 0.009))
            near_min_bounce = float(params.get("patch_near_sma_min_bounce", 0.002))
            near_vol_floor = float(params.get("patch_near_sma_vol_floor", 0.62))
            near_require_momentum = bool(params.get("patch_near_sma_require_momentum", True))
            dist_sma20 = close[i] / sma20[i] - 1 if not np.isnan(sma20[i]) else np.nan
            bounced_from_low = not np.isnan(local_low_20[i]) and close[i] >= local_low_20[i] * (
                1 + near_min_bounce
            )
            close_not_bearish = (
                close[i] >= opn[i] or close[i] >= close[i - 1] if i > 0 else close[i] >= opn[i]
            )
            momentum_ok = (
                not near_require_momentum
                or rs > 0
                or (not np.isnan(macd_hist[i]) and macd_hist[i] > 0)
            )
            volume_ok = np.isnan(avg_vol20[i]) or volume[i] >= avg_vol20[i] * near_vol_floor
            near_sma_setup = (
                near_min_drop <= drop_from_peak_20[i] <= near_max_drop
                and near_min_ret5 <= ret_5d[i] <= near_max_ret5
                and near_min_ret20 <= ret_20d[i] <= near_max_ret20
                and not np.isnan(dist_sma20)
                and near_sma_low <= dist_sma20 <= near_sma_high
                and trend in ("strong", "moderate", "weak")
                and bounced_from_low
                and close_not_bearish
                and momentum_ok
                and volume_ok
            )
            if near_sma_setup:
                new_position = 1
                near_sma_continuation_entry = True
                counters["n_near_sma_continuation_entries"] = (
                    counters.get("n_near_sma_continuation_entries", 0) + 1
                )

        if bool(params.get("patch_above_sma_continuation_entry", False)) and new_position == 0:
            above_min_drop = float(params.get("patch_above_sma_min_drop20", -0.010))
            above_max_drop = float(params.get("patch_above_sma_max_drop20", 0.000))
            above_min_ret5 = float(params.get("patch_above_sma_min_ret5", 0.0045))
            above_max_ret5 = float(params.get("patch_above_sma_max_ret5", 0.021))
            above_min_ret20 = float(params.get("patch_above_sma_min_ret20", -0.004))
            above_max_ret20 = float(params.get("patch_above_sma_max_ret20", 0.026))
            above_sma_low = float(params.get("patch_above_sma_low", 0.004))
            above_sma_high = float(params.get("patch_above_sma_high", 0.022))
            above_min_bounce = float(params.get("patch_above_sma_min_bounce", 0.002))
            above_vol_floor = float(params.get("patch_above_sma_vol_floor", 0.90))
            above_require_momentum = bool(params.get("patch_above_sma_require_momentum", True))
            dist_sma20 = close[i] / sma20[i] - 1 if not np.isnan(sma20[i]) else np.nan
            bounced_from_low = not np.isnan(local_low_20[i]) and close[i] >= local_low_20[i] * (
                1 + above_min_bounce
            )
            close_not_bearish = (
                close[i] >= opn[i] or close[i] >= close[i - 1] if i > 0 else close[i] >= opn[i]
            )
            momentum_ok = (
                not above_require_momentum
                or rs > 0
                or (not np.isnan(macd_hist[i]) and macd_hist[i] > 0)
            )
            volume_ok = np.isnan(avg_vol20[i]) or volume[i] >= avg_vol20[i] * above_vol_floor
            above_sma_setup = (
                above_min_drop <= drop_from_peak_20[i] <= above_max_drop
                and above_min_ret5 <= ret_5d[i] <= above_max_ret5
                and above_min_ret20 <= ret_20d[i] <= above_max_ret20
                and not np.isnan(dist_sma20)
                and above_sma_low <= dist_sma20 <= above_sma_high
                and trend in ("strong", "moderate", "weak")
                and bounced_from_low
                and close_not_bearish
                and momentum_ok
                and volume_ok
            )
            if above_sma_setup:
                new_position = 1
                above_sma_continuation_entry = True
                counters["n_above_sma_continuation_entries"] = (
                    counters.get("n_above_sma_continuation_entries", 0) + 1
                )

        if bool(params.get("patch_deep_bottom_entry", False)) and new_position == 0:
            deep_min_drop = float(params.get("patch_deep_bottom_min_drop20", -0.080))
            deep_max_drop = float(params.get("patch_deep_bottom_max_drop20", -0.035))
            deep_min_ret5 = float(params.get("patch_deep_bottom_min_ret5", -0.020))
            deep_max_ret5 = float(params.get("patch_deep_bottom_max_ret5", 0.005))
            deep_sma_low = float(params.get("patch_deep_bottom_sma_low", -0.025))
            deep_sma_high = float(params.get("patch_deep_bottom_sma_high", 0.005))
            deep_min_bounce = float(params.get("patch_deep_bottom_min_bounce", 0.003))
            deep_vol_floor = float(params.get("patch_deep_bottom_vol_floor", 0.60))
            dist_sma20 = close[i] / sma20[i] - 1 if not np.isnan(sma20[i]) else np.nan
            bounced_from_low = not np.isnan(local_low_20[i]) and close[i] >= local_low_20[i] * (
                1 + deep_min_bounce
            )
            close_not_bearish = (
                close[i] >= opn[i] or close[i] >= close[i - 1] if i > 0 else close[i] >= opn[i]
            )
            volume_ok = np.isnan(avg_vol20[i]) or volume[i] >= avg_vol20[i] * deep_vol_floor
            deep_bottom_setup = (
                deep_min_drop <= drop_from_peak_20[i] <= deep_max_drop
                and deep_min_ret5 <= ret_5d[i] <= deep_max_ret5
                and not np.isnan(dist_sma20)
                and deep_sma_low <= dist_sma20 <= deep_sma_high
                and trend in ("strong", "moderate", "weak")
                and bounced_from_low
                and close_not_bearish
                and volume_ok
            )
            if deep_bottom_setup:
                new_position = 1
                deep_bottom_entry = True
                counters["n_deep_bottom_entries"] = counters.get("n_deep_bottom_entries", 0) + 1

        special_patch_entry = (
            washout_reversal_entry
            or pullback_reclaim_entry
            or early_pullback_start_entry
            or near_sma_continuation_entry
            or above_sma_continuation_entry
            or deep_bottom_entry
        )

        if (
            bool(params.get("patch_late_entry_guard", False))
            and new_position == 1
            and not special_patch_entry
        ):
            late_min_ret5 = float(params.get("patch_late_entry_min_ret5", 0.035))
            late_min_ret20 = float(params.get("patch_late_entry_min_ret20", 0.060))
            late_min_dist_sma20 = float(params.get("patch_late_entry_min_dist_sma20", 0.020))
            late_min_dist_ema8 = float(params.get("patch_late_entry_min_dist_ema8", 0.006))
            dist_sma20 = close[i] / sma20[i] - 1 if not np.isnan(sma20[i]) else np.nan
            dist_ema8 = close[i] / ema8[i] - 1 if not np.isnan(ema8[i]) else np.nan
            late_extended = (
                ret_5d[i] >= late_min_ret5
                and ret_20d[i] >= late_min_ret20
                and not np.isnan(dist_sma20)
                and dist_sma20 >= late_min_dist_sma20
                and not np.isnan(dist_ema8)
                and dist_ema8 >= late_min_dist_ema8
            )
            if late_extended:
                new_position = 0
                counters["n_late_entry_guard_blocked"] = (
                    counters.get("n_late_entry_guard_blocked", 0) + 1
                )

        # Cooldown filter (skip quick_reentry/vshape).
        if new_position == 1 and not quick_reentry and not vshape_entry and not special_patch_entry:
            if cooldown_remaining > 0:
                if disable_cooldown_filter:
                    counters["n_v19_relaxed_cooldown_entries"] = (
                        counters.get("n_v19_relaxed_cooldown_entries", 0) + 1
                    )
                else:
                    new_position = 0

        # Price proximity filter.
        if new_position == 1 and not quick_reentry and not vshape_entry and not special_patch_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:
                    entry_score_price_prox = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
                    can_relax_price_prox = (
                        relax_price_proximity_strong
                        and trend == "strong"
                        and entry_score_price_prox >= relax_price_proximity_min_score
                        and dp >= relax_price_proximity_min_dp
                    )
                    can_relax_price_prox_moderate = (
                        relax_price_proximity_moderate
                        and trend == "moderate"
                        and entry_score_price_prox >= relax_price_proximity_moderate_min_score
                        and dp >= relax_price_proximity_min_dp
                        and bs >= relax_price_proximity_moderate_min_bs
                    )
                    if (
                        disable_price_proximity_filter
                        or can_relax_price_prox
                        or can_relax_price_prox_moderate
                    ):
                        counters["n_v19_relaxed_price_prox_entries"] = (
                            counters.get("n_v19_relaxed_price_prox_entries", 0) + 1
                        )
                    else:
                        new_position = 0

        # Prev-signal continuation (skip quick/breakout/vshape).
        if (
            new_position == 1
            and not quick_reentry
            and not breakout_entry
            and not vshape_entry
            and not special_patch_entry
        ):
            if (bs >= 4 and vs > 1.2) or (trend == "strong" and rs > 0):
                pass
            elif prev_pred != 1:
                entry_score_prev = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
                can_relax_prev_pred = (
                    relax_prev_pred_strong
                    and trend == "strong"
                    and entry_score_prev >= relax_prev_pred_min_score
                )
                if can_relax_prev_pred:
                    counters["n_v19_relaxed_prev_pred_entries"] = (
                        counters.get("n_v19_relaxed_prev_pred_entries", 0) + 1
                    )
                else:
                    new_position = 0

        # SMA-below filter.
        if new_position == 1 and not quick_reentry and not vshape_entry and not special_patch_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry:
                        new_position = 0

        entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
        strong_breakout_context = trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry)
        entry_alpha_ok = True
        relax_dp_floor_strong = bool(params.get("patch_relax_dp_floor_strong", False))
        relax_hot_ret5_strong = bool(params.get("patch_relax_hot_ret5_strong", False))
        min_entry_score_override = params.get("v19_min_entry_score")

        # Entry alpha gate.
        if new_position == 1 and not quick_reentry and not vshape_entry and not special_patch_entry:
            near_sma_support = (
                not np.isnan(sma20[i])
                and close[i] <= sma20[i] * 1.02
                and close[i] >= sma20[i] * 0.97
            )
            near_local_low = not np.isnan(local_low_20[i]) and close[i] <= local_low_20[i] * 1.05
            in_uptrend_macro = (
                not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i]
            )

            if trend == "strong":
                min_score = 1
            elif (
                (near_sma_support or near_local_low)
                and in_uptrend_macro
                or in_uptrend_macro
                and rs > 0
            ):
                min_score = 2
            else:
                min_score = 3

            if entry_score < min_score and not breakout_entry:
                entry_alpha_ok = False
            if wp > 0.9 and rs <= 0 and bs < 2 and trend != "strong" and not breakout_entry:
                entry_alpha_ok = False
            if (
                bb > 0.85
                and bs < 2
                and entry_score < 4
                and trend != "strong"
                and not breakout_entry
            ):
                entry_alpha_ok = False
            if entry_alpha_ok:
                if wp > 0.78 and bb < 0.35 and trend == "weak" and not breakout_entry:
                    entry_alpha_ok = False
            if entry_alpha_ok and dp < dp_floor:
                can_relax_dp_floor = (
                    relax_dp_floor_strong
                    and trend == "strong"
                    and entry_score >= 3
                    and rs > 0
                    and vs > 1.0
                )
                if entry_score < 4 and not strong_breakout_context and not can_relax_dp_floor:
                    entry_alpha_ok = False
                elif entry_score < 4 and (strong_breakout_context or can_relax_dp_floor):
                    counters["n_v18_relaxed_dp_entries"] = (
                        counters.get("n_v18_relaxed_dp_entries", 0) + 1
                    )

        # Hot ret5 gate.
        if new_position == 1 and not vshape_entry:
            if ret_5d[i] > ret5_hot and not strong_breakout_context:
                can_relax_hot_ret5 = (
                    relax_hot_ret5_strong
                    and trend == "strong"
                    and entry_score >= 3
                    and dp >= max(0.01, dp_floor * 0.5)
                    and vs > 1.0
                )
                if can_relax_hot_ret5:
                    counters["n_v18_relaxed_ret5_entries"] = (
                        counters.get("n_v18_relaxed_ret5_entries", 0) + 1
                    )
                else:
                    entry_alpha_ok = False
            elif ret_5d[i] > ret5_hot and strong_breakout_context:
                counters["n_v18_relaxed_ret5_entries"] = (
                    counters.get("n_v18_relaxed_ret5_entries", 0) + 1
                )

        # Drop from peak gate.
        if new_position == 1 and not vshape_entry and not special_patch_entry and entry_alpha_ok:
            drop_threshold = float(params.get("v19_drop_from_peak_threshold", -0.15))
            if drop_from_peak_20[i] <= drop_threshold and not stabilized_sideways[i]:
                entry_alpha_ok = False

        # Volume floor.
        if new_position == 1 and entry_alpha_ok:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor:
                entry_alpha_ok = False

        # Bear regime defense (mod_g).
        if (
            mods.get("g", True)
            and new_position == 1
            and not vshape_entry
            and not special_patch_entry
            and entry_alpha_ok
        ):
            sma20_below_50 = (
                not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] < sma50[i]
            )
            close_below_50 = not np.isnan(sma50[i]) and close[i] < sma50[i]
            deep_60d_loss = ret_60d[i] < -0.10
            if sma20_below_50 and close_below_50 and deep_60d_loss:
                entry_alpha_ok = False
                counters["n_bear_blocked"] = counters.get("n_bear_blocked", 0) + 1

        # Anti-chop (mod_j).
        if (
            mods.get("j", True)
            and new_position == 1
            and not vshape_entry
            and not breakout_entry
            and not special_patch_entry
            and entry_alpha_ok
        ):
            ma_flat = (
                not np.isnan(sma20[i])
                and not np.isnan(sma50[i])
                and abs(sma20[i] / sma50[i] - 1) < 0.02
            )
            weak_momo = abs(ret_20d[i]) < 0.06
            narrow_vol = bb < 0.45
            weak_trend = trend == "weak"
            if ma_flat and weak_momo and narrow_vol and weak_trend:
                entry_alpha_ok = False
                counters["n_chop_blocked"] = counters.get("n_chop_blocked", 0) + 1

        if new_position == 1 and not entry_alpha_ok:
            new_position = 0
            counters["n_v19_alpha_blocked"] = counters.get("n_v19_alpha_blocked", 0) + 1

        # Global entry_score threshold override (phase15+)
        if new_position == 1 and min_entry_score_override is not None:
            if (
                entry_score < min_entry_score_override
                and not vshape_entry
                and not breakout_entry
                and not special_patch_entry
            ):
                new_position = 0
                counters["n_v19_score_filtered"] = counters.get("n_v19_score_filtered", 0) + 1

        # Entry type filters (phase16+)
        if new_position == 1 and bool(params.get("v19_block_breakout_only", False)):
            if breakout_entry and not vshape_entry:
                new_position = 0
                counters["n_v19_breakout_filtered"] = counters.get("n_v19_breakout_filtered", 0) + 1

        if new_position == 1 and bool(params.get("v19_block_strong_only", False)):
            if trend == "strong" and not vshape_entry:
                new_position = 0
                counters["n_v19_strong_filtered"] = counters.get("n_v19_strong_filtered", 0) + 1

        # ATR volatility gate (phase17+)
        max_atr_ratio = params.get("v19_max_atr_ratio")
        if new_position == 1 and max_atr_ratio is not None:
            atr_ratio_check = (
                (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03
            )
            if atr_ratio_check > max_atr_ratio:
                new_position = 0
                counters["n_v19_atr_filtered"] = counters.get("n_v19_atr_filtered", 0) + 1

        # Breakout-specific score floor (phase17+): require higher score for breakout entries
        min_score_breakout = params.get("v19_min_score_breakout")
        if (
            new_position == 1
            and min_score_breakout is not None
            and breakout_entry
            and not vshape_entry
        ):
            if entry_score < min_score_breakout:
                new_position = 0
                counters["n_v19_bo_score_filtered"] = counters.get("n_v19_bo_score_filtered", 0) + 1

        if new_position == 0:
            return FusionResult(action="pass", reason="", metadata={"counters": counters})

        # Position sizing.
        entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
        atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

        if vshape_entry:
            position_size = 0.50
        elif washout_reversal_entry:
            position_size = float(params.get("patch_washout_position_size", 0.35))
        elif pullback_reclaim_entry:
            position_size = float(params.get("patch_pullback_position_size", 0.40))
        elif early_pullback_start_entry:
            position_size = float(params.get("patch_early_pullback_position_size", 0.40))
        elif near_sma_continuation_entry:
            position_size = float(params.get("patch_near_sma_position_size", 0.40))
        elif deep_bottom_entry:
            position_size = float(params.get("patch_deep_bottom_position_size", 0.40))
        elif trend == "strong" and entry_score >= 4:
            position_size = 0.95
        elif trend == "strong" and entry_score >= 3:
            position_size = 0.90
        elif trend == "moderate" and entry_score >= 3:
            position_size = 0.50
        elif trend == "weak":
            position_size = 0.30
        else:
            position_size = 0.50

        if atr_ratio > 0.055:
            position_size = min(position_size, 0.35)
        elif atr_ratio > 0.040:
            position_size = min(position_size, 0.50)

        if trend == "weak":
            position_size = min(position_size, 0.40)
        elif trend == "moderate":
            position_size = min(position_size, 0.70)

        if close[i] <= opn[i] and not vshape_entry and not special_patch_entry:
            position_size *= 0.75

        if ret_5d[i] > ret5_hot:
            position_size = min(position_size, 0.40)
            counters["n_v19_overheat_entries"] = counters.get("n_v19_overheat_entries", 0) + 1

        position_size *= size_mult
        position_size *= float(params.get("patch_position_size_mult", 1.0))
        position_size = max(0.25, min(position_size, 1.0))

        # Build entry_features snapshot for trade record.
        dates = ind.get("dates")
        symbols = ind.get("symbols")
        entry_features = {
            "entry_wp": wp,
            "entry_dp": dp,
            "entry_rs": rs,
            "entry_vs": vs,
            "entry_bs": bs,
            "entry_hl": hl,
            "entry_od": od,
            "entry_bb": bb,
            "entry_score": entry_score,
            "entry_date": _format_date(dates[i]) if dates is not None else "",
            "entry_symbol": str(symbols[i]) if symbols is not None else "",
            "position_size": position_size,
            "entry_trend": trend,
            "quick_reentry": quick_reentry,
            "breakout_entry": breakout_entry,
            "vshape_entry": vshape_entry,
            "washout_reversal_entry": washout_reversal_entry,
            "pullback_reclaim_entry": pullback_reclaim_entry,
            "early_pullback_start_entry": early_pullback_start_entry,
            "near_sma_continuation_entry": near_sma_continuation_entry,
            "above_sma_continuation_entry": above_sma_continuation_entry,
            "deep_bottom_entry": deep_bottom_entry,
            "entry_ret_5d": round(float(ret_5d[i]) * 100, 2),
            "entry_drop20d": round(float(drop_from_peak_20[i]) * 100, 2),
            "entry_dist_sma20": round(float(ind["dist_sma20"][i]) * 100, 2),
            "entry_profile": regime_cfg.get("profile", "balanced"),
            "entry_choppy_regime": bool(regime_cfg.get("choppy_regime", False)),
        }

        reason = (
            "vshape"
            if vshape_entry
            else "breakout"
            if breakout_entry
            else "quick_reentry"
            if quick_reentry
            else "washout_reversal"
            if washout_reversal_entry
            else "pullback_reclaim"
            if pullback_reclaim_entry
            else "early_pullback_start"
            if early_pullback_start_entry
            else "near_sma_continuation"
            if near_sma_continuation_entry
            else "deep_bottom"
            if deep_bottom_entry
            else "ml_signal"
        )

        return FusionResult(
            action="enter",
            reason=reason,
            metadata={
                "size": position_size,
                "entry_features": entry_features,
                "flags": {
                    "quick_reentry": quick_reentry,
                    "breakout_entry": breakout_entry,
                    "vshape_entry": vshape_entry,
                    "washout_reversal_entry": washout_reversal_entry,
                    "pullback_reclaim_entry": pullback_reclaim_entry,
                    "early_pullback_start_entry": early_pullback_start_entry,
                    "near_sma_continuation_entry": near_sma_continuation_entry,
                    "above_sma_continuation_entry": above_sma_continuation_entry,
                },
                "counters": counters,
            },
        )
