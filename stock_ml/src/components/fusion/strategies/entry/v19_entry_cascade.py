"""V19_3 entry cascade: gộp 5 entry sources + 8 filters + sizing thành 1 strategy.

Map 1-1 với legacy backtest_v19_3 (src/strategies/legacy.py:2718-2947) để đảm bảo
parity exact với golden 1910 trades. Logic chia sẻ state (entry_alpha_ok,
breakout_entry, vshape_entry, strong_breakout_context) khó tách rời nên giữ
dạng monolithic — đổi lại an toàn parity.

Đầu vào trong ctx.config:
  - indicators: dict ndarray từ helpers.compute_v19_indicators
  - mods: {"a","b","c","d","e","f","g","h","i","j": bool}
  - entry_state: {"cooldown_remaining": int, "last_exit_price": float,
                  "last_exit_reason": str, "last_exit_bar": int,
                  "prev_pred": int}
  - regime_cfg: dict từ helpers.get_regime_adapter (đã cache theo bar)

Đầu ra FusionResult.metadata:
  - size: float (position_size sau khi clip)
  - entry_features: dict (entry_wp/dp/rs/vs/bs/hl/od/bb/score/profile/...)
  - flags: {"quick_reentry","breakout_entry","vshape_entry"}
  - counters: dict[str,int] (n_vshape_entries/n_secondary_breakout/
              n_v19_alpha_blocked/n_v18_relaxed_*/n_bear_blocked/n_chop_blocked/
              n_v19_overheat_entries) — driver cộng dồn vào counters globals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

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

        # Cooldown filter (skip quick_reentry/vshape).
        if new_position == 1 and not quick_reentry and not vshape_entry:
            if cooldown_remaining > 0:
                new_position = 0

        # Price proximity filter.
        if new_position == 1 and not quick_reentry and not vshape_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                price_diff = abs(close[i] / last_exit_price - 1)
                if price_diff < 0.03:
                    new_position = 0

        # Prev-signal continuation (skip quick/breakout/vshape).
        if new_position == 1 and not quick_reentry and not breakout_entry and not vshape_entry:
            if (bs >= 4 and vs > 1.2) or (trend == "strong" and rs > 0):
                pass
            elif prev_pred != 1:
                new_position = 0

        # SMA-below filter.
        if new_position == 1 and not quick_reentry and not vshape_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry:
                        new_position = 0

        entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
        strong_breakout_context = trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry)
        entry_alpha_ok = True
        relax_dp_floor_strong = bool(params.get("patch_relax_dp_floor_strong", False))
        relax_hot_ret5_strong = bool(params.get("patch_relax_hot_ret5_strong", False))

        # Entry alpha gate.
        if new_position == 1 and not quick_reentry and not vshape_entry:
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
        if new_position == 1 and not vshape_entry and entry_alpha_ok:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]:
                entry_alpha_ok = False

        # Volume floor.
        if new_position == 1 and entry_alpha_ok:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor:
                entry_alpha_ok = False

        # Bear regime defense (mod_g).
        if mods.get("g", True) and new_position == 1 and not vshape_entry and entry_alpha_ok:
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

        if new_position == 0:
            return FusionResult(action="pass", reason="", metadata={"counters": counters})

        # Position sizing.
        entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
        atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

        if vshape_entry:
            position_size = 0.50
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

        if close[i] <= opn[i] and not vshape_entry:
            position_size *= 0.75

        if ret_5d[i] > ret5_hot:
            position_size = min(position_size, 0.40)
            counters["n_v19_overheat_entries"] = counters.get("n_v19_overheat_entries", 0) + 1

        position_size *= size_mult
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
            "entry_date": str(dates[i])[:10] if dates is not None else "",
            "entry_symbol": str(symbols[i]) if symbols is not None else "",
            "position_size": position_size,
            "entry_trend": trend,
            "quick_reentry": quick_reentry,
            "breakout_entry": breakout_entry,
            "vshape_entry": vshape_entry,
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
                },
                "counters": counters,
            },
        )
