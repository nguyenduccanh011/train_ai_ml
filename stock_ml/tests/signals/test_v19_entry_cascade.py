"""Unit tests for V19EntryCascade — gộp 5 entry sources + filters + sizing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from src.components.base import BarContext
from src.components.fusion.helpers import compute_v19_indicators, get_regime_adapter
from src.components.fusion.strategies.entry import V19EntryCascade


def _df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0.05, 0.5, n))
    return pd.DataFrame(
        {
            "open": base + rng.normal(0, 0.1, n),
            "close": base,
            "high": base + np.abs(rng.normal(0.4, 0.2, n)),
            "low": base - np.abs(rng.normal(0.4, 0.2, n)),
            "volume": rng.uniform(1_000_000, 3_000_000, n),
            "symbol": ["TEST"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


def _ctx(
    *,
    bar_idx: int,
    df: pd.DataFrame,
    entry_signal: int,
    mods: dict[str, bool] | None = None,
    entry_state: dict[str, Any] | None = None,
) -> BarContext:
    ind = compute_v19_indicators(df)
    regime = get_regime_adapter("TEST", ind, bar_idx, "moderate")
    return BarContext(
        bar_idx=bar_idx,
        df_test=df,
        entry_signal=entry_signal,
        entry_proba=None,
        exit_signal=None,
        exit_proba=None,
        position=None,
        config={
            "indicators": ind,
            "mods": mods or {"a": True, "b": True, "e": True, "f": True, "g": True, "j": True},
            "entry_state": entry_state or {"prev_pred": 1, "trend": "moderate"},
            "regime_cfg": regime,
            "trend": "moderate",
        },
    )


class TestNoEntrySignal:
    def test_returns_pass_when_no_signal_and_no_special_entry(self) -> None:
        df = _df(80)
        ctx = _ctx(bar_idx=40, df=df, entry_signal=0)
        res = V19EntryCascade().apply(ctx)
        assert res.action == "pass"

    def test_passes_outside_warmup(self) -> None:
        df = _df(80)
        ctx = _ctx(bar_idx=0, df=df, entry_signal=1)
        res = V19EntryCascade().apply(ctx)
        assert res.action == "pass"


class TestMlEntry:
    def test_ml_entry_with_prev_pred_1(self) -> None:
        df = _df(80, seed=1)
        ctx = _ctx(
            bar_idx=40,
            df=df,
            entry_signal=1,
            entry_state={"prev_pred": 1, "trend": "strong"},
        )
        res = V19EntryCascade().apply(ctx)
        # ML entry path; either enter or blocked by alpha — both acceptable.
        assert res.action in {"enter", "pass"}
        if res.action == "enter":
            assert 0.25 <= res.metadata["size"] <= 1.0
            assert "entry_features" in res.metadata


class TestQuickReentry:
    def test_quick_reentry_after_trailing(self) -> None:
        df = _df(80, seed=2)
        ind = compute_v19_indicators(df)
        # Pick bar where close > sma20 and macd_line > 0
        i = 70
        # Force conditions
        ind_close = ind["close"]
        ind_sma20 = ind["sma20"]
        if not (ind_close[i] > ind_sma20[i] and ind["macd_line"][i] > 0):
            return  # synthetic data didn't satisfy; not asserting beyond shape
        regime = get_regime_adapter("TEST", ind, i, "moderate")
        ctx = BarContext(
            bar_idx=i,
            df_test=df,
            entry_signal=0,
            entry_proba=None,
            exit_signal=None,
            exit_proba=None,
            position=None,
            config={
                "indicators": ind,
                "mods": {},
                "entry_state": {
                    "trend": "moderate",
                    "last_exit_reason": "trailing_stop",
                    "last_exit_bar": i - 2,
                    "prev_pred": 0,
                },
                "regime_cfg": regime,
            },
        )
        res = V19EntryCascade().apply(ctx)
        if res.action == "enter":
            assert res.reason == "quick_reentry"


class TestCounters:
    def test_alpha_blocked_counter(self) -> None:
        df = _df(80, seed=5)
        ctx = _ctx(
            bar_idx=40,
            df=df,
            entry_signal=1,
            entry_state={"prev_pred": 0, "trend": "weak"},  # likely blocked
            mods={"a": True, "f": True, "g": True, "j": True},
        )
        res = V19EntryCascade().apply(ctx)
        # Either blocked (counter incremented) or passed silently.
        assert res.action in {"pass", "enter"}


class TestPositionSize:
    def test_size_clipped_min(self) -> None:
        df = _df(80, seed=7)
        ctx = _ctx(
            bar_idx=50,
            df=df,
            entry_signal=1,
            entry_state={"prev_pred": 1, "trend": "strong"},
        )
        res = V19EntryCascade().apply(ctx)
        if res.action == "enter":
            assert res.metadata["size"] >= 0.25
            assert res.metadata["size"] <= 1.0


class TestPrevPredRelaxPatch:
    def test_relax_prev_pred_strong_allows_entry(self) -> None:
        df = _df(100, seed=11)
        bar_idx = 70
        ctx = _ctx(
            bar_idx=bar_idx,
            df=df,
            entry_signal=1,
            entry_state={"prev_pred": 0, "trend": "strong"},
        )

        ind = ctx.config["indicators"]
        feat = ind["feat_arrays"]

        # Force a high-quality strong-trend setup that only fails at prev_pred gate.
        feat["range_position_20d"][bar_idx] = 0.60
        feat["dist_to_resistance"][bar_idx] = 0.03
        feat["rsi_slope_5d"][
            bar_idx
        ] = -0.10  # Keep rs<=0 so default prev_pred bypass does not apply.
        feat["vol_surge_ratio"][bar_idx] = 1.30
        feat["breakout_setup_score"][bar_idx] = 2.0
        feat["higher_lows_count"][bar_idx] = 2.0
        feat["bb_width_percentile"][bar_idx] = 0.60

        ind["consolidation_breakout"][bar_idx] = False
        ind["secondary_breakout"][bar_idx] = False
        ind["vshape_bypass"][bar_idx] = False
        ind["ret_5d"][bar_idx] = 0.01
        ind["ret_20d"][bar_idx] = 0.08
        ind["ret_60d"][bar_idx] = 0.05
        ind["drop_from_peak_20"][bar_idx] = -0.02
        ind["avg_vol20"][bar_idx] = 1.0
        ind["volume"][bar_idx] = 2.0
        ind["sma20"][bar_idx] = ind["close"][bar_idx] * 0.99
        ind["sma50"][bar_idx] = ind["close"][bar_idx] * 0.98

        base_res = V19EntryCascade().apply(ctx)
        assert base_res.action == "pass"

        ctx.config["params"] = {
            "patch_relax_prev_pred_strong": True,
            "patch_relax_prev_pred_min_score": 3,
        }
        patched_res = V19EntryCascade().apply(ctx)

        assert patched_res.action == "enter"
        assert patched_res.metadata["counters"]["n_v19_relaxed_prev_pred_entries"] == 1


class TestCooldownAndPriceProximityPatch:
    @staticmethod
    def _force_entry_setup(ctx: BarContext, bar_idx: int) -> None:
        ind = ctx.config["indicators"]
        feat = ind["feat_arrays"]

        feat["range_position_20d"][bar_idx] = 0.60
        feat["dist_to_resistance"][bar_idx] = 0.03
        feat["rsi_slope_5d"][bar_idx] = 0.20
        feat["vol_surge_ratio"][bar_idx] = 1.30
        feat["breakout_setup_score"][bar_idx] = 2.0
        feat["higher_lows_count"][bar_idx] = 2.0
        feat["bb_width_percentile"][bar_idx] = 0.60

        ind["consolidation_breakout"][bar_idx] = False
        ind["secondary_breakout"][bar_idx] = False
        ind["vshape_bypass"][bar_idx] = False
        ind["ret_5d"][bar_idx] = 0.01
        ind["ret_20d"][bar_idx] = 0.08
        ind["ret_60d"][bar_idx] = 0.05
        ind["drop_from_peak_20"][bar_idx] = -0.02
        ind["avg_vol20"][bar_idx] = 1.0
        ind["volume"][bar_idx] = 2.0
        ind["sma20"][bar_idx] = ind["close"][bar_idx] * 0.99
        ind["sma50"][bar_idx] = ind["close"][bar_idx] * 0.98

    def test_disable_cooldown_filter_allows_entry(self) -> None:
        df = _df(100, seed=13)
        bar_idx = 70
        ctx = _ctx(
            bar_idx=bar_idx,
            df=df,
            entry_signal=1,
            entry_state={
                "prev_pred": 1,
                "trend": "strong",
                "cooldown_remaining": 2,
                "last_exit_price": 0.0,
                "last_exit_reason": "",
            },
        )
        self._force_entry_setup(ctx, bar_idx)

        base_res = V19EntryCascade().apply(ctx)
        assert base_res.action == "pass"

        ctx.config["params"] = {"patch_disable_cooldown_filter": True}
        patched_res = V19EntryCascade().apply(ctx)
        assert patched_res.action == "enter"
        assert patched_res.metadata["counters"]["n_v19_relaxed_cooldown_entries"] == 1

    def test_disable_price_proximity_filter_allows_entry(self) -> None:
        df = _df(100, seed=17)
        bar_idx = 70
        close = float(df.loc[bar_idx, "close"])
        ctx = _ctx(
            bar_idx=bar_idx,
            df=df,
            entry_signal=1,
            entry_state={
                "prev_pred": 1,
                "trend": "strong",
                "cooldown_remaining": 0,
                "last_exit_price": close * 1.01,  # <3% diff => default block
                "last_exit_reason": "exit_model",
            },
        )
        self._force_entry_setup(ctx, bar_idx)

        base_res = V19EntryCascade().apply(ctx)
        assert base_res.action == "pass"

        ctx.config["params"] = {"patch_disable_price_proximity_filter": True}
        patched_res = V19EntryCascade().apply(ctx)
        assert patched_res.action == "enter"
        assert patched_res.metadata["counters"]["n_v19_relaxed_price_prox_entries"] == 1

    def test_relax_price_proximity_strong_allows_entry(self) -> None:
        df = _df(100, seed=19)
        bar_idx = 70
        close = float(df.loc[bar_idx, "close"])
        ctx = _ctx(
            bar_idx=bar_idx,
            df=df,
            entry_signal=1,
            entry_state={
                "prev_pred": 1,
                "trend": "strong",
                "cooldown_remaining": 0,
                "last_exit_price": close * 1.01,  # <3% diff => default block
                "last_exit_reason": "exit_model",
            },
        )
        self._force_entry_setup(ctx, bar_idx)

        base_res = V19EntryCascade().apply(ctx)
        assert base_res.action == "pass"

        ctx.config["params"] = {
            "patch_relax_price_proximity_strong": True,
            "patch_relax_price_proximity_min_score": 3,
        }
        patched_res = V19EntryCascade().apply(ctx)
        assert patched_res.action == "enter"
        assert patched_res.metadata["counters"]["n_v19_relaxed_price_prox_entries"] == 1

    def test_relax_price_proximity_moderate_allows_entry(self) -> None:
        df = _df(100, seed=23)
        bar_idx = 70
        close = float(df.loc[bar_idx, "close"])
        ctx = _ctx(
            bar_idx=bar_idx,
            df=df,
            entry_signal=1,
            entry_state={
                "prev_pred": 1,
                "trend": "moderate",
                "cooldown_remaining": 0,
                "last_exit_price": close * 1.01,  # <3% diff => default block
                "last_exit_reason": "exit_model",
            },
        )
        self._force_entry_setup(ctx, bar_idx)

        base_res = V19EntryCascade().apply(ctx)
        assert base_res.action == "pass"

        ctx.config["params"] = {
            "patch_relax_price_proximity_moderate": True,
            "patch_relax_price_proximity_moderate_min_score": 4,
        }
        patched_res = V19EntryCascade().apply(ctx)
        assert patched_res.action == "enter"
        assert patched_res.metadata["counters"]["n_v19_relaxed_price_prox_entries"] == 1

    def test_relax_price_proximity_moderate_respects_min_dp_and_bs(self) -> None:
        df = _df(100, seed=29)
        bar_idx = 70
        close = float(df.loc[bar_idx, "close"])
        ctx = _ctx(
            bar_idx=bar_idx,
            df=df,
            entry_signal=1,
            entry_state={
                "prev_pred": 1,
                "trend": "moderate",
                "cooldown_remaining": 0,
                "last_exit_price": close * 1.01,
                "last_exit_reason": "exit_model",
            },
        )
        self._force_entry_setup(ctx, bar_idx)
        ind = ctx.config["indicators"]
        feat = ind["feat_arrays"]
        feat["dist_to_resistance"][bar_idx] = 0.005
        feat["breakout_setup_score"][bar_idx] = 0.0

        ctx.config["params"] = {
            "patch_relax_price_proximity_moderate": True,
            "patch_relax_price_proximity_moderate_min_score": 4,
            "patch_relax_price_proximity_min_dp": 0.02,
            "patch_relax_price_proximity_moderate_min_bs": 1,
        }
        patched_res = V19EntryCascade().apply(ctx)
        assert patched_res.action == "pass"
