"""Equivalence tests: new target components must match legacy TargetGenerator output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.components.targets.early_wave import EarlyWaveTarget
from src.components.targets.early_wave_dual import EarlyWaveDualTarget
from src.components.targets.early_wave_v2 import EarlyWaveV2Target
from src.components.targets.registry import get_target, list_targets
from src.components.targets.trend_regime import TrendRegimeTarget


def _make_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "TEST",
        }
    )


def _legacy(target_type: str, df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    from src.data.target import TargetGenerator

    gen = TargetGenerator(target_type=target_type, **kwargs)  # type: ignore[arg-type]
    return gen.generate(df.copy())


# ── TrendRegimeTarget ──────────────────────────────────────────────────────────


class TestTrendRegimeTarget:
    def test_dual_ma_matches_legacy(self) -> None:
        df = _make_df()
        legacy_df = _legacy("trend_regime", df)
        new_labels = TrendRegimeTarget().generate_entry_labels(df)
        pd.testing.assert_series_equal(
            new_labels.reset_index(drop=True),
            legacy_df["target"].reset_index(drop=True),
            check_names=False,
        )

    def test_hhll_matches_legacy(self) -> None:
        df = _make_df()
        legacy_df = _legacy("trend_regime", df, trend_method="hhll")
        new_labels = TrendRegimeTarget(trend_method="hhll").generate_entry_labels(df)
        pd.testing.assert_series_equal(
            new_labels.reset_index(drop=True),
            legacy_df["target"].reset_index(drop=True),
            check_names=False,
        )

    def test_binary_mode(self) -> None:
        df = _make_df()
        legacy_df = _legacy("trend_regime", df, n_classes=2)
        new_labels = TrendRegimeTarget(n_classes=2).generate_entry_labels(df)
        pd.testing.assert_series_equal(
            new_labels.reset_index(drop=True),
            legacy_df["target"].reset_index(drop=True),
            check_names=False,
        )

    def test_no_exit_labels(self) -> None:
        t = TrendRegimeTarget()
        assert t.supports_exit_labels is False
        assert t.generate_exit_labels(_make_df(), 10, 0.05) is None

    def test_num_classes(self) -> None:
        assert TrendRegimeTarget(n_classes=3).num_classes == 3
        assert TrendRegimeTarget(n_classes=2).num_classes == 2


# ── EarlyWaveTarget ────────────────────────────────────────────────────────────


class TestEarlyWaveTarget:
    def test_entry_labels_match_legacy(self) -> None:
        df = _make_df()
        legacy_df = _legacy(
            "early_wave",
            df,
            short_window=10,
            long_window=20,
            forward_window=10,
            gain_threshold=0.08,
            loss_threshold=0.05,
            n_classes=3,
        )
        new_labels = EarlyWaveTarget(
            short_window=10,
            long_window=20,
            forward_window=10,
            gain_threshold=0.08,
            loss_threshold=0.05,
            n_classes=3,
        ).generate_entry_labels(df)

        # Legacy drops NaN rows; compare where both are non-NaN
        mask = ~(new_labels.isna() | legacy_df["target"].isna())
        pd.testing.assert_series_equal(
            new_labels[mask].reset_index(drop=True),
            legacy_df["target"][mask].reset_index(drop=True),
            check_names=False,
        )

    def test_exit_labels_match_legacy(self) -> None:
        df = _make_df()
        from src.data.target import TargetGenerator

        legacy_df = TargetGenerator.generate_exit_labels(
            df.copy(), forward_window=15, loss_threshold=0.05
        )
        new_labels = EarlyWaveTarget().generate_exit_labels(
            df, forward_window=15, loss_threshold=0.05
        )
        assert new_labels is not None

        mask = ~(new_labels.isna() | legacy_df["target_sell"].isna())
        pd.testing.assert_series_equal(
            new_labels[mask].reset_index(drop=True),
            legacy_df["target_sell"][mask].reset_index(drop=True),
            check_names=False,
        )

    def test_supports_exit(self) -> None:
        assert EarlyWaveTarget().supports_exit_labels is True


# ── EarlyWaveV2Target ──────────────────────────────────────────────────────────


class TestEarlyWaveV2Target:
    def test_entry_labels_match_legacy(self) -> None:
        df = _make_df()
        legacy_df = _legacy(
            "early_wave_v2",
            df,
            short_window=10,
            long_window=20,
            forward_window=10,
            gain_threshold=0.05,
            loss_threshold=0.03,
            n_classes=3,
        )
        new_labels = EarlyWaveV2Target(
            short_window=10,
            long_window=20,
            forward_window=10,
            gain_threshold=0.05,
            loss_threshold=0.03,
            n_classes=3,
        ).generate_entry_labels(df)

        mask = ~(new_labels.isna() | legacy_df["target"].isna())
        pd.testing.assert_series_equal(
            new_labels[mask].reset_index(drop=True),
            legacy_df["target"][mask].reset_index(drop=True),
            check_names=False,
        )

    def test_supports_exit(self) -> None:
        assert EarlyWaveV2Target().supports_exit_labels is True


# ── EarlyWaveDualTarget ────────────────────────────────────────────────────────


class TestEarlyWaveDualTarget:
    def test_entry_equals_early_wave(self) -> None:
        df = _make_df()
        dual = EarlyWaveDualTarget()
        base = EarlyWaveTarget()
        pd.testing.assert_series_equal(
            dual.generate_entry_labels(df).reset_index(drop=True),
            base.generate_entry_labels(df).reset_index(drop=True),
            check_names=False,
        )

    def test_exit_labels_present(self) -> None:
        df = _make_df()
        dual = EarlyWaveDualTarget()
        sell = dual.generate_exit_labels(df, forward_window=15, loss_threshold=0.05)
        assert sell is not None
        assert len(sell) == len(df)

    def test_supports_exit(self) -> None:
        assert EarlyWaveDualTarget().supports_exit_labels is True


# ── Registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_all_names_registered(self) -> None:
        names = list_targets()
        assert set(names) == {"trend_regime", "early_wave", "early_wave_v2", "early_wave_dual"}

    def test_get_returns_correct_type(self) -> None:
        assert isinstance(get_target("trend_regime"), TrendRegimeTarget)
        assert isinstance(get_target("early_wave"), EarlyWaveTarget)
        assert isinstance(get_target("early_wave_v2"), EarlyWaveV2Target)
        assert isinstance(get_target("early_wave_dual"), EarlyWaveDualTarget)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown target"):
            get_target("nonexistent_target")

    def test_kwargs_forwarded(self) -> None:
        t = get_target("trend_regime", n_classes=2)
        assert t.num_classes == 2  # type: ignore[union-attr]
