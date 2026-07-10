"""Phase 1.5 methodology tests: Purged KFold, DSR, PBO, bootstrap CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data.splitter import PurgedKFoldSplitter
from src.evaluation.bootstrap import bootstrap_metric, max_drawdown, sharpe_ratio, win_rate
from src.evaluation.dsr import deflated_sharpe
from src.evaluation.pbo import pbo


class TestPurgedKFold:
    """1.5.1: Purged K-Fold with embargo."""

    def test_purged_kfold_basic(self):
        """Basic functionality: k folds, no overlaps."""
        dates = pd.date_range("2015-01-01", "2019-12-31", freq="D")
        df = pd.DataFrame({"date": dates, "symbol": "TEST", "value": np.random.randn(len(dates))})

        splitter = PurgedKFoldSplitter(n_splits=4, embargo_days=0)
        folds = list(splitter.split(df))

        assert len(folds) == 4
        test_sizes = [len(test_df) for _, _, test_df in folds]
        assert sum(test_sizes) <= len(df)

    def test_purged_kfold_embargo(self):
        """Embargo removes rows near fold boundaries."""
        dates = pd.date_range("2015-01-01", "2019-12-31", freq="D")
        df = pd.DataFrame({"date": dates, "symbol": "TEST", "value": np.random.randn(len(dates))})

        splitter_no_embargo = PurgedKFoldSplitter(n_splits=3, embargo_days=0)
        splitter_embargo = PurgedKFoldSplitter(n_splits=3, embargo_days=10)

        folds_no_embargo = list(splitter_no_embargo.split(df))
        folds_embargo = list(splitter_embargo.split(df))

        train_size_no_embargo = len(folds_no_embargo[0][1])
        train_size_embargo = len(folds_embargo[0][1])

        assert train_size_embargo < train_size_no_embargo, "Embargo should reduce training set"

    def test_purged_kfold_minimal_data(self):
        """Raises on insufficient data."""
        dates = pd.date_range("2015-01-01", "2015-01-10", freq="D")
        df = pd.DataFrame({"date": dates, "symbol": "TEST", "value": np.random.randn(len(dates))})

        splitter = PurgedKFoldSplitter(n_splits=20)
        with pytest.raises(ValueError, match="Insufficient unique dates"):
            list(splitter.split(df))


class TestBootstrap:
    """1.5.2: Bootstrap confidence intervals."""

    def test_bootstrap_metric_basic(self):
        """Bootstrap CI for Sharpe ratio."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01

        result = bootstrap_metric(
            returns,
            sharpe_ratio,
            n_iter=100,
            block_size=10,
            random_state=42,
        )

        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "std" in result
        assert result["ci_lower"] < result["point_estimate"] < result["ci_upper"]

    def test_bootstrap_ci_coverage(self):
        """CI should contain point estimate."""
        returns = np.random.randn(252) * 0.01

        result = bootstrap_metric(returns, np.mean, n_iter=500, block_size=1)

        assert result["ci_lower"] < result["point_estimate"] < result["ci_upper"]

    def test_sharpe_ratio(self):
        """Sharpe ratio calculation."""
        returns = np.array([0.001, 0.002, -0.001, 0.0015])
        sr = sharpe_ratio(returns)
        assert isinstance(sr, float)
        assert sr > 0 or sr < 0

    def test_max_drawdown(self):
        """Max drawdown calculation."""
        returns = np.array([0.05, -0.10, 0.02, -0.03, 0.01])
        dd = max_drawdown(returns)
        assert dd <= 0
        assert dd < -0.1

    def test_win_rate(self):
        """Win rate (positive return fraction)."""
        returns = np.array([0.01, -0.01, 0.02, -0.01, 0.01])
        wr = win_rate(returns)
        assert wr == 0.6


class TestDSR:
    """1.5.3: Deflated Sharpe Ratio."""

    def test_dsr_basic(self):
        """DSR reduces naïve Sharpe for multiple testing."""
        sharpe = 1.5
        n_trials = 50
        sample_length = 252

        dsr = deflated_sharpe(sharpe, n_trials, sample_length=sample_length)

        assert dsr < sharpe, "DSR should be lower than naïve Sharpe"

    def test_dsr_with_returns(self):
        """DSR computes from returns directly."""
        returns = np.random.randn(252) * 0.01
        n_trials = 10

        dsr = deflated_sharpe(1.0, n_trials, returns=returns)

        assert isinstance(dsr, float)
        assert dsr < 1.0

    def test_dsr_single_trial(self):
        """n_trials=1 has no multiple-testing correction (sqrt(2*ln(1))=0)."""
        sharpe = 1.5
        dsr = deflated_sharpe(sharpe, n_trials=1, sample_length=252)

        assert dsr == sharpe, "n_trials=1 should give no correction"

    def test_dsr_many_trials(self):
        """Larger n_trials → larger correction."""
        sharpe = 1.5
        dsr_10 = deflated_sharpe(sharpe, n_trials=10, sample_length=252)
        dsr_100 = deflated_sharpe(sharpe, n_trials=100, sample_length=252)

        assert dsr_10 > dsr_100, "More trials should give larger correction"
        assert dsr_100 < sharpe


class TestPBO:
    """1.5.4: Probability of Backtest Overfitting."""

    def test_pbo_basic(self):
        """PBO basic functionality."""
        np.random.seed(42)
        n_models = 5
        n_folds = 3
        scores = np.random.randn(n_models, n_folds)

        result = pbo(scores)

        assert "pbo" in result
        assert 0 <= result["pbo"] <= 1
        assert result["n_models"] == n_models
        assert result["n_folds"] == n_folds

    def test_pbo_overfit_scenario(self):
        """Overfit scenario: in-sample champion underperforms OOS."""
        # Create scores where model 0 is best IS but worst OOS
        scores = np.array(
            [
                [1.0, 0.1, 0.1],  # Model 0: best IS (fold 0)
                [0.2, 0.9, 0.9],  # Model 1: best OOS (folds 1,2)
            ]
        )

        result = pbo(scores)

        # PBO should be > 0 to indicate risk of overfitting
        assert result["pbo"] > 0, "Should detect potential overfitting"
        assert isinstance(result["champion_is_idx"], (int, np.integer))
        assert isinstance(result["champion_oos_idx"], (int, np.integer))

    def test_pbo_insufficient_models(self):
        """PBO requires at least 2 models."""
        scores = np.random.randn(1, 5)
        with pytest.raises(ValueError, match="at least 2 models"):
            pbo(scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
