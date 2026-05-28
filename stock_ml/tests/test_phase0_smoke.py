"""Phase 0 smoke test: MLflow tracking, reproducibility, logging."""

import tempfile
from pathlib import Path

import pytest
from src.pipeline.run import build_default_config, run
from src.seed import set_global_seed
from src.tracking.mlflow_logger import MLFlowLogger, data_fingerprint, get_git_commit


def test_git_commit():
    """Test git commit detection."""
    commit = get_git_commit()
    assert len(commit) == 40 or commit == "unknown"


def test_data_fingerprint():
    """Test data fingerprint is stable."""
    symbols = ["AAA", "SSI", "VND"]
    start = "2020-01-01"
    end = "2025-12-31"

    fp1 = data_fingerprint(symbols, start, end)
    fp2 = data_fingerprint(symbols, start, end)
    assert fp1 == fp2, "fingerprint should be deterministic"
    assert len(fp1) == 16, "fingerprint should be 16 hex chars"


def test_data_fingerprint_different():
    """Test fingerprint changes with different data."""
    fp1 = data_fingerprint(["AAA", "SSI"], "2020-01-01", "2025-12-31")
    fp2 = data_fingerprint(["BBB", "SSI"], "2020-01-01", "2025-12-31")
    assert fp1 != fp2


def test_set_global_seed():
    """Test seed setter returns seed."""
    seed = set_global_seed(42)
    assert seed == 42

    seed = set_global_seed(None)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**31


def test_mlflow_logger_context():
    """Test MLFlowLogger context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with MLFlowLogger(tracking_uri=tmpdir, experiment_name="test") as mlf:
            assert mlf._in_context is True
            mlf.log_metrics({"test_metric": 1.5})
            run_id = mlf.get_run_id()
            assert run_id in ("no_mlflow",) or len(run_id) > 0

        assert mlf._in_context is False


def test_mlflow_logger_config_logging():
    """Test MLFlowLogger logs config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = {"name": "test", "seed": 42, "target": {"horizon": 5}}
        with MLFlowLogger(tracking_uri=tmpdir, experiment_name="test") as mlf:
            mlf.log_config(config_dict)


def test_run_config_minimal():
    """Test building minimal config."""
    cfg = build_default_config(
        data_root="/tmp/data",
        symbols=["AAA", "SSI"],
        out_dir="/tmp/results",
        name="test_run",
    )
    assert cfg.name == "test_run"
    assert cfg.seed == 42
    assert cfg.target.horizon == 5


def test_run_output_structure():
    """Test that run creates expected output structure.

    This requires actual data. Skipped if data unavailable.
    """
    data_root = Path("portable_data/vn_stock_ai_dataset_cleaned")
    if not data_root.exists():
        pytest.skip("data not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = build_default_config(
            data_root=str(data_root),
            symbols=["AAA"],
            out_dir=tmpdir,
            name="smoke_test",
            first_test_year=2024,
            last_test_year=2024,
            train_years=1,
            test_years=1,
        )
        summary = run(cfg)

        assert summary.get("ok") is not False or summary.get("n_trades") >= 0
        assert "run_id" in summary, "summary should have run_id"
        assert "data_fingerprint" in summary
        assert "git_commit" in summary
        assert "mlflow_run_id" in summary

        run_dir = Path(tmpdir) / summary["run_id"]
        assert run_dir.exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "data_fingerprint.txt").exists()
        assert (run_dir / "trades.csv").exists()
        assert (run_dir / "signals.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
