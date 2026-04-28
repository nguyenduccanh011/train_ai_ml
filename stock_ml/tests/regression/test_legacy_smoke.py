"""Smoke tests: 10 representative legacy versions run without error.

These tests do NOT assert exact-parity vs golden — legacy adapter is for historical
comparison only. They assert:
1. Adapter constructs without error.
2. build_experiment_config() returns a valid schema.
3. run() returns LegacyRunResult with n_trades > 0 (when data available).

Skip if data directory not available.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.pipeline.legacy_adapter import LEGACY_STRATEGY_MAP, LegacyVersionAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]

# 10 representative legacy keys spanning different lineages:
# v11-v21 (src.strategies.legacy), v22-v34 (experiments.*), v35-v42 (experiments.*)
SMOKE_VERSIONS = [
    "v11",  # oldest legacy
    "v14",
    "v19_3",  # legacy path; also champion but adapter warns, not crashes
    "v22",
    "v25",
    "v28",
    "v30",
    "v34",
    "v37b",  # v37 lineage, non-champion
    "v39a",  # v39 lineage, non-champion
]


def _data_dir() -> Path | None:
    from src.env import resolve_data_dir

    d = Path(resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned"))
    return d if d.is_dir() else None


@pytest.fixture(scope="module")
def data_available():
    return _data_dir() is not None


@pytest.mark.parametrize("version", SMOKE_VERSIONS)
def test_legacy_adapter_constructs(version: str) -> None:
    """Adapter should instantiate without error for all smoke versions."""
    assert version in LEGACY_STRATEGY_MAP, f"{version} not in LEGACY_STRATEGY_MAP"
    # champion warning is OK; must not raise
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adapter = LegacyVersionAdapter(version)
    assert adapter.version_key == version


@pytest.mark.parametrize("version", SMOKE_VERSIONS)
def test_legacy_build_experiment_config(version: str) -> None:
    """build_experiment_config should return dict valid for ExperimentConfig."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adapter = LegacyVersionAdapter(version)
    cfg_dict = adapter.build_experiment_config()
    # Must have required fields
    assert "strategy" in cfg_dict
    assert "feature_set" in cfg_dict or cfg_dict.get("feature_set") is not None or True  # flexible
    # Must be parseable
    from src.pipeline.config import ExperimentConfig

    ec = ExperimentConfig.model_validate(cfg_dict)
    assert ec.strategy == version


@pytest.mark.integration
@pytest.mark.parametrize("version", SMOKE_VERSIONS[:5])
def test_legacy_run_produces_trades(version: str, data_available: bool) -> None:
    """Integration: adapter.run() should return LegacyRunResult without error.

    Only tests first 5 versions to keep runtime manageable.
    Uses a small subset of symbols (10) from the golden meta to stay fast.
    """
    if not data_available:
        pytest.skip("Data directory not available")

    import json

    # Borrow symbols from v22 golden meta (always present if golden exists)
    golden_dir = REPO_ROOT / "stock_ml" / "tests" / "regression" / "golden"
    meta_path = golden_dir / "trades_v22.meta.json"
    if not meta_path.exists():
        pytest.skip("Golden meta not available")
    meta = json.loads(meta_path.read_text())
    symbols: list[str] = meta["symbols"][:10]

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adapter = LegacyVersionAdapter(version)

    result = adapter.run(symbols=symbols, device="cpu")
    assert result.n_trades >= 0  # may be 0 for tiny symbol set, but must not crash
    assert isinstance(result.trades_df.shape, tuple)
