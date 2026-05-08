from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from src.data.loader import DataLoader


def _write_symbol_csv(root: Path, symbol: str, timeframe: str = "1H") -> None:
    symbol_dir = root / f"symbol={symbol}" / f"timeframe={timeframe}"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
            "symbol": [symbol, symbol],
        }
    )
    df.to_csv(symbol_dir / "data.csv", index=False)


def test_loader_supports_flat_symbol_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "flat_dataset"
    _write_symbol_csv(data_dir, "VN30F1M")

    loader = DataLoader(str(data_dir), timeframe="1H")

    assert loader.symbols == ["VN30F1M"]
    df = loader.load_symbol("VN30F1M")
    assert len(df) == 2
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)


def test_loader_prefers_all_symbols_when_present(tmp_path: Path) -> None:
    data_dir = tmp_path / "mixed_dataset"
    # Root layout should be ignored when all_symbols layout exists.
    _write_symbol_csv(data_dir, "ROOT_ONLY")
    _write_symbol_csv(data_dir / "all_symbols", "VN30F2M")

    loader = DataLoader(str(data_dir), timeframe="1H")

    assert loader.symbols == ["VN30F2M"]
    df = loader.load_symbol("VN30F2M")
    assert len(df) == 2


def test_vn_derivatives_phase0_manifest_and_schema_sanity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    market_path = repo_root / "config" / "markets" / "vn_derivatives.yaml"
    market = yaml.safe_load(market_path.read_text(encoding="utf-8"))
    data_dir = (market_path.parent / market["data"]["data_dir"]).resolve()
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"derivatives dataset manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == 2
    assert "VN30F1M" in manifest["symbols"]
    assert market["data"]["default_timeframe"] in manifest["timeframes"]

    target_entry = next(
        item
        for item in manifest["files"]
        if item["symbol"] == "VN30F1M" and item["timeframe"] == market["data"]["default_timeframe"]
    )
    assert target_entry["rows"] == 9603
    assert Path(target_entry["csv"]).as_posix() == "symbol=VN30F1M/timeframe=1H/data.csv"

    loader = DataLoader(
        str(data_dir),
        timeframe=market["data"]["default_timeframe"],
        timestamp_column=market["data"]["timestamp_column"],
        timezone=market["data"].get("timezone"),
        required_columns=market["data"]["required_columns"],
        optional_columns=market["data"].get("optional_columns"),
    )
    df = loader.load_symbol("VN30F1M", use_cache=False)

    assert len(df) == target_entry["rows"]
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()
    assert df[market["data"]["required_columns"]].notna().all().all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] >= 0).all()
