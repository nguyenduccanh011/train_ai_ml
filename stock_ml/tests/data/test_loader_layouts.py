from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from src.data.loader import DataLoader


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
