"""FeatureStore tests: content-addressed save/load, data_version, JOIN matrix."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_ml.src.features.store import FeatureStore, content_fingerprint


def _frame(values, symbol="AAA"):
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"symbol": symbol, "date": dates, "value": values})


def test_path_layout(tmp_path):
    store = FeatureStore(tmp_path)
    p = store.path("abc123", "ver1")
    assert p == tmp_path / "abc123" / "ver1.parquet"


def test_save_then_load_roundtrip(tmp_path):
    store = FeatureStore(tmp_path)
    frame = _frame([1.0, 2.0, 3.0])
    assert not store.exists("h1", "v1")
    path = store.save(expr_hash="h1", data_version="v1", frame=frame)
    assert path.exists()
    assert store.exists("h1", "v1")
    loaded = store.load("h1", "v1")
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False
    )


def test_two_loads_are_cache_hits(tmp_path):
    store = FeatureStore(tmp_path)
    store.save(expr_hash="h", data_version="v", frame=_frame([0.5, 0.6]))
    first = store.load("h", "v")
    second = store.load("h", "v")
    assert first is not None and second is not None
    pd.testing.assert_frame_equal(first, second)
    # exactly one physical file under the expr_hash dir
    files = list((tmp_path / "h").glob("*.parquet"))
    assert len(files) == 1


def test_load_miss_returns_none(tmp_path):
    assert FeatureStore(tmp_path).load("nope", "nope") is None


def test_atomic_write_leaves_no_temp(tmp_path):
    store = FeatureStore(tmp_path)
    store.save(expr_hash="h", data_version="v", frame=_frame([1.0]))
    leftovers = [p for p in (tmp_path / "h").iterdir() if p.suffix != ".parquet"]
    assert leftovers == []


def test_save_validates_schema(tmp_path):
    store = FeatureStore(tmp_path)
    bad = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2020-01-01")], "val": [1.0]})
    with pytest.raises(ValueError):
        store.save(expr_hash="h", data_version="v", frame=bad)


def test_data_version_deterministic_and_universe_sensitive():
    kw = dict(symbols=["BBB", "AAA"], timeframe="1d", start="2020-01-01", end="2021-01-01")
    v1 = FeatureStore.compute_data_version(**kw)
    v2 = FeatureStore.compute_data_version(
        symbols=["AAA", "BBB"], timeframe="1d", start="2020-01-01", end="2021-01-01"
    )
    assert v1 == v2  # order-independent
    v3 = FeatureStore.compute_data_version(
        symbols=["AAA", "BBB", "CCC"], timeframe="1d", start="2020-01-01", end="2021-01-01"
    )
    assert v1 != v3  # different universe -> different version
    v4 = FeatureStore.compute_data_version(**{**kw, "timeframe": "1h"})
    assert v1 != v4


def test_data_version_requires_symbols():
    with pytest.raises(ValueError):
        FeatureStore.compute_data_version(symbols=[], timeframe="1d")


def test_dedup_shared_feature_is_one_file(tmp_path):
    store = FeatureStore(tmp_path)
    frame = _frame([1.0, 2.0, 3.0])
    # same expr_hash + data_version saved "by two sets" overwrites the one file
    store.save(expr_hash="rsi14", data_version="v", frame=frame)
    store.save(expr_hash="rsi14", data_version="v", frame=frame)
    files = list((tmp_path / "rsi14").glob("*.parquet"))
    assert len(files) == 1


def test_join_matrix(tmp_path):
    store = FeatureStore(tmp_path)
    dates = pd.bdate_range("2020-01-01", periods=3)
    f1 = pd.DataFrame({"symbol": "AAA", "date": dates, "value": [1.0, 2.0, 3.0]})
    f2 = pd.DataFrame({"symbol": "AAA", "date": dates, "value": [10.0, 20.0, 30.0]})
    store.save(expr_hash="ha", data_version="v", frame=f1)
    store.save(expr_hash="hb", data_version="v", frame=f2)
    matrix = store.join_matrix(
        [
            {"name": "feat_a", "expr_hash": "ha", "data_version": "v"},
            {"name": "feat_b", "expr_hash": "hb", "data_version": "v"},
        ]
    )
    assert list(matrix.columns) == ["symbol", "date", "feat_a", "feat_b"]
    assert len(matrix) == 3
    row = matrix.sort_values("date").iloc[1]
    assert row["feat_a"] == 2.0 and row["feat_b"] == 20.0


def test_join_matrix_missing_member_raises(tmp_path):
    store = FeatureStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.join_matrix([{"name": "x", "expr_hash": "nope", "data_version": "v"}])


def test_content_fingerprint_deterministic_sensitive_and_order_independent():
    dates = pd.bdate_range("2020-01-01", periods=4)
    base = pd.DataFrame({"symbol": "AAA", "date": dates, "close": [1.0, 2.0, 3.0, 4.0]})

    fp = content_fingerprint(base)
    assert fp == content_fingerprint(base.copy())  # deterministic

    changed = base.copy()
    changed.loc[0, "close"] = 1.5
    assert content_fingerprint(changed) != fp  # any value change flips it

    reordered = base.iloc[::-1].reset_index(drop=True)
    assert content_fingerprint(reordered) == fp  # order-independent
