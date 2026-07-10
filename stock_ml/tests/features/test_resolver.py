"""FeatureResolver tests: leading_v2 parity, per-feature caching/dedup, fail-loud."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_ml.src.features.resolver import FeatureResolver
from stock_ml.src.features.store import FeatureStore

from ._helpers import assert_close, golden_frame, make_ohlcv

DF = make_ohlcv()


def test_resolve_leading_v2_matches_builder(tmp_path):
    resolver = FeatureResolver.from_catalog(FeatureStore(tmp_path))
    feat, cols, hits = resolver.resolve(DF, ["leading_v2"], data_root="ds")
    assert hits == 0  # empty store on first run
    assert len(cols["leading_v2"]) == 37

    v2 = golden_frame()
    got = feat.set_index(["symbol", "date"])
    # Parity is against the legacy builder's columns only; is_limit_lock is a new
    # feature with no legacy counterpart (the count is asserted separately above).
    for c in v2.columns:
        assert_close(got[c], v2[c])


def test_second_resolve_is_all_cache_hits(tmp_path):
    store = FeatureStore(tmp_path)
    resolver = FeatureResolver.from_catalog(store)
    resolver.resolve(DF, ["leading_v2"], data_root="ds")
    _feat, _cols, hits = resolver.resolve(DF, ["leading_v2"], data_root="ds")
    assert hits == 37  # every feature served from the content-addressed store


def test_shared_feature_resolved_once(tmp_path):
    # basic_v1 ⊂ leading_v2 → union has no duplicates, total = 37 unique features.
    resolver = FeatureResolver.from_catalog(FeatureStore(tmp_path))
    feat, cols, _hits = resolver.resolve(DF, ["leading_v2", "basic_v1"], data_root="ds")
    base = {"symbol", "date", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in feat.columns if c not in base]
    assert len(feature_cols) == 37
    assert set(cols["basic_v1"]).issubset(set(cols["leading_v2"]))


def test_leading_v3_requires_market_and_sector(tmp_path):
    resolver = FeatureResolver.from_catalog(FeatureStore(tmp_path))
    with pytest.raises(ValueError):
        resolver.resolve(DF, ["leading_v3"], data_root="ds")


def test_leading_v3_resolves_with_market_and_sector(tmp_path):
    resolver = FeatureResolver.from_catalog(FeatureStore(tmp_path))
    dates = sorted(DF["date"].unique())
    market_df = pd.DataFrame({"date": dates, "close": range(100, 100 + len(dates))})
    sector_map = {"AAA": "Finance", "BBB": "Finance", "CCC": "Energy"}
    feat, cols, _hits = resolver.resolve(
        DF, ["leading_v3"], market_df=market_df, sector_map=sector_map, data_root="ds"
    )
    assert "momentum_rank" in feat.columns
    assert "return_vs_sector" in feat.columns
    assert "market_trend" in feat.columns
    assert len(cols["leading_v3"]) == 56


def test_unknown_set_raises(tmp_path):
    resolver = FeatureResolver.from_catalog(FeatureStore(tmp_path))
    with pytest.raises(KeyError):
        resolver.resolve(DF, ["does_not_exist"], data_root="ds")


def test_leading_alias_resolves(tmp_path):
    resolver = FeatureResolver.from_catalog(FeatureStore(tmp_path))
    _feat, cols, _hits = resolver.resolve(DF, ["leading"], data_root="ds")
    assert len(cols["leading"]) == 37


def test_leading_v3_via_pipeline_inputs(tmp_path):
    # Mirror the experiment.py wiring: equal-weight market index + sector map.
    from stock_ml.src.features.market import build_equal_weight_index
    from stock_ml.src.features.sectors import build_sector_map

    resolver = FeatureResolver.from_catalog(FeatureStore(tmp_path))
    needed = resolver.required_raw_inputs(["leading_v3"])
    assert {"market_close", "sector"}.issubset(needed)

    sector_map = build_sector_map(sorted(DF["symbol"].unique()))
    market_df = build_equal_weight_index(DF)
    feat, _cols, _hits = resolver.resolve(
        DF, ["leading_v3"], market_df=market_df, sector_map=sector_map, data_root="ds"
    )
    # CSRank is a within-date percentile in [0, 1]
    assert feat["momentum_rank"].dropna().between(0.0, 1.0).all()
    # market regime flags are binary
    assert set(feat["market_trend"].dropna().unique()).issubset({0.0, 1.0})
    assert feat["return_vs_sector"].notna().any()


def test_data_change_busts_cache(tmp_path):
    store = FeatureStore(tmp_path)
    resolver = FeatureResolver.from_catalog(store)
    n = len(resolver.feature_cols("basic_v1"))

    resolver.resolve(DF, ["basic_v1"], data_root="ds")
    _f, _c, hits_same = resolver.resolve(DF, ["basic_v1"], data_root="ds")
    assert hits_same == n  # unchanged data → all hits

    df2 = DF.copy()
    df2.loc[df2.index[0], "close"] = df2.loc[df2.index[0], "close"] * 1.05
    _f2, _c2, hits_changed = resolver.resolve(df2, ["basic_v1"], data_root="ds")
    assert hits_changed == 0  # mutated data → content fingerprint flips → recompute
