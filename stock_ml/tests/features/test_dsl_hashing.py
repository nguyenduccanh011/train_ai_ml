"""Canonicalisation + expr_hash tests."""

from __future__ import annotations

from stock_ml.src.features.dsl.hashing import canonicalize, expr_hash


def test_commutative_add_canonical_equal():
    assert canonicalize("$close + $open") == canonicalize("$open + $close")
    assert expr_hash("$a + $b") == expr_hash("$b + $a")


def test_commutative_mul_canonical_equal():
    assert expr_hash("2 * $close") == expr_hash("$close * 2")


def test_non_commutative_order_matters():
    assert expr_hash("$close - $open") != expr_hash("$open - $close")
    assert expr_hash("$close / $open") != expr_hash("$open / $close")


def test_whitespace_irrelevant():
    assert expr_hash("$close/Ref($close,5)-1") == expr_hash("$close / Ref($close, 5) - 1")


def test_literal_normalisation():
    assert canonicalize("14") == canonicalize("14.0")


def test_hash_changes_with_operator():
    assert expr_hash("Mean($close, 20)") != expr_hash("Mean($close, 21)")
    assert expr_hash("Mean($close, 20)") != expr_hash("Std($close, 20)")


def test_hash_is_deterministic_hex():
    h = expr_hash("RSI($close, 14)")
    assert isinstance(h, str) and len(h) == 40
    assert h == expr_hash("RSI($close, 14)")


def test_dep_hashes_affect_hash():
    base = expr_hash("#ret_20d")
    with_dep = expr_hash("#ret_20d", dep_hashes=["abc123"])
    assert base != with_dep


def test_engine_version_affects_hash():
    assert expr_hash("RSI($close, 14)", engine_version="v1") != expr_hash(
        "RSI($close, 14)", engine_version="v2"
    )
