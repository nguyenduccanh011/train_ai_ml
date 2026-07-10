"""Parser/lexer unit tests for the feature DSL."""

from __future__ import annotations

import pytest

from stock_ml.src.features.dsl import ast as dslast
from stock_ml.src.features.dsl.engine import extract_deps, infer_kind
from stock_ml.src.features.dsl.parser import DSLSyntaxError, parse


def test_field_and_const():
    node = parse("$close")
    assert isinstance(node, dslast.Field) and node.name == "close"
    assert isinstance(parse("14"), dslast.Const)
    assert parse("1e-8").value == pytest.approx(1e-8)


def test_bare_and_hash_feature_ref():
    assert isinstance(parse("#ret_20d"), dslast.FeatureRef)
    bare = parse("ret_20d")
    assert isinstance(bare, dslast.FeatureRef) and bare.name == "ret_20d"


def test_arithmetic_precedence():
    # $close / Ref($close, 5) - 1  ==  (($close / Ref(...)) - 1)
    node = parse("$close / Ref($close, 5) - 1")
    assert isinstance(node, dslast.BinOp) and node.op == "-"
    assert isinstance(node.left, dslast.BinOp) and node.left.op == "/"
    assert isinstance(node.right, dslast.Const)


def test_multiplicative_binds_tighter_than_additive():
    node = parse("1 + 2 * 3")
    assert node.op == "+"
    assert isinstance(node.right, dslast.BinOp) and node.right.op == "*"


def test_call_with_args_and_attribute():
    node = parse("MACD($close, 12, 26, 9).hist")
    assert isinstance(node, dslast.Attribute) and node.attr == "hist"
    call = node.value
    assert isinstance(call, dslast.Call) and call.name == "MACD"
    assert len(call.args) == 4


def test_kwarg_call():
    node = parse("CSGroupMedian(#ret_20d, by=$sector)")
    assert isinstance(node, dslast.Call)
    assert "by" in node.kwargs
    assert isinstance(node.kwargs["by"], dslast.Field)


def test_comparison_returns_node():
    node = parse("$market_close > Mean($market_close, 200)")
    assert isinstance(node, dslast.BinOp) and node.op == ">"


def test_unary_minus():
    node = parse("-$close")
    assert isinstance(node, dslast.UnaryOp) and node.op == "-"


@pytest.mark.parametrize("bad", ["", "$close +", "Func(,)", "1 2", "$close.", "(", "$"])
def test_syntax_errors(bad):
    with pytest.raises(DSLSyntaxError):
        parse(bad)


def test_infer_kind():
    assert infer_kind(parse("RSI($close, 14)")) == "per_symbol"
    assert infer_kind(parse("CSRank($close)")) == "cross_sectional"
    assert infer_kind(parse("$market_close > Mean($market_close, 200)")) == "market"


def test_extract_deps():
    feats, raws = extract_deps(parse("#ret_20d - CSGroupMedian(#ret_20d, by=$sector)"))
    assert feats == {"ret_20d"}
    assert raws == {"sector"}

    feats2, raws2 = extract_deps(parse("$close / Ref($close, 5) - 1"))
    assert feats2 == set()
    assert raws2 == {"close"}
