"""Expression DSL for feature definitions (Qlib-style).

A feature is a string formula (e.g. ``$close / Ref($close, 5) - 1``) parsed into
an AST, evaluated against an OHLCV DataFrame, and content-addressed via expr_hash.

Public surface:
    parse(expr)              -> AST root node
    Engine().eval(node, ctx) -> pd.Series
    infer_kind(node)         -> 'per_symbol' | 'cross_sectional' | 'market'
    extract_deps(node)       -> (feature_refs, raw_fields)
    expr_hash(expr, ...)     -> sha1 hex
"""

from __future__ import annotations

from stock_ml.src.features.dsl.engine import (
    ENGINE_VERSION,
    Engine,
    EvalContext,
    extract_deps,
    infer_kind,
)
from stock_ml.src.features.dsl.hashing import canonicalize, expr_hash
from stock_ml.src.features.dsl.parser import DSLSyntaxError, parse

__all__ = [
    "ENGINE_VERSION",
    "DSLSyntaxError",
    "Engine",
    "EvalContext",
    "canonicalize",
    "expr_hash",
    "extract_deps",
    "infer_kind",
    "parse",
]
