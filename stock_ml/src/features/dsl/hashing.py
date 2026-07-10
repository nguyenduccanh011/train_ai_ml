"""Canonicalisation + content-addressed hashing of feature expressions.

``canonicalize`` renders an AST to a normalised string so that semantically equal
expressions collapse to one form (commutative operands sorted, literals
normalised, whitespace removed). ``expr_hash`` folds that canonical form together
with the engine version and the hashes of any dependency features, so changing an
operator or an upstream feature flips the hash and invalidates downstream caches.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from stock_ml.src.features.dsl import ast as dslast
from stock_ml.src.features.dsl.engine import ENGINE_VERSION
from stock_ml.src.features.dsl.parser import parse


def canonicalize(expr: str | dslast.Node) -> str:
    """Return the canonical string form of an expression (string or AST)."""
    node = parse(expr) if isinstance(expr, str) else expr
    return dslast.canonical(node)


def expr_hash(
    expr: str | dslast.Node,
    dep_hashes: Iterable[str] = (),
    engine_version: str = ENGINE_VERSION,
) -> str:
    """sha1 of the canonical form + engine version + sorted dependency hashes."""
    canon = canonicalize(expr)
    payload = "|".join([canon, engine_version, ",".join(sorted(dep_hashes))])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
