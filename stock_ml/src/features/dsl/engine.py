"""Evaluator for the feature DSL.

``Engine.eval(node, ctx)`` walks an AST and returns a ``pd.Series`` aligned to
``ctx.df.index`` (or a ``dict`` for a multi-output op awaiting an ``.attr``).
Numeric literals stay Python ``float`` scalars so window arguments like the 20 in
``Mean($close, 20)`` are not materialised as constant columns.

Also provides ``infer_kind`` (per_symbol / cross_sectional / market — derived from
the operators used) and ``extract_deps`` (feature refs + raw fields) for the DB
layer.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from stock_ml.src.features.dsl import ast as dslast
from stock_ml.src.features.dsl.ops import get_op

ENGINE_VERSION = "dsl-1.0.0"

# DSL source files whose content affects computed feature *values* (operator
# implementations live here, not in the stored expression strings).
_CODE_FILES = ("ast.py", "parser.py", "ops.py", "engine.py", "hashing.py")


@functools.lru_cache(maxsize=1)
def engine_code_fingerprint() -> str:
    """sha1 over the DSL source files.

    Folded into the feature store's ``data_version`` so editing an operator (e.g.
    fixing the RSI smoothing) invalidates materialised caches automatically —
    without relying on someone remembering to bump ``ENGINE_VERSION``.
    """
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha1(ENGINE_VERSION.encode("utf-8"))
    for name in _CODE_FILES:
        try:
            h.update((pkg / name).read_bytes())
        except OSError:
            h.update(b"\x00missing\x00")
    return h.hexdigest()[:12]


_OHLCV = frozenset({"open", "high", "low", "close", "volume"})

_COMPARE = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class EvalContext:
    """Holds the data and resolved feature values an expression evaluates against."""

    df: pd.DataFrame
    features: dict[str, pd.Series] = field(default_factory=dict)
    # Sub-expression cache (keyed by canonical form) so a repeated sub-tree like
    # ``Bollinger($close,20,2)`` in bb_squeeze is evaluated once, not 4×.
    memo: dict = field(default_factory=dict)

    @property
    def index(self) -> pd.Index:
        return self.df.index

    @property
    def symbol(self) -> pd.Series:
        return self.df["symbol"]

    @property
    def date(self) -> pd.Series:
        return self.df["date"]

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        features: dict[str, pd.Series] | None = None,
        *,
        sort: bool = True,
    ) -> EvalContext:
        """Build a context; sorts by [symbol, date] so time-series ops see ordered history."""
        if "symbol" not in df.columns or "date" not in df.columns:
            raise ValueError("EvalContext requires 'symbol' and 'date' columns")
        if sort:
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        feats = {}
        if features:
            # Re-align provided feature series to the (possibly sorted) frame index.
            for k, v in features.items():
                feats[k] = v.reindex(df.index) if isinstance(v, pd.Series) else v
        return cls(df=df, features=feats)


class Engine:
    """Recursive AST evaluator."""

    def eval(self, node: dslast.Node, ctx: EvalContext):
        if isinstance(node, dslast.Const):
            return node.value
        if isinstance(node, dslast.Field):
            if node.name not in ctx.df.columns:
                raise KeyError(f"Field ${node.name} not present in data columns")
            return ctx.df[node.name]
        if isinstance(node, dslast.FeatureRef):
            if node.name not in ctx.features:
                raise KeyError(f"Feature #{node.name} not resolved in context")
            return ctx.features[node.name]
        if isinstance(node, dslast.UnaryOp):
            return -self._series(node.operand, ctx)
        if isinstance(node, dslast.BinOp):
            return self._binop(node, ctx)
        if isinstance(node, dslast.Call):
            key = dslast.canonical(node)
            if key in ctx.memo:
                return ctx.memo[key]
            result = self._call(node, ctx)
            ctx.memo[key] = result
            return result
        if isinstance(node, dslast.Attribute):
            value = self.eval(node.value, ctx)
            if not isinstance(value, dict):
                raise TypeError(f".{node.attr} used on a single-output expression")
            if node.attr not in value:
                raise KeyError(f"Unknown attribute .{node.attr}; have {sorted(value)}")
            return value[node.attr]
        raise TypeError(f"Cannot evaluate node {type(node).__name__}")

    def _series(self, node: dslast.Node, ctx: EvalContext):
        val = self.eval(node, ctx)
        if isinstance(val, dict):
            raise TypeError("Multi-output op requires an attribute selector (e.g. .hist)")
        return val

    def _binop(self, node: dslast.BinOp, ctx: EvalContext):
        left = self._series(node.left, ctx)
        right = self._series(node.right, ctx)
        op = node.op
        if op in _COMPARE:
            res = _COMPARE[op](left, right)
            return res.astype(float) if isinstance(res, pd.Series) else float(bool(res))
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        raise ValueError(f"Unknown binary operator {op!r}")

    def _call(self, node: dslast.Call, ctx: EvalContext):
        spec = get_op(node.name)
        if spec is None:
            raise KeyError(f"Unknown operator {node.name!r}")
        args = [self._arg(a, ctx) for a in node.args]
        kwargs = {k: self._arg(v, ctx) for k, v in node.kwargs.items()}
        return spec.fn(ctx, args, kwargs)

    def _arg(self, node: dslast.Node, ctx: EvalContext):
        # Numeric literals stay scalars; everything else is a Series (never a dict).
        if isinstance(node, dslast.Const):
            return node.value
        return self._series(node, ctx)


# ---------------------------------------------------------------------------
# Static analysis (no evaluation)
# ---------------------------------------------------------------------------


def infer_kind(node: dslast.Node) -> str:
    """Derive the eval context a feature needs from the operators it uses."""
    has_cross_sectional = False
    has_market = False
    for n in dslast.walk(node):
        if isinstance(n, dslast.Call):
            spec = get_op(n.name)
            if spec is not None and spec.axis == "date":
                has_cross_sectional = True
        elif isinstance(n, dslast.Field) and n.name.startswith("market_"):
            has_market = True
    if has_cross_sectional:
        return "cross_sectional"
    if has_market:
        return "market"
    return "per_symbol"


def extract_deps(node: dslast.Node) -> tuple[set[str], set[str]]:
    """Return (feature_refs, raw_fields) referenced anywhere in the expression."""
    feature_refs: set[str] = set()
    raw_fields: set[str] = set()
    for n in dslast.walk(node):
        if isinstance(n, dslast.FeatureRef):
            feature_refs.add(n.name)
        elif isinstance(n, dslast.Field):
            raw_fields.add(n.name)
    return feature_refs, raw_fields
