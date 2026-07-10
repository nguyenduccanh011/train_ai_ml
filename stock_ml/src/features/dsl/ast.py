"""AST node types for the feature expression DSL.

Nodes are plain dataclasses. The parser builds a tree of these; the engine walks
it to evaluate, infer ``kind``, extract dependencies, and canonicalize for hashing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


class Node:
    """Base class for all AST nodes."""


@dataclass
class Const(Node):
    """Numeric literal, e.g. ``14``, ``2.0``, ``1e-8``."""

    value: float


@dataclass
class Field(Node):
    """Raw column reference ``$name`` (e.g. ``$close``, ``$market_close``, ``$sector``)."""

    name: str  # without the leading '$'


@dataclass
class FeatureRef(Node):
    """Reference to another defined feature: ``#name`` or a bare ``name``.

    Creates a DAG edge to the named feature_def.
    """

    name: str  # without the leading '#'


@dataclass
class UnaryOp(Node):
    """Unary operation, currently only negation ``-x``."""

    op: str  # '-'
    operand: Node


@dataclass
class BinOp(Node):
    """Binary operation: arithmetic (+ - * /) or comparison (> < >= <= == !=)."""

    op: str
    left: Node
    right: Node


@dataclass
class Call(Node):
    """Function call ``Func(arg1, arg2, ..., by=$sector)``."""

    name: str
    args: list[Node] = field(default_factory=list)
    kwargs: dict[str, Node] = field(default_factory=dict)


@dataclass
class Attribute(Node):
    """Attribute access on a multi-output call, e.g. ``MACD($close,12,26,9).hist``."""

    value: Node  # expected to be a Call
    attr: str


COMPARISON_OPS = frozenset({">", "<", ">=", "<=", "==", "!="})
ARITHMETIC_OPS = frozenset({"+", "-", "*", "/"})
COMMUTATIVE_OPS = frozenset({"+", "*", "==", "!="})


def canonical(node: Node) -> str:
    """Normalised string form of an AST.

    Semantically equal expressions collapse to one form (commutative operands
    sorted, literals normalised, whitespace removed). Used both for content
    hashing (``hashing.expr_hash``) and for sub-expression memoisation in the
    engine — one source of truth so the two never drift.
    """
    if isinstance(node, Const):
        return repr(float(node.value))
    if isinstance(node, Field):
        return f"${node.name}"
    if isinstance(node, FeatureRef):
        return f"#{node.name}"
    if isinstance(node, UnaryOp):
        return f"(u{node.op} {canonical(node.operand)})"
    if isinstance(node, BinOp):
        lc = canonical(node.left)
        rc = canonical(node.right)
        if node.op in COMMUTATIVE_OPS:
            lc, rc = sorted((lc, rc))
        return f"({node.op} {lc} {rc})"
    if isinstance(node, Call):
        parts = [canonical(a) for a in node.args]
        parts += [f"{k}={canonical(v)}" for k, v in sorted(node.kwargs.items())]
        return f"{node.name}({','.join(parts)})"
    if isinstance(node, Attribute):
        return f"{canonical(node.value)}.{node.attr}"
    raise TypeError(f"Cannot canonicalize {type(node).__name__}")


def walk(node: Node) -> Iterator[Node]:
    """Yield ``node`` and all descendant nodes (pre-order)."""
    yield node
    if isinstance(node, UnaryOp):
        yield from walk(node.operand)
    elif isinstance(node, BinOp):
        yield from walk(node.left)
        yield from walk(node.right)
    elif isinstance(node, Call):
        for a in node.args:
            yield from walk(a)
        for v in node.kwargs.values():
            yield from walk(v)
    elif isinstance(node, Attribute):
        yield from walk(node.value)
