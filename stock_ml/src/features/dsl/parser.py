"""Lexer + recursive-descent parser for the feature expression DSL.

Grammar (lowest to highest precedence)::

    expr        := comparison
    comparison  := additive ( (> | < | >= | <= | == | !=) additive )*
    additive    := multiplicative ( (+ | -) multiplicative )*
    multiplicative := unary ( (* | /) unary )*
    unary       := '-' unary | postfix
    postfix     := atom ( '.' IDENT )*
    atom        := NUMBER
                 | FIELD            ; $close, $market_close, $sector
                 | FEATURE          ; #ret_20d
                 | IDENT '(' args ')'   ; function call
                 | IDENT            ; bare feature reference
                 | '(' expr ')'
    args        := (arg (',' arg)*)?
    arg         := IDENT '=' expr   ; keyword arg (must follow positionals)
                 | expr             ; positional
"""

from __future__ import annotations

import re

from stock_ml.src.features.dsl.ast import (
    Attribute,
    BinOp,
    Call,
    Const,
    FeatureRef,
    Field,
    Node,
    UnaryOp,
)


class DSLSyntaxError(ValueError):
    """Raised when an expression cannot be tokenized or parsed."""


_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<NUMBER>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
  | (?P<FIELD>\$[A-Za-z_][A-Za-z0-9_]*)
  | (?P<FEATURE>\#[A-Za-z_][A-Za-z0-9_]*)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<OP>>=|<=|==|!=|[-+*/<>(),.=])
    """,
    re.VERBOSE,
)


class _Token:
    __slots__ = ("kind", "value", "pos")

    def __init__(self, kind: str, value: str, pos: int):
        self.kind = kind
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Token({self.kind}, {self.value!r})"


def _tokenize(expr: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    n = len(expr)
    while pos < n:
        m = _TOKEN_RE.match(expr, pos)
        if not m or m.start() != pos:
            raise DSLSyntaxError(f"Unexpected character at {pos}: {expr[pos : pos + 10]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append(_Token(kind, m.group(), m.start()))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token], expr: str):
        self.tokens = tokens
        self.expr = expr
        self.i = 0

    def _peek(self) -> _Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self) -> _Token:
        tok = self._peek()
        if tok is None:
            raise DSLSyntaxError(f"Unexpected end of expression: {self.expr!r}")
        self.i += 1
        return tok

    def _expect_op(self, value: str) -> _Token:
        tok = self._next()
        if tok.kind != "OP" or tok.value != value:
            raise DSLSyntaxError(f"Expected {value!r} but got {tok.value!r} at {tok.pos}")
        return tok

    def parse(self) -> Node:
        node = self._comparison()
        if self._peek() is not None:
            tok = self._peek()
            raise DSLSyntaxError(f"Unexpected token {tok.value!r} at {tok.pos}")
        return node

    def _comparison(self) -> Node:
        node = self._additive()
        while True:
            tok = self._peek()
            if tok and tok.kind == "OP" and tok.value in (">", "<", ">=", "<=", "==", "!="):
                self._next()
                node = BinOp(tok.value, node, self._additive())
            else:
                return node

    def _additive(self) -> Node:
        node = self._multiplicative()
        while True:
            tok = self._peek()
            if tok and tok.kind == "OP" and tok.value in ("+", "-"):
                self._next()
                node = BinOp(tok.value, node, self._multiplicative())
            else:
                return node

    def _multiplicative(self) -> Node:
        node = self._unary()
        while True:
            tok = self._peek()
            if tok and tok.kind == "OP" and tok.value in ("*", "/"):
                self._next()
                node = BinOp(tok.value, node, self._unary())
            else:
                return node

    def _unary(self) -> Node:
        tok = self._peek()
        if tok and tok.kind == "OP" and tok.value == "-":
            self._next()
            return UnaryOp("-", self._unary())
        return self._postfix()

    def _postfix(self) -> Node:
        node = self._atom()
        while True:
            tok = self._peek()
            if tok and tok.kind == "OP" and tok.value == ".":
                self._next()
                attr = self._next()
                if attr.kind != "IDENT":
                    raise DSLSyntaxError(f"Expected attribute name after '.' at {attr.pos}")
                node = Attribute(node, attr.value)
            else:
                return node

    def _atom(self) -> Node:
        tok = self._next()
        if tok.kind == "NUMBER":
            return Const(float(tok.value))
        if tok.kind == "FIELD":
            return Field(tok.value[1:])
        if tok.kind == "FEATURE":
            return FeatureRef(tok.value[1:])
        if tok.kind == "IDENT":
            nxt = self._peek()
            if nxt and nxt.kind == "OP" and nxt.value == "(":
                return self._call(tok.value)
            return FeatureRef(tok.value)
        if tok.kind == "OP" and tok.value == "(":
            node = self._comparison()
            self._expect_op(")")
            return node
        raise DSLSyntaxError(f"Unexpected token {tok.value!r} at {tok.pos}")

    def _call(self, name: str) -> Call:
        self._expect_op("(")
        args: list[Node] = []
        kwargs: dict[str, Node] = {}
        if self._peek() and self._peek().value == ")":
            self._next()
            return Call(name, args, kwargs)
        while True:
            args_before = self._maybe_kwarg()
            if args_before is not None:
                key, value = args_before
                if key in kwargs:
                    raise DSLSyntaxError(f"Duplicate keyword arg {key!r} in {name}()")
                kwargs[key] = value
            else:
                if kwargs:
                    raise DSLSyntaxError(f"Positional arg after keyword arg in {name}()")
                args.append(self._comparison())
            tok = self._next()
            if tok.value == ")":
                break
            if tok.value != ",":
                raise DSLSyntaxError(f"Expected ',' or ')' but got {tok.value!r} at {tok.pos}")
        return Call(name, args, kwargs)

    def _maybe_kwarg(self) -> tuple[str, Node] | None:
        tok = self._peek()
        nxt = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
        if tok and tok.kind == "IDENT" and nxt and nxt.kind == "OP" and nxt.value == "=":
            self._next()  # ident
            self._next()  # '='
            return tok.value, self._comparison()
        return None


def parse(expr: str) -> Node:
    """Parse a DSL expression string into an AST root node.

    Raises:
        DSLSyntaxError: on any lexing/parsing error.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise DSLSyntaxError("Empty expression")
    tokens = _tokenize(expr)
    if not tokens:
        raise DSLSyntaxError(f"Empty expression: {expr!r}")
    return _Parser(tokens, expr).parse()
