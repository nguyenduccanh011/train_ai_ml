"""FeatureResolver — turn feature set names into a feature matrix.

Pipeline:  load members → build DAG → topo-sort → evaluate per kind → cache each
feature in the content-addressed FeatureStore → assemble the set matrix.

A feature shared by the entry and exit sets is evaluated and stored exactly once
(``cache hits`` reported), realising the "sharing a feature is free" principle.

Definitions come from the canonical ``catalog`` module — the same source that
seeds the DB mirror (feature_def/feature_set) used by the API/UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stock_ml.src.features.catalog import FEATURES, SETS, compute_expr_hashes
from stock_ml.src.features.dsl import ast as dslast
from stock_ml.src.features.dsl.engine import (
    ENGINE_VERSION,
    Engine,
    EvalContext,
    engine_code_fingerprint,
    extract_deps,
    infer_kind,
)
from stock_ml.src.features.dsl.parser import parse
from stock_ml.src.features.store import FeatureStore, content_fingerprint

_OHLCV = frozenset({"open", "high", "low", "close", "volume"})


@dataclass
class FeatureDefinition:
    name: str
    expr: str
    kind: str
    expr_hash: str
    node: dslast.Node
    feature_deps: list[str] = field(default_factory=list)
    raw_deps: list[str] = field(default_factory=list)


class FeatureResolver:
    """Resolve feature sets to a materialised matrix, caching per feature."""

    def __init__(
        self,
        definitions: dict[str, FeatureDefinition],
        set_members: dict[str, list[str]],
        store: FeatureStore | None = None,
        engine_version: str = ENGINE_VERSION,
    ):
        self.defs = definitions
        self.set_members = set_members
        self.store = store or FeatureStore()
        self.engine = Engine()
        self.engine_version = engine_version

    # -- construction -----------------------------------------------------

    @classmethod
    def from_catalog(cls, store: FeatureStore | None = None) -> FeatureResolver:
        hashes = compute_expr_hashes(FEATURES)
        defs: dict[str, FeatureDefinition] = {}
        for name, expr in FEATURES.items():
            node = parse(expr)
            refs, raws = extract_deps(node)
            defs[name] = FeatureDefinition(
                name=name,
                expr=expr,
                kind=infer_kind(node),
                expr_hash=hashes[name],
                node=node,
                feature_deps=sorted(refs),
                raw_deps=sorted(raws),
            )
        sets = {name: list(members) for name, (_desc, members) in SETS.items()}
        return cls(defs, sets, store)

    # -- public API -------------------------------------------------------

    def feature_cols(self, set_name: str) -> list[str]:
        if set_name not in self.set_members:
            raise KeyError(f"Unknown feature set '{set_name}'")
        return list(self.set_members[set_name])

    def required_raw_inputs(self, set_names: list[str]) -> set[str]:
        """Raw ``$``-fields the union of the given sets depends on (full closure).

        Lets callers build only the external inputs actually needed — e.g. skip
        constructing a market index for per-symbol-only sets.
        """
        requested: list[str] = []
        for s in set_names:
            for m in self.feature_cols(s):
                if m not in requested:
                    requested.append(m)
        order = self._topo_order(requested)
        return {r for n in order for r in self.defs[n].raw_deps}

    def resolve(
        self,
        ohlcv: pd.DataFrame,
        set_names: list[str],
        *,
        market_df: pd.DataFrame | None = None,
        sector_map: dict[str, str] | None = None,
        timeframe: str = "1D",
        data_root: str | None = None,
        cache: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, list[str]], int]:
        """Materialise the union of the requested sets onto ``ohlcv``.

        Args:
            cache: when False, evaluate fresh without reading/writing the on-disk
                feature store (used by live-sim and tests to avoid side effects).

        Returns:
            (feat_df, {set_name: member_cols}, cache_hits)
            feat_df = ohlcv columns + one column per resolved member feature.
        """
        requested: list[str] = []
        for s in set_names:
            for m in self.feature_cols(s):
                if m not in requested:
                    requested.append(m)

        order = self._topo_order(requested)
        frame = self._prepare_frame(ohlcv, order, market_df, sector_map)
        ctx = EvalContext.from_df(frame, sort=True)

        data_version = self.store.compute_data_version(
            symbols=ctx.df["symbol"].unique().tolist(),
            timeframe=timeframe,
            start=ctx.df["date"].min(),
            end=ctx.df["date"].max(),
            # Content fingerprint of the actual inputs (OHLCV + any market/sector
            # columns) so edited data never reuses stale cached features; engine
            # code fingerprint so editing an operator invalidates too.
            source_fingerprint=content_fingerprint(ctx.df),
            extra={
                "data_root": str(data_root) if data_root else None,
                "engine_code": engine_code_fingerprint(),
            },
        )

        cache_hits = 0
        for name in order:
            d = self.defs[name]
            cached = self.store.load(d.expr_hash, data_version) if cache else None
            if cached is not None:
                series = self._align_cached(cached, ctx)
                cache_hits += 1
            else:
                series = self.engine.eval(d.node, ctx)
                if isinstance(series, dict):
                    raise TypeError(f"Feature '{name}' resolved to a multi-output op without .attr")
                if cache:
                    self._store_feature(d, data_version, ctx, series)
            ctx.features[name] = series

        out = ctx.df.copy()
        for name in requested:
            out[name] = ctx.features[name].to_numpy()
        return out, {s: list(self.feature_cols(s)) for s in set_names}, cache_hits

    # -- internals --------------------------------------------------------

    def _topo_order(self, requested: list[str]) -> list[str]:
        """Closure of requested features in dependency order (deps first)."""
        visited: set[str] = set()
        temp: set[str] = set()
        order: list[str] = []

        def visit(n: str) -> None:
            if n in visited:
                return
            if n in temp:
                raise ValueError(f"Cyclic feature dependency at '{n}'")
            if n not in self.defs:
                raise KeyError(f"Unknown feature '{n}'")
            temp.add(n)
            for dep in self.defs[n].feature_deps:
                visit(dep)
            temp.discard(n)
            visited.add(n)
            order.append(n)

        for r in requested:
            visit(r)
        return order

    def _prepare_frame(
        self,
        ohlcv: pd.DataFrame,
        order: list[str],
        market_df: pd.DataFrame | None,
        sector_map: dict[str, str] | None,
    ) -> pd.DataFrame:
        base_cols = ["symbol", "date", *sorted(_OHLCV)]
        missing = [c for c in base_cols if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"ohlcv frame missing required columns {missing}")
        frame = ohlcv.loc[:, base_cols].copy()

        raw_deps = {r for n in order for r in self.defs[n].raw_deps}

        if "market_close" in raw_deps:
            if market_df is None or "market_close" not in self._market_columns(market_df):
                raise ValueError(
                    "Feature set needs $market_close but no market index was provided"
                )
            md = market_df.rename(columns={"close": "market_close"})
            frame = frame.merge(md[["date", "market_close"]], on="date", how="left")
            if frame["market_close"].isna().all():
                raise ValueError("market index does not overlap the OHLCV date range")

        if "sector" in raw_deps:
            if not sector_map:
                raise ValueError("Feature set needs $sector but no sector_map was provided")
            frame["sector"] = frame["symbol"].map(sector_map)
            if frame["sector"].isna().any():
                unknown = sorted(frame.loc[frame["sector"].isna(), "symbol"].unique())
                raise ValueError(f"sector_map missing symbols: {unknown}")

        unknown_raw = raw_deps - _OHLCV - {"market_close", "sector"}
        if unknown_raw:
            raise ValueError(f"Unknown raw fields referenced: {sorted(unknown_raw)}")
        return frame

    @staticmethod
    def _market_columns(market_df: pd.DataFrame) -> set[str]:
        cols = set(market_df.columns)
        if "close" in cols:
            cols.add("market_close")
        return cols

    @staticmethod
    def _align_cached(cached: pd.DataFrame, ctx: EvalContext) -> pd.Series:
        merged = ctx.df[["symbol", "date"]].merge(cached, on=["symbol", "date"], how="left")
        return pd.Series(merged["value"].to_numpy(), index=ctx.index)

    def _store_feature(
        self, d: FeatureDefinition, data_version: str, ctx: EvalContext, series: pd.Series
    ) -> None:
        frame = ctx.df[["symbol", "date"]].copy()
        frame["value"] = series.to_numpy()
        self.store.save(expr_hash=d.expr_hash, data_version=data_version, frame=frame)


def add_features(ohlcv: pd.DataFrame, set_name: str = "basic_v1", **kwargs) -> pd.DataFrame:
    """Compute one feature set onto an OHLCV frame (in-memory, no store side effects).

    Drop-in for the legacy per-set builders: returns ``ohlcv`` columns + the set's
    feature columns. For per-symbol sets (basic_v1/leading_v2) no market/sector data
    is required.
    """
    resolver = FeatureResolver.from_catalog()
    feat, _cols, _hits = resolver.resolve(ohlcv, [set_name], cache=False, **kwargs)
    return feat
