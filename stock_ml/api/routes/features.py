"""Read-only API over the Feature Store + DSL catalog.

Feature definitions are authored in code — ``src/features/catalog.py`` is the
single source of truth the backtest resolver reads — and projected one-way into
the DB by ``scripts/seed_features.py``. These endpoints therefore only *read* that
mirror (plus a stateless ``/validate`` expression checker). Features are not
created through the API, so the DB can never diverge from the catalog backtests
actually run on.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.dependencies import get_db
from stock_ml.db.models.feature import (
    FeatureDefModel,
    FeatureMaterializationModel,
    FeatureSetMemberModel,
)
from stock_ml.db.repositories.feature_repo import FeatureDefRepository, FeatureSetRepository
from stock_ml.src.features.dsl.engine import extract_deps, infer_kind
from stock_ml.src.features.dsl.parser import DSLSyntaxError, parse

router = APIRouter(prefix="/api/v1/features", tags=["features"])


def _validate_expr(expr: str) -> dict:
    """Parse an expression and return {valid, kind, deps, error} for live UI checks."""
    try:
        node = parse(expr)
    except DSLSyntaxError as exc:
        return {"valid": False, "kind": None, "deps": None, "error": str(exc)}
    refs, raws = extract_deps(node)
    return {
        "valid": True,
        "kind": infer_kind(node),
        "deps": {"features": sorted(refs), "raw": sorted(raws)},
        "error": None,
    }


# --- Definitions -----------------------------------------------------------


@router.get("/definitions")
async def list_definitions(
    kind: str | None = None,
    search: str | None = None,
    session: AsyncSession = Depends(get_db),
):
    """List feature definitions (filter by kind, search name)."""
    repo = FeatureDefRepository(session)
    defs = await repo.list(kind=kind, search=search)
    usage = await repo.usage_counts()
    return [
        {
            "id": d.id,
            "name": d.name,
            "kind": d.kind,
            "version": d.version,
            "expr": d.expr,
            "usedByCount": usage.get(d.id, 0),
        }
        for d in defs
    ]


@router.get("/definitions/{feature_id}")
async def get_definition(feature_id: int, session: AsyncSession = Depends(get_db)):
    """Definition detail: expression, deps, sets using it, materialize status."""
    repo = FeatureDefRepository(session)
    d = await repo.get_by_id(feature_id)
    if not d:
        raise HTTPException(status_code=404, detail="Feature not found")

    refs, raws = extract_deps(parse(d.expr))
    materialized = (
        await session.execute(
            select(FeatureMaterializationModel.id).where(
                FeatureMaterializationModel.feature_id == feature_id
            )
        )
    ).first() is not None

    set_rows = (
        (
            await session.execute(
                select(FeatureSetMemberModel.feature_set_id).where(
                    FeatureSetMemberModel.feature_id == feature_id
                )
            )
        )
        .scalars()
        .all()
    )

    return {
        "id": d.id,
        "name": d.name,
        "expr": d.expr,
        "kind": d.kind,
        "version": d.version,
        "outputDtype": d.output_dtype,
        "description": d.description,
        "deps": {"features": sorted(refs), "raw": sorted(raws)},
        "setIds": list(set_rows),
        "materialized": materialized,
    }


@router.post("/validate")
async def validate_expr(body: dict = Body(...)):
    """Parse a single expression → {valid, kind, deps, error} for UI live-check."""
    expr = (body.get("expr") or "").strip()
    if not expr:
        return {"valid": False, "kind": None, "deps": None, "error": "empty expression"}
    return _validate_expr(expr)


# --- Sets ------------------------------------------------------------------


@router.get("/sets")
async def list_sets(session: AsyncSession = Depends(get_db)):
    """List feature sets with member counts."""
    repo = FeatureSetRepository(session)
    sets = await repo.list_all(is_active=True)
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "memberCount": len(s.members),
        }
        for s in sets
    ]


@router.get("/sets/{set_id}")
async def get_set(set_id: int, session: AsyncSession = Depends(get_db)):
    """Set detail with members ordered by position."""
    repo = FeatureSetRepository(session)
    s = await repo.get_by_id(set_id)
    if not s:
        raise HTTPException(status_code=404, detail="Feature set not found")
    members = sorted(s.members, key=lambda m: m.position)
    return {
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "members": [
            {"position": m.position, "id": m.feature.id, "name": m.feature.name, "kind": m.feature.kind}
            for m in members
        ],
    }


# --- Materializations ------------------------------------------------------


@router.get("/materializations")
async def list_materializations(session: AsyncSession = Depends(get_db)):
    """Feature store status: which features have been materialised."""
    rows = (
        await session.execute(
            select(FeatureMaterializationModel, FeatureDefModel.name)
            .join(FeatureDefModel, FeatureMaterializationModel.feature_id == FeatureDefModel.id)
            .order_by(FeatureMaterializationModel.computed_at.desc())
        )
    ).all()
    return [
        {
            "featureId": mat.feature_id,
            "featureName": name,
            "exprHash": mat.expr_hash,
            "dataVersion": mat.data_version,
            "rows": mat.rows,
            "engineVersion": mat.engine_version,
            "storageUri": mat.storage_uri,
            "computedAt": mat.computed_at.isoformat() if mat.computed_at else None,
        }
        for mat, name in rows
    ]


# --- Values (per-symbol time series, for verification) ---------------------


@router.get("/values")
async def feature_values(
    symbol: str,
    set_name: str = Query("leading_v2", alias="set"),
    features: str | None = None,
    market: str = "vn_stock",
    start: str | None = None,
    end: str | None = None,
    limit: int = 500,
):
    """Per-symbol feature time series for verification.

    Computes the feature set on the symbol's *full* OHLCV history (so rolling
    features are warm), then returns the requested date window. Single-symbol view
    supports per-symbol features only — cross-sectional / market features need the
    whole universe, so a set containing them is rejected with 400.
    """
    import pandas as pd

    from stock_ml.src.data.loader import get_loader
    from stock_ml.src.features.resolver import FeatureResolver
    from stock_ml.src.market_profile import load_market_profile
    from stock_ml.src.utils.env import resolve_data_dir

    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    resolver = FeatureResolver.from_catalog()
    if set_name not in resolver.set_members:
        raise HTTPException(status_code=404, detail=f"Unknown feature set '{set_name}'")

    members = resolver.feature_cols(set_name)
    non_per_symbol = [m for m in members if resolver.defs[m].kind != "per_symbol"]
    if non_per_symbol:
        raise HTTPException(
            status_code=400,
            detail=(
                f"set '{set_name}' has non per-symbol features {non_per_symbol[:5]} — a "
                "single-symbol view can't compute cross-sectional/market features (they need "
                "the whole universe). Use a per-symbol set like basic_v1 or leading_v2."
            ),
        )

    if features:
        cols = [f.strip() for f in features.split(",") if f.strip()]
        unknown = [c for c in cols if c not in members]
        if unknown:
            raise HTTPException(
                status_code=400, detail=f"features not in set '{set_name}': {unknown}"
            )
    else:
        cols = members

    profile = load_market_profile(market)
    data_dir = profile.data.data_dir
    if not data_dir:
        raise HTTPException(status_code=404, detail=f"market '{market}' has no data dir")
    loader = get_loader(str(resolve_data_dir(data_dir)))
    if symbol not in loader.list_symbols():
        raise HTTPException(status_code=404, detail=f"symbol '{symbol}' not in market '{market}'")

    raw = loader.load_many([symbol])
    ohlcv = raw[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()

    # Resolve on full history (warm), no on-disk store side effects.
    feat, _cols, _hits = resolver.resolve(ohlcv, [set_name], cache=False)
    feat["date"] = pd.to_datetime(feat["date"])
    if start:
        feat = feat[feat["date"] >= pd.Timestamp(start)]
    if end:
        feat = feat[feat["date"] <= pd.Timestamp(end)]
    feat = feat.sort_values("date").tail(max(1, min(int(limit), 5000)))

    def _num(v):
        return None if pd.isna(v) else round(float(v), 6)

    rows = [
        {
            "date": pd.Timestamp(rec["date"]).strftime("%Y-%m-%d"),
            "close": _num(rec.get("close")),
            **{c: _num(rec.get(c)) for c in cols},
        }
        for rec in feat.to_dict("records")
    ]
    return {
        "symbol": symbol,
        "set": set_name,
        "market": market,
        "count": len(rows),
        "columns": cols,
        "rows": rows,
    }
