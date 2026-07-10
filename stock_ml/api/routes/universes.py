"""Universe (symbol set) management routes."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.dependencies import get_db

router = APIRouter(prefix="/api/v1", tags=["universes"])


def _validate_slug(slug: str) -> str:
    """Validate slug format: alphanumeric, underscore, hyphen, max 128 chars.

    Args:
        slug: slug to validate

    Returns:
        normalized slug (lowercased)

    Raises:
        ValueError: if invalid format
    """
    slug = str(slug).strip().lower()
    if not slug or len(slug) > 128:
        raise ValueError("slug must be 1-128 characters")
    if not re.match(r"^[a-z0-9_-]+$", slug):
        raise ValueError("slug must contain only alphanumeric, underscore, hyphen")
    return slug


def _list_markets() -> list[str]:
    """List available market profile names."""
    from stock_ml.src.market_profile import list_markets

    return list_markets()


def _list_symbols_in_market(market: str) -> list[str]:
    """List symbols available in dataset for a market."""

    from stock_ml.src.data.loader import get_loader
    from stock_ml.src.market_profile import load_market_profile
    from stock_ml.src.utils.env import resolve_data_dir

    profile = load_market_profile(market)
    data_dir = profile.data.data_dir
    if not data_dir:
        return []

    abs_data_dir = resolve_data_dir(data_dir)
    try:
        loader = get_loader(str(abs_data_dir))
        return sorted(loader.list_symbols())
    except Exception:
        return []


@router.get("/universes")
async def list_universes(
    market: str = "", active_only: bool = True, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """List all universes, optionally filtered by market.

    Args:
        market: filter by market name (e.g., "vn_stock"), empty = all markets
        active_only: if True, exclude soft-deleted universes

    Returns:
        {universes: [{id, slug, name, market, symbol_count, version, is_locked, is_active, created_at}]}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        universes = await repo.list_all(market=market if market else None, active_only=active_only)
        return {
            "universes": [
                {
                    "id": u.id,
                    "slug": u.slug,
                    "name": u.name,
                    "description": u.description,
                    "market": u.market,
                    "symbol_count": u.symbol_count,
                    "version": u.version,
                    "is_locked": u.is_locked,
                    "is_active": u.is_active,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "updated_at": u.updated_at.isoformat() if u.updated_at else None,
                }
                for u in universes
            ]
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to list universes: {e}") from e


@router.post("/universes")
async def create_universe(
    payload: dict = Body(...), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Create a new universe.

    Payload:
    {
        "slug": "main_50",
        "name": "Main 50 Symbols",
        "market": "vn_stock",
        "description": "Top 50 liquid symbols",
        "notes": "For daily backtesting",
        "symbols": [{"symbol": "ACB", "group": "bank"}, ...]
    }

    Returns:
        {id, slug, name, market, symbol_count, version, created_at}
    """
    try:
        slug = payload.get("slug", "")
        name = payload.get("name", "")
        market = payload.get("market", "")
        description = payload.get("description")
        notes = payload.get("notes")
        symbols = payload.get("symbols", [])

        if not slug or not name or not market:
            raise HTTPException(400, detail="Missing required fields: slug, name, market")

        slug = _validate_slug(slug)

        available_markets = _list_markets()
        if market not in available_markets:
            raise HTTPException(
                400, detail=f"Unknown market: {market}. Available: {available_markets}"
            )

        try:
            from stock_ml.db.repositories.universe_repo import UniverseRepository

            repo = UniverseRepository(session)
            universe = await repo.create(
                slug=slug,
                name=name,
                market=market,
                description=description,
                notes=notes,
                symbols=symbols if symbols else None,
            )
            await session.commit()
            return {
                "id": universe.id,
                "slug": universe.slug,
                "name": universe.name,
                "market": universe.market,
                "symbol_count": universe.symbol_count,
                "version": universe.version,
                "created_at": universe.created_at.isoformat() if universe.created_at else None,
            }
        except ValueError as ve:
            await session.rollback()
            raise HTTPException(409, detail=str(ve)) from ve
        except Exception as e:
            await session.rollback()
            raise HTTPException(500, detail=f"Failed to create universe: {e}") from e
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(500, detail=f"Unexpected error: {e}") from e


@router.get("/universes/{slug}")
async def get_universe(slug: str, session: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Get universe details and symbols.

    Returns:
        {id, slug, name, description, market, symbol_count, version, is_locked, is_active, symbols: [{symbol, group?, weight?, rank?}], created_at}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        universe = await repo.get_by_slug(slug)
        if universe is None:
            raise HTTPException(404, detail=f"Universe not found: {slug}")

        symbols = await repo.get_symbols(universe.id)
        return {
            "id": universe.id,
            "slug": universe.slug,
            "name": universe.name,
            "description": universe.description,
            "market": universe.market,
            "symbol_count": universe.symbol_count,
            "version": universe.version,
            "is_locked": universe.is_locked,
            "is_active": universe.is_active,
            "notes": universe.notes,
            "symbols": [
                {
                    "symbol": s.symbol,
                    "group": s.symbol_group,
                    "weight": s.weight,
                    "rank": s.rank,
                    "notes": s.notes,
                }
                for s in symbols
            ],
            "created_at": universe.created_at.isoformat() if universe.created_at else None,
            "updated_at": universe.updated_at.isoformat() if universe.updated_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get universe: {e}") from e


@router.put("/universes/{slug}")
async def update_universe(
    slug: str, payload: dict = Body(...), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Update universe metadata (name, description, notes, is_locked).

    Payload:
    {
        "name": "New name",
        "description": "New description",
        "notes": "New notes",
        "is_locked": true
    }

    Returns:
        {id, slug, name, market, version, is_locked, updated_at}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        universe = await repo.update_meta(
            slug=slug,
            name=payload.get("name"),
            description=payload.get("description"),
            notes=payload.get("notes"),
            is_locked=payload.get("is_locked"),
        )
        if universe is None:
            raise HTTPException(404, detail=f"Universe not found: {slug}")

        await session.commit()
        # updated_at uses a server-side onupdate; it is expired after flush, so
        # refresh in async context before reading to avoid a lazy-load (greenlet) error.
        await session.refresh(universe)
        return {
            "id": universe.id,
            "slug": universe.slug,
            "name": universe.name,
            "market": universe.market,
            "version": universe.version,
            "is_locked": universe.is_locked,
            "updated_at": universe.updated_at.isoformat() if universe.updated_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to update universe: {e}") from e


@router.delete("/universes/{slug}")
async def delete_universe(slug: str, session: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Soft-delete a universe (set is_active=False).

    Fails if universe is locked.

    Returns:
        {deleted: slug}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        success = await repo.soft_delete(slug)
        if not success:
            universe = await repo.get_by_slug_all(slug)
            if universe is None:
                raise HTTPException(404, detail=f"Universe not found: {slug}")
            if universe.is_locked:
                raise HTTPException(400, detail=f"Universe is locked: {slug}")
            raise HTTPException(500, detail=f"Failed to delete universe: {slug}")

        await session.commit()
        return {"deleted": slug}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to delete universe: {e}") from e


@router.get("/universes/{slug}/symbols")
async def get_universe_symbols(
    slug: str, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Get symbols in a universe.

    Returns:
        {slug, symbol_count, version, symbols: [{symbol, group?, weight?, rank?}]}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        universe = await repo.get_by_slug(slug)
        if universe is None:
            raise HTTPException(404, detail=f"Universe not found: {slug}")

        symbols = await repo.get_symbols(universe.id)
        return {
            "slug": universe.slug,
            "symbol_count": universe.symbol_count,
            "version": universe.version,
            "symbols": [
                {
                    "symbol": s.symbol,
                    "group": s.symbol_group,
                    "weight": s.weight,
                    "rank": s.rank,
                }
                for s in symbols
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to get symbols: {e}") from e


@router.post("/universes/{slug}/symbols")
async def add_symbols(
    slug: str, payload: dict = Body(...), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Add symbols to universe.

    Payload:
    {
        "symbols": [{"symbol": "ACB", "group": "bank", "weight": null, "rank": 1}, ...]
    }

    Returns:
        {slug, symbol_count, version, added_count}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        universe = await repo.get_by_slug_all(slug)
        if universe is None:
            raise HTTPException(404, detail=f"Universe not found: {slug}")

        symbols = payload.get("symbols", [])
        if not symbols:
            raise HTTPException(400, detail="Missing 'symbols' in payload")

        added_count = await repo.add_symbols(universe.id, symbols)
        await session.commit()
        return {
            "slug": universe.slug,
            # add_symbols already bumped universe.symbol_count by added_count;
            # adding it again here double-counts.
            "symbol_count": universe.symbol_count,
            "version": universe.version,
            "added_count": added_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to add symbols: {e}") from e


@router.delete("/universes/{slug}/symbols/{symbol}")
async def remove_symbol(
    slug: str, symbol: str, session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Remove a symbol from universe.

    Args:
        slug: universe slug
        symbol: symbol to remove (case-insensitive)

    Returns:
        {slug, symbol_count, version, removed: symbol}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        universe = await repo.get_by_slug_all(slug)
        if universe is None:
            raise HTTPException(404, detail=f"Universe not found: {slug}")

        success = await repo.remove_symbol(universe.id, symbol)
        if not success:
            raise HTTPException(404, detail=f"Symbol not found in universe: {symbol}")

        await session.commit()
        return {
            "slug": universe.slug,
            "symbol_count": universe.symbol_count,
            "version": universe.version,
            "removed": symbol.upper(),
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to remove symbol: {e}") from e


@router.put("/universes/{slug}/symbols")
async def replace_symbols(
    slug: str, payload: dict = Body(...), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Replace all symbols in universe.

    Payload:
    {
        "symbols": [{"symbol": "ACB", "group": "bank"}, ...]
    }

    Returns:
        {slug, symbol_count, version}
    """
    try:
        from stock_ml.db.repositories.universe_repo import UniverseRepository

        repo = UniverseRepository(session)
        universe = await repo.get_by_slug_all(slug)
        if universe is None:
            raise HTTPException(404, detail=f"Universe not found: {slug}")

        symbols = payload.get("symbols", [])
        if not isinstance(symbols, list):
            raise HTTPException(400, detail="'symbols' must be a list")

        count = await repo.replace_symbols(universe.id, symbols)
        await session.commit()
        return {
            "slug": universe.slug,
            "symbol_count": count,
            "version": universe.version,
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(500, detail=f"Failed to replace symbols: {e}") from e


@router.post("/universes/{slug}/clone")
async def clone_universe(
    slug: str, payload: dict = Body(...), session: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """Clone a universe (copy symbols, new slug).

    Payload:
    {
        "new_slug": "main_50_backup",
        "new_name": "Main 50 Backup",
        "new_market": "vn_stock"  # optional, defaults to source market
    }

    Returns:
        {id, slug, name, market, symbol_count, version}
    """
    try:
        new_slug = payload.get("new_slug", "")
        new_name = payload.get("new_name", "")

        if not new_slug or not new_name:
            raise HTTPException(400, detail="Missing required fields: new_slug, new_name")

        new_slug = _validate_slug(new_slug)

        try:
            from stock_ml.db.repositories.universe_repo import UniverseRepository

            repo = UniverseRepository(session)
            universe = await repo.clone(
                source_slug=slug,
                new_slug=new_slug,
                new_name=new_name,
                new_market=payload.get("new_market"),
            )
            if universe is None:
                raise HTTPException(404, detail=f"Source universe not found: {slug}")

            await session.commit()
            return {
                "id": universe.id,
                "slug": universe.slug,
                "name": universe.name,
                "market": universe.market,
                "symbol_count": universe.symbol_count,
                "version": universe.version,
            }
        except ValueError as ve:
            await session.rollback()
            raise HTTPException(409, detail=str(ve)) from ve
        except Exception as e:
            await session.rollback()
            raise HTTPException(500, detail=f"Failed to clone universe: {e}") from e
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(500, detail=f"Unexpected error: {e}") from e


@router.get("/markets")
def list_markets_endpoint() -> dict[str, Any]:
    """List available market profiles.

    Returns:
        {markets: [market_name, ...]}
    """
    try:
        markets = _list_markets()
        return {"markets": markets}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to list markets: {e}") from e


@router.get("/markets/{market}/symbols")
def list_market_symbols(market: str) -> dict[str, Any]:
    """List symbols available in dataset for a market.

    Returns:
        {market, symbol_count, symbols: [symbol, ...]}
    """
    try:
        available_markets = _list_markets()
        if market not in available_markets:
            raise HTTPException(
                404, detail=f"Unknown market: {market}. Available: {available_markets}"
            )

        symbols = _list_symbols_in_market(market)
        return {
            "market": market,
            "symbol_count": len(symbols),
            "symbols": symbols,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to list symbols: {e}") from e
