"""Universe (symbol set) CRUD repository."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import and_, delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.models.universe import UniverseSetModel, UniverseSymbolModel
from stock_ml.db.models.universe_version import UniverseVersionModel


class UniverseRepository:
    """CRUD operations for universe sets and their symbols."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def _save_version_snapshot(self, universe: UniverseSetModel) -> None:
        """Save snapshot of universe symbols at current version.

        Args:
            universe: UniverseSetModel instance (must be already flushed)
        """
        symbols = await self.get_symbols(universe.id)
        snapshot = UniverseVersionModel(
            universe_id=universe.id,
            version=universe.version,
            symbols_json=json.dumps(
                [
                    {
                        "symbol": s.symbol,
                        "group": s.symbol_group,
                        "rank": s.rank,
                        "weight": s.weight,
                    }
                    for s in symbols
                ]
            ),
            symbol_count=universe.symbol_count,
            # universe_versions.created_at is a naive TIMESTAMP column; asyncpg
            # rejects a tz-aware value for it, so store naive UTC.
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        self._session.add(snapshot)

    async def list_all(
        self, market: str | None = None, active_only: bool = True
    ) -> list[UniverseSetModel]:
        """List all universes, optionally filtered by market and active status.

        Args:
            market: filter by market (e.g., "vn_stock"), None = all markets
            active_only: if True, exclude soft-deleted (is_active=False)

        Returns:
            list of UniverseSetModel ordered by created_at DESC
        """
        stmt = select(UniverseSetModel)

        if active_only:
            stmt = stmt.where(UniverseSetModel.is_active)

        if market:
            stmt = stmt.where(UniverseSetModel.market == market)

        stmt = stmt.order_by(desc(UniverseSetModel.created_at))
        return (await self._session.execute(stmt)).scalars().all()

    async def get_by_slug(self, slug: str) -> UniverseSetModel | None:
        """Get universe by slug (case-sensitive).

        Args:
            slug: universe slug

        Returns:
            UniverseSetModel or None if not found or soft-deleted
        """
        stmt = select(UniverseSetModel).where(
            and_(UniverseSetModel.slug == slug, UniverseSetModel.is_active)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_slug_all(self, slug: str) -> UniverseSetModel | None:
        """Get universe by slug, including soft-deleted.

        Args:
            slug: universe slug

        Returns:
            UniverseSetModel or None if not found
        """
        stmt = select(UniverseSetModel).where(UniverseSetModel.slug == slug)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_symbols(self, universe_id: int) -> list[UniverseSymbolModel]:
        """Get all symbols in a universe, ordered by rank then symbol.

        Args:
            universe_id: universe PK

        Returns:
            list of UniverseSymbolModel
        """
        stmt = (
            select(UniverseSymbolModel)
            .where(UniverseSymbolModel.universe_id == universe_id)
            .order_by(UniverseSymbolModel.rank, UniverseSymbolModel.symbol)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def create(
        self,
        slug: str,
        name: str,
        market: str,
        description: str | None = None,
        notes: str | None = None,
        symbols: list[dict] | None = None,
    ) -> UniverseSetModel:
        """Create new universe.

        Args:
            slug: URL-safe identifier (alphanumeric, underscore, hyphen)
            name: display name
            market: market name (e.g., "vn_stock")
            description: optional description
            notes: optional notes
            symbols: optional list of dicts {symbol, group?, weight?, rank?}

        Returns:
            created UniverseSetModel

        Raises:
            ValueError: if slug already exists
        """
        existing = (
            await self._session.execute(
                select(UniverseSetModel).where(UniverseSetModel.slug == slug)
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise ValueError(f"Universe slug already exists: {slug}")

        universe = UniverseSetModel(
            slug=slug,
            name=name,
            market=market,
            description=description,
            notes=notes,
            symbol_count=0,
            version=1,
        )
        self._session.add(universe)
        await self._session.flush()

        if symbols:
            # add_symbols already bumps the version and writes the version
            # snapshot; calling _save_version_snapshot again here would duplicate
            # the (universe_id, version) row (unique-constraint violation on PG).
            await self.add_symbols(universe.id, symbols)
            await self._session.flush()

        return universe

    async def update_meta(
        self,
        slug: str,
        name: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        is_locked: bool | None = None,
    ) -> UniverseSetModel | None:
        """Update universe metadata (not symbols).

        Args:
            slug: universe slug
            name: new name
            description: new description
            notes: new notes
            is_locked: new lock status

        Returns:
            updated UniverseSetModel or None if not found
        """
        universe = await self.get_by_slug_all(slug)
        if universe is None:
            return None

        if name is not None:
            universe.name = name
        if description is not None:
            universe.description = description
        if notes is not None:
            universe.notes = notes
        if is_locked is not None:
            universe.is_locked = is_locked

        await self._session.flush()
        return universe

    async def soft_delete(self, slug: str) -> bool:
        """Soft-delete universe (set is_active=False).

        Args:
            slug: universe slug

        Returns:
            True if deleted, False if not found or is_locked
        """
        universe = await self.get_by_slug_all(slug)
        if universe is None or universe.is_locked:
            return False

        universe.is_active = False
        await self._session.flush()
        return True

    async def hard_delete(self, slug: str) -> bool:
        """Hard-delete universe (cascade deletes symbols).

        Args:
            slug: universe slug

        Returns:
            True if deleted, False if not found or is_locked

        Note:
            This cascades and deletes all related UniverseSymbolModel rows.
        """
        universe = await self.get_by_slug_all(slug)
        if universe is None or universe.is_locked:
            return False

        self._session.delete(universe)
        await self._session.flush()
        return True

    async def add_symbols(self, universe_id: int, symbols: list[dict]) -> int:
        """Add symbols to universe (deduped).

        Args:
            universe_id: universe PK
            symbols: list of dicts {symbol, group?, weight?, rank?, notes?}

        Returns:
            count of added symbols

        Note:
            Ignores duplicates (symbol already in universe).
            Increments version and updates symbol_count.
        """
        universe = (
            await self._session.execute(
                select(UniverseSetModel).where(UniverseSetModel.id == universe_id)
            )
        ).scalar_one_or_none()
        if universe is None:
            return 0

        # select(...symbol) + .scalars() yields the symbol strings directly.
        existing_symbols = set(
            (
                await self._session.execute(
                    select(UniverseSymbolModel.symbol).where(
                        UniverseSymbolModel.universe_id == universe_id
                    )
                )
            ).scalars()
        )

        added_count = 0
        for item in symbols:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol or symbol in existing_symbols:
                continue

            uni_sym = UniverseSymbolModel(
                universe_id=universe_id,
                symbol=symbol,
                symbol_group=item.get("group"),
                weight=item.get("weight"),
                rank=item.get("rank"),
                notes=item.get("notes"),
            )
            self._session.add(uni_sym)
            existing_symbols.add(symbol)
            added_count += 1

        if added_count > 0:
            universe.symbol_count += added_count
            universe.version += 1
            await self._session.flush()
            await self._save_version_snapshot(universe)
            await self._session.flush()

        return added_count

    async def remove_symbol(self, universe_id: int, symbol: str) -> bool:
        """Remove a symbol from universe.

        Args:
            universe_id: universe PK
            symbol: symbol to remove (case-insensitive, uppercased)

        Returns:
            True if removed, False if not found

        Note:
            Decrements symbol_count and increments version.
        """
        symbol = symbol.upper()
        stmt = delete(UniverseSymbolModel).where(
            and_(
                UniverseSymbolModel.universe_id == universe_id,
                UniverseSymbolModel.symbol == symbol,
            )
        )
        result = await self._session.execute(stmt)

        if result.rowcount > 0:
            universe = (
                await self._session.execute(
                    select(UniverseSetModel).where(UniverseSetModel.id == universe_id)
                )
            ).scalar_one_or_none()
            if universe:
                universe.symbol_count = max(0, universe.symbol_count - 1)
                universe.version += 1
                await self._session.flush()
                await self._save_version_snapshot(universe)
                await self._session.flush()
            return True

        return False

    async def replace_symbols(self, universe_id: int, symbols: list[dict]) -> int:
        """Replace all symbols in universe.

        Args:
            universe_id: universe PK
            symbols: list of dicts {symbol, group?, weight?, rank?, notes?}

        Returns:
            count of symbols after replacement

        Note:
            Deletes all existing symbols and adds new ones.
            Increments version, updates symbol_count.
        """
        universe = (
            await self._session.execute(
                select(UniverseSetModel).where(UniverseSetModel.id == universe_id)
            )
        ).scalar_one_or_none()
        if universe is None:
            return 0

        await self._session.execute(
            delete(UniverseSymbolModel).where(UniverseSymbolModel.universe_id == universe_id)
        )

        added_count = 0
        seen = set()
        for item in symbols:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)

            uni_sym = UniverseSymbolModel(
                universe_id=universe_id,
                symbol=symbol,
                symbol_group=item.get("group"),
                weight=item.get("weight"),
                rank=item.get("rank"),
                notes=item.get("notes"),
            )
            self._session.add(uni_sym)
            added_count += 1

        universe.symbol_count = added_count
        universe.version += 1
        await self._session.flush()
        await self._save_version_snapshot(universe)
        await self._session.flush()

        return added_count

    async def get_version_snapshot(self, slug: str, version: int) -> list[dict] | None:
        """Get symbol list for a specific universe version.

        Args:
            slug: universe slug
            version: version number

        Returns:
            list of symbol dicts or None if version not found
        """
        universe = await self.get_by_slug_all(slug)
        if universe is None:
            return None

        snapshot = (
            await self._session.execute(
                select(UniverseVersionModel).where(
                    and_(
                        UniverseVersionModel.universe_id == universe.id,
                        UniverseVersionModel.version == version,
                    )
                )
            )
        ).scalar_one_or_none()

        if snapshot is None:
            return None

        return json.loads(snapshot.symbols_json)

    async def clone(
        self, source_slug: str, new_slug: str, new_name: str, new_market: str | None = None
    ) -> UniverseSetModel | None:
        """Clone a universe (copy symbols, increment version counter).

        Args:
            source_slug: source universe slug
            new_slug: new universe slug (must not exist)
            new_name: new universe name
            new_market: new market (defaults to source market)

        Returns:
            new UniverseSetModel or None if source not found

        Raises:
            ValueError: if new_slug already exists
        """
        source = await self.get_by_slug_all(source_slug)
        if source is None:
            return None

        symbols = await self.get_symbols(source.id)
        symbol_list = [
            {
                "symbol": s.symbol,
                "group": s.symbol_group,
                "weight": s.weight,
                "rank": s.rank,
                "notes": s.notes,
            }
            for s in symbols
        ]

        target_market = new_market or source.market
        return await self.create(
            slug=new_slug,
            name=new_name,
            market=target_market,
            description=source.description,
            notes=f"Cloned from {source_slug}",
            symbols=symbol_list,
        )
