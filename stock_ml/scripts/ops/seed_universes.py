"""Seed universe management DB from existing YAML configs.

Populates universe_sets and universe_symbols tables from:
1. config/universes/*.yaml files
2. config/markets/*.yaml market profiles (creates {market}_default universe)

Usage:
    python stock_ml/scripts/seed_universes.py [--reset]

Flags:
    --reset: drop and recreate tables before seeding (for development)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    """Get project root."""
    return Path(__file__).resolve().parents[3]


async def seed_from_universes_yaml(session, repo) -> tuple[int, int]:
    """Seed from config/universes/*.yaml files.

    Returns:
        (created_count, skipped_count)
    """
    universes_dir = _repo_root() / "stock_ml" / "config" / "universes"
    if not universes_dir.exists():
        print(f"  [skip] universes_dir not found: {universes_dir}")
        return 0, 0

    created = 0
    skipped = 0

    for yaml_file in sorted(universes_dir.glob("*.yaml")):
        if yaml_file.name.startswith("."):
            continue

        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            slug = yaml_file.stem
            name = data.get("name", slug)
            description = data.get("description", "")
            symbols_list = data.get("symbols", [])

            if not isinstance(symbols_list, list):
                print(f"  [warn] {yaml_file.name}: symbols must be a list, skipping")
                skipped += 1
                continue

            universe = await repo.get_by_slug_all(slug)
            if universe is not None:
                print(f"  [skip] {slug}: universe already exists")
                skipped += 1
                continue

            universe = await repo.create(
                slug=slug,
                name=name,
                market="vn_stock",  # default market for YAML-based universes
                description=description,
                symbols=[{"symbol": s} for s in symbols_list],
            )
            await session.commit()
            print(f"  [create] {slug}: {len(symbols_list)} symbols")
            created += 1

        except Exception as e:
            await session.rollback()
            print(f"  [error] {yaml_file.name}: {e}")
            skipped += 1

    return created, skipped


async def seed_from_markets(session, repo) -> tuple[int, int]:
    """Seed from config/markets/*.yaml market profiles.

    Creates a '{market}_default' universe for each market's default_list.

    Returns:
        (created_count, skipped_count)
    """
    from stock_ml.src.market_profile import list_markets, load_market_profile

    created = 0
    skipped = 0

    for market in list_markets():
        try:
            profile = load_market_profile(market)
            default_symbols = profile.symbols.default_list

            if not default_symbols:
                print(f"  [skip] {market}: no default_list")
                skipped += 1
                continue

            slug = f"{market}_default"
            universe = await repo.get_by_slug_all(slug)
            if universe is not None:
                print(f"  [skip] {slug}: universe already exists")
                skipped += 1
                continue

            universe = await repo.create(
                slug=slug,
                name=f"{market} Default",
                market=market,
                description=f"Default symbol list for {market} market",
                symbols=[{"symbol": s} for s in default_symbols],
            )
            await session.commit()
            print(f"  [create] {slug}: {len(default_symbols)} symbols")
            created += 1

        except Exception as e:
            await session.rollback()
            print(f"  [error] {market}: {e}")
            skipped += 1

    return created, skipped


def main():
    """Seed the database."""
    parser = argparse.ArgumentParser(description="Seed universe management DB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate universe tables (for development)",
    )
    args = parser.parse_args()

    if args.reset:
        print("[reset] Dropping and recreating universe tables...")
        try:
            from stock_ml.db.base import Base
            from stock_ml.db.engine import sync_engine
            from stock_ml.db.models.universe import UniverseSetModel, UniverseSymbolModel

            Base.metadata.drop_all(
                bind=sync_engine, tables=[UniverseSymbolModel.__table__, UniverseSetModel.__table__]
            )
            Base.metadata.create_all(
                bind=sync_engine, tables=[UniverseSetModel.__table__, UniverseSymbolModel.__table__]
            )
            print("[reset] Done")
        except Exception as e:
            print(f"[error] Reset failed: {e}")
            return 1

    print("[seed] Starting universe seeding...")
    return asyncio.run(_seed_async())


async def _seed_async() -> int:
    from stock_ml.db.engine import AsyncSessionLocal
    from stock_ml.db.repositories.universe_repo import UniverseRepository

    async with AsyncSessionLocal() as session:
        try:
            repo = UniverseRepository(session)

            total_created = 0
            total_skipped = 0

            print("[seed] Loading from config/universes/*.yaml...")
            created, skipped = await seed_from_universes_yaml(session, repo)
            total_created += created
            total_skipped += skipped

            print("[seed] Loading from config/markets/*.yaml...")
            created, skipped = await seed_from_markets(session, repo)
            total_created += created
            total_skipped += skipped

            print(f"\n[seed] Complete: {total_created} created, {total_skipped} skipped")
            return 0

        except Exception as e:
            print(f"[error] Seeding failed: {e}")
            import traceback

            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
