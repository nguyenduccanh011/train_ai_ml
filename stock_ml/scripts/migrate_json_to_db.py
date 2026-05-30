"""One-shot script: import results/leaderboard.json into PostgreSQL.

Usage:
    DB_ENABLED=true python -m stock_ml.scripts.migrate_json_to_db
    python -m stock_ml.scripts.migrate_json_to_db --results-dir ./results

Idempotent: safe to re-run (uses ON CONFLICT DO UPDATE).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 100


async def migrate(results_dir: Path) -> None:
    from stock_ml.db.engine import AsyncSessionLocal, async_engine
    from stock_ml.db.base import Base
    from stock_ml.db.repositories.run_repo import LeaderboardRunRepository
    from stock_ml.db.repositories.trade_repo import RunTradeRepository
    from stock_ml.src.leaderboard.schema import LeaderboardRow

    # Ensure tables exist (idempotent via checkfirst)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
    logger.info("Schema verified")

    leaderboard_file = results_dir / "leaderboard.json"
    if not leaderboard_file.exists():
        logger.error(f"Not found: {leaderboard_file}")
        sys.exit(1)

    raw = json.loads(leaderboard_file.read_text(encoding="utf-8"))

    # Support both list[LeaderboardRow] and {models: [...]} formats
    if isinstance(raw, list):
        rows_data = raw
    elif isinstance(raw, dict) and "models" in raw:
        rows_data = raw["models"]
    else:
        logger.error("Unexpected leaderboard.json format")
        sys.exit(1)

    logger.info(f"Found {len(rows_data)} runs to migrate")

    ok = 0
    errors = 0
    trade_total = 0

    for i in range(0, len(rows_data), BATCH_SIZE):
        batch = rows_data[i:i + BATCH_SIZE]
        async with AsyncSessionLocal() as session:
            run_repo = LeaderboardRunRepository(session)
            trade_repo = RunTradeRepository(session)

            for item in batch:
                try:
                    row = LeaderboardRow.model_validate(item)
                    await run_repo.upsert(row)

                    # Load trades CSV if artifact path is set
                    trades_rel = row.artifacts.trades_csv
                    if trades_rel:
                        trades_path = results_dir / trades_rel
                        if trades_path.exists():
                            import csv
                            trade_rows = []
                            with open(trades_path, newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                for t in reader:
                                    trade_rows.append({
                                        "symbol": t.get("symbol", ""),
                                        "entry_date": t.get("entry_date") or None,
                                        "entry_price": _float(t.get("entry_price")),
                                        "exit_date": t.get("exit_date") or None,
                                        "exit_price": _float(t.get("exit_price")),
                                        "holding_days": _float(t.get("holding_days")),
                                        "pnl_pct": float(t.get("pnl_pct", 0)),
                                        "exit_reason": t.get("exit_reason") or None,
                                        "entry_signal_date": t.get("entry_signal_date") or None,
                                    })
                            n = await trade_repo.bulk_insert(row.run_id, trade_rows)
                            trade_total += n

                    ok += 1
                except Exception as e:
                    logger.warning(f"Skipped {item.get('run_id', '?')}: {e}")
                    errors += 1

            await session.commit()
        logger.info(f"Progress: {min(i + BATCH_SIZE, len(rows_data))}/{len(rows_data)}")

    logger.info(f"Done: {ok} runs imported, {trade_total} trades inserted, {errors} errors")


def _float(v) -> float | None:
    try:
        return float(v) if v not in (None, "", "nan") else None
    except (ValueError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate leaderboard JSON to PostgreSQL")
    parser.add_argument("--results-dir", default="./results", help="Path to results/ folder")
    args = parser.parse_args()
    asyncio.run(migrate(Path(args.results_dir)))


if __name__ == "__main__":
    main()
