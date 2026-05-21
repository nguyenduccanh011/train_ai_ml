"""Ingest Sieu Tin Hieu OHLCV data into Train61 PostgreSQL tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.providers.sieutinhieu import SieuTinHieuProvider  # noqa: E402
from tools.env import load_dotenv  # noqa: E402

DEFAULT_DATABASE_URL = "postgresql://train61:train61_local_password@localhost:15432/train61"
DEFAULT_SYMBOLS = ROOT / "config" / "train61_symbols.json"


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    database_url = args.database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    provider = SieuTinHieuProvider(base_url=args.api_base, timeout=args.timeout)
    symbols = load_symbols(args.symbols)
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        run_id = create_ingestion_run(conn, args.mode, args.timeframe)
        totals = {"symbols": 0, "inserted": 0, "updated": 0, "errors": 0, "invalid": 0}
        try:
            for symbol in symbols:
                try:
                    inserted, updated, invalid = ingest_symbol(conn, provider, symbol, args, run_id)
                    recompute_data_version(conn, symbol, args.timeframe, provider.provider_name)
                    totals["symbols"] += 1
                    totals["inserted"] += inserted
                    totals["updated"] += updated
                    totals["invalid"] += invalid
                    print(
                        f"{symbol}: inserted={inserted} updated={updated} invalid_skipped={invalid}"
                    )
                except Exception as exc:  # Keep the batch moving and log per-symbol failures.
                    totals["errors"] += 1
                    log_quality(
                        conn,
                        run_id,
                        symbol,
                        args.timeframe,
                        None,
                        "ingest_symbol_error",
                        "error",
                        str(exc),
                        {"symbol": symbol},
                    )
                    print(f"{symbol}: ERROR {exc}", file=sys.stderr)
            finish_ingestion_run(
                conn, run_id, "success" if totals["errors"] == 0 else "partial", totals
            )
        except Exception as exc:
            finish_ingestion_run(conn, run_id, "failed", totals, error=str(exc))
            raise

    print(
        "done "
        f"symbols={totals['symbols']} inserted={totals['inserted']} "
        f"updated={totals['updated']} invalid_skipped={totals['invalid']} errors={totals['errors']}"
    )
    return 0 if totals["errors"] == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["latest", "backfill"], default="latest")
    parser.add_argument("--timeframe", default=os.getenv("DEFAULT_TIMEFRAME", "1D"))
    parser.add_argument(
        "--symbols", default=str(DEFAULT_SYMBOLS), help="JSON/TXT file or comma-separated symbols"
    )
    parser.add_argument("--latest-limit", type=int, default=10)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--database-url")
    parser.add_argument("--api-base", default=os.getenv("SIEUTINHIEU_API_BASE"))
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def load_symbols(value: str) -> list[str]:
    path = Path(value)
    if path.exists():
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw = payload.get("symbols", payload) if isinstance(payload, dict) else payload
        else:
            raw = path.read_text(encoding="utf-8").splitlines()
    else:
        raw = value.split(",")
    symbols = sorted({str(symbol).strip().upper() for symbol in raw if str(symbol).strip()})
    if not symbols:
        raise ValueError(f"No symbols found from {value}")
    return symbols


def ingest_symbol(
    conn: psycopg.Connection,
    provider: SieuTinHieuProvider,
    symbol: str,
    args: argparse.Namespace,
    run_id: int,
) -> tuple[int, int, int]:
    if args.mode == "latest":
        items = provider.fetch_latest(symbol, args.timeframe, args.latest_limit)
    else:
        items = list(fetch_backfill_pages(provider, symbol, args))

    inserted = 0
    updated = 0
    invalid = 0
    for item in items:
        bar = provider.normalize_bar(symbol, item)
        validation_error = validate_bar(bar)
        if validation_error:
            invalid += 1
            log_quality(
                conn,
                run_id,
                bar.symbol,
                bar.timeframe,
                bar.timestamp,
                "invalid_ohlcv",
                "warning",
                validation_error,
                asdict(bar),
            )
            continue
        was_inserted = upsert_market_bar(conn, bar)
        if was_inserted:
            inserted += 1
        else:
            updated += 1
    return inserted, updated, invalid


def fetch_backfill_pages(
    provider: SieuTinHieuProvider,
    symbol: str,
    args: argparse.Namespace,
) -> Iterable[dict]:
    offset = 0
    while True:
        payload = provider.fetch_ohlcv(
            symbol=symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.page_size,
            offset=offset,
        )
        items = extract_items(payload)
        if not items:
            break
        yield from items
        if len(items) < args.page_size:
            break
        offset += args.page_size


def extract_items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("value", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Provider payload does not contain a list of bars")


def validate_bar(bar) -> str | None:
    if not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
        return f"Invalid OHLC for {bar.symbol} {bar.timestamp}"
    if bar.volume < 0:
        return f"Negative volume for {bar.symbol} {bar.timestamp}: {bar.volume}"
    return None


def upsert_market_bar(conn: psycopg.Connection, bar) -> bool:
    row = conn.execute(
        """
        insert into market_bars (
            symbol, symbol_id, timeframe, timestamp, open, high, low, close, volume,
            traded_value, provider, provider_bar_id, provider_created_at, updated_at
        )
        values (
            %(symbol)s, %(symbol_id)s, %(timeframe)s, %(timestamp)s, %(open)s, %(high)s,
            %(low)s, %(close)s, %(volume)s, %(traded_value)s, %(provider)s,
            %(provider_bar_id)s, %(provider_created_at)s, now()
        )
        on conflict (symbol, timeframe, timestamp, provider) do update set
            symbol_id = excluded.symbol_id,
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            traded_value = excluded.traded_value,
            provider_bar_id = excluded.provider_bar_id,
            provider_created_at = excluded.provider_created_at,
            updated_at = now()
        returning (xmax = 0) as inserted
        """,
        asdict(bar),
    ).fetchone()
    return bool(row["inserted"])


def recompute_data_version(
    conn: psycopg.Connection,
    symbol: str,
    timeframe: str,
    provider: str,
) -> None:
    row = conn.execute(
        """
        select timestamp, close, count(*) over () as row_count
        from market_bars
        where symbol = %s and timeframe = %s and provider = %s
        order by timestamp desc
        limit 1
        """,
        (symbol, timeframe, provider),
    ).fetchone()
    if row is None:
        return
    version_hash = build_version_hash(
        symbol,
        timeframe,
        provider,
        row["timestamp"].isoformat(),
        row["close"],
        row["row_count"],
    )
    conn.execute(
        """
        insert into data_versions (
            symbol, timeframe, provider, latest_timestamp, latest_close,
            row_count, version_hash, updated_at
        )
        values (%s, %s, %s, %s, %s, %s, %s, now())
        on conflict (symbol, timeframe, provider) do update set
            latest_timestamp = excluded.latest_timestamp,
            latest_close = excluded.latest_close,
            row_count = excluded.row_count,
            version_hash = excluded.version_hash,
            updated_at = now()
        """,
        (
            symbol,
            timeframe,
            provider,
            row["timestamp"],
            row["close"],
            row["row_count"],
            version_hash,
        ),
    )


def build_version_hash(
    symbol: str,
    timeframe: str,
    provider: str,
    latest_timestamp: str,
    latest_close: Decimal,
    row_count: int,
) -> str:
    raw = f"{symbol}|{timeframe}|{provider}|{latest_timestamp}|{latest_close}|{row_count}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def create_ingestion_run(conn: psycopg.Connection, mode: str, timeframe: str) -> int:
    row = conn.execute(
        """
        insert into ingestion_runs (provider, timeframe, mode)
        values ('sieutinhieu', %s, %s)
        returning id
        """,
        (timeframe, mode),
    ).fetchone()
    return int(row["id"])


def finish_ingestion_run(
    conn: psycopg.Connection,
    run_id: int,
    status: str,
    totals: dict[str, int],
    error: str | None = None,
) -> None:
    conn.execute(
        """
        update ingestion_runs
        set finished_at = now(),
            status = %s,
            symbol_count = %s,
            inserted_count = %s,
            updated_count = %s,
            error_count = %s,
            error = %s
        where id = %s
        """,
        (
            status,
            totals["symbols"],
            totals["inserted"],
            totals["updated"],
            totals["errors"],
            error,
            run_id,
        ),
    )


def log_quality(
    conn: psycopg.Connection,
    ingestion_run_id: int,
    symbol: str,
    timeframe: str,
    timestamp: str | None,
    check_type: str,
    severity: str,
    message: str,
    payload: dict,
) -> None:
    conn.execute(
        """
        insert into data_quality_checks (
            ingestion_run_id, symbol, timeframe, timestamp, check_type,
            severity, message, payload
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            ingestion_run_id,
            symbol,
            timeframe,
            timestamp,
            check_type,
            severity,
            message,
            json.dumps(payload, default=str),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
