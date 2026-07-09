"""Migrate CSV OHLCV data to DuckDB format.

Usage:
    python stock_ml/scripts/migrate_csv_to_duckdb.py \
      --source portable_data/vn_stock_ai_dataset \
      --out portable_data/vn_stock.duckdb
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm


def _find_symbol_dirs(source_dir: Path) -> list[Path]:
    """Locate the symbol=* directories.

    Supports both layouts:
      - <source>/all_symbols/symbol=<SYM>/...   (vn_stock_ai_dataset_cleaned)
      - <source>/symbol=<SYM>/...               (derivatives_ai_dataset)
    """
    base = source_dir / "all_symbols"
    if not base.exists():
        base = source_dir
    return sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("symbol="))


def _ingest_csv(conn, csv_path: Path, symbol: str, timeframe: str) -> bool:
    """Read one symbol/timeframe CSV and INSERT OR IGNORE into ohlcv.

    Cleaning mirrors src/data/loader.py exactly so the DuckDB path is row-for-row
    identical to the CSV path (parity requirement): tz-naive normalized date,
    dropna on [date, open, high, low, close], dedup on date keeping the last,
    sorted by date.
    """
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        print(f"  [warn] {csv_path}: missing 'timestamp' column, skipped")
        return False

    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()

    required = ["symbol", "timeframe", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in ["open", "high", "low", "close", "volume"] if c not in df.columns]
    if missing:
        print(f"  [warn] {csv_path}: missing columns {missing}, skipped")
        return False

    select_cols = required + (["traded_value"] if "traded_value" in df.columns else [])
    df = df[select_cols].copy()
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if "traded_value" not in df.columns:
        df["traded_value"] = None  # keep column order aligned with the table

    conn.register("df_in", df)
    conn.execute(
        "INSERT OR IGNORE INTO ohlcv "
        "SELECT symbol, timeframe, date, open, high, low, close, volume, traded_value FROM df_in"
    )
    conn.unregister("df_in")
    return True


def migrate_market(source_dir: Path, out_db: Path, timeframe: str | None = None) -> int:
    """Migrate all symbol CSVs from source_dir to DuckDB.

    Args:
        source_dir: dataset root (with or without an all_symbols/ wrapper)
        out_db: path to output .duckdb file
        timeframe: a single timeframe to migrate, or None to auto-discover every
            timeframe=* directory present per symbol

    Returns:
        count of (symbol, timeframe) CSVs migrated
    """
    conn = duckdb.connect(str(out_db))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol      VARCHAR NOT NULL,
            timeframe   VARCHAR NOT NULL DEFAULT '1D',
            date        DATE NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            traded_value DOUBLE,
            PRIMARY KEY (symbol, timeframe, date)
        )
    """)

    symbol_dirs = _find_symbol_dirs(source_dir)
    if not symbol_dirs:
        print(f"Error: no symbol=* directories under {source_dir}")
        conn.close()
        return 0

    migrated = 0
    for symbol_dir in tqdm(symbol_dirs, desc="Migrating symbols"):
        symbol = symbol_dir.name.split("=", 1)[1]

        if timeframe is not None:
            tf_dirs = [symbol_dir / f"timeframe={timeframe}"]
        else:
            tf_dirs = sorted(
                d for d in symbol_dir.iterdir() if d.is_dir() and d.name.startswith("timeframe=")
            )

        for tf_dir in tf_dirs:
            csv_path = tf_dir / "data.csv"
            if not csv_path.exists():
                continue
            tf = tf_dir.name.split("=", 1)[1]
            try:
                if _ingest_csv(conn, csv_path, symbol, tf):
                    migrated += 1
            except Exception as e:
                print(f"  [warn] Failed to migrate {symbol} [{tf}]: {e}")

    # Index for fast (symbol, timeframe, date) range scans.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf_date ON ohlcv(symbol, timeframe, date)"
    )
    conn.close()
    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate CSV OHLCV data to DuckDB")
    parser.add_argument(
        "--source", required=True, help="Source directory (e.g., portable_data/vn_stock_ai_dataset)"
    )
    parser.add_argument(
        "--out", required=True, help="Output DuckDB path (e.g., portable_data/vn_stock.duckdb)"
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Single timeframe to migrate (default: auto-discover all timeframe=* dirs)",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Backup existing DB before overwriting"
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    out_db = Path(args.out)

    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        return 1

    # Backup existing DB if requested
    if args.backup and out_db.exists():
        backup_path = out_db.with_suffix(out_db.suffix + ".backup")
        print(f"[backup] Copying {out_db} → {backup_path}")
        shutil.copy2(out_db, backup_path)

    print(f"[migrate] Starting migration from {source_dir} → {out_db}")
    print(f"  Timeframe: {args.timeframe}")

    migrated = migrate_market(source_dir, out_db, timeframe=args.timeframe)

    print(f"[done] Migrated {migrated} symbols")
    print(f"  Output: {out_db}")
    print(f"  Size: {out_db.stat().st_size / 1e9:.2f} GB")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
