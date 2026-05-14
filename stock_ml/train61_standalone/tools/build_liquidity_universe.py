from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "vn_stock_ai_dataset" / "all_symbols"
DEFAULT_OUTPUT = ROOT / "config" / "liquidity_1b_symbols.json"
DEFAULT_EXCLUDED_SYMBOLS = {"VNINDEX", "HNXINDEX", "HNXUPCOM"}
DEFAULT_EXCLUDED_PREFIXES = ("VN30F",)


def _symbol_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("symbol="):
            return part.split("=", 1)[1].upper()
    raise ValueError(f"Cannot infer symbol from {path}")


def _resolve_common_date(data_dir: Path, common_date: str) -> tuple[str, str]:
    if common_date.lower() != "auto":
        return common_date, "cli"

    coverage_path = data_dir / "metadata" / "coverage.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path, usecols=["last_timestamp"])
        timestamps = pd.to_datetime(coverage["last_timestamp"], utc=True, errors="coerce").dropna()
        if not timestamps.empty:
            return timestamps.max().date().isoformat(), str(coverage_path)

    latest: pd.Timestamp | None = None
    for path in sorted(data_dir.glob("symbol=*/timeframe=1D/data.csv")):
        df = pd.read_csv(path, usecols=["timestamp"])
        if df.empty:
            continue
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna().max()
        if pd.notna(ts) and (latest is None or ts > latest):
            latest = ts
    if latest is None:
        raise ValueError(f"Cannot resolve common date from {data_dir}")
    return latest.date().isoformat(), "scanned_data_files"


def _load_common_date_row(path: Path, common_date: pd.Timestamp) -> dict[str, Any] | None:
    df = pd.read_csv(
        path,
        usecols=lambda col: col in {"timestamp", "exchange", "asset_type", "traded_value"},
    )
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["traded_value"] = pd.to_numeric(df["traded_value"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return None

    latest_ts = df.iloc[-1]["timestamp"]
    common_rows = df[df["timestamp"].dt.date == common_date.date()]
    common_traded_value = (
        float(common_rows.iloc[-1]["traded_value"]) if not common_rows.empty else None
    )
    common_row = common_rows.iloc[-1] if not common_rows.empty else None
    return {
        "latest_date": latest_ts.date().isoformat(),
        "has_common_date": common_traded_value is not None,
        "common_traded_value": common_traded_value,
        "asset_type": str(common_row["asset_type"]).lower()
        if common_row is not None and "asset_type" in common_row
        else None,
        "exchange": str(common_row["exchange"]).upper()
        if common_row is not None and "exchange" in common_row
        else None,
    }


def _is_stock_symbol(
    symbol: str,
    row: dict[str, Any],
    stock_only: bool,
    excluded_symbols: set[str],
    excluded_prefixes: tuple[str, ...],
) -> bool:
    if symbol in excluded_symbols:
        return False
    if any(symbol.startswith(prefix) for prefix in excluded_prefixes):
        return False
    return not (stock_only and row.get("asset_type") not in {None, "stock"})


def build_universe(
    data_dir: Path,
    common_date: str,
    min_traded_value_vnd: float,
    stock_only: bool = True,
    excluded_symbols: set[str] | None = None,
    excluded_prefixes: tuple[str, ...] = DEFAULT_EXCLUDED_PREFIXES,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    resolved_common_date, common_date_source = _resolve_common_date(data_dir, common_date)
    common_ts = pd.Timestamp(resolved_common_date, tz="UTC")
    # CSV prices are in thousand VND, so traded_value is also in thousand VND.
    min_traded_value_column = min_traded_value_vnd / 1000.0
    excluded_symbols = excluded_symbols or set()

    eligible: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("symbol=*/timeframe=1D/data.csv")):
        symbol = _symbol_from_path(path)
        row = _load_common_date_row(path, common_ts)
        if row is None:
            diagnostics.append({"symbol": symbol, "status": "empty_or_invalid"})
            continue

        latest_ts = pd.Timestamp(row["latest_date"], tz="UTC")
        date_ok = latest_ts >= common_ts
        traded_value = row["common_traded_value"]
        liquidity_ok = traded_value is not None and float(traded_value) >= min_traded_value_column
        stock_ok = _is_stock_symbol(
            symbol,
            row,
            stock_only=stock_only,
            excluded_symbols=excluded_symbols,
            excluded_prefixes=excluded_prefixes,
        )
        status = (
            "eligible"
            if date_ok and row["has_common_date"] and liquidity_ok and stock_ok
            else "excluded"
        )
        if status == "eligible":
            eligible.append(symbol)
        diagnostics.append(
            {
                "symbol": symbol,
                "status": status,
                "latest_date": row["latest_date"],
                "has_common_date": row["has_common_date"],
                "asset_type": row.get("asset_type"),
                "exchange": row.get("exchange"),
                "traded_value_column_on_common_date": traded_value,
                "date_ok": date_ok,
                "liquidity_ok": liquidity_ok,
                "stock_ok": stock_ok,
            }
        )

    meta = {
        "common_date": resolved_common_date,
        "common_date_source": common_date_source,
        "min_traded_value_vnd": min_traded_value_vnd,
        "min_traded_value_column": min_traded_value_column,
        "dataset_dir": str(data_dir),
        "stock_only": stock_only,
        "excluded_symbols": sorted(excluded_symbols),
        "excluded_prefixes": list(excluded_prefixes),
        "eligible_count": len(eligible),
    }
    return sorted(eligible), diagnostics, meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a symbol universe by common-date coverage and traded value."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--common-date", default="auto")
    parser.add_argument("--min-traded-value-vnd", type=float, default=1_000_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument(
        "--stock-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only stock rows and exclude known index/futures symbols.",
    )
    parser.add_argument(
        "--exclude-symbols",
        default=",".join(sorted(DEFAULT_EXCLUDED_SYMBOLS)),
        help="Comma-separated symbols to exclude from the universe.",
    )
    parser.add_argument(
        "--exclude-prefixes",
        default=",".join(DEFAULT_EXCLUDED_PREFIXES),
        help="Comma-separated symbol prefixes to exclude from the universe.",
    )
    args = parser.parse_args()

    excluded_symbols = {
        item.strip().upper() for item in str(args.exclude_symbols).split(",") if item.strip()
    }
    excluded_prefixes = tuple(
        item.strip().upper() for item in str(args.exclude_prefixes).split(",") if item.strip()
    )
    symbols, diagnostics, meta = build_universe(
        args.data_dir,
        args.common_date,
        args.min_traded_value_vnd,
        stock_only=args.stock_only,
        excluded_symbols=excluded_symbols,
        excluded_prefixes=excluded_prefixes,
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols_count": len(symbols),
        "symbols": symbols,
        "selection": meta,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {len(symbols)} symbols to {args.output}")
    print(",".join(symbols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
