"""Phase 2 parity check: CSV DataLoader vs DuckDBLoader must agree byte-for-byte.

Compares OHLCV rows symbol-by-symbol through the exact loader code paths the
pipeline uses, so any divergence introduced by the CSV->DuckDB ETL surfaces
before we flip ``data_dir`` to the DuckDB file. Fails loud (non-zero exit) on
the first mismatch — no silent dropping of rows.

Usage (repo root on PYTHONPATH):
    python stock_ml/scripts/verify_ohlcv_parity.py \
        --csv-root portable_data/vn_stock_ai_dataset_cleaned \
        --duckdb market_data/market.duckdb \
        --timeframe 1D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on sys.path so ``stock_ml`` resolves as a namespace package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stock_ml.src.data.duckdb_loader import DuckDBLoader  # noqa: E402
from stock_ml.src.data.loader import DataLoader  # noqa: E402

_VALUE_COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Index by plain date, keep only the OHLCV columns, sorted ascending."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.set_index("date").sort_index()
    return out[_VALUE_COLS]


def _compare_symbol(sym: str, csv: DataLoader, duck: DuckDBLoader, atol: float) -> list[str]:
    """Return a list of human-readable mismatch reasons (empty == parity)."""
    try:
        c = _normalize(csv.load_symbol(sym))
    except Exception as e:  # noqa: BLE001
        return [f"CSV load failed: {e}"]
    try:
        d = _normalize(duck.load_symbol(sym))
    except Exception as e:  # noqa: BLE001
        return [f"DuckDB load failed: {e}"]

    reasons: list[str] = []
    if len(c) != len(d):
        reasons.append(f"row count {len(c)} (csv) != {len(d)} (duckdb)")

    only_csv = c.index.difference(d.index)
    only_duck = d.index.difference(c.index)
    if len(only_csv):
        reasons.append(f"{len(only_csv)} dates only in csv (first={only_csv[0].date()})")
    if len(only_duck):
        reasons.append(f"{len(only_duck)} dates only in duckdb (first={only_duck[0].date()})")

    common = c.index.intersection(d.index)
    cc, dd = c.loc[common], d.loc[common]
    for col in _VALUE_COLS:
        diff = ~np.isclose(
            cc[col].to_numpy(), dd[col].to_numpy(), atol=atol, rtol=0, equal_nan=True
        )
        if diff.any():
            n = int(diff.sum())
            first = common[diff][0].date()
            reasons.append(f"{col}: {n} values differ (first={first})")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify CSV<->DuckDB OHLCV parity")
    ap.add_argument("--csv-root", default="portable_data/vn_stock_ai_dataset_cleaned")
    ap.add_argument("--duckdb", default="market_data/market.duckdb")
    ap.add_argument("--timeframe", default="1D")
    ap.add_argument("--atol", type=float, default=1e-9, help="absolute float tolerance")
    ap.add_argument("--limit", type=int, default=0, help="check only first N symbols (0=all)")
    args = ap.parse_args()

    csv = DataLoader(args.csv_root, timeframe=args.timeframe)
    duck = DuckDBLoader(args.duckdb, timeframe=args.timeframe)

    csv_syms = set(csv.list_symbols())
    duck_syms = set(duck.list_symbols())
    missing_in_duck = sorted(csv_syms - duck_syms)
    missing_in_csv = sorted(duck_syms - csv_syms)

    print(f"CSV symbols:    {len(csv_syms)}")
    print(f"DuckDB symbols: {len(duck_syms)}")
    if missing_in_duck:
        print(
            f"[FAIL] {len(missing_in_duck)} symbols in CSV but not DuckDB: {missing_in_duck[:10]}"
        )
    if missing_in_csv:
        print(f"[WARN] {len(missing_in_csv)} symbols in DuckDB but not CSV: {missing_in_csv[:10]}")

    symbols = sorted(csv_syms & duck_syms)
    if args.limit:
        symbols = symbols[: args.limit]

    failed: dict[str, list[str]] = {}
    for i, sym in enumerate(symbols, 1):
        reasons = _compare_symbol(sym, csv, duck, args.atol)
        if reasons:
            failed[sym] = reasons
        if i % 100 == 0:
            print(f"  checked {i}/{len(symbols)} ... {len(failed)} mismatches so far")

    print(f"\nChecked {len(symbols)} symbols — {len(failed)} mismatched.")
    for sym, reasons in list(failed.items())[:20]:
        print(f"  [DIFF] {sym}: {'; '.join(reasons)}")

    if failed or missing_in_duck:
        print("\nRESULT: PARITY FAILED")
        return 1
    print("\nRESULT: PARITY OK (diff = 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
