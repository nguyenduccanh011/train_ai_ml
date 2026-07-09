from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "portable_data" / "derivatives_ai_dataset"
VIZ_DIR = REPO_ROOT / "visualization"


def _format_time(value: str) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def export_symbol(symbol: str = "VN30F1M", timeframe: str = "1H") -> int:
    data_path = DATASET_ROOT / f"symbol={symbol}" / f"timeframe={timeframe}" / "data.csv"
    df = pd.read_csv(data_path)
    ohlcv = [
        {
            "time": _format_time(row.timestamp),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": int(row.volume),
        }
        for row in df.itertuples(index=False)
    ]

    output_dir = VIZ_DIR / "data_derivatives"
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"{symbol}.json").write_text(
        json.dumps({"symbol": symbol, "ohlcv": ohlcv}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "index.json").write_text(
        json.dumps(
            {"symbols": [{"symbol": symbol, "file": f"data_derivatives/{symbol}.json"}]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported {len(ohlcv)} candles to {output_dir / f'{symbol}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(export_symbol())
