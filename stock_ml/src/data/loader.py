"""OHLCV loader for the partitioned CSV dataset.

Dataset layout:
    <data_root>/all_symbols/symbol=<SYM>/timeframe=<TF>/data.csv

Each CSV has columns: timestamp, symbol, exchange, asset_type, data_provider,
timeframe, open, high, low, close, volume, traded_value.

The loader returns a clean per-symbol DataFrame indexed by tz-naive UTC date
with columns [open, high, low, close, volume].
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


class DataLoader:
    def __init__(self, data_root: str | Path, timeframe: str = "1D") -> None:
        self.root = Path(data_root)
        self.timeframe = timeframe
        if not self.root.exists():
            raise FileNotFoundError(f"data_root does not exist: {self.root}")

    @property
    def symbols_root(self) -> Path:
        return self.root / "all_symbols"

    def list_symbols(self) -> list[str]:
        if not self.symbols_root.exists():
            return []
        out: list[str] = []
        for p in sorted(self.symbols_root.iterdir()):
            if p.is_dir() and p.name.startswith("symbol="):
                sym = p.name.split("=", 1)[1]
                if (p / f"timeframe={self.timeframe}" / "data.csv").exists():
                    out.append(sym)
        return out

    def symbol_path(self, symbol: str) -> Path:
        return self.symbols_root / f"symbol={symbol}" / f"timeframe={self.timeframe}" / "data.csv"

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        path = self.symbol_path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"missing data file for {symbol}: {path}")
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{path} missing 'timestamp' column")
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns {missing}")
        df = df[["date", *REQUIRED_COLS]].copy()
        df = df.dropna(subset=["date", "open", "high", "low", "close"])
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        df["symbol"] = symbol
        return df

    def load_many(self, symbols: list[str]) -> pd.DataFrame:
        frames = [self.load_symbol(s) for s in symbols]
        if not frames:
            return pd.DataFrame(columns=["date", *REQUIRED_COLS, "symbol"])
        return pd.concat(frames, ignore_index=True)
