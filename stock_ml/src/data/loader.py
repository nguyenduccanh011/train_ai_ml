"""
Data loader for VN stock dataset.
Loads OHLCV data from Hive-partitioned parquet/csv structure.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm


class DataLoader:
    """Load stock data from cleaned dataset directory."""

    def __init__(
        self,
        data_dir: str,
        timeframe: str = "1D",
        timestamp_column: str = "timestamp",
        timezone: str | None = None,
        required_columns: list[str] | None = None,
        optional_columns: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.all_symbols_dir = self.data_dir / "all_symbols"
        self.context_dir = self.data_dir / "context_features"
        self.timeframe = timeframe
        self.timestamp_column = timestamp_column
        self.timezone = timezone
        self.required_columns = required_columns or ["open", "high", "low", "close", "volume"]
        self.optional_columns = optional_columns or []
        self._symbols_cache: list[str] | None = None
        self._data_cache: dict[str, pd.DataFrame] = {}

    @property
    def symbols(self) -> list[str]:
        """Get list of available symbols."""
        if self._symbols_cache is None:
            symbols_file = self.data_dir / "clean_symbols.txt"
            if symbols_file.exists():
                self._symbols_cache = symbols_file.read_text().strip().split("\n")
            else:
                # Discover from directory
                self._symbols_cache = sorted(
                    [
                        d.name.replace("symbol=", "")
                        for d in self.all_symbols_dir.iterdir()
                        if d.is_dir() and d.name.startswith("symbol=")
                    ]
                )
        return self._symbols_cache

    def load_symbol(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """Load data for a single symbol."""
        if use_cache and symbol in self._data_cache:
            return self._data_cache[symbol].copy()

        csv_path = (
            self.all_symbols_dir / f"symbol={symbol}" / f"timeframe={self.timeframe}" / "data.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(f"No data for {symbol} at {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=[self.timestamp_column])
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {symbol}: {missing}")
        df = df.sort_values(self.timestamp_column).reset_index(drop=True)
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], utc=True)
        if self.timestamp_column != "timestamp" and "timestamp" not in df.columns:
            df["timestamp"] = df[self.timestamp_column]

        if use_cache:
            self._data_cache[symbol] = df

        return df.copy()

    def load_all(
        self,
        symbols: list[str] | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Load data for multiple symbols, concatenated into one DataFrame."""
        symbols = symbols or self.symbols
        dfs = []
        iterator = tqdm(symbols, desc="Loading symbols") if show_progress else symbols

        for sym in iterator:
            try:
                df = self.load_symbol(sym)
                dfs.append(df)
            except FileNotFoundError:
                continue

        if not dfs:
            raise ValueError("No data loaded!")

        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return result

    def load_context(self, context_symbol: str | None = None) -> pd.DataFrame:
        """Load context/market data (indices, futures)."""
        if context_symbol is None:
            return pd.DataFrame()
        csv_path = (
            self.context_dir
            / f"symbol={context_symbol}"
            / f"timeframe={self.timeframe}"
            / "data.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"No context data for {context_symbol}")

        df = pd.read_csv(csv_path, parse_dates=[self.timestamp_column])
        df = df.sort_values(self.timestamp_column).reset_index(drop=True)
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], utc=True)
        if self.timestamp_column != "timestamp" and "timestamp" not in df.columns:
            df["timestamp"] = df[self.timestamp_column]
        return df

    def load_all_context(self) -> dict[str, pd.DataFrame]:
        """Load all context features (indices + futures)."""
        result = {}
        for d in self.context_dir.iterdir():
            if d.is_dir() and d.name.startswith("symbol="):
                sym = d.name.replace("symbol=", "")
                try:
                    result[sym] = self.load_context(sym)
                except FileNotFoundError:
                    continue
        return result

    def clear_cache(self):
        """Clear data cache to free memory."""
        self._data_cache.clear()

    def get_date_range(self) -> tuple:
        """Get overall date range from first symbol."""
        df = self.load_symbol(self.symbols[0])
        return df["timestamp"].min(), df["timestamp"].max()

    def summary(self) -> dict:
        """Quick dataset summary."""
        return {
            "n_symbols": len(self.symbols),
            "data_dir": str(self.data_dir),
            "timeframe": self.timeframe,
            "context_symbols": [
                d.name.replace("symbol=", "") for d in self.context_dir.iterdir() if d.is_dir()
            ]
            if self.context_dir.exists()
            else [],
        }
