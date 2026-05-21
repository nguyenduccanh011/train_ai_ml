"""Provider interface for external market data sources."""

from __future__ import annotations

from typing import Protocol


class MarketDataProvider(Protocol):
    """Read market symbols and OHLCV bars from a provider."""

    def list_symbols(self, limit: int = 1000, offset: int = 0) -> dict:
        """Return provider symbols with provider-specific metadata."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1D",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> dict:
        """Return historical OHLCV bars for a symbol."""

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 10,
    ) -> list[dict]:
        """Return latest OHLCV bars for a symbol."""
