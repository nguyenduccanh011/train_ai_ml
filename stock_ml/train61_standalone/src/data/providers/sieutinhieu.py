"""Sieu Tin Hieu market data provider adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import requests

DEFAULT_API_BASE = "https://sieutinhieu.vn/api/v1"


@dataclass(frozen=True)
class NormalizedBar:
    symbol: str
    symbol_id: int | None
    timeframe: str
    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    traded_value: Decimal | None
    provider: str
    provider_bar_id: int | None
    provider_created_at: str | None


class SieuTinHieuProvider:
    """HTTP client for https://sieutinhieu.vn/api/v1."""

    provider_name = "sieutinhieu"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("SIEUTINHIEU_API_BASE") or DEFAULT_API_BASE).rstrip(
            "/"
        )
        self.timeout = timeout
        self.session = session or requests.Session()

    def list_symbols(self, limit: int = 1000, offset: int = 0) -> dict:
        return self._get("/symbols/", {"limit": limit, "offset": offset})

    def search_symbols(self, query: str, limit: int = 20, offset: int = 0) -> dict:
        return self._get("/symbols/search", {"query": query, "limit": limit, "offset": offset})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1D",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> dict:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "limit": min(limit, 1000),
            "offset": offset,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._get("/ohlcv/", params)

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 10,
    ) -> list[dict]:
        payload = self._get(
            "/ohlcv/latest",
            {"symbol": symbol.upper(), "timeframe": timeframe, "limit": min(limit, 1000)},
        )
        return _extract_items(payload)

    def normalize_bar(self, symbol: str, item: dict[str, Any]) -> NormalizedBar:
        """Normalize one provider OHLCV item into DB-ready fields."""
        return NormalizedBar(
            symbol=(item.get("symbol") or symbol).upper(),
            symbol_id=_optional_int(item.get("symbol_id")),
            timeframe=str(item.get("timeframe") or "1D"),
            timestamp=str(item["timestamp"]),
            open=Decimal(str(item["open"])),
            high=Decimal(str(item["high"])),
            low=Decimal(str(item["low"])),
            close=Decimal(str(item["close"])),
            volume=int(item.get("volume") or 0),
            traded_value=_optional_decimal(item.get("traded_value")),
            provider=self.provider_name,
            provider_bar_id=_optional_int(item.get("id")),
            provider_created_at=item.get("created_at"),
        )

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()


def _extract_items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected provider payload type: {type(payload).__name__}")
    for key in ("value", "data", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError(
        f"Could not find OHLCV list in provider payload keys: {sorted(payload.keys())}"
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))
