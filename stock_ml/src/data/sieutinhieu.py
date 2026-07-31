"""Siêu Tín Hiệu data client — the SINGLE source of truth (§13.3.1).

Local ``market.duckdb`` / CSV are a secondary CACHE, allowed to be stale; this module reaches the
authoritative, point-in-time source for the two things the cache gets WRONG:

- **Universe** (`/symbols/universe`): ranks by MATCHED average daily traded value (``basis=matched``,
  from ``market_flow_1d``) and gates by ``min_sessions_traded`` (real matched sessions) — fixing the
  §13.9 defect where the resolver used ``avg(volume*close)`` (back-adjust-distorted, block-trade-
  inflated) and ``count(*)`` (ghost bars: 22% volume=0). This endpoint is point-in-time & append-only
  (``market_flow_1d`` is NEVER re-synced), so a given ``as_of`` is stable/reproducible.
- **Prices** (`/ohlcv/`): back-adjusted history for symbols the local cache lacks (the survivorship-
  correct names the correct universe surfaces — ROS, delisted tickers — §13.3.1 fetch-on-miss).

Read path is public (no token). Fail-loud on any network/parse error: an engine-integrity run must
never silently proceed on partial data (mirrors ``MARKET_CONTEXT_REQUIRED``, §1.1).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

API_BASE = "https://sieutinhieu.vn/api/v1"
_TIMEOUT = 60
# The server raised the per-call cap to 50,000 bars (was 1,000) — a full VN daily history (~5.6k bars)
# now comes back in ONE call, so fetch-on-miss needs no paging in practice. Paging is kept as a fallback
# for anything deeper than the cap. See https://sieutinhieu.vn/docs (OpenAPI: /ohlcv/ limit max=50000).
_OHLCV_API_MAX = 50000
_OHLCV_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def _get(path: str, params: dict) -> object:
    """GET a public read endpoint. Fail loud (no silent fallback) on HTTP/JSON error."""
    url = f"{API_BASE}/{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception as e:  # noqa: BLE001 — surface the endpoint that failed, then stop
        raise RuntimeError(f"sieutinhieu: GET {path} failed ({e}); params={params}") from e


# --------------------------------------------------------------------------------------------------
# Universe (point-in-time, matched basis) — §13.9 correct measuring stick
# --------------------------------------------------------------------------------------------------
def fetch_universe(
    as_of: list[str] | str,
    *,
    sessions: int,
    min_sessions_traded: int | None = None,
    min_adtv: float | None = None,
    top_n: int | None = None,
    basis: str = "matched",
    require_full_window: bool = False,
    asset_type: str = "stock",
) -> dict[str, list[dict]]:
    """Resolve the tradable universe at one or more ``as_of`` dates.

    Returns ``{as_of_str: [ {symbol, adtv_value, sessions_traded, exchange, ...}, ... ]}`` with each
    list ranked by matched ADTV descending. ``sessions`` sets the trailing window length (≤ 250, the
    server cap; 250 ≈ one prior trading year). ``min_sessions_traded`` is the real-liquidity gate that
    replaces the old ``count(*)`` (drops symbols that stopped trading). ``top_n`` is left None here so
    the caller can apply lookback-min / hysteresis over the FULL ranked list in Python.

    Fails loud if the server omits a requested ``as_of`` (never a silent universe shrink).
    """
    dates = [as_of] if isinstance(as_of, str) else list(as_of)
    params: dict = {"as_of": ",".join(dates), "sessions": int(sessions), "basis": basis,
                    "asset_type": asset_type, "require_full_window": str(require_full_window).lower()}
    if min_sessions_traded is not None:
        params["min_sessions_traded"] = int(min_sessions_traded)
    if min_adtv is not None:
        params["min_adtv"] = float(min_adtv)
    if top_n is not None:
        params["top_n"] = int(top_n)

    payload = _get("symbols/universe", params)
    if not isinstance(payload, dict) or "dates" not in payload:
        raise RuntimeError(f"sieutinhieu: unexpected /symbols/universe payload: {str(payload)[:200]}")
    out: dict[str, list[dict]] = {d.get("as_of"): d.get("symbols", []) for d in payload["dates"]}
    missing = [d for d in dates if d not in out]
    if missing:
        raise RuntimeError(f"sieutinhieu: /symbols/universe returned no rows for as_of={missing}")
    return out


# --------------------------------------------------------------------------------------------------
# Prices (back-adjusted OHLCV) — §13.3.1 fetch-on-miss for symbols the local cache lacks
# --------------------------------------------------------------------------------------------------
def _rows_to_df(symbol: str, items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        if it.get("id", None) == 0:  # today's still-forming (preliminary) bar
            continue
        try:
            rows.append({
                "symbol": symbol,
                "date": pd.to_datetime(it["timestamp"]).tz_localize(None).normalize(),
                "open": float(it["open"]), "high": float(it["high"]),
                "low": float(it["low"]), "close": float(it["close"]),
                "volume": int(it["volume"]),
            })
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"sieutinhieu: bad OHLCV row for {symbol}: {it!r} ({e})") from e
    if not rows:
        raise ValueError(f"sieutinhieu: no bars returned for {symbol}")
    return pd.DataFrame(rows, columns=_OHLCV_COLS).sort_values("date").reset_index(drop=True)


def fetch_history(symbol: str, *, limit: int = _OHLCV_API_MAX) -> pd.DataFrame:
    """Fetch up to ``limit`` most-recent CLOSED daily bars (back-adjusted) for one symbol.

    Default pulls the full available history in ONE call (server cap 50k >> VN history). Fail loud.
    """
    items: list = []
    offset = 0
    while len(items) < limit:
        page = _get("ohlcv/", {"symbol": symbol, "timeframe": "1D", "limit": _OHLCV_API_MAX, "offset": offset})
        page_items = page["items"] if isinstance(page, dict) else page
        if not page_items:
            break
        items.extend(page_items)
        if len(page_items) < _OHLCV_API_MAX:
            break
        offset += _OHLCV_API_MAX
    return _rows_to_df(symbol, items[:limit] if limit else items)


def fetch_ohlcv(symbols: list[str], *, limit: int = _OHLCV_API_MAX) -> pd.DataFrame:
    """Fetch full back-adjusted history for many symbols, concatenated. Fail loud on any miss."""
    frames, failed = [], []
    for sym in symbols:
        try:
            frames.append(fetch_history(sym, limit=limit))
        except Exception as e:  # noqa: BLE001
            failed.append((sym, str(e)))
    if failed:
        raise RuntimeError(f"sieutinhieu: {len(failed)} symbol(s) failed to fetch: {failed[:5]}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_OHLCV_COLS)
