"""§13.3.1 fetch-on-miss cache upsert (``ensure_symbols_cached``).

Mocks the source (``fetch_ohlcv``) so the cache logic — detect missing symbols, upsert their bars with
``timeframe``/``traded_value`` filled, idempotence — is tested without a network call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.data.duckdb_loader import DuckDBLoader, ensure_symbols_cached  # noqa: E402


def _seed_duck(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE ohlcv(symbol VARCHAR, timeframe VARCHAR, date DATE, open DOUBLE, high DOUBLE, "
        "low DOUBLE, close DOUBLE, volume DOUBLE, traded_value DOUBLE)"
    )
    con.execute("INSERT INTO ohlcv VALUES ('FPT','1D','2020-01-02',1,1,1,1,100,100)")
    con.close()


def _fake_history(symbol: str, **_kw) -> pd.DataFrame:
    # real intraday range (high>low) so the loader's leading-phantom trim keeps the bars
    rows = [
        {
            "symbol": symbol,
            "date": d,
            "open": 2.0,
            "high": 2.1,
            "low": 1.9,
            "close": 2.0,
            "volume": 500,
        }
        for d in pd.bdate_range("2019-01-02", periods=3)
    ]
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])


def test_fetch_on_miss_upserts_missing_only(tmp_path, monkeypatch):
    import src.data.sieutinhieu as sth

    monkeypatch.setattr(sth, "fetch_history", _fake_history)
    db = tmp_path / "cache.duckdb"
    _seed_duck(db)

    fetched = ensure_symbols_cached(str(db), ["FPT", "ROS", "FLC"])
    assert fetched == ["FLC", "ROS"]  # FPT already present, only the two missing fetched

    loader = DuckDBLoader(str(db))
    assert set(loader.list_symbols()) == {"FPT", "FLC", "ROS"}
    ros = loader.load_symbol("ROS")
    assert len(ros) == 3
    # traded_value filled with the local convention volume*close (500 * 2.0)
    con = duckdb.connect(str(db), read_only=True)
    tv = con.execute("SELECT DISTINCT traded_value FROM ohlcv WHERE symbol='ROS'").fetchone()[0]
    con.close()
    assert tv == 1000.0


def test_fetch_on_miss_idempotent(tmp_path, monkeypatch):
    import src.data.sieutinhieu as sth

    monkeypatch.setattr(sth, "fetch_history", _fake_history)
    db = tmp_path / "cache.duckdb"
    _seed_duck(db)

    ensure_symbols_cached(str(db), ["ROS"])
    assert ensure_symbols_cached(str(db), ["FPT", "ROS"]) == []  # nothing missing the second time


def test_fetch_on_miss_tolerates_untradeable_symbol(tmp_path, monkeypatch):
    # A symbol whose /ohlcv/ 500s (no price series) is dropped with a warning; the rest still land.
    import src.data.sieutinhieu as sth

    def _flaky(symbol: str, **kw):
        if symbol == "DEAD":
            raise RuntimeError("sieutinhieu: GET ohlcv/ failed (HTTP Error 500)")
        return _fake_history(symbol, **kw)

    monkeypatch.setattr(sth, "fetch_history", _flaky)
    db = tmp_path / "cache.duckdb"
    _seed_duck(db)

    fetched = ensure_symbols_cached(str(db), ["ROS", "DEAD", "FLC"])
    assert fetched == ["FLC", "ROS"]  # DEAD dropped, others inserted


def test_fetch_on_miss_all_untradeable_no_abort(tmp_path, monkeypatch):
    # When every missing symbol 500s (no price series), proceed with zero fetched — do NOT abort
    # (they are untradeable, not a down source). Regression: the survivorship 7-symbol dyn900 case.
    import src.data.sieutinhieu as sth

    def _all_500(symbol: str, **kw):
        raise RuntimeError("sieutinhieu: GET ohlcv/ failed (HTTP Error 500: Internal Server Error)")

    monkeypatch.setattr(sth, "fetch_history", _all_500)
    db = tmp_path / "cache.duckdb"
    _seed_duck(db)

    assert ensure_symbols_cached(str(db), ["CI5", "H11"]) == []  # both dropped, no raise


def test_fetch_on_miss_aborts_when_source_down(tmp_path, monkeypatch):
    import src.data.sieutinhieu as sth

    def _all_fail(symbol: str, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(sth, "fetch_history", _all_fail)
    db = tmp_path / "cache.duckdb"
    _seed_duck(db)

    import pytest

    with pytest.raises(RuntimeError, match="source unreachable"):
        ensure_symbols_cached(str(db), ["ROS", "FLC"])
