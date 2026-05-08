from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.leaderboard.schema import LeaderboardRow

DEFAULT_CONFIG = {
    "fairness_dims": ["market_family", "symbols", "window", "cost_profile", "target"],
    "market_families": {},
}


def load_config(root: str | Path | None = None) -> dict[str, Any]:
    config_path = _config_path(root)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {**DEFAULT_CONFIG, **data}


def resolve_market_family(
    market: str, timeframe: str | None = None, config: dict[str, Any] | None = None
) -> str:
    cfg = config or DEFAULT_CONFIG
    market_text = str(market or "unknown")
    timeframe_text = str(timeframe or "").lower()
    for family, family_cfg in (cfg.get("market_families") or {}).items():
        for member in family_cfg.get("members", []) or []:
            if str(member.get("market")) != market_text:
                continue
            member_timeframe = str(member.get("timeframe") or "").lower()
            if not member_timeframe or not timeframe_text or member_timeframe == timeframe_text:
                return str(family)
    return market_text if market_text != "unknown" else "vn_stock"


def backtest_window_key(first_year: int, last_year: int) -> str:
    if first_year <= 0 or last_year <= 0:
        return "unknown"
    return f"{first_year}-{last_year}"


def resolve_baseline(
    rows: list[LeaderboardRow], config: dict[str, Any] | None = None
) -> LeaderboardRow | None:
    """Resolve baseline from global config (legacy single-baseline path)."""
    cfg = config or DEFAULT_CONFIG
    baseline = cfg.get("baseline", {})
    if not baseline:
        return None
    return _find_baseline(rows, baseline.get("bundle"), baseline.get("run_name"))


def resolve_baseline_for_market(
    rows: list[LeaderboardRow], market: str, config: dict[str, Any] | None = None
) -> LeaderboardRow | None:
    """Resolve baseline for a specific market from the markets config section."""
    cfg = config or DEFAULT_CONFIG
    market_cfg = cfg.get("markets", {}).get(market, {})
    baseline = market_cfg.get("baseline") or {}
    return _find_baseline(rows, baseline.get("bundle"), baseline.get("run_name"))


def annotate_rows(
    rows: list[LeaderboardRow], baseline: LeaderboardRow | None
) -> list[LeaderboardRow]:
    if baseline is None:
        return rows

    return [
        row.model_copy(
            update={
                "is_baseline": row.run_id == baseline.run_id,
                "same_symbols_as_baseline": row.n_symbols == baseline.n_symbols,
                "same_window_as_baseline": row.first_test_year == baseline.first_test_year
                and row.last_test_year == baseline.last_test_year,
                "same_cost_as_baseline": row.cost_profile == baseline.cost_profile,
                "same_target_as_baseline": row.target == baseline.target,
                "same_timeframe_as_baseline": row.timeframe == baseline.timeframe,
                "same_market_family_as_baseline": row.market_family == baseline.market_family,
            }
        )
        for row in rows
    ]


def _find_baseline(
    rows: list[LeaderboardRow], bundle: str | None, run_name: str | None
) -> LeaderboardRow | None:
    if not bundle or not run_name:
        return None
    active_rows = [row for row in rows if not row.superseded]
    candidates = active_rows or rows
    for row in candidates:
        if row.bundle == bundle and row.run_name == run_name:
            return row
    return None


def _config_path(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root) / "config" / "leaderboard.yaml"
    return Path(__file__).resolve().parents[2] / "config" / "leaderboard.yaml"
