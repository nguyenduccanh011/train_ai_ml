from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.leaderboard.schema import LeaderboardRow

DEFAULT_CONFIG = {
    "baseline": {
        "bundle": "champions_2020_2025_fair",
        "run_name": "v22",
    },
    "fairness_dims": ["symbols", "window", "cost_profile", "target"],
}


def load_config(root: str | Path | None = None) -> dict[str, Any]:
    config_path = _config_path(root)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {**DEFAULT_CONFIG, **data}


def resolve_baseline(
    rows: list[LeaderboardRow], config: dict[str, Any] | None = None
) -> LeaderboardRow | None:
    active_rows = [row for row in rows if not row.superseded]
    candidates = active_rows or rows
    if not candidates:
        return None

    baseline = (config or DEFAULT_CONFIG).get("baseline", {})
    bundle = baseline.get("bundle")
    run_name = baseline.get("run_name")
    if bundle and run_name:
        for row in candidates:
            if row.bundle == bundle and row.run_name == run_name:
                return row

    return max(candidates, key=lambda row: row.composite_score)


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
            }
        )
        for row in rows
    ]


def _config_path(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root) / "config" / "leaderboard.yaml"
    return Path(__file__).resolve().parents[2] / "config" / "leaderboard.yaml"
