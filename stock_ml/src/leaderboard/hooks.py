from __future__ import annotations

from pathlib import Path

from stock_ml.src.leaderboard.aggregator import append_or_update as _append_or_update
from stock_ml.src.leaderboard.schema import LeaderboardRow


def append_or_update(
    run_dir: str | Path, output_dir: str | Path, *, bundle: str | None = None
) -> LeaderboardRow:
    return _append_or_update(run_dir, output_dir, bundle=bundle)
