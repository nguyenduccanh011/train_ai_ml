from pathlib import Path

from src.leaderboard.fairness import annotate_rows, load_config, resolve_baseline
from src.leaderboard.schema import LeaderboardRow

ROOT = Path(__file__).resolve().parents[2]


def _row(**overrides) -> LeaderboardRow:
    data = {
        "run_id": "champions_2020_2025_fair/v22#abc12345",
        "bundle": "champions_2020_2025_fair",
        "run_name": "v22",
        "config_hash": "abc12345def6",
        "generated_at": "2026-05-01T10:00:00",
        "superseded": False,
        "strategy": "v22",
        "feature_set": "leading_v2",
        "entry_model": "lightgbm",
        "exit_model_type": "lightgbm",
        "exit_model_enabled": True,
        "target": {
            "type": "trend_regime",
            "forward_window": 8,
            "gain_threshold": 0.06,
            "loss_threshold": 0.03,
        },
        "trades": 1000,
        "wr": 55.0,
        "avg_pnl": 3.5,
        "total_pnl": 3500.0,
        "pf": 2.5,
        "avg_hold": 10.0,
        "sharpe": 1.2,
        "max_drawdown": 200.0,
        "mdd_per_symbol": 20.0,
        "yearly_consistency": 0.6,
        "composite_score": 400.0,
        "score_mode": "live",
        "n_symbols": 61,
        "first_test_year": 2020,
        "last_test_year": 2025,
        "cost_profile": {
            "commission": "unknown",
            "tax": "unknown",
            "slippage": "unknown",
        },
        "fairness_group_key": "fair-a",
        "warnings": [],
    }
    data.update(overrides)
    return LeaderboardRow(**data)


def test_load_config_reads_repo_config():
    config = load_config(ROOT)

    assert config["baseline"] == {"bundle": "champions_2020_2025_fair", "run_name": "v22"}
    assert config["fairness_dims"] == ["symbols", "window", "cost_profile", "target"]


def test_resolve_baseline_found():
    rows = [
        _row(bundle="other", run_name="top", run_id="other/top#abc12345", composite_score=900.0),
        _row(),
    ]

    baseline = resolve_baseline(rows, load_config(ROOT))

    assert baseline is not None
    assert baseline.bundle == "champions_2020_2025_fair"
    assert baseline.run_name == "v22"


def test_annotate_rows_flags():
    baseline = _row()
    cross_group = _row(
        run_id="other/run#def67890",
        bundle="other",
        run_name="run",
        n_symbols=30,
        first_test_year=2021,
        fairness_group_key="fair-b",
    )

    annotated = annotate_rows([baseline, cross_group], baseline)

    assert annotated[0].is_baseline is True
    assert annotated[0].same_window_as_baseline is True
    assert annotated[1].is_baseline is False
    assert annotated[1].same_symbols_as_baseline is False
    assert annotated[1].same_window_as_baseline is False
    assert annotated[1].same_cost_as_baseline is True
    assert annotated[1].same_target_as_baseline is True
