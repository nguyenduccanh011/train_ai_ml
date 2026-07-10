"""Adapters between LeaderboardRow (Pydantic) and LeaderboardRunModel (SQLAlchemy)."""

from __future__ import annotations

from datetime import UTC, datetime

from stock_ml.db.models.run import LeaderboardRunModel
from stock_ml.src.leaderboard.schema import (
    Artifacts,
    CacheKeys,
    CostProfile,
    LeaderboardRow,
    LifecycleState,
    TargetConfig,
)


def row_to_model(
    row: LeaderboardRow,
    *,
    parent_run_id: str | None = None,
    run_seed: int | None = None,
    template_config_hash: str | None = None,
    raw_config: str | None = None,
    template_id: int | None = None,
    git_sha: str | None = None,
    data_snapshot_date=None,
    lib_versions: dict | None = None,
) -> LeaderboardRunModel:
    """Convert LeaderboardRow Pydantic → LeaderboardRunModel ORM."""
    now = datetime.now(UTC)
    generated = datetime.fromisoformat(row.generated_at)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)

    # Parse ISO timestamps if present
    test_start = None
    if row.test_start_date:
        test_start = datetime.fromisoformat(row.test_start_date)
        if test_start.tzinfo is None:
            test_start = test_start.replace(tzinfo=UTC)

    test_end = None
    if row.test_end_date:
        test_end = datetime.fromisoformat(row.test_end_date)
        if test_end.tzinfo is None:
            test_end = test_end.replace(tzinfo=UTC)

    train_start = None
    if row.train_start_date:
        train_start = datetime.fromisoformat(row.train_start_date)
        if train_start.tzinfo is None:
            train_start = train_start.replace(tzinfo=UTC)

    train_end = None
    if row.train_end_date:
        train_end = datetime.fromisoformat(row.train_end_date)
        if train_end.tzinfo is None:
            train_end = train_end.replace(tzinfo=UTC)

    return LeaderboardRunModel(
        run_id=row.run_id,
        bundle=row.bundle,
        run_name=row.run_name,
        config_hash=row.config_hash,
        generated_at=generated,
        superseded=row.superseded,
        state=row.state.value,
        cache_key_features=row.cache_keys.features,
        cache_key_predictions=row.cache_keys.predictions,
        artifact_trades_csv=row.artifacts.trades_csv,
        artifact_meta_json=row.artifacts.meta_json,
        artifact_model_pkl=row.artifacts.model_pkl,
        market=row.market,
        market_family=row.market_family,
        currency=row.currency,
        pnl_mode=row.pnl_mode,
        schema_ver=row.schema,
        timeframe=row.timeframe,
        strategy=row.strategy,
        feature_set=row.feature_set,
        entry_model=row.entry_model,
        target_type=row.target.type,
        target_forward_window=row.target.forward_window,
        trades=row.trades,
        wr=row.wr,
        avg_pnl=row.avg_pnl,
        total_pnl=row.total_pnl,
        pnl_pct=row.pnl_pct,
        pf=row.pf,
        avg_hold=row.avg_hold,
        max_win=row.max_win,
        max_loss=row.max_loss,
        sharpe=row.sharpe,
        max_drawdown=row.max_drawdown,
        mdd_per_symbol=row.mdd_per_symbol,
        yearly_consistency=row.yearly_consistency,
        composite_score=row.composite_score,
        score_mode=row.score_mode,
        n_symbols=row.n_symbols,
        first_test_year=row.first_test_year,
        last_test_year=row.last_test_year,
        backtest_window_key=row.backtest_window_key,
        cost_commission=str(row.cost_profile.commission),
        cost_tax=str(row.cost_profile.tax),
        cost_slippage=str(row.cost_profile.slippage),
        fairness_group_key=row.fairness_group_key,
        is_baseline=row.is_baseline,
        same_symbols_as_baseline=row.same_symbols_as_baseline,
        same_window_as_baseline=row.same_window_as_baseline,
        same_cost_as_baseline=row.same_cost_as_baseline,
        same_target_as_baseline=row.same_target_as_baseline,
        same_timeframe_as_baseline=row.same_timeframe_as_baseline,
        same_market_family_as_baseline=row.same_market_family_as_baseline,
        warnings=list(row.warnings),
        parent_run_id=parent_run_id,
        experiment_group=row.experiment_group,
        variant_type=row.variant_type,
        metadata_notes=row.metadata_notes,
        direction=row.direction,
        model_mode=row.model_mode,
        signal_mode=row.signal_mode,
        test_start_date=test_start,
        test_end_date=test_end,
        train_start_date=train_start,
        train_end_date=train_end,
        universe_slug=row.universe_slug,
        universe_version=row.universe_version,
        run_seed=run_seed,
        template_config_hash=template_config_hash,
        raw_config=raw_config,
        template_id=template_id,
        git_sha=git_sha,
        data_snapshot_date=data_snapshot_date,
        lib_versions=lib_versions,
        created_at=now,
        updated_at=now,
    )


def model_to_row(m: LeaderboardRunModel) -> LeaderboardRow:
    """Convert LeaderboardRunModel ORM → LeaderboardRow Pydantic."""
    return LeaderboardRow(
        run_id=m.run_id,
        bundle=m.bundle,
        run_name=m.run_name,
        config_hash=m.config_hash,
        generated_at=m.generated_at.isoformat(),
        superseded=m.superseded,
        state=LifecycleState(m.state),
        cache_keys=CacheKeys(features=m.cache_key_features, predictions=m.cache_key_predictions),
        artifacts=Artifacts(
            trades_csv=m.artifact_trades_csv,
            meta_json=m.artifact_meta_json,
            model_pkl=m.artifact_model_pkl,
        ),
        market=m.market,
        market_family=m.market_family,
        currency=m.currency,
        pnl_mode=m.pnl_mode,
        schema=m.schema_ver,
        timeframe=m.timeframe,
        strategy=m.strategy,
        feature_set=m.feature_set,
        entry_model=m.entry_model,
        target=TargetConfig(
            type=m.target_type,
            forward_window=m.target_forward_window,
        ),
        trades=m.trades,
        wr=m.wr,
        avg_pnl=m.avg_pnl,
        total_pnl=m.total_pnl,
        pnl_pct=m.pnl_pct,
        pf=m.pf,
        avg_hold=m.avg_hold,
        max_win=m.max_win,
        max_loss=m.max_loss,
        sharpe=m.sharpe,
        max_drawdown=m.max_drawdown,
        mdd_per_symbol=m.mdd_per_symbol,
        yearly_consistency=m.yearly_consistency,
        composite_score=m.composite_score,
        score_mode=m.score_mode,
        n_symbols=m.n_symbols,
        first_test_year=m.first_test_year,
        last_test_year=m.last_test_year,
        backtest_window_key=m.backtest_window_key,
        cost_profile=CostProfile(
            commission=m.cost_commission,
            tax=m.cost_tax,
            slippage=m.cost_slippage,
        ),
        fairness_group_key=m.fairness_group_key,
        is_baseline=m.is_baseline,
        same_symbols_as_baseline=m.same_symbols_as_baseline,
        same_window_as_baseline=m.same_window_as_baseline,
        same_cost_as_baseline=m.same_cost_as_baseline,
        same_target_as_baseline=m.same_target_as_baseline,
        same_timeframe_as_baseline=m.same_timeframe_as_baseline,
        same_market_family_as_baseline=m.same_market_family_as_baseline,
        warnings=list(m.warnings or []),
        experiment_group=m.experiment_group or "ungrouped",
        variant_type=m.variant_type,
        parent_run_id=m.parent_run_id,
        metadata_notes=m.metadata_notes,
        direction=m.direction or "long",
        model_mode=m.model_mode,
        signal_mode=m.signal_mode,
        test_start_date=m.test_start_date.isoformat() if m.test_start_date else None,
        test_end_date=m.test_end_date.isoformat() if m.test_end_date else None,
        train_start_date=m.train_start_date.isoformat() if m.train_start_date else None,
        train_end_date=m.train_end_date.isoformat() if m.train_end_date else None,
        universe_slug=m.universe_slug,
        universe_version=m.universe_version,
    )
