"""ModelTrainer — builds walk-forward prediction cache from ExperimentConfig.

This is a clean wrapper around the legacy _build_predictions logic,
parameterized by ExperimentConfig instead of positional args.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.config import ExperimentConfig


def build_prediction_cache(
    cfg: ExperimentConfig,
    symbols: list[str],
    *,
    device: str = "cpu",
    out_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train walk-forward models and return per-(fold, symbol) prediction cache.

    Returns list of dicts compatible with legacy runner format:
        {symbol, y_pred, y_pred_exit, y_proba, classes, returns, sym_test_df, feature_cols}

    If out_meta is provided, populates out_meta["feature_cache_key"] with the
    FeatureCacheManager key used for this run (for cache GC attribution).
    """
    import src.data.target as target_module
    import src.features.engine as feature_engine_module
    from src.cache.feature_cache import FeatureCacheManager
    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.env import get_results_dir, resolve_data_dir
    from src.features.engine import FeatureEngine
    from src.market_profile import resolve_run_context
    from src.models.registry import detect_device

    run_context = resolve_run_context(cfg)
    if run_context.resolved_data_dir is None:
        raise ValueError(f"Market {run_context.market!r} does not define data.data_dir")
    abs_data_dir = resolve_data_dir(run_context.resolved_data_dir)

    split_cfg = cfg.split
    target_config = run_context.target_config
    legacy_split = {
        "split": {
            "method": split_cfg.method,
            "train_years": split_cfg.train_years,
            "test_years": split_cfg.test_years,
            "gap_days": split_cfg.gap_days,
            "first_test_year": split_cfg.first_test_year,
            "last_test_year": split_cfg.last_test_year,
        },
        "target": target_config,
    }

    target_type = target_config.get("type", "trend_regime")
    if str(target_type).lower() == "return_regression":
        raise ValueError("signals.target.type='return_regression' is not supported.")

    resolved_device = detect_device(device)
    print(f"    Training device: {resolved_device.upper()}")

    # Model: run_context.model_stack already resolved via experiment > profile priority
    effective_model_type = (
        run_context.model_stack[0] if run_context.model_stack else cfg.entry_model_type()
    )
    entry_model_extras = cfg.signals.entry_model.extras
    exit_model_dict = cfg.exit_model_dict()

    loader = DataLoader(
        abs_data_dir,
        timeframe=run_context.timeframe,
        timestamp_column=run_context.market_profile.data.timestamp_column,
        timezone=run_context.market_profile.data.timezone,
        required_columns=run_context.market_profile.data.required_columns,
        optional_columns=run_context.market_profile.data.optional_columns,
    )
    splitter = WalkForwardSplitter(
        method=split_cfg.method,
        train_years=split_cfg.train_years,
        test_years=split_cfg.test_years,
        gap_days=split_cfg.gap_days,
        first_test_year=split_cfg.first_test_year,
        last_test_year=split_cfg.last_test_year,
    )
    target_gen = TargetGenerator.from_config(legacy_split)

    # Feature set: list of blocks (from MarketProfile) or named string (from experiment)
    resolved_feature_set = run_context.feature_set
    if isinstance(resolved_feature_set, list):
        from src.components.features.registry import build_engine_from_blocks

        engine = build_engine_from_blocks(resolved_feature_set)
        feature_set_key = "|".join(resolved_feature_set)
    else:
        feature_set_str = resolved_feature_set or cfg.feature_set()
        engine = FeatureEngine(feature_set=feature_set_str)
        feature_set_key = feature_set_str

    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__]

    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=symbols,
        timeframe=loader.timeframe,
        feature_set=feature_set_key,
        target_config=target_config,
        code_paths=code_paths,
    )
    if df is None:
        print(f"    Feature cache: MISS ({feature_set_key}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=symbols)
        df = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=symbols,
            timeframe=loader.timeframe,
            feature_set=feature_set_key,
            target_config=target_config,
            code_paths=code_paths,
        )
        print(
            f"    Feature cache: STORED ({feature_set_key}) key={saved_key[:8]} format={saved_fmt}"
        )
    else:
        print(f"    Feature cache: HIT ({feature_set_key}) key={cache_key[:8]}")

    if out_meta is not None:
        out_meta["feature_cache_key"] = cache_key
        out_meta["feature_set_key"] = feature_set_key

    df = target_gen.generate_for_all_symbols(df, drop_na=False)

    if exit_model_dict:
        from src.data.target import TargetGenerator as _TG

        df = _TG.generate_exit_labels(
            df,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
            drop_na=False,
        )

    feature_cols = engine.get_feature_columns(df)
    has_exit = "target_sell" in df.columns

    from src.pipeline._train_loop import train_predict_walk_forward

    exit_model_cfg = cfg.signals.exit_model

    return train_predict_walk_forward(
        df=df,
        splitter=splitter,
        symbols=symbols,
        feature_cols=feature_cols,
        target_cfg=target_config,
        entry_model_factory=lambda: get_model(
            effective_model_type, device=device, **entry_model_extras
        ),
        exit_model_factory=(
            lambda: get_exit_model(exit_model_cfg.type, device=device, **exit_model_cfg.extras)
        )
        if has_exit
        else None,
        has_exit=has_exit,
    )
