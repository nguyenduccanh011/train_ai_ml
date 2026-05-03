"""ModelTrainer — builds walk-forward prediction cache from ExperimentConfig.

This is a clean wrapper around the legacy _build_predictions logic,
parameterized by ExperimentConfig instead of positional args.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.pipeline.config import ExperimentConfig


def build_prediction_cache(
    cfg: ExperimentConfig,
    symbols: list[str],
    *,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Train walk-forward models and return per-(fold, symbol) prediction cache.

    Returns list of dicts compatible with legacy runner format:
        {symbol, y_pred, y_pred_exit, y_proba, classes, returns, sym_test_df, feature_cols}
    """
    import src.data.target as target_module
    import src.features.engine as feature_engine_module
    from src.cache.feature_cache import FeatureCacheManager
    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.config_loader import load_config
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.env import get_results_dir, resolve_data_dir
    from src.features.engine import FeatureEngine
    from src.models.registry import detect_device
    from src.signal_adapter import canonicalize_predictions

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)

    split_cfg = cfg.split
    legacy_split = {
        "split": {
            "method": split_cfg.method,
            "train_years": split_cfg.train_years,
            "test_years": split_cfg.test_years,
            "gap_days": split_cfg.gap_days,
            "first_test_year": split_cfg.first_test_year,
            "last_test_year": split_cfg.last_test_year,
        },
        "target": cfg.target_dict(),
    }

    target_type = cfg.signals.target.type
    if str(target_type).lower() == "return_regression":
        raise ValueError("signals.target.type='return_regression' is not supported.")

    resolved_device = detect_device(device)
    print(f"    Training device: {resolved_device.upper()}")

    effective_model_type = cfg.entry_model_type()
    entry_model_extras = cfg.signals.entry_model.extras
    exit_model_dict = cfg.exit_model_dict()

    loader = DataLoader(abs_data_dir)
    splitter = WalkForwardSplitter(
        method=split_cfg.method,
        train_years=split_cfg.train_years,
        test_years=split_cfg.test_years,
        gap_days=split_cfg.gap_days,
        first_test_year=split_cfg.first_test_year,
        last_test_year=split_cfg.last_test_year,
    )
    target_gen = TargetGenerator.from_config(legacy_split)
    feature_set = cfg.feature_set()
    engine = FeatureEngine(feature_set=feature_set)

    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__]

    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=symbols,
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=legacy_split.get("target", {}),
        code_paths=code_paths,
    )
    if df is None:
        print(f"    Feature cache: MISS ({feature_set}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=symbols)
        df = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=symbols,
            timeframe=loader.timeframe,
            feature_set=feature_set,
            target_config=legacy_split.get("target", {}),
            code_paths=code_paths,
        )
        print(f"    Feature cache: STORED ({feature_set}) key={saved_key[:8]} format={saved_fmt}")
    else:
        print(f"    Feature cache: HIT ({feature_set}) key={cache_key[:8]}")

    df = target_gen.generate_for_all_symbols(df)

    if exit_model_dict:
        from src.data.target import TargetGenerator as _TG

        df = _TG.generate_exit_labels(
            df,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    feature_cols = engine.get_feature_columns(df)
    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in df.columns
    if has_exit:
        drop_cols.append("target_sell")
    df = df.dropna(subset=drop_cols)

    results: list[dict[str, Any]] = []
    target_cfg_dict = legacy_split.get("target", {})

    for _window, train_df, test_df in splitter.split(df):
        model = get_model(effective_model_type, device=device, **entry_model_extras)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        sell_model = None
        if has_exit:
            exit_model_cfg = cfg.signals.exit_model
            sell_model = get_exit_model(exit_model_cfg.type, device=device, **exit_model_cfg.extras)
            sell_model.fit(X_train, train_df["target_sell"].values.astype(int))

        for sym in test_df["symbol"].unique():
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred_raw = model.predict(X_sym)
            y_pred = canonicalize_predictions(y_pred_raw, target_cfg_dict)
            rets = sym_test["return_1d"].values

            y_proba = None
            classes = None
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_sym)
                    final_est = model.steps[-1][1] if hasattr(model, "steps") else model
                    classes = list(final_est.classes_)
            except Exception:
                y_proba = None

            results.append(
                {
                    "symbol": sym,
                    "y_pred": y_pred,
                    "y_pred_exit": (
                        sell_model.predict(X_sym).astype(int) if sell_model is not None else None
                    ),
                    "y_proba": y_proba,
                    "classes": classes,
                    "returns": rets,
                    "sym_test_df": sym_test,
                    "feature_cols": feature_cols,
                }
            )

    return results
