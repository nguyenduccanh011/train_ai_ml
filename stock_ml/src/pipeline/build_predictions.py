from __future__ import annotations

from src.config_loader import load_config
from src.env import get_results_dir, resolve_data_dir
from src.signal_adapter import canonicalize_predictions


def _build_predictions(
    symbols_list,
    feature_set,
    target_cfg,
    device,
    model_type=None,
    exit_model_cfg=None,
    model_extras=None,
):
    """Train ML model once and return per-(fold, symbol) predictions + test data.

    Args:
        model_type: override ML model (e.g. "xgboost"). None = read from config.
        exit_model_cfg: dict with keys {forward_window, loss_threshold} to train
            an independent exit model. None = no exit model.
        model_extras: optional model constructor overrides from model config.

    Returns:
        list of dict: each dict has keys
            {symbol, y_pred, y_pred_exit, returns, sym_test_df, feature_cols}
    """
    from pathlib import Path

    import numpy as np

    import src.data.target as target_module
    import src.features.engine as feature_engine_module
    from src.cache.feature_cache import FeatureCacheManager
    from src.components.models.registry import get_model
    from src.config_loader import get_training_device
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.features.engine import FeatureEngine
    from src.models.registry import detect_device

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)

    config = {
        "split": {
            "method": "walk_forward",
            "train_years": pipeline_cfg.get("train_years", 4),
            "test_years": pipeline_cfg.get("test_years", 1),
            "gap_days": 0,
            "first_test_year": pipeline_cfg.get("first_test_year", 2020),
            "last_test_year": pipeline_cfg.get("last_test_year", 2025),
        },
        "target": target_cfg
        or pipeline_cfg.get(
            "target",
            {
                "type": "trend_regime",
                "trend_method": "dual_ma",
                "short_window": 5,
                "long_window": 20,
                "classes": 3,
            },
        ),
    }
    target_type = config["target"].get("type", "trend_regime")
    if str(target_type).lower() == "return_regression":
        raise ValueError(
            "target.type='return_regression' is not supported by current classifier pipeline. "
            "Use a regression model path before enabling this target."
        )

    if device is None:
        device = get_training_device()
    resolved_device = detect_device(device)
    print(
        f"    Training device: {resolved_device.upper()}"
        f"{' (auto-detected)' if device == 'auto' else ''}"
    )

    effective_model_type = model_type or pipeline_cfg.get("model_type", "lightgbm")
    model_extras = model_extras or {}

    loader = DataLoader(abs_data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    engine = FeatureEngine(feature_set=feature_set)

    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__]
    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=symbols_list,
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=config.get("target", {}),
        code_paths=code_paths,
    )
    if df is None:
        print(f"    Feature cache: MISS ({feature_set}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=symbols_list)
        df = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=symbols_list,
            timeframe=loader.timeframe,
            feature_set=feature_set,
            target_config=config.get("target", {}),
            code_paths=code_paths,
        )
        print(f"    Feature cache: STORED ({feature_set}) key={saved_key[:8]} format={saved_fmt}")
    else:
        print(f"    Feature cache: HIT ({feature_set}) key={cache_key[:8]}")

    df = target_gen.generate_for_all_symbols(df)

    # Generate exit labels independently if exit_model_cfg provided
    if exit_model_cfg:
        from src.data.target import TargetGenerator as _TG

        df = _TG.generate_exit_labels(
            df,
            forward_window=exit_model_cfg.get("forward_window", 15),
            loss_threshold=exit_model_cfg.get("loss_threshold", 0.05),
        )

    feature_cols = engine.get_feature_columns(df)
    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in df.columns
    if has_exit:
        drop_cols.append("target_sell")
    df = df.dropna(subset=drop_cols)

    results = []
    for window, train_df, test_df in splitter.split(df):
        model = get_model(effective_model_type, device=device, **model_extras)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        sell_model = None
        if has_exit:
            sell_model = get_model(effective_model_type, device=device, **model_extras)
            sell_model.fit(X_train, train_df["target_sell"].values.astype(int))

        for sym in test_df["symbol"].unique():
            if sym not in symbols_list:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred_raw = model.predict(X_sym)
            y_pred = canonicalize_predictions(y_pred_raw, config["target"])
            rets = sym_test["return_1d"].values

            # V37c: capture proba + class mapping for per-profile threshold tuning
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
