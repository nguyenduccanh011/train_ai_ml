from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from paths import (
    CONFIG_PATH,
    CONTEXT_CACHE_TAG,
    FEATURE_CACHE_ROOT,
    MARKET,
    MODEL_PATH,
    STANDALONE_DATASET_DIR,
    STOCK_ML_ROOT,
    SYMBOLS_PATH,
    TIMEFRAME,
)

if str(STOCK_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCK_ML_ROOT))

import src.data.target as target_module
import src.features.engine as feature_engine_module
from src.cache.feature_cache import FeatureCacheManager
from src.components.exit_models.registry import get_exit_model
from src.components.models.registry import get_model
from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.pipeline import ExperimentConfig
from src.pipeline.trainer import build_prediction_cache

FULL_HISTORY_SCOPE = "full_history"
LAST_FOLD_SCOPE = "last_fold"
FOLD_CHAIN_SCOPE = "fold_chain"
VALID_TRAIN_SCOPES = {FULL_HISTORY_SCOPE, LAST_FOLD_SCOPE, FOLD_CHAIN_SCOPE}
VALID_CONTEXT_MODES = {CONTEXT_CACHE_TAG}


def _safe_ts_str(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        return ts.isoformat()
    except Exception:
        return str(value)


def _safe_date_str(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        return ts.date().isoformat()
    except Exception:
        return str(value)[:10]


def _load_symbols(path: Path) -> tuple[list[str], str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = [str(s).upper() for s in payload.get("symbols", []) if str(s).strip()]
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    cutoff_date = payload.get("window", {}).get("cutoff_date")
    return sorted(set(symbols)), str(cutoff_date) if cutoff_date else None


def _legacy_split(cfg: ExperimentConfig) -> dict[str, Any]:
    split_cfg = cfg.split
    return {
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


def _prepare_training_frame(
    cfg: ExperimentConfig,
    symbols: list[str],
    cutoff_date: str | None,
    context_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    if context_mode != CONTEXT_CACHE_TAG:
        raise ValueError(f"Unsupported context_mode={context_mode}; use {CONTEXT_CACHE_TAG}")
    if not STANDALONE_DATASET_DIR.exists():
        raise FileNotFoundError(f"Missing dataset directory: {STANDALONE_DATASET_DIR}")

    abs_data_dir = str(STANDALONE_DATASET_DIR)
    loader = DataLoader(abs_data_dir, timeframe=TIMEFRAME)
    feature_set = cfg.feature_set()
    cutoff_tag = (str(cutoff_date).strip() if cutoff_date else "none").replace("-", "")
    cache_feature_key = f"{feature_set}__{context_mode}__{MARKET}__{TIMEFRAME}__cutoff_{cutoff_tag}"
    legacy_split = _legacy_split(cfg)

    cache_mgr = FeatureCacheManager(str(FEATURE_CACHE_ROOT))
    code_paths = [feature_engine_module.__file__, target_module.__file__]
    feat, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=symbols,
        timeframe=loader.timeframe,
        feature_set=cache_feature_key,
        target_config=legacy_split.get("target", {}),
        code_paths=code_paths,
    )
    if feat is None:
        print(f"Feature cache MISS ({cache_feature_key}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=symbols, show_progress=True)
        raw_df["symbol"] = raw_df["symbol"].astype(str).str.upper()
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
        raw_df = raw_df.dropna(subset=["timestamp"]).copy()
        if cutoff_date:
            cutoff_ts = pd.Timestamp(cutoff_date, tz="UTC")
            raw_df = raw_df[raw_df["timestamp"] <= cutoff_ts].copy()
        engine = FeatureEngine(feature_set=feature_set)
        feat = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=feat,
            data_dir=abs_data_dir,
            symbols=symbols,
            timeframe=loader.timeframe,
            feature_set=cache_feature_key,
            target_config=legacy_split.get("target", {}),
            code_paths=code_paths,
        )
        print(f"Feature cache STORED key={saved_key[:8]} format={saved_fmt}")
    else:
        print(f"Feature cache HIT ({cache_feature_key}) key={cache_key[:8]}")

    feat = feat.copy()
    feat["symbol"] = feat["symbol"].astype(str).str.upper()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True, errors="coerce")
    feat = feat.dropna(subset=["timestamp"]).copy()
    if cutoff_date:
        cutoff_ts = pd.Timestamp(cutoff_date, tz="UTC")
        feat = feat[feat["timestamp"] <= cutoff_ts].copy()

    engine = FeatureEngine(feature_set=feature_set)
    feature_cols = engine.get_feature_columns(feat)
    target_gen = TargetGenerator.from_config(legacy_split)
    labeled = target_gen.generate_for_all_symbols(feat.copy())
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        labeled = TargetGenerator.generate_exit_labels(
            labeled,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    labeled["symbol"] = labeled["symbol"].astype(str).str.upper()
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], utc=True, errors="coerce")
    drop_cols = feature_cols + ["target"]
    if "target_sell" in labeled.columns:
        drop_cols.append("target_sell")
    labeled = labeled.dropna(subset=drop_cols).copy()
    return labeled, feature_cols


def _build_splitter(cfg: ExperimentConfig) -> WalkForwardSplitter:
    split_cfg = cfg.split
    return WalkForwardSplitter(
        method=split_cfg.method,
        train_years=split_cfg.train_years,
        test_years=split_cfg.test_years,
        gap_days=split_cfg.gap_days,
        first_test_year=split_cfg.first_test_year,
        last_test_year=split_cfg.last_test_year,
    )


def _select_last_fold_train_rows(
    cfg: ExperimentConfig, labeled_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    last_train_df: pd.DataFrame | None = None
    last_window_info: dict[str, str] | None = None
    for window, train_df, _test_df in _build_splitter(cfg).split(labeled_df, time_col="timestamp"):
        last_train_df = train_df.copy()
        last_window_info = {
            "window_label": window.label,
            "window_train_start": window.train_start.date().isoformat(),
            "window_train_end": window.train_end.date().isoformat(),
            "window_test_start": window.test_start.date().isoformat(),
            "window_test_end": window.test_end.date().isoformat(),
        }
    if last_train_df is None or last_window_info is None:
        raise ValueError("Cannot derive last fold training rows from current split and data range.")
    return last_train_df, last_window_info


def _collect_fold_train_test_sets(
    cfg: ExperimentConfig, labeled_df: pd.DataFrame
) -> list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    splitter = _build_splitter(cfg)
    rows: list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]] = []
    for window, train_df, test_df in splitter.split(labeled_df, time_col="timestamp"):
        rows.append(
            (
                {
                    "fold_id": len(rows),
                    "window_label": window.label,
                    "train_start": window.train_start.date().isoformat(),
                    "train_end": window.train_end.date().isoformat(),
                    "test_start": window.test_start.date().isoformat(),
                    "test_end": window.test_end.date().isoformat(),
                    "is_extended": False,
                },
                train_df.copy(),
                test_df.copy(),
            )
        )
    if not rows:
        raise ValueError("No valid walk-forward folds found from current split and data range.")
    return rows


def _fit_entry_exit(
    cfg: ExperimentConfig, train_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[Any, Any, dict[str, int]]:
    y_train = train_df["target"].values.astype(int)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Target has < 2 classes; cannot train entry model.")
    X_train = np.nan_to_num(train_df[feature_cols].values)
    entry_model = get_model(cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras)
    entry_model.fit(X_train, y_train)

    exit_model = None
    if cfg.exit_model_dict() and "target_sell" in train_df.columns:
        y_exit = train_df["target_sell"].values.astype(int)
        if len(np.unique(y_exit)) >= 2:
            exit_cfg = cfg.signals.exit_model
            exit_model = get_exit_model(exit_cfg.type, device="cpu", **exit_cfg.extras)
            exit_model.fit(X_train, y_exit)
    class_counts = {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()}
    return entry_model, exit_model, class_counts


def _build_fold_chain_artifact(
    cfg: ExperimentConfig, labeled_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    started = time.time()
    fold_models: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for info, train_df, test_df in _collect_fold_train_test_sets(cfg, labeled_df):
        if len(train_df) < 20 or len(test_df) < 10:
            skipped.append(
                {
                    **info,
                    "reason": "insufficient_rows",
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue
        try:
            entry_model, exit_model, class_counts = _fit_entry_exit(cfg, train_df, feature_cols)
        except ValueError as exc:
            skipped.append(
                {
                    **info,
                    "reason": str(exc),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue
        fold_models.append(
            {
                **info,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_end_date": _safe_ts_str(train_df["timestamp"].max()),
                "target_class_counts": class_counts,
                "entry_model": entry_model,
                "exit_model": exit_model,
            }
        )
    if not fold_models:
        raise ValueError("No valid fold models were trained for train_scope=fold_chain.")

    trained_windows = [
        {
            k: v
            for k, v in fold.items()
            if k not in {"entry_model", "exit_model", "target_class_counts"}
        }
        for fold in fold_models
    ]
    stats = {
        "window_count": int(len(fold_models)),
        "train_rows": int(fold_models[-1].get("train_rows", 0)),
        "train_end_date": str(fold_models[-1].get("train_end_date", "")),
        "build_runtime_sec": round(time.time() - started, 2),
    }
    return (
        fold_models,
        {"fold_chain": {"trained_windows": trained_windows, "skipped_windows": skipped}},
        stats,
    )


def build_model_artifact(
    cfg_path: Path,
    symbols_path: Path,
    model_out: Path,
    context_mode: str = CONTEXT_CACHE_TAG,
    train_scope: str = FOLD_CHAIN_SCOPE,
) -> dict[str, Any]:
    started = time.time()
    cfg = ExperimentConfig.from_yaml(cfg_path)
    symbols, cutoff_date = _load_symbols(symbols_path)
    if train_scope not in VALID_TRAIN_SCOPES:
        raise ValueError(
            f"Unsupported train_scope={train_scope}. Valid: {sorted(VALID_TRAIN_SCOPES)}"
        )

    labeled_df, feature_cols = _prepare_training_frame(cfg, symbols, cutoff_date, context_mode)
    if labeled_df.empty or len(labeled_df) < 20:
        raise ValueError("Not enough rows to train derivatives30m model.")

    entry_model = None
    exit_model = None
    fold_models = None
    scope_meta: dict[str, Any] = {}
    artifact_mode = "single_model_full_history"
    artifact_version = 1

    if train_scope == FOLD_CHAIN_SCOPE:
        artifact_mode = "multi_fold_chain"
        artifact_version = 2
        fold_models, scope_meta, stats = _build_fold_chain_artifact(cfg, labeled_df, feature_cols)
        scope_meta["prediction_cache"] = build_prediction_cache(cfg, symbols, device="cpu")
    else:
        train_df = labeled_df.copy()
        if train_scope == LAST_FOLD_SCOPE:
            train_df, fold_meta = _select_last_fold_train_rows(cfg, train_df)
            scope_meta = {"last_fold": fold_meta}
        entry_model, exit_model, class_counts = _fit_entry_exit(cfg, train_df, feature_cols)
        stats = {
            "window_count": 1,
            "train_rows": int(len(train_df)),
            "train_end_date": _safe_ts_str(train_df["timestamp"].max()),
            "target_class_counts": class_counts,
        }

    stats["build_runtime_sec"] = round(time.time() - started, 2)
    artifact = {
        "artifact_version": artifact_version,
        "mode": artifact_mode,
        "train_scope": train_scope,
        "context_mode": context_mode,
        "market": MARKET,
        "timeframe": TIMEFRAME,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(cfg_path),
        "config_name": cfg.name,
        "train_symbols": symbols,
        "train_symbol_count": len(symbols),
        "cutoff_date": cutoff_date,
        "feature_set": cfg.feature_set(),
        "feature_cols": feature_cols,
        "target_config": cfg.target_dict(),
        "entry_model_type": cfg.entry_model_type(),
        "entry_model_extras": dict(cfg.signals.entry_model.extras or {}),
        "exit_model_type": cfg.signals.exit_model.type if cfg.exit_model_dict() else "",
        "exit_model_extras": dict(cfg.signals.exit_model.extras or {})
        if cfg.exit_model_dict()
        else {},
        "entry_model": entry_model,
        "exit_model": exit_model,
        "stats": stats,
    }
    if fold_models is not None:
        artifact["fold_models"] = fold_models
    prediction_cache = scope_meta.pop("prediction_cache", None)
    if prediction_cache is not None:
        artifact["prediction_cache"] = prediction_cache
    if scope_meta:
        artifact["scope_meta"] = scope_meta

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VN derivatives 30m top1 model artifact.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to resolved config yaml")
    parser.add_argument("--symbols", default=str(SYMBOLS_PATH), help="Path to symbols JSON")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Output model pickle path")
    parser.add_argument(
        "--context-mode", default=CONTEXT_CACHE_TAG, choices=sorted(VALID_CONTEXT_MODES)
    )
    parser.add_argument(
        "--train-scope", default=FOLD_CHAIN_SCOPE, choices=sorted(VALID_TRAIN_SCOPES)
    )
    args = parser.parse_args()

    artifact = build_model_artifact(
        cfg_path=Path(args.config),
        symbols_path=Path(args.symbols),
        model_out=Path(args.output),
        context_mode=str(args.context_mode),
        train_scope=str(args.train_scope),
    )
    print(f"Saved model: {args.output}")
    print(
        "Summary:",
        json.dumps(
            {
                "market": artifact.get("market"),
                "timeframe": artifact.get("timeframe"),
                "mode": artifact.get("mode"),
                "train_scope": artifact.get("train_scope"),
                "train_symbol_count": artifact.get("train_symbol_count"),
                "train_rows": artifact.get("stats", {}).get("train_rows"),
                "train_end_date": artifact.get("stats", {}).get("train_end_date"),
                "window_count": artifact.get("stats", {}).get("window_count"),
                "build_runtime_sec": artifact.get("stats", {}).get("build_runtime_sec"),
            },
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
