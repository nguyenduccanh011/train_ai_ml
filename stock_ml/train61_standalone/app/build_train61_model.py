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
    MODEL_PATH,
    ROOT,
    STANDALONE_DATASET_DIR,
    TRAIN61_SYMBOLS_PATH,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

DEFAULT_CONFIG_PATH = CONFIG_PATH
DEFAULT_SYMBOLS_PATH = TRAIN61_SYMBOLS_PATH
DEFAULT_MODEL_PATH = MODEL_PATH
DEFAULT_CONTEXT_MODE = CONTEXT_CACHE_TAG
NO_CONTEXT_MODE = "no_context_v1"
VALID_CONTEXT_MODES = {DEFAULT_CONTEXT_MODE, NO_CONTEXT_MODE}
FULL_HISTORY_SCOPE = "full_history"
LAST_FOLD_SCOPE = "last_fold"
FOLD_CHAIN_SCOPE = "fold_chain"
VALID_TRAIN_SCOPES = {FULL_HISTORY_SCOPE, LAST_FOLD_SCOPE, FOLD_CHAIN_SCOPE}


def _safe_date_str(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        return ts.date().isoformat()
    except Exception:
        return str(value)[:10]


def _load_train_symbols(path: Path) -> tuple[list[str], str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = [str(s).upper() for s in payload.get("symbols", []) if str(s).strip()]
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    cutoff_date = payload.get("window", {}).get("cutoff_date")
    return sorted(set(symbols)), str(cutoff_date) if cutoff_date else None


def _prepare_training_frame(
    cfg: ExperimentConfig,
    train_symbols: list[str],
    cutoff_date: str | None,
    context_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not STANDALONE_DATASET_DIR.exists():
        raise FileNotFoundError(
            "Missing standalone dataset directory. "
            f"Expected: {STANDALONE_DATASET_DIR}. "
            "This standalone trainer only reads local data inside train61_standalone/data."
        )
    abs_data_dir = str(STANDALONE_DATASET_DIR)

    loader = DataLoader(abs_data_dir)
    feature_set = cfg.feature_set()
    cutoff_tag = (str(cutoff_date).strip() if cutoff_date else "none").replace("-", "")
    cache_feature_key = f"{feature_set}__{context_mode}__cutoff_{cutoff_tag}"
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

    cache_mgr = FeatureCacheManager(str(FEATURE_CACHE_ROOT))
    code_paths = [feature_engine_module.__file__, target_module.__file__]
    feat, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=train_symbols,
        timeframe=loader.timeframe,
        feature_set=cache_feature_key,
        target_config=legacy_split.get("target", {}),
        code_paths=code_paths,
    )
    if feat is None:
        print(f"Feature cache MISS ({cache_feature_key}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=train_symbols, show_progress=True)
        raw_df["symbol"] = raw_df["symbol"].astype(str).str.upper()
        if cutoff_date:
            cutoff_ts = pd.Timestamp(cutoff_date, tz="UTC")
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
            raw_df = raw_df[raw_df["timestamp"] <= cutoff_ts].copy()
        engine = FeatureEngine(feature_set=feature_set)
        feat = engine.compute_for_all_symbols(raw_df)
        if context_mode == DEFAULT_CONTEXT_MODE:
            context_data = loader.load_all_context()
            if not context_data:
                raise ValueError(
                    "No context data found under context_features; cannot train context-aware model."
                )
            feat = engine.add_market_context(feat, context_data)
        saved_key, saved_fmt = cache_mgr.save(
            df=feat,
            data_dir=abs_data_dir,
            symbols=train_symbols,
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


def _select_last_fold_train_rows(
    cfg: ExperimentConfig, labeled_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    split_cfg = cfg.split
    splitter = WalkForwardSplitter(
        method=split_cfg.method,
        train_years=split_cfg.train_years,
        test_years=split_cfg.test_years,
        gap_days=split_cfg.gap_days,
        first_test_year=split_cfg.first_test_year,
        last_test_year=split_cfg.last_test_year,
    )
    last_window_info: dict[str, str] | None = None
    last_train_df: pd.DataFrame | None = None
    for window, train_df, _test_df in splitter.split(labeled_df, time_col="timestamp"):
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


def _window_info_dict(
    *,
    label: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    is_extended: bool,
) -> dict[str, Any]:
    return {
        "window_label": label,
        "window_train_start": train_start.date().isoformat(),
        "window_train_end": train_end.date().isoformat(),
        "window_test_start": test_start.date().isoformat(),
        "window_test_end": test_end.date().isoformat(),
        "is_extended": bool(is_extended),
    }


def _collect_fold_train_test_sets(
    cfg: ExperimentConfig,
    labeled_df: pd.DataFrame,
    *,
    extend_to_latest: bool,
) -> list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    splitter = _build_splitter(cfg)
    rows: list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]] = []

    for window, train_df, test_df in splitter.split(labeled_df, time_col="timestamp"):
        info = _window_info_dict(
            label=window.label,
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            is_extended=False,
        )
        rows.append((info, train_df.copy(), test_df.copy()))

    if not rows:
        raise ValueError("No valid walk-forward folds found from current split and data range.")

    if not extend_to_latest:
        for idx, (info, _tr, _te) in enumerate(rows):
            info["fold_id"] = idx
        return rows

    latest_ts = pd.Timestamp(labeled_df["timestamp"].max())
    if pd.isna(latest_ts):
        for idx, (info, _tr, _te) in enumerate(rows):
            info["fold_id"] = idx
        return rows

    latest_year = int(latest_ts.year)
    next_test_year = int(splitter.last_test_year) + 1
    if latest_year >= next_test_year:
        for test_year in range(next_test_year, latest_year + 1):
            test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
            test_end = pd.Timestamp(f"{test_year + splitter.test_years - 1}-12-31", tz="UTC")
            if splitter.method == "expanding":
                train_start = pd.Timestamp(
                    f"{splitter.first_test_year - splitter.train_years}-01-01", tz="UTC"
                )
            else:
                train_start = pd.Timestamp(f"{test_year - splitter.train_years}-01-01", tz="UTC")
            train_end = test_start - pd.Timedelta(days=splitter.gap_days + 1)
            train_mask = (labeled_df["timestamp"] >= train_start) & (
                labeled_df["timestamp"] <= train_end
            )
            test_mask = (labeled_df["timestamp"] >= test_start) & (
                labeled_df["timestamp"] <= test_end
            )
            ext_train_df = labeled_df[train_mask].copy()
            ext_test_df = labeled_df[test_mask].copy()
            if len(ext_train_df) == 0 or len(ext_test_df) == 0:
                continue
            label = f"train_{train_start.year}-{train_end.year}_test_{test_year}"
            info = _window_info_dict(
                label=label,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_extended=True,
            )
            rows.append((info, ext_train_df, ext_test_df))

    rows.sort(key=lambda x: str(x[0].get("window_test_start", "")))
    for idx, (info, _tr, _te) in enumerate(rows):
        info["fold_id"] = idx
    return rows


def _build_fold_chain_artifact(
    cfg: ExperimentConfig,
    labeled_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    started = time.time()
    fold_sets = _collect_fold_train_test_sets(cfg, labeled_df, extend_to_latest=True)
    fold_models: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    exit_cfg = cfg.signals.exit_model
    has_exit_target = "target_sell" in labeled_df.columns

    for info, train_df, test_df in fold_sets:
        fold_label = str(info.get("window_label", ""))
        if len(train_df) < 20 or len(test_df) < 10:
            skipped.append(
                {
                    "window_label": fold_label,
                    "reason": "insufficient_rows",
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue

        y_train = train_df["target"].values.astype(int)
        if len(np.unique(y_train)) < 2:
            skipped.append(
                {
                    "window_label": fold_label,
                    "reason": "single_target_class",
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue

        X_train = np.nan_to_num(train_df[feature_cols].values)
        entry_model = get_model(
            cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras
        )
        entry_model.fit(X_train, y_train)

        exit_model = None
        if cfg.exit_model_dict() and has_exit_target:
            y_exit = train_df["target_sell"].values.astype(int)
            if len(np.unique(y_exit)) >= 2:
                exit_model = get_exit_model(exit_cfg.type, device="cpu", **exit_cfg.extras)
                exit_model.fit(X_train, y_exit)

        class_counts = {
            str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()
        }
        fold_models.append(
            {
                "fold_id": int(info.get("fold_id", len(fold_models))),
                "window_label": fold_label,
                "is_extended": bool(info.get("is_extended", False)),
                "train_start": str(info.get("window_train_start", "")),
                "train_end": str(info.get("window_train_end", "")),
                "test_start": str(info.get("window_test_start", "")),
                "test_end": str(info.get("window_test_end", "")),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_end_date": _safe_date_str(train_df["timestamp"].max()),
                "target_class_counts": class_counts,
                "entry_model": entry_model,
                "exit_model": exit_model,
            }
        )

    if not fold_models:
        raise ValueError("No valid fold models were trained for train_scope=fold_chain.")

    latest_ts = pd.Timestamp(labeled_df["timestamp"].max())
    latest_year = int(latest_ts.year) if not pd.isna(latest_ts) else None
    configured_last_year = int(cfg.split.last_test_year)
    trained_windows = [
        {
            "fold_id": int(fold["fold_id"]),
            "window_label": str(fold["window_label"]),
            "is_extended": bool(fold["is_extended"]),
            "train_start": str(fold["train_start"]),
            "train_end": str(fold["train_end"]),
            "test_start": str(fold["test_start"]),
            "test_end": str(fold["test_end"]),
            "train_rows": int(fold["train_rows"]),
            "test_rows": int(fold["test_rows"]),
            "train_end_date": str(fold["train_end_date"]),
        }
        for fold in fold_models
    ]
    scope_meta = {
        "fold_chain": {
            "configured_last_test_year": configured_last_year,
            "latest_data_year": latest_year,
            "extended_fold_count": int(sum(1 for f in fold_models if bool(f.get("is_extended")))),
            "trained_fold_count": int(len(fold_models)),
            "skipped_fold_count": int(len(skipped)),
            "trained_windows": trained_windows,
            "skipped_windows": skipped,
        }
    }

    latest_fold = max(
        fold_models,
        key=lambda fold: str(fold.get("test_end", "")),
    )
    stats = {
        "window_count": int(len(fold_models)),
        "train_rows": int(latest_fold.get("train_rows", 0)),
        "train_end_date": str(latest_fold.get("train_end_date", "")),
        "build_runtime_sec": round(time.time() - started, 2),
    }
    return fold_models, scope_meta, stats


def build_model_artifact(
    cfg_path: Path,
    symbols_path: Path,
    model_out: Path,
    context_mode: str = DEFAULT_CONTEXT_MODE,
    train_scope: str = FULL_HISTORY_SCOPE,
) -> dict[str, Any]:
    started = time.time()
    cfg = ExperimentConfig.from_yaml(cfg_path)
    train_symbols, cutoff_date = _load_train_symbols(symbols_path)
    if context_mode not in VALID_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported context_mode={context_mode}. Valid: {sorted(VALID_CONTEXT_MODES)}"
        )
    if train_scope not in VALID_TRAIN_SCOPES:
        raise ValueError(
            f"Unsupported train_scope={train_scope}. Valid: {sorted(VALID_TRAIN_SCOPES)}"
        )
    labeled_df, feature_cols = _prepare_training_frame(
        cfg,
        train_symbols,
        cutoff_date,
        context_mode=context_mode,
    )
    if labeled_df.empty or len(labeled_df) < 20:
        raise ValueError("Not enough rows to train pooled train61 model.")

    train_scope_meta: dict[str, Any] = {}
    artifact_mode = "single_model_full_history"
    artifact_version = 2
    entry_model = None
    exit_model = None
    stats: dict[str, Any]

    if train_scope == FOLD_CHAIN_SCOPE:
        artifact_mode = "multi_fold_chain"
        artifact_version = 3
        fold_models, train_scope_meta, stats = _build_fold_chain_artifact(
            cfg, labeled_df, feature_cols
        )
    else:
        train_df = labeled_df.copy()
        if train_scope == LAST_FOLD_SCOPE:
            train_df, fold_meta = _select_last_fold_train_rows(cfg, train_df)
            train_scope_meta = {"last_fold": fold_meta}

        if train_df.empty or len(train_df) < 20:
            raise ValueError("Not enough rows to train pooled train61 model.")

        y_train = train_df["target"].values.astype(int)
        if len(np.unique(y_train)) < 2:
            raise ValueError("Target has < 2 classes; cannot train entry model.")

        X_train = np.nan_to_num(train_df[feature_cols].values)
        entry_model = get_model(
            cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras
        )
        entry_model.fit(X_train, y_train)

        has_exit = bool(cfg.exit_model_dict()) and "target_sell" in train_df.columns
        if has_exit:
            y_exit = train_df["target_sell"].values.astype(int)
            if len(np.unique(y_exit)) >= 2:
                exit_cfg = cfg.signals.exit_model
                exit_model = get_exit_model(exit_cfg.type, device="cpu", **exit_cfg.extras)
                exit_model.fit(X_train, y_exit)

        train_end_date = _safe_date_str(train_df["timestamp"].max())
        class_counts = {
            str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()
        }
        stats = {
            "window_count": 1,
            "train_rows": int(len(train_df)),
            "train_end_date": train_end_date,
            "target_class_counts": class_counts,
            "build_runtime_sec": 0.0,
        }

    stats["build_runtime_sec"] = round(time.time() - started, 2)

    artifact = {
        "artifact_version": artifact_version,
        "mode": artifact_mode,
        "train_scope": train_scope,
        "context_mode": context_mode,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(cfg_path),
        "config_name": cfg.name,
        "train_symbols": train_symbols,
        "train_symbol_count": len(train_symbols),
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
    if train_scope == FOLD_CHAIN_SCOPE:
        artifact["fold_models"] = fold_models
    if train_scope_meta:
        artifact["scope_meta"] = train_scope_meta

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pooled train61 single-model artifact.")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to resolved config yaml"
    )
    parser.add_argument(
        "--symbols",
        default=str(DEFAULT_SYMBOLS_PATH),
        help="Path to JSON containing train61 symbols",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MODEL_PATH),
        help="Output model pickle path",
    )
    parser.add_argument(
        "--context-mode",
        default=DEFAULT_CONTEXT_MODE,
        choices=sorted(VALID_CONTEXT_MODES),
        help="Feature context mode: with_context_v1 or no_context_v1",
    )
    parser.add_argument(
        "--train-scope",
        default=FULL_HISTORY_SCOPE,
        choices=sorted(VALID_TRAIN_SCOPES),
        help=(
            "Training scope: full_history, last_fold "
            "(last walk-forward train window), or fold_chain "
            "(train one model per fold + auto-extend folds to latest data year)"
        ),
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
                "mode": artifact.get("mode"),
                "train_scope": artifact.get("train_scope"),
                "context_mode": artifact.get("context_mode"),
                "train_symbol_count": artifact["train_symbol_count"],
                "train_rows": artifact["stats"]["train_rows"],
                "train_end_date": artifact["stats"]["train_end_date"],
                "window_count": artifact["stats"].get("window_count", 1),
                "build_runtime_sec": artifact["stats"]["build_runtime_sec"],
            },
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
