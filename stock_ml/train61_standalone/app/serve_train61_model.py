from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from paths import (
    BASE_DATA_DIR,
    CONFIG_PATH,
    CONTEXT_CACHE_TAG,
    FEATURE_CACHE_ROOT,
    ROOT,
    SIGNAL_CACHE_DIR,
    STANDALONE_DATASET_DIR,
    TRAIN61_SYMBOLS_PATH,
    VIZ_DIR,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.data.target as target_module
import src.features.engine as feature_engine_module
from model_registry import DEFAULT_MODEL, MODELS, get_model_cfg, model_availability
from src.cache.feature_cache import FeatureCacheManager
from src.data.loader import DataLoader
from src.features.engine import FeatureEngine
from src.pipeline import ExperimentConfig, Pipeline
from src.signal_adapter import canonicalize_predictions

app = Flask(__name__, static_folder=str(VIZ_DIR), static_url_path="")


@dataclass
class JobState:
    status: str = "running"
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


jobs: dict[str, JobState] = {}
signal_results: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
artifact_lock = threading.Lock()
artifact_cached_by_model: dict[str, dict[str, Any]] = {}
context_lock = threading.Lock()
context_cached_by_mode: dict[str, dict[str, pd.DataFrame]] = {}
backtest_trades_cached_by_model: dict[str, pd.DataFrame] = {}
pooled_global_payloads_cached_by_model: dict[str, dict[str, dict[str, Any]]] = {}


def _json_response(payload: Any):
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response


def _html_response(filename: str):
    response = send_from_directory(VIZ_DIR, filename)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _dataset_dir_for_model(model_cfg: dict[str, Any] | None = None) -> Path:
    if model_cfg and model_cfg.get("dataset"):
        return Path(model_cfg["dataset"])
    return STANDALONE_DATASET_DIR


def _symbols_path_for_model(model_cfg: dict[str, Any] | None = None) -> Path:
    if model_cfg and model_cfg.get("symbols"):
        return Path(model_cfg["symbols"])
    return TRAIN61_SYMBOLS_PATH


def _load_symbols_from_path(path: Path) -> list[str]:
    payload = _read_json(path)
    symbols = [str(s).upper() for s in payload.get("symbols", []) if str(s).strip()]
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    return sorted(set(symbols))


def _load_symbols_for_model(model_cfg: dict[str, Any] | None = None) -> list[str]:
    return _load_symbols_from_path(_symbols_path_for_model(model_cfg))


def _symbol_file_name(symbol: str) -> str:
    return f"{symbol.upper()}.json"


def _cache_key(model_id: str, symbol: str) -> str:
    """Generate cache key for model+symbol combination"""
    return f"{model_id}:{symbol.upper()}"


def _signal_cache_path(symbol: str, model_id: str = DEFAULT_MODEL) -> Path:
    """Get signal cache path for specific model and symbol"""
    return SIGNAL_CACHE_DIR / model_id / _symbol_file_name(symbol)


def _legacy_signal_cache_path(symbol: str) -> Path:
    """Backward-compatible cache file path used before per-model folders."""
    return SIGNAL_CACHE_DIR / _symbol_file_name(symbol)


def _version_key_for_model(model_id: str) -> str:
    return model_id


def _marker_color_for_model(model_id: str) -> str:
    model_cfg = MODELS.get(model_id, {})
    return str(model_cfg.get("color", "#00E5FF"))


_get_model_cfg = get_model_cfg


def _normalize_cached_payload_for_model(model_id: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    version_key = _version_key_for_model(model_id)
    if f"{version_key}_stats" in payload:
        payload.setdefault("model_id", version_key)
        return payload
    # Legacy payload from single-model server used train61_model_* keys.
    legacy_key = "train61_model"
    if model_id == DEFAULT_MODEL and f"{legacy_key}_stats" in payload:
        converted = dict(payload)
        for suffix in ("markers", "trades", "stats"):
            old_key = f"{legacy_key}_{suffix}"
            new_key = f"{version_key}_{suffix}"
            if old_key in converted and new_key not in converted:
                converted[new_key] = converted.pop(old_key)
        converted["model_id"] = version_key
        converted.setdefault("source", "pkl:legacy_cache")
        return converted
    return payload


def _base_ohlcv_path(symbol: str) -> Path:
    return BASE_DATA_DIR / _symbol_file_name(symbol)


def _data_loader(data_dir: Path | None = None) -> DataLoader:
    # DataLoader requires parquet/csv dataset layout:
    #   symbol=XXX/timeframe=1D/data.csv
    resolved_data_dir = data_dir or STANDALONE_DATASET_DIR
    if not resolved_data_dir.exists():
        raise FileNotFoundError(
            "Missing standalone dataset directory. "
            f"Expected: {resolved_data_dir}. "
            "This standalone runtime only reads local data inside train61_standalone/data."
        )
    abs_data_dir = str(resolved_data_dir)
    return DataLoader(abs_data_dir)


def _load_context_data(loader: DataLoader, context_mode: str) -> dict[str, pd.DataFrame]:
    cached = context_cached_by_mode.get(context_mode)
    if cached is not None:
        return cached
    with context_lock:
        cached = context_cached_by_mode.get(context_mode)
        if cached is not None:
            return cached
        context_data = loader.load_all_context()
        if not context_data:
            raise ValueError(
                "No context data found under context_features for context-aware model."
            )
        context_cached_by_mode[context_mode] = context_data
        return context_data


def _load_cfg(model_id: str, model_cfg: dict[str, Any] | None = None) -> ExperimentConfig:
    cfg_dict = model_cfg if model_cfg is not None else _get_model_cfg(model_id)
    cfg_path = Path(cfg_dict.get("config", CONFIG_PATH))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = ExperimentConfig.from_yaml(cfg_path)
    cfg.name = model_id
    return cfg


def _latest_data_year_for_symbols(symbols: list[str], data_dir: Path | None = None) -> int | None:
    if not symbols:
        return None
    loader = _data_loader(data_dir)
    raw_df = loader.load_all(symbols=symbols)
    if raw_df is None or raw_df.empty:
        return None
    ts = pd.to_datetime(raw_df.get("timestamp"), utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return None
    return int(ts.max().year)


def _align_cfg_last_test_year_to_data(
    cfg: ExperimentConfig, symbols: list[str], data_dir: Path | None = None
) -> ExperimentConfig:
    latest_year = _latest_data_year_for_symbols(symbols, data_dir)
    if latest_year is None:
        return cfg

    configured_year = int(cfg.split.last_test_year)
    if latest_year > configured_year:
        cfg.split.last_test_year = latest_year
        if cfg.execution is not None:
            cfg.execution.split.last_test_year = latest_year
        print(
            "[INFO] Auto-extended split.last_test_year "
            f"from {configured_year} to {latest_year} based on latest market data."
        )
    return cfg


def _load_backtest_trades(model_id: str, model_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cached = backtest_trades_cached_by_model.get(model_id)
    if cached is not None:
        return cached

    cfg_dict = model_cfg if model_cfg is not None else _get_model_cfg(model_id)
    if str(cfg_dict.get("type", "")).lower() != "backtest_replay":
        raise ValueError(f"Model '{model_id}' is not a backtest_replay model.")

    trades_path = Path(cfg_dict.get("trades_path", ""))
    if not trades_path.exists():
        raise FileNotFoundError(f"Backtest trades file not found: {trades_path}")

    df = pd.read_csv(trades_path)
    if "symbol" not in df.columns:
        raise ValueError(f"Invalid trades file (missing symbol column): {trades_path}")

    df["symbol"] = df["symbol"].astype(str).str.upper()
    backtest_trades_cached_by_model[model_id] = df
    return df


def _load_model_artifact(model_id: str, model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cached = artifact_cached_by_model.get(model_id)
    if cached is not None:
        return cached

    cfg_dict = model_cfg if model_cfg is not None else _get_model_cfg(model_id)
    model_type = str(cfg_dict.get("type", "")).lower()
    if model_type != "pkl":
        raise ValueError(f"Model '{model_id}' is not a pkl artifact model.")
    model_path = Path(cfg_dict.get("path", ""))

    with artifact_lock:
        cached = artifact_cached_by_model.get(model_id)
        if cached is not None:
            return cached
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run: python train61_standalone/app/build_train61_model.py"
            )
        with model_path.open("rb") as f:
            artifact = pickle.load(f)
        artifact.setdefault("context_mode", CONTEXT_CACHE_TAG)
        artifact_cached_by_model[model_id] = artifact
        return artifact


def _load_train61_symbols() -> list[str]:
    return _load_symbols_from_path(TRAIN61_SYMBOLS_PATH)


def _load_ohlcv(symbol: str, model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    symbol = symbol.upper()
    path = _base_ohlcv_path(symbol)
    if path.exists():
        return _read_json(path)

    loader = _data_loader(_dataset_dir_for_model(model_cfg))
    df = loader.load_symbol(symbol)
    ohlcv = []
    for row in df.to_dict("records"):
        ts = row.get("timestamp") or row.get("date")
        if hasattr(ts, "date"):
            ts = ts.date().isoformat()
        ohlcv.append(
            {
                "time": str(ts)[:10],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0) or 0),
            }
        )
    payload = {"symbol": symbol, "ohlcv": ohlcv}
    _write_json(path, payload)
    return payload


def _load_all_symbols(model_cfg: dict[str, Any] | None = None) -> list[str]:
    symbols = {
        path.stem.upper()
        for path in BASE_DATA_DIR.glob("*.json")
        if path.stem.lower() != "index" and path.stem.strip()
    }
    try:
        loader = _data_loader(_dataset_dir_for_model(model_cfg))
        symbols.update(str(s).upper() for s in loader.symbols)
    except Exception:
        pass
    return sorted(symbols)


def _safe_date_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        return ts.date().isoformat()
    except Exception:
        return str(value)[:10]


def _trade_date_key(trade: dict[str, Any]) -> pd.Timestamp:
    raw = trade.get("entry_date") or trade.get("exit_date")
    if not raw:
        return pd.Timestamp.min
    try:
        ts = pd.Timestamp(raw)
        if pd.isna(ts):
            return pd.Timestamp.min
        return ts
    except Exception:
        return pd.Timestamp.min


def _normalize_open_position(
    symbol: str,
    trades: list[dict[str, Any]],
    realtime: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    symbol = symbol.upper()
    if not trades:
        return trades, None

    latest_bar_date = ""
    latest_close = None
    mode = ""
    fold_boundary_end_dates: set[str] = set()
    if realtime:
        mode = str(realtime.get("mode", "") or "")
        latest_bar_date = str(realtime.get("latest_bar_date", "") or "")
        try:
            raw_close = realtime.get("latest_close")
            latest_close = float(raw_close) if raw_close is not None else None
        except Exception:
            latest_close = None
        boundary_vals = realtime.get("fold_boundary_end_dates", [])
        if isinstance(boundary_vals, list):
            fold_boundary_end_dates = {str(v)[:10] for v in boundary_vals if str(v).strip()}

    updated = [dict(t) for t in trades]
    for idx, t in enumerate(updated):
        exit_reason = str(t.get("exit_reason", "") or "").lower()
        exit_date = str(t.get("exit_date", "") or "")[:10]
        if exit_reason == "end" and exit_date and exit_date in fold_boundary_end_dates:
            t["exit_reason"] = "rollover"
            t["is_fold_rollover"] = True
            updated[idx] = t

    last_idx = max(range(len(trades)), key=lambda i: _trade_date_key(trades[i]))
    t = updated[last_idx]

    exit_reason = str(t.get("exit_reason", "") or "").lower()
    exit_date = str(t.get("exit_date", "") or "")[:10]
    should_mark_open = bool(
        latest_bar_date and exit_reason == "end" and exit_date == latest_bar_date
    )
    if mode == "pooled_global_rerun_realtime" and exit_reason in {"end", "rollover"}:
        should_mark_open = True

    if not should_mark_open:
        return updated, None

    entry_price = None
    try:
        raw_entry = t.get("entry_price")
        entry_price = float(raw_entry) if raw_entry is not None else None
    except Exception:
        entry_price = None
    unrealized = t.get("pnl_pct", 0.0)
    if latest_close is not None and entry_price is not None and entry_price > 0:
        try:
            unrealized = round((latest_close / entry_price - 1.0) * 100.0, 2)
        except Exception:
            unrealized = t.get("pnl_pct", 0.0)

    t["exit_date"] = ""
    t["exit_reason"] = "open"
    t["is_open_position"] = True
    t["pnl_pct"] = unrealized
    updated[last_idx] = t

    open_position = {
        "symbol": symbol,
        "entry_date": str(t.get("entry_date", "") or "")[:10],
        "latest_bar_date": latest_bar_date,
        "holding_days": t.get("holding_days", 0),
        "unrealized_pnl_pct": unrealized,
        "position_size": t.get("position_size", 1.0),
        "entry_trend": t.get("entry_trend"),
    }
    return updated, open_position


def _to_date_or_none(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _stitch_rollover_trades(
    trades: list[dict[str, Any]],
    realtime: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not trades:
        return trades
    fold_boundary_end_dates: set[str] = set()
    if realtime:
        boundary_vals = realtime.get("fold_boundary_end_dates", [])
        if isinstance(boundary_vals, list):
            fold_boundary_end_dates = {str(v)[:10] for v in boundary_vals if str(v).strip()}
    if not fold_boundary_end_dates:
        return [dict(t) for t in trades]

    ordered = [dict(t) for t in trades]
    ordered.sort(key=_trade_date_key)
    stitched: list[dict[str, Any]] = []
    i = 0
    while i < len(ordered):
        cur = dict(ordered[i])
        while i + 1 < len(ordered):
            nxt = dict(ordered[i + 1])
            cur_exit_reason = str(cur.get("exit_reason", "") or "").lower()
            cur_exit_date = str(cur.get("exit_date", "") or "")[:10]
            if (
                cur_exit_reason not in {"end", "rollover"}
                or cur_exit_date not in fold_boundary_end_dates
            ):
                break

            cur_exit_ts = _to_date_or_none(cur.get("exit_date"))
            nxt_entry_ts = _to_date_or_none(nxt.get("entry_date"))
            if cur_exit_ts is None or nxt_entry_ts is None:
                break

            # Allow normal calendar gaps across year-end / holidays.
            day_gap = (nxt_entry_ts - cur_exit_ts).days
            if day_gap < 0 or day_gap > 45:
                break

            merged = dict(cur)
            merged["exit_date"] = nxt.get("exit_date", "")
            merged["exit_reason"] = nxt.get("exit_reason", "")
            if nxt.get("exit_price") is not None:
                merged["exit_price"] = nxt.get("exit_price")

            p1 = _to_float_or_none(cur.get("pnl_pct"))
            p2 = _to_float_or_none(nxt.get("pnl_pct"))
            if p1 is not None and p2 is not None:
                merged["pnl_pct"] = round(
                    ((1.0 + p1 / 100.0) * (1.0 + p2 / 100.0) - 1.0) * 100.0, 2
                )
            elif p2 is not None:
                merged["pnl_pct"] = p2

            h1 = cur.get("holding_days")
            h2 = nxt.get("holding_days")
            if isinstance(h1, (int, float)) and isinstance(h2, (int, float)):
                merged["holding_days"] = int(h1) + int(h2)
            else:
                merged["holding_days"] = nxt.get("holding_days", h1)

            merged["stitched"] = True
            merged["stitch_segments"] = int(cur.get("stitch_segments", 1)) + int(
                nxt.get("stitch_segments", 1)
            )

            cur = merged
            i += 1
        stitched.append(cur)
        i += 1
    return stitched


def _records_for_symbol(trades: Any, symbol: str) -> list[dict[str, Any]]:
    if isinstance(trades, pd.DataFrame):
        if trades.empty:
            return []
        records = trades.to_dict("records")
    elif isinstance(trades, list):
        records = trades
    else:
        return []

    symbol = symbol.upper()
    rows = []
    for record in records:
        if not isinstance(record, dict):
            if hasattr(record, "__dict__"):
                record = vars(record)
            else:
                continue
        if str(record.get("symbol", symbol)).upper() == symbol:
            rows.append(dict(record))
    return rows


def _build_signal_payload(
    symbol: str,
    trades_df: Any,
    version_key: str,
    model_color: str,
    source: str,
    realtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.export.unified_export import compute_stats, make_markers, select_fields

    symbol = symbol.upper()
    records = _records_for_symbol(trades_df, symbol)
    trades = select_fields(pd.DataFrame(records)) if records else []
    for trade in trades:
        trade["symbol"] = symbol
    trades = _stitch_rollover_trades(trades, realtime=realtime)
    trades, open_position = _normalize_open_position(symbol, trades, realtime=realtime)

    payload = {
        "symbol": symbol,
        "model_id": version_key,
        f"{version_key}_markers": make_markers(trades, version_key, model_color, "arrowUp"),
        f"{version_key}_trades": trades,
        f"{version_key}_stats": compute_stats(trades, version_key),
        "has_open_position": open_position is not None,
        "open_position": open_position,
        "source": source,
    }
    if realtime is not None:
        payload["realtime"] = realtime
    return payload


def _load_symbol_features(
    symbol: str,
    feature_set: str,
    target_cfg: dict[str, Any],
    *,
    context_mode: str,
    cache_namespace: str,
) -> pd.DataFrame:
    symbol = symbol.upper()
    loader = _data_loader()
    abs_data_dir = str(loader.data_dir)
    cache_mgr = FeatureCacheManager(str(FEATURE_CACHE_ROOT))
    cache_feature_key = f"{feature_set}__{context_mode}__{cache_namespace}"
    code_paths = [feature_engine_module.__file__, target_module.__file__]
    feat_df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=[symbol],
        timeframe=loader.timeframe,
        feature_set=cache_feature_key,
        target_config=target_cfg,
        code_paths=code_paths,
    )
    if feat_df is None:
        print(f"    Feature cache MISS ({cache_feature_key}) key={cache_key[:8]} symbol={symbol}")
        raw_df = loader.load_symbol(symbol, use_cache=False).copy()
        raw_df["symbol"] = raw_df.get("symbol", symbol)
        raw_df["symbol"] = raw_df["symbol"].astype(str).str.upper()
        engine = FeatureEngine(feature_set=feature_set)
        feat_df = engine.compute_for_all_symbols(raw_df)
        if context_mode == CONTEXT_CACHE_TAG:
            context_data = _load_context_data(loader, context_mode=context_mode)
            feat_df = engine.add_market_context(feat_df, context_data)
        saved_key, saved_fmt = cache_mgr.save(
            df=feat_df,
            data_dir=abs_data_dir,
            symbols=[symbol],
            timeframe=loader.timeframe,
            feature_set=cache_feature_key,
            target_config=target_cfg,
            code_paths=code_paths,
        )
        print(f"    Feature cache STORED key={saved_key[:8]} format={saved_fmt} symbol={symbol}")
    else:
        print(f"    Feature cache HIT ({cache_feature_key}) key={cache_key[:8]} symbol={symbol}")
    return feat_df.copy()


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _cache_item_latest_ts_value(item: dict[str, Any]) -> int:
    df = item.get("sym_test_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return -1
    time_col = "timestamp" if "timestamp" in df.columns else "date"
    ts = _to_utc_timestamp(df.iloc[-1].get(time_col))
    if ts is None:
        return -1
    return int(ts.value)


def _predict_cache_item_from_models(
    *,
    symbol: str,
    sym_df: pd.DataFrame,
    feature_cols: list[str],
    target_cfg: dict[str, Any],
    entry_model: Any,
    exit_model: Any,
    model_mode: str,
    train_rows: int,
    train_end_date: str,
    window_label: str = "",
    test_start: str = "",
    test_end: str = "",
) -> dict[str, Any] | None:
    if entry_model is None or sym_df.empty:
        return None

    X = np.nan_to_num(sym_df[feature_cols].values)
    y_pred = canonicalize_predictions(entry_model.predict(X), target_cfg)

    y_proba = None
    classes = None
    try:
        if hasattr(entry_model, "predict_proba"):
            y_proba = entry_model.predict_proba(X)
            final_est = entry_model.steps[-1][1] if hasattr(entry_model, "steps") else entry_model
            classes = list(final_est.classes_)
    except Exception:
        y_proba = None

    y_pred_exit = None
    if exit_model is not None:
        try:
            y_pred_exit = exit_model.predict(X).astype(int)
        except Exception:
            y_pred_exit = None

    payload = {
        "symbol": symbol,
        "y_pred": y_pred,
        "y_pred_exit": y_pred_exit,
        "y_proba": y_proba,
        "classes": classes,
        "returns": sym_df["return_1d"].values,
        "sym_test_df": sym_df,
        "feature_cols": feature_cols,
        "train_rows": int(train_rows),
        "train_end_date": str(train_end_date),
        "model_mode": str(model_mode),
    }
    if window_label:
        payload["window_label"] = window_label
    if test_start:
        payload["test_start"] = test_start
    if test_end:
        payload["test_end"] = test_end
    return payload


def _build_prediction_cache_from_model(
    model_id: str,
    symbol: str,
    model_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    artifact = _load_model_artifact(model_id, model_cfg=model_cfg)
    symbol = symbol.upper()
    feature_cols = list(artifact.get("feature_cols", []))
    if not feature_cols:
        return []

    # Backward-compat: some historical artifacts may have inconsistent
    # context_mode metadata while feature columns clearly require context.
    context_prefixes = ("vnindex_", "hnxindex_", "hnxupcom_", "vn30f1m_", "vn30f2m_")
    inferred_need_context = any(str(col).startswith(context_prefixes) for col in feature_cols)
    artifact_context_mode = str(artifact.get("context_mode", CONTEXT_CACHE_TAG))
    effective_context_mode = (
        CONTEXT_CACHE_TAG
        if inferred_need_context and artifact_context_mode != CONTEXT_CACHE_TAG
        else artifact_context_mode
    )

    feat_df = _load_symbol_features(
        symbol=symbol,
        feature_set=str(artifact.get("feature_set", "leading")),
        target_cfg=dict(artifact.get("target_config", {})),
        context_mode=effective_context_mode,
        cache_namespace=model_id,
    )
    if feat_df.empty:
        return []
    feat_df["symbol"] = feat_df["symbol"].astype(str).str.upper()
    feat_df = feat_df[feat_df["symbol"] == symbol].copy()
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"], utc=True, errors="coerce")
    feat_df = feat_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    feat_df = feat_df.dropna(subset=feature_cols).reset_index(drop=True)
    if feat_df.empty:
        return []

    if "return_1d" not in feat_df.columns:
        feat_df["return_1d"] = feat_df["close"].pct_change().fillna(0.0)

    target_cfg = dict(artifact.get("target_config", {}))
    model_mode = str(artifact.get("mode", "single_model_full_history"))
    fold_models = artifact.get("fold_models")
    if isinstance(fold_models, list) and fold_models:
        cache_items: list[dict[str, Any]] = []
        for fold in fold_models:
            if not isinstance(fold, dict):
                continue
            entry_model = fold.get("entry_model")
            if entry_model is None:
                continue

            test_start_ts = _to_utc_timestamp(fold.get("test_start"))
            test_end_ts = _to_utc_timestamp(fold.get("test_end"))
            fold_df = feat_df
            if test_start_ts is not None:
                fold_df = fold_df[fold_df["timestamp"] >= test_start_ts]
            if test_end_ts is not None:
                fold_df = fold_df[fold_df["timestamp"] <= test_end_ts]
            fold_df = fold_df.reset_index(drop=True)
            if fold_df.empty:
                continue

            item = _predict_cache_item_from_models(
                symbol=symbol,
                sym_df=fold_df,
                feature_cols=feature_cols,
                target_cfg=target_cfg,
                entry_model=entry_model,
                exit_model=fold.get("exit_model"),
                model_mode=model_mode,
                train_rows=int(fold.get("train_rows", 0)),
                train_end_date=str(fold.get("train_end_date", "")),
                window_label=str(fold.get("window_label", "")),
                test_start=_safe_date_str(test_start_ts),
                test_end=_safe_date_str(test_end_ts),
            )
            if item is not None:
                cache_items.append(item)

        cache_items.sort(key=_cache_item_latest_ts_value)
        return cache_items

    entry_model = artifact.get("entry_model")
    item = _predict_cache_item_from_models(
        symbol=symbol,
        sym_df=feat_df,
        feature_cols=feature_cols,
        target_cfg=target_cfg,
        entry_model=entry_model,
        exit_model=artifact.get("exit_model"),
        model_mode=model_mode,
        train_rows=int(artifact.get("stats", {}).get("train_rows", 0)),
        train_end_date=str(artifact.get("stats", {}).get("train_end_date", "")),
    )
    return [item] if item is not None else []


def _build_live_prediction_cache_on_demand(
    model_id: str,
    cfg: ExperimentConfig,
    symbol: str,
) -> list[dict[str, Any]]:
    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.data.target import TargetGenerator

    symbol = symbol.upper()
    loader = _data_loader()
    abs_data_dir = str(loader.data_dir)

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

    loader = DataLoader(abs_data_dir)
    feature_set = cfg.feature_set()
    engine = FeatureEngine(feature_set=feature_set)
    target_gen = TargetGenerator.from_config(legacy_split)

    cache_mgr = FeatureCacheManager(str(FEATURE_CACHE_ROOT))
    code_paths = [feature_engine_module.__file__, target_module.__file__]
    cache_feature_key = f"{feature_set}__{model_id}"
    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=[symbol],
        timeframe=loader.timeframe,
        feature_set=cache_feature_key,
        target_config=legacy_split.get("target", {}),
        code_paths=code_paths,
    )
    if df is None:
        print(
            f"    Live feature cache MISS ({cache_feature_key}) key={cache_key[:8]} symbol={symbol}"
        )
        raw_df = loader.load_all(symbols=[symbol])
        df = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=[symbol],
            timeframe=loader.timeframe,
            feature_set=cache_feature_key,
            target_config=legacy_split.get("target", {}),
            code_paths=code_paths,
        )
        print(
            f"    Live feature cache STORED key={saved_key[:8]} format={saved_fmt} symbol={symbol}"
        )
    else:
        print(
            f"    Live feature cache HIT ({cache_feature_key}) key={cache_key[:8]} symbol={symbol}"
        )

    feature_df = df.copy()
    feature_cols = engine.get_feature_columns(feature_df)

    train_df = target_gen.generate_for_all_symbols(feature_df.copy())
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        train_df = TargetGenerator.generate_exit_labels(
            train_df,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in train_df.columns
    if has_exit:
        drop_cols.append("target_sell")
    train_df = train_df.dropna(subset=drop_cols)
    train_sym_df = train_df[train_df["symbol"].astype(str).str.upper() == symbol].reset_index(
        drop=True
    )
    if len(train_sym_df) < 20:
        return []

    infer_df = feature_df[feature_df["symbol"].astype(str).str.upper() == symbol].reset_index(
        drop=True
    )
    infer_df = infer_df.dropna(subset=feature_cols).reset_index(drop=True)
    if infer_df.empty:
        return []

    if "return_1d" not in infer_df.columns:
        infer_df["return_1d"] = infer_df["close"].pct_change().fillna(0.0)

    X_train = np.nan_to_num(train_sym_df[feature_cols].values)
    y_train = train_sym_df["target"].values.astype(int)

    model = get_model(cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras)
    model.fit(X_train, y_train)

    X_infer = np.nan_to_num(infer_df[feature_cols].values)
    y_pred = canonicalize_predictions(model.predict(X_infer), legacy_split.get("target", {}))

    sell_model = None
    if has_exit:
        exit_model_cfg = cfg.signals.exit_model
        sell_model = get_exit_model(exit_model_cfg.type, device="cpu", **exit_model_cfg.extras)
        sell_model.fit(X_train, train_sym_df["target_sell"].values.astype(int))

    y_proba = None
    classes = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_infer)
            final_est = model.steps[-1][1] if hasattr(model, "steps") else model
            classes = list(final_est.classes_)
    except Exception:
        y_proba = None

    train_time_col = "timestamp" if "timestamp" in train_sym_df.columns else "date"
    train_end_date = ""
    if train_time_col in train_sym_df.columns and not train_sym_df.empty:
        train_end_date = _safe_date_str(train_sym_df.iloc[-1].get(train_time_col))

    return [
        {
            "symbol": symbol,
            "y_pred": y_pred,
            "y_pred_exit": sell_model.predict(X_infer).astype(int)
            if sell_model is not None
            else None,
            "y_proba": y_proba,
            "classes": classes,
            "returns": infer_df["return_1d"].values,
            "sym_test_df": infer_df,
            "feature_cols": feature_cols,
            "train_rows": len(train_sym_df),
            "train_end_date": train_end_date,
            "model_mode": "on_demand",
        }
    ]


def _build_realtime_summary(cache_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not cache_items:
        return {"mode": "realtime", "error": "No inference rows"}

    valid_items: list[dict[str, Any]] = []
    for item in cache_items:
        df = item.get("sym_test_df")
        y_pred = item.get("y_pred")
        if isinstance(df, pd.DataFrame) and not df.empty and y_pred is not None and len(y_pred) > 0:
            valid_items.append(item)
    if not valid_items:
        return {"mode": "realtime", "error": "No predictions"}

    last_item = max(valid_items, key=_cache_item_latest_ts_value)
    df = last_item.get("sym_test_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"mode": "realtime", "error": "No inference rows"}

    y_pred = last_item.get("y_pred")
    y_pred_exit = last_item.get("y_pred_exit")
    y_proba = last_item.get("y_proba")
    classes = last_item.get("classes")
    if y_pred is None or len(y_pred) == 0:
        return {"mode": "realtime", "error": "No predictions"}

    time_col = "timestamp" if "timestamp" in df.columns else "date"
    latest_bar_date = _safe_date_str(df.iloc[-1].get(time_col))
    latest_close = None
    if "close" in df.columns:
        try:
            latest_close = float(df.iloc[-1]["close"])
        except Exception:
            latest_close = None

    buy_proba = None
    if y_proba is not None and classes is not None and len(classes) > 0 and len(y_proba) > 0:
        try:
            classes_list = list(classes)
            if 1 in classes_list:
                idx = classes_list.index(1)
                buy_proba = float(y_proba[-1][idx])
        except Exception:
            buy_proba = None

    infer_rows = sum(
        int(len(item["sym_test_df"]))
        for item in cache_items
        if isinstance(item.get("sym_test_df"), pd.DataFrame)
    )
    summary = {
        "mode": "realtime",
        "model_mode": str(last_item.get("model_mode", "single_model_full_history")),
        "latest_bar_date": latest_bar_date,
        "latest_close": latest_close,
        "entry_signal_for_next_bar": int(y_pred[-1]),
        "entry_signal_used_for_latest_bar": int(y_pred[-2]) if len(y_pred) >= 2 else None,
        "exit_signal_for_next_bar": (
            int(y_pred_exit[-1]) if y_pred_exit is not None and len(y_pred_exit) > 0 else None
        ),
        "buy_proba_for_next_bar": buy_proba,
        "train_rows": int(last_item.get("train_rows", 0)),
        "train_end_date": str(last_item.get("train_end_date", "")),
        "inference_rows": infer_rows,
        "window_count": len(valid_items),
    }
    if last_item.get("window_label"):
        summary["active_window"] = str(last_item.get("window_label", ""))
    if last_item.get("test_start"):
        summary["active_window_test_start"] = str(last_item.get("test_start", ""))
    if last_item.get("test_end"):
        summary["active_window_test_end"] = str(last_item.get("test_end", ""))
    return summary


def _generate_signal_pkl(model_id: str, symbol: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = _load_cfg(model_id, model_cfg=model_cfg)
    cache_items = _build_prediction_cache_from_model(model_id, symbol, model_cfg)
    if not cache_items:
        raise ValueError(f"No model predictions for {symbol}. Check symbol data availability.")
    result = Pipeline(cfg, symbols=[symbol], device="cpu", prediction_cache=cache_items).run()
    artifact = _load_model_artifact(model_id, model_cfg=model_cfg)
    artifact_mode = str(artifact.get("mode", "single_model_full_history"))
    return _build_signal_payload(
        symbol=symbol,
        trades_df=result.trades_df,
        version_key=_version_key_for_model(model_id),
        model_color=_marker_color_for_model(model_id),
        source=f"pkl:{artifact_mode}",
        realtime=_build_realtime_summary(cache_items),
    )


def _generate_signal_on_demand(
    model_id: str, symbol: str, model_cfg: dict[str, Any]
) -> dict[str, Any]:
    cfg = _load_cfg(model_id, model_cfg=model_cfg)
    cache_items = _build_live_prediction_cache_on_demand(model_id, cfg, symbol)
    if not cache_items:
        raise ValueError(f"Not enough data to generate realtime signal for {symbol}")
    result = Pipeline(cfg, symbols=[symbol], device="cpu", prediction_cache=cache_items).run()
    return _build_signal_payload(
        symbol=symbol,
        trades_df=result.trades_df,
        version_key=_version_key_for_model(model_id),
        model_color=_marker_color_for_model(model_id),
        source="live_on_demand",
        realtime=_build_realtime_summary(cache_items),
    )


def _generate_signal_backtest_replay(
    model_id: str, symbol: str, model_cfg: dict[str, Any]
) -> dict[str, Any]:
    df = _load_backtest_trades(model_id, model_cfg=model_cfg)
    symbol = symbol.upper()
    sym_df = df[df["symbol"] == symbol].copy()
    return _build_signal_payload(
        symbol=symbol,
        trades_df=sym_df,
        version_key=_version_key_for_model(model_id),
        model_color=_marker_color_for_model(model_id),
        source="backtest_replay",
        realtime={"mode": "backtest_replay", "as_of": "2025-12-31"},
    )


def _build_pooled_global_payloads(
    model_id: str, model_cfg: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    cached = pooled_global_payloads_cached_by_model.get(model_id)
    if cached is not None:
        return cached

    symbols = _load_symbols_for_model(model_cfg)
    data_dir = _dataset_dir_for_model(model_cfg)
    cfg = _load_cfg(model_id, model_cfg=model_cfg)
    cfg = _align_cfg_last_test_year_to_data(cfg, symbols, data_dir)
    previous_data_dir = os.environ.get("STOCK_DATA_DIR")
    os.environ["STOCK_DATA_DIR"] = str(data_dir.resolve())
    try:
        result = Pipeline(cfg, symbols=symbols, device="cpu").run()
    finally:
        if previous_data_dir is None:
            os.environ.pop("STOCK_DATA_DIR", None)
        else:
            os.environ["STOCK_DATA_DIR"] = previous_data_dir
    trades_df = result.trades_df.copy()
    fold_boundary_end_dates: list[str] = []
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        tmp = trades_df.copy()
        tmp["exit_reason"] = tmp.get("exit_reason", "").astype(str).str.lower()
        tmp["exit_date"] = pd.to_datetime(tmp.get("exit_date"), errors="coerce")
        end_rows = tmp[(tmp["exit_reason"] == "end") & (tmp["exit_date"].notna())].copy()
        if not end_rows.empty:
            counts = (
                end_rows.groupby(end_rows["exit_date"].dt.strftime("%Y-%m-%d"))
                .size()
                .sort_values(ascending=False)
            )
            boundary_min_count = max(5, int(len(symbols) * 0.2))
            fold_boundary_end_dates = [
                str(day) for day, cnt in counts.items() if int(cnt) >= boundary_min_count
            ]

    payloads: dict[str, dict[str, Any]] = {}
    version_key = _version_key_for_model(model_id)
    model_color = _marker_color_for_model(model_id)
    for sym in symbols:
        ohlcv = _load_ohlcv(sym, model_cfg=model_cfg)
        latest_bar_date = ""
        latest_close = None
        rows = ohlcv.get("ohlcv", []) if isinstance(ohlcv, dict) else []
        if isinstance(rows, list) and rows:
            last_bar = rows[-1] if isinstance(rows[-1], dict) else {}
            latest_bar_date = str(last_bar.get("time", "") or "")[:10]
            try:
                raw_close = last_bar.get("close")
                latest_close = float(raw_close) if raw_close is not None else None
            except Exception:
                latest_close = None
        payloads[sym] = _build_signal_payload(
            symbol=sym,
            trades_df=trades_df,
            version_key=version_key,
            model_color=model_color,
            source="pooled_global_rerun",
            realtime={
                "mode": "pooled_global_rerun_realtime",
                "latest_bar_date": latest_bar_date,
                "latest_close": latest_close,
                "fold_boundary_end_dates": fold_boundary_end_dates,
            },
        )
    pooled_global_payloads_cached_by_model[model_id] = payloads
    return payloads


def _generate_signal_pooled_global_rerun(
    model_id: str, symbol: str, model_cfg: dict[str, Any]
) -> dict[str, Any]:
    payloads = _build_pooled_global_payloads(model_id, model_cfg)
    symbol = symbol.upper()
    if symbol not in payloads:
        return _build_signal_payload(
            symbol=symbol,
            trades_df=[],
            version_key=_version_key_for_model(model_id),
            model_color=_marker_color_for_model(model_id),
            source="pooled_global_rerun",
            realtime={"mode": "pooled_global_rerun", "warning": "symbol_not_in_model_universe"},
        )
    return payloads[symbol]


def _generate_signal_for_model(model_id: str, symbol: str) -> dict[str, Any]:
    model_cfg = _get_model_cfg(model_id)
    model_type = str(model_cfg.get("type", "")).lower()
    symbol = symbol.upper()
    if model_type == "pkl":
        return _generate_signal_pkl(model_id, symbol, model_cfg)
    if model_type == "on_demand":
        return _generate_signal_on_demand(model_id, symbol, model_cfg)
    if model_type == "backtest_replay":
        return _generate_signal_backtest_replay(model_id, symbol, model_cfg)
    if model_type == "pooled_global_rerun":
        return _generate_signal_pooled_global_rerun(model_id, symbol, model_cfg)
    raise ValueError(f"Unsupported model type '{model_type}' for model '{model_id}'")


def _generate_signal_threaded(model_id: str, symbol: str) -> None:
    symbol = symbol.upper()
    cache_key = _cache_key(model_id, symbol)
    try:
        with jobs_lock:
            jobs.setdefault(cache_key, JobState())

        model_cfg = _get_model_cfg(model_id)
        _load_ohlcv(symbol, model_cfg=model_cfg)
        payload = _generate_signal_for_model(model_id, symbol)
        _write_json(_signal_cache_path(symbol, model_id), payload)

        with jobs_lock:
            signal_results[cache_key] = payload
            state = jobs[cache_key]
            state.status = "done"
            state.finished_at = time.time()
    except Exception as exc:
        with jobs_lock:
            state = jobs.setdefault(cache_key, JobState())
            state.status = "error"
            state.error = str(exc)
            state.finished_at = time.time()


def _ensure_job_for_model(model_id: str, symbol: str) -> JobState:
    symbol = symbol.upper()
    cache_key = _cache_key(model_id, symbol)
    with jobs_lock:
        state = jobs.get(cache_key)
        if state and state.status == "running":
            return state
        signal_results.pop(cache_key, None)
        state = JobState()
        jobs[cache_key] = state
    threading.Thread(target=_generate_signal_threaded, args=(model_id, symbol), daemon=True).start()
    return state


@app.get("/")
def root():
    return _html_response("train61_model.html")


@app.get("/train61_model.html")
def page():
    return _html_response("train61_model.html")


@app.get("/api/model-info")
def api_model_info():
    model_id = str(request.args.get("model_id", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    try:
        model_cfg = _get_model_cfg(model_id)
    except ValueError as exc:
        return _json_response({"error": str(exc), "model_id": model_id}), 400

    model_type = str(model_cfg.get("type", "")).lower()
    if model_type == "pkl":
        artifact = _load_model_artifact(model_id, model_cfg=model_cfg)
        payload = {
            "model_id": model_id,
            "config_name": artifact.get("config_name", ""),
            "mode": artifact.get("mode", ""),
            "context_mode": artifact.get("context_mode", ""),
            "train_symbol_count": artifact.get("train_symbol_count", 0),
            "feature_set": artifact.get("feature_set", ""),
            "window_count": artifact.get("stats", {}).get("window_count", 1),
            "train_rows": artifact.get("stats", {}).get("train_rows", 0),
            "train_end_date": artifact.get("stats", {}).get("train_end_date", ""),
            "created_at": artifact.get("created_at", ""),
            "model_path": str(model_cfg.get("path", "")),
            "config_path": str(model_cfg.get("config", "")),
        }
        return _json_response(payload)
    return _json_response(
        {
            "model_id": model_id,
            "type": model_type,
            "config_path": str(model_cfg.get("config", "")),
            "dataset_path": str(_dataset_dir_for_model(model_cfg)),
            "symbols_path": str(_symbols_path_for_model(model_cfg)),
            "symbol_count": len(_load_symbols_for_model(model_cfg)),
        }
    )


@app.get("/api/models")
def api_models():
    rows = []
    for model_id, model_cfg in MODELS.items():
        availability = model_availability(model_id)
        rows.append(
            {
                "id": model_id,
                "type": str(model_cfg.get("type", "")),
                "label": str(model_cfg.get("label", model_id)),
                "color": str(model_cfg.get("color", "#00E5FF")),
                "path": str(model_cfg.get("path", "")),
                "config": str(model_cfg.get("config", "")),
                "dataset": str(model_cfg.get("dataset", "")),
                "symbols": str(model_cfg.get("symbols", "")),
                "is_default": model_id == DEFAULT_MODEL,
                "available": bool(availability["available"]),
                "missing": availability["missing"],
            }
        )
    return _json_response(rows)


@app.get("/api/symbols")
def api_symbols():
    model_id = str(request.args.get("model_id", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    try:
        model_cfg = _get_model_cfg(model_id)
    except ValueError as exc:
        return _json_response({"error": str(exc), "model_id": model_id}), 400

    version_key = _version_key_for_model(model_id)
    model_symbols_set: set[str] = set()
    try:
        model_symbols_set = set(_load_symbols_for_model(model_cfg))
    except Exception:
        model_symbols_set = set()

    rows = []
    for symbol in _load_all_symbols(model_cfg):
        cache_key = _cache_key(model_id, symbol)
        disk_cache = _signal_cache_path(symbol, model_id)
        legacy_cache = _legacy_signal_cache_path(symbol)
        has_legacy_cache = model_id == DEFAULT_MODEL and legacy_cache.exists()
        payload: dict[str, Any] | None = None
        with jobs_lock:
            payload = signal_results.get(cache_key)
        if payload is None and disk_cache.exists():
            try:
                payload = _read_json(disk_cache)
            except Exception:
                payload = None
        if payload is None and has_legacy_cache:
            try:
                payload = _normalize_cached_payload_for_model(model_id, _read_json(legacy_cache))
            except Exception:
                payload = None
        if isinstance(payload, dict):
            payload = _normalize_cached_payload_for_model(model_id, payload)
        stats = payload.get(f"{version_key}_stats", {}) if isinstance(payload, dict) else {}
        trades = int(stats.get("total_trades", 0) or 0)
        pnl = float(stats.get("total_pnl_pct", 0.0) or 0.0)
        wr = float(stats.get("win_rate", 0.0) or 0.0)

        rows.append(
            {
                "symbol": symbol,
                f"{version_key}_trades": trades,
                f"{version_key}_pnl": pnl,
                f"{version_key}_wr": wr,
                "is_train61": symbol in model_symbols_set,
                "is_model_symbol": symbol in model_symbols_set,
                "cached": payload is not None,
                "has_historical_export": disk_cache.exists() or has_legacy_cache,
            }
        )
    rows.sort(key=lambda row: (not row["is_train61"], row["symbol"]))
    return _json_response(rows)


@app.get("/api/ohlcv/<symbol>")
def api_ohlcv(symbol: str):
    try:
        model_id = str(request.args.get("model_id", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
        try:
            model_cfg = _get_model_cfg(model_id)
        except ValueError:
            model_cfg = None
        return _json_response(_load_ohlcv(symbol, model_cfg=model_cfg))
    except Exception as exc:
        return _json_response({"error": str(exc), "symbol": symbol.upper()}), 404


@app.get("/api/signal/<model_id>/<symbol>")
def api_signal_with_model(model_id: str, symbol: str):
    try:
        _get_model_cfg(model_id)
    except ValueError as exc:
        return _json_response(
            {"error": str(exc), "model_id": model_id, "symbol": symbol.upper()}
        ), 400

    symbol = symbol.upper()
    refresh = str(request.args.get("refresh", "0") or "0").strip().lower() in {"1", "true", "yes"}
    cache_key = _cache_key(model_id, symbol)
    with jobs_lock:
        payload = signal_results.get(cache_key)
        if payload is not None and not refresh:
            return _json_response(payload)
        if refresh:
            signal_results.pop(cache_key, None)

    disk_cache = _signal_cache_path(symbol, model_id)
    legacy_cache = _legacy_signal_cache_path(symbol)
    if disk_cache.exists() and not refresh:
        try:
            payload = _normalize_cached_payload_for_model(model_id, _read_json(disk_cache))
            with jobs_lock:
                signal_results[cache_key] = payload
            return _json_response(payload)
        except Exception:
            pass
    if model_id == DEFAULT_MODEL and legacy_cache.exists() and not refresh:
        try:
            payload = _normalize_cached_payload_for_model(model_id, _read_json(legacy_cache))
            _write_json(disk_cache, payload)
            with jobs_lock:
                signal_results[cache_key] = payload
            return _json_response(payload)
        except Exception:
            pass

    if refresh:
        disk_cache.unlink(missing_ok=True)
        if model_id == DEFAULT_MODEL:
            legacy_cache.unlink(missing_ok=True)

    state = _ensure_job_for_model(model_id, symbol)
    return _json_response(
        {
            "status": state.status,
            "job_id": cache_key,
            "model_id": model_id,
            "symbol": symbol,
            "source": str(MODELS[model_id].get("type", "unknown")),
        }
    ), 202


@app.get("/api/signal/<model_id>/<symbol>/status")
def api_signal_status_with_model(model_id: str, symbol: str):
    try:
        _get_model_cfg(model_id)
    except ValueError as exc:
        return _json_response(
            {"error": str(exc), "model_id": model_id, "symbol": symbol.upper()}
        ), 400

    cache_key = _cache_key(model_id, symbol.upper())
    with jobs_lock:
        if cache_key in signal_results:
            return _json_response({"status": "done", "job_id": cache_key, "model_id": model_id})
        state = jobs.get(cache_key)
        if not state:
            return _json_response(
                {"status": "not_started", "job_id": cache_key, "model_id": model_id}
            )
        return _json_response(
            {
                "status": state.status,
                "job_id": cache_key,
                "model_id": model_id,
                "symbol": symbol.upper(),
                "error": state.error,
                "elapsed_seconds": round((state.finished_at or time.time()) - state.started_at, 1),
            }
        )


@app.get("/api/signal/<symbol>/status")
def api_signal_status(symbol: str):
    return api_signal_status_with_model(DEFAULT_MODEL, symbol)


@app.get("/api/signal/<symbol>")
def api_signal(symbol: str):
    return api_signal_with_model(DEFAULT_MODEL, symbol)


if __name__ == "__main__":
    print("=" * 60)
    print("TRAIN61 STANDALONE SERVER")
    print("=" * 60)
    print(f"Default model: {DEFAULT_MODEL}")
    for model_id, model_cfg in MODELS.items():
        model_type = str(model_cfg.get("type", "unknown"))
        path_or_cfg = model_cfg.get("path") or model_cfg.get("config") or ""
        print(f"- {model_id} [{model_type}] -> {path_or_cfg}")
    print("URL: http://127.0.0.1:5012")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5012, debug=True, threaded=True)
