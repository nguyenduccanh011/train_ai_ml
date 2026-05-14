from __future__ import annotations

import json
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, Response, request
from model_registry import DEFAULT_MODEL, MODELS, get_model_config, model_availability
from paths import (
    BASE_DATA_DIR,
    CONFIG_PATH,
    CONTEXT_CACHE_TAG,
    FEATURE_CACHE_ROOT,
    MARKET,
    SIGNAL_CACHE_DIR,
    STANDALONE_DATASET_DIR,
    STOCK_ML_ROOT,
    SYMBOLS_PATH,
    TIMEFRAME,
    VIZ_DIR,
)

if str(STOCK_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCK_ML_ROOT))

import src.data.target as target_module
import src.features.engine as feature_engine_module
from src.cache.feature_cache import FeatureCacheManager
from src.data.loader import DataLoader
from src.export.unified_export import compute_stats, make_markers, select_fields
from src.features.engine import FeatureEngine
from src.pipeline import ExperimentConfig, Pipeline
from src.signal_adapter import canonicalize_predictions

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=2)
artifact_cache: dict[str, dict[str, Any]] = {}
signal_results: dict[str, dict[str, Any]] = {}
jobs: dict[str, JobState] = {}
jobs_lock = threading.Lock()


@dataclass
class JobState:
    status: str = "running"
    error: str | None = None
    started_at: float = time.time()
    finished_at: float | None = None


def _json_response(payload: Any, status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False, default=str),
        status=status,
        mimetype="application/json",
    )


def _html_response(name: str) -> Response:
    path = VIZ_DIR / name
    if not path.exists():
        return Response(f"Missing page: {path}", status=404)
    return Response(path.read_text(encoding="utf-8"), mimetype="text/html")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _symbol_file_name(symbol: str) -> str:
    return f"{symbol.upper()}.json"


def _cache_key(model_id: str, symbol: str) -> str:
    return f"{model_id}:{symbol.upper()}"


def _signal_cache_path(symbol: str, model_id: str) -> Path:
    return SIGNAL_CACHE_DIR / model_id / _symbol_file_name(symbol)


def _data_loader() -> DataLoader:
    if not STANDALONE_DATASET_DIR.exists():
        raise FileNotFoundError(f"Missing dataset directory: {STANDALONE_DATASET_DIR}")
    return DataLoader(str(STANDALONE_DATASET_DIR), timeframe=TIMEFRAME)


def _source_path(symbol: str) -> Path | None:
    symbol = symbol.upper()
    candidates = [
        STANDALONE_DATASET_DIR
        / "all_symbols"
        / f"symbol={symbol}"
        / f"timeframe={TIMEFRAME}"
        / "data.csv",
        STANDALONE_DATASET_DIR / f"symbol={symbol}" / f"timeframe={TIMEFRAME}" / "data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _format_ts(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.isoformat()


def _data_fingerprint(symbol: str) -> dict[str, Any]:
    path = _source_path(symbol)
    fp: dict[str, Any] = {"symbol": symbol.upper(), "source_path": str(path) if path else None}
    if path is None:
        return fp
    stat = path.stat()
    fp["size"] = stat.st_size
    fp["mtime_ns"] = stat.st_mtime_ns
    try:
        df = pd.read_csv(path)
        if not df.empty:
            last = df.iloc[-1]
            fp["latest_bar_time"] = _format_ts(last.get("timestamp"))
            fp["latest_close"] = float(last.get("close")) if pd.notna(last.get("close")) else None
    except Exception as exc:
        fp["warning"] = str(exc)
    return fp


def _load_symbols() -> list[str]:
    payload = _read_json(SYMBOLS_PATH)
    symbols = [str(s).upper() for s in payload.get("symbols", []) if str(s).strip()]
    return sorted(set(symbols))


def _load_cfg(model_id: str, model_cfg: dict[str, Any] | None = None) -> ExperimentConfig:
    cfg_dict = model_cfg if model_cfg is not None else get_model_config(model_id)[1]
    cfg_path = Path(cfg_dict.get("config", CONFIG_PATH))
    cfg = ExperimentConfig.from_yaml(cfg_path)
    cfg.name = model_id
    return cfg


def _load_model_artifact(model_id: str, model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cached = artifact_cache.get(model_id)
    if cached is not None:
        return cached
    cfg_dict = model_cfg if model_cfg is not None else get_model_config(model_id)[1]
    model_path = Path(cfg_dict.get("path", ""))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run build_derivatives30m_model.py first."
        )
    with model_path.open("rb") as f:
        artifact = pickle.load(f)
    artifact.setdefault("context_mode", CONTEXT_CACHE_TAG)
    artifact_cache[model_id] = artifact
    return artifact


def _load_ohlcv(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper()
    path = BASE_DATA_DIR / _symbol_file_name(symbol)
    source_fp = _data_fingerprint(symbol)
    if path.exists():
        cached = _read_json(path)
        if cached.get("data_fingerprint") == source_fp:
            return cached

    df = _data_loader().load_symbol(symbol)
    ohlcv = []
    for row in df.to_dict("records"):
        ohlcv.append(
            {
                "time": _format_ts(row.get("timestamp") or row.get("date")),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0) or 0),
            }
        )
    payload = {
        "symbol": symbol,
        "market": MARKET,
        "timeframe": TIMEFRAME,
        "ohlcv": ohlcv,
        "data_fingerprint": source_fp,
        "latest_bar_time": source_fp.get("latest_bar_time"),
    }
    _write_json(path, payload)
    return payload


def _load_universe_features(
    symbols: list[str], artifact: dict[str, Any], model_id: str
) -> pd.DataFrame:
    symbols = sorted({str(symbol).upper() for symbol in symbols})
    loader = _data_loader()
    feature_set = str(artifact.get("feature_set", "all_features"))
    target_cfg = dict(artifact.get("target_config", {}))
    cache_key_name = f"{feature_set}__{CONTEXT_CACHE_TAG}__{model_id}__{TIMEFRAME}__universe"
    cache_mgr = FeatureCacheManager(str(FEATURE_CACHE_ROOT))
    feat_df, cache_key = cache_mgr.load(
        data_dir=str(loader.data_dir),
        symbols=symbols,
        timeframe=loader.timeframe,
        feature_set=cache_key_name,
        target_config=target_cfg,
        code_paths=[feature_engine_module.__file__, target_module.__file__],
    )
    if feat_df is None:
        print(
            f"Feature cache MISS ({cache_key_name}) key={cache_key[:8]} symbols={','.join(symbols)}"
        )
        raw_df = loader.load_all(symbols=symbols, show_progress=False).copy()
        raw_df["symbol"] = raw_df["symbol"].astype(str).str.upper()
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
        engine = FeatureEngine(feature_set=feature_set)
        feat_df = engine.compute_for_all_symbols(raw_df)
        cache_mgr.save(
            df=feat_df,
            data_dir=str(loader.data_dir),
            symbols=symbols,
            timeframe=loader.timeframe,
            feature_set=cache_key_name,
            target_config=target_cfg,
            code_paths=[feature_engine_module.__file__, target_module.__file__],
        )
    return feat_df.copy()


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _safe_date(value: Any) -> str:
    ts = _to_utc_timestamp(value)
    return ts.date().isoformat() if ts is not None else ""


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
        pass
    y_pred_exit = None
    if exit_model is not None:
        with suppress(Exception):
            y_pred_exit = exit_model.predict(X).astype(int)
    returns = (
        sym_df["return_1d"].values
        if "return_1d" in sym_df.columns
        else sym_df["close"].pct_change().fillna(0.0).values
    )
    return {
        "symbol": symbol,
        "y_pred": y_pred,
        "y_pred_exit": y_pred_exit,
        "y_proba": y_proba,
        "classes": classes,
        "returns": returns,
        "sym_test_df": sym_df,
        "feature_cols": feature_cols,
        "train_rows": int(train_rows),
        "train_end_date": str(train_end_date),
        "model_mode": model_mode,
        "window_label": window_label,
        "test_start": test_start,
        "test_end": test_end,
    }


def _build_prediction_cache_from_model(
    model_id: str, symbols: list[str], model_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    artifact = _load_model_artifact(model_id, model_cfg)
    symbols = sorted({str(symbol).upper() for symbol in symbols})
    cached_predictions = artifact.get("prediction_cache")
    if isinstance(cached_predictions, list) and cached_predictions:
        symbol_set = set(symbols)
        return [
            item for item in cached_predictions if str(item.get("symbol", "")).upper() in symbol_set
        ]
    feature_cols = list(artifact.get("feature_cols", []))
    if not feature_cols:
        return []
    feat_df = _load_universe_features(symbols, artifact, model_id)
    feat_df["symbol"] = feat_df["symbol"].astype(str).str.upper()
    feat_df = feat_df[feat_df["symbol"].isin(symbols)].copy()
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"], utc=True, errors="coerce")
    feat_df = (
        feat_df.dropna(subset=["timestamp", *feature_cols])
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )
    if feat_df.empty:
        return []

    target_cfg = dict(artifact.get("target_config", {}))
    model_mode = str(artifact.get("mode", "single_model_full_history"))
    fold_models = artifact.get("fold_models")
    if isinstance(fold_models, list) and fold_models:
        items: list[dict[str, Any]] = []
        max_test_end_ts: pd.Timestamp | None = None
        last_fold: dict[str, Any] | None = None
        for fold in fold_models:
            test_start_ts = _to_utc_timestamp(fold.get("test_start"))
            test_end_ts = _to_utc_timestamp(fold.get("test_end"))
            if test_end_ts is not None and (
                max_test_end_ts is None or test_end_ts > max_test_end_ts
            ):
                max_test_end_ts = test_end_ts
                last_fold = fold
            fold_df = feat_df
            if test_start_ts is not None:
                fold_df = fold_df[fold_df["timestamp"] >= test_start_ts]
            if test_end_ts is not None:
                fold_df = fold_df[fold_df["timestamp"] <= test_end_ts]
            for symbol in symbols:
                sym_fold_df = (
                    fold_df[fold_df["symbol"] == symbol]
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                item = _predict_cache_item_from_models(
                    symbol=symbol,
                    sym_df=sym_fold_df,
                    feature_cols=feature_cols,
                    target_cfg=target_cfg,
                    entry_model=fold.get("entry_model"),
                    exit_model=fold.get("exit_model"),
                    model_mode=model_mode,
                    train_rows=int(fold.get("train_rows", 0)),
                    train_end_date=str(fold.get("train_end_date", "")),
                    window_label=str(fold.get("window_label", "")),
                    test_start=_safe_date(test_start_ts),
                    test_end=_safe_date(test_end_ts),
                )
                if item is not None:
                    items.append(item)

        if last_fold is not None and max_test_end_ts is not None:
            tail_df = feat_df[feat_df["timestamp"] > max_test_end_ts]
            for symbol in symbols:
                sym_tail_df = (
                    tail_df[tail_df["symbol"] == symbol]
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                tail_item = _predict_cache_item_from_models(
                    symbol=symbol,
                    sym_df=sym_tail_df,
                    feature_cols=feature_cols,
                    target_cfg=target_cfg,
                    entry_model=last_fold.get("entry_model"),
                    exit_model=last_fold.get("exit_model"),
                    model_mode=f"{model_mode}_realtime_tail",
                    train_rows=int(last_fold.get("train_rows", 0)),
                    train_end_date=str(last_fold.get("train_end_date", "")),
                    window_label=f"{str(last_fold.get('window_label', ''))}_tail_after_test",
                    test_start=_safe_date(max_test_end_ts + pd.Timedelta(minutes=1)),
                    test_end="",
                )
                if tail_item is not None:
                    items.append(tail_item)
        return items

    items = []
    for symbol in symbols:
        sym_df = (
            feat_df[feat_df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        )
        item = _predict_cache_item_from_models(
            symbol=symbol,
            sym_df=sym_df,
            feature_cols=feature_cols,
            target_cfg=target_cfg,
            entry_model=artifact.get("entry_model"),
            exit_model=artifact.get("exit_model"),
            model_mode=model_mode,
            train_rows=int(artifact.get("stats", {}).get("train_rows", 0)),
            train_end_date=str(artifact.get("stats", {}).get("train_end_date", "")),
        )
        if item is not None:
            items.append(item)
    return items


def _cache_item_latest_ts_value(item: dict[str, Any]) -> int:
    df = item.get("sym_test_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return -1
    ts = _to_utc_timestamp(df.iloc[-1].get("timestamp"))
    return int(ts.value) if ts is not None else -1


def _realtime_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"mode": "realtime", "error": "No predictions"}
    item = max(items, key=_cache_item_latest_ts_value)
    df = item.get("sym_test_df")
    y_pred = item.get("y_pred")
    y_pred_exit = item.get("y_pred_exit")
    y_proba = item.get("y_proba")
    classes = item.get("classes")
    if not isinstance(df, pd.DataFrame) or df.empty or y_pred is None or len(y_pred) == 0:
        return {"mode": "realtime", "error": "No predictions"}
    latest_ts = _format_ts(df.iloc[-1].get("timestamp"))
    buy_proba = None
    if y_proba is not None and classes is not None and len(y_proba) > 0:
        try:
            classes_list = list(classes)
            if 1 in classes_list:
                buy_proba = float(y_proba[-1][classes_list.index(1)])
        except Exception:
            pass
    return {
        "mode": "realtime",
        "model_mode": str(item.get("model_mode", "")),
        "latest_bar_time": latest_ts,
        "latest_close": float(df.iloc[-1]["close"]),
        "entry_signal_for_next_bar": int(y_pred[-1]),
        "entry_signal_used_for_latest_bar": int(y_pred[-2]) if len(y_pred) >= 2 else None,
        "exit_signal_for_next_bar": int(y_pred_exit[-1])
        if y_pred_exit is not None and len(y_pred_exit)
        else None,
        "buy_proba_for_next_bar": buy_proba,
        "train_rows": int(item.get("train_rows", 0)),
        "train_end_date": str(item.get("train_end_date", "")),
        "inference_rows": sum(
            int(len(i["sym_test_df"]))
            for i in items
            if isinstance(i.get("sym_test_df"), pd.DataFrame)
        ),
        "window_count": len(items),
        "active_window": str(item.get("window_label", "")),
    }


def _records_for_symbol(trades: Any, symbol: str) -> list[dict[str, Any]]:
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return []
    rows = []
    symbol = symbol.upper()
    for record in trades.to_dict("records"):
        if str(record.get("symbol", symbol)).upper() == symbol:
            rows.append(record)
    return rows


def _build_signal_payload(
    symbol: str, trades_df: Any, model_id: str, color: str, source: str, realtime: dict[str, Any]
) -> dict[str, Any]:
    symbol = symbol.upper()
    records = _records_for_symbol(trades_df, symbol)
    trades = select_fields(pd.DataFrame(records)) if records else []
    for trade in trades:
        trade["symbol"] = symbol
    return {
        "symbol": symbol,
        "market": MARKET,
        "timeframe": TIMEFRAME,
        "model_id": model_id,
        f"{model_id}_markers": make_markers(trades, model_id, color, "arrowUp"),
        f"{model_id}_trades": trades,
        f"{model_id}_stats": compute_stats(trades, model_id),
        "source": source,
        "realtime": realtime,
        "data_fingerprint": _data_fingerprint(symbol),
    }


def _generate_signal(model_id: str, symbol: str) -> dict[str, Any]:
    model_id, model_cfg = get_model_config(model_id)
    symbols = _load_symbols()
    cache_items = _build_prediction_cache_from_model(model_id, symbols, model_cfg)
    if not cache_items:
        raise ValueError("No model predictions")
    cfg = _load_cfg(model_id, model_cfg)
    result = Pipeline(cfg, symbols=symbols, device="cpu", prediction_cache=cache_items).run()
    source = f"pkl:{_load_model_artifact(model_id, model_cfg).get('mode', '')}"
    symbol_items = [
        item for item in cache_items if str(item.get("symbol", "")).upper() == symbol.upper()
    ]
    return _build_signal_payload(
        symbol=symbol,
        trades_df=result.trades_df,
        model_id=model_id,
        color=str(model_cfg.get("color", "#FFD54F")),
        source=source,
        realtime=_realtime_summary(symbol_items),
    )


def _generate_signal_threaded(model_id: str, symbol: str) -> None:
    key = _cache_key(model_id, symbol)
    try:
        _load_ohlcv(symbol)
        payload = _generate_signal(model_id, symbol)
        _write_json(_signal_cache_path(symbol, model_id), payload)
        with jobs_lock:
            signal_results[key] = payload
            state = jobs[key]
            state.status = "done"
            state.finished_at = time.time()
    except Exception as exc:
        with jobs_lock:
            state = jobs.setdefault(key, JobState())
            state.status = "error"
            state.error = str(exc)
            state.finished_at = time.time()


def _ensure_job(model_id: str, symbol: str) -> JobState:
    key = _cache_key(model_id, symbol)
    with jobs_lock:
        state = jobs.get(key)
        if state and state.status == "running":
            return state
        jobs[key] = JobState()
        signal_results.pop(key, None)
    executor.submit(_generate_signal_threaded, model_id, symbol.upper())
    return jobs[key]


def _load_signal_cache(model_id: str, symbol: str) -> dict[str, Any] | None:
    key = _cache_key(model_id, symbol)
    if key in signal_results:
        return signal_results[key]
    path = _signal_cache_path(symbol, model_id)
    if path.exists():
        try:
            return _read_json(path)
        except Exception:
            return None
    return None


def _symbol_summary(symbol: str, model_id: str) -> dict[str, Any]:
    payload = _load_signal_cache(model_id, symbol)
    stats = payload.get(f"{model_id}_stats", {}) if isinstance(payload, dict) else {}
    trades = payload.get(f"{model_id}_trades", []) if isinstance(payload, dict) else []
    markers = payload.get(f"{model_id}_markers", []) if isinstance(payload, dict) else []
    source_fp = _data_fingerprint(symbol)
    cached_fp = payload.get("data_fingerprint", {}) if isinstance(payload, dict) else {}
    stale = bool(payload) and source_fp != cached_fp
    total_trades = stats.get("total_trades", stats.get("trades", len(trades)))
    pnl = stats.get("total_pnl_pct", stats.get("total_pnl", stats.get("pnl", None)))
    wr = stats.get("win_rate", stats.get("wr", None))
    return {
        "symbol": symbol.upper(),
        "cached": payload is not None,
        "stale": stale,
        "signal_cached": payload is not None,
        "latest_bar_time": source_fp.get("latest_bar_time"),
        f"{model_id}_trades": total_trades,
        f"{model_id}_pnl": pnl,
        f"{model_id}_wr": wr,
        f"{model_id}_markers": len(markers) if isinstance(markers, list) else 0,
    }


def _cache_status_payload() -> dict[str, Any]:
    symbols = _load_symbols()
    rows = []
    running = 0
    done = 0
    errors = 0
    with jobs_lock:
        states = list(jobs.values())
    for state in states:
        if state.status == "running":
            running += 1
        elif state.status == "done":
            done += 1
        elif state.status == "error":
            errors += 1
    cached_count = 0
    for symbol in symbols:
        row = {"symbol": symbol, "models": {}}
        for model_id in MODELS:
            cache_path = _signal_cache_path(symbol, model_id)
            cached = cache_path.exists() or _cache_key(model_id, symbol) in signal_results
            cached_count += int(cached)
            row["models"][model_id] = {"cached": cached, "path": str(cache_path)}
        rows.append(row)
    return {
        "market": MARKET,
        "timeframe": TIMEFRAME,
        "total_symbols": len(symbols),
        "total_models": len(MODELS),
        "cached_count": cached_count,
        "running_jobs": running,
        "done_jobs": done,
        "error_jobs": errors,
        "symbols": rows,
    }


@app.get("/")
def root():
    return _html_response("derivatives30m_model.html")


@app.get("/api/models")
def api_models():
    rows = []
    for model_id, cfg in MODELS.items():
        availability = model_availability(model_id)
        rows.append(
            {
                "id": model_id,
                "type": cfg.get("type", ""),
                "label": cfg.get("label", model_id),
                "color": cfg.get("color", "#FFD54F"),
                "path": str(cfg.get("path", "")),
                "config": str(cfg.get("config", "")),
                "is_default": model_id == DEFAULT_MODEL,
                "available": availability["available"],
                "missing": availability["missing"],
            }
        )
    return _json_response(rows)


@app.get("/api/model-info")
def api_model_info():
    model_id, model_cfg = get_model_config(request.args.get("model_id") or DEFAULT_MODEL)
    artifact = _load_model_artifact(model_id, model_cfg)
    return _json_response(
        {
            "model_id": model_id,
            "market": artifact.get("market", MARKET),
            "timeframe": artifact.get("timeframe", TIMEFRAME),
            "config_name": artifact.get("config_name", ""),
            "mode": artifact.get("mode", ""),
            "context_mode": artifact.get("context_mode", ""),
            "train_symbol_count": artifact.get("train_symbol_count", 0),
            "feature_set": artifact.get("feature_set", ""),
            "entry_model_type": artifact.get("entry_model_type", ""),
            "exit_model_type": artifact.get("exit_model_type", ""),
            "window_count": artifact.get("stats", {}).get("window_count", 1),
            "train_rows": artifact.get("stats", {}).get("train_rows", 0),
            "train_end_date": artifact.get("stats", {}).get("train_end_date", ""),
            "created_at": artifact.get("created_at", ""),
            "model_path": str(model_cfg.get("path", "")),
            "config_path": str(model_cfg.get("config", "")),
        }
    )


@app.get("/api/symbols")
def api_symbols():
    model_id = str(request.args.get("model_id", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    if str(request.args.get("format", "")).lower() in {"list", "plain"}:
        return _json_response(_load_symbols())
    rows = [_symbol_summary(symbol, model_id) for symbol in _load_symbols()]
    return _json_response(rows)


@app.get("/api/data/<symbol>")
def api_data(symbol: str):
    model_id = str(request.args.get("model_id", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    try:
        ohlcv = _load_ohlcv(symbol)
        payload: dict[str, Any] = dict(ohlcv)
        signal_payload = _load_signal_cache(model_id, symbol)
        payload["signal_cached"] = signal_payload is not None
        payload["model_id"] = model_id
        if signal_payload:
            payload.update(signal_payload)
        return _json_response(payload)
    except Exception as exc:
        return _json_response({"error": str(exc), "symbol": symbol.upper()}, 500)


@app.get("/api/ohlcv/<symbol>")
def api_ohlcv(symbol: str):
    try:
        return _json_response(_load_ohlcv(symbol))
    except Exception as exc:
        return _json_response({"error": str(exc), "symbol": symbol.upper()}, 500)


@app.get("/api/signal/<model_id>/<symbol>")
def api_signal(model_id: str, symbol: str):
    refresh = str(request.args.get("refresh", "")).lower() in {"1", "true", "yes"}
    key = _cache_key(model_id, symbol)
    cache_path = _signal_cache_path(symbol, model_id)
    if not refresh and key in signal_results:
        return _json_response(signal_results[key])
    if not refresh and cache_path.exists():
        return _json_response(_read_json(cache_path))
    state = _ensure_job(model_id, symbol)
    return _json_response(
        {"status": state.status, "model_id": model_id, "symbol": symbol.upper()}, 202
    )


@app.get("/api/signal/<model_id>/<symbol>/status")
def api_signal_status(model_id: str, symbol: str):
    key = _cache_key(model_id, symbol)
    with jobs_lock:
        state = jobs.get(key)
    if state is None:
        return _json_response({"status": "idle", "model_id": model_id, "symbol": symbol.upper()})
    payload = {
        "status": state.status,
        "error": state.error,
        "model_id": model_id,
        "symbol": symbol.upper(),
        "started_at": state.started_at,
        "finished_at": state.finished_at,
    }
    if state.status == "done" and key in signal_results:
        payload["result"] = signal_results[key]
    return _json_response(payload)


@app.get("/api/signal/<symbol>")
def api_signal_default(symbol: str):
    return api_signal(DEFAULT_MODEL, symbol)


@app.get("/api/signal/<symbol>/status")
def api_signal_default_status(symbol: str):
    return api_signal_status(DEFAULT_MODEL, symbol)


@app.get("/api/signal-cache/status")
def api_signal_cache_status():
    return _json_response(_cache_status_payload())


@app.post("/api/signal-cache/build")
def api_signal_cache_build():
    payload = request.get_json(silent=True) or {}
    model_id = str(payload.get("model_id") or request.args.get("model_id") or DEFAULT_MODEL).strip()
    symbol_arg = payload.get("symbol") or request.args.get("symbol")
    symbols = [str(symbol_arg).upper()] if symbol_arg else _load_symbols()
    started = []
    for symbol in symbols:
        _ensure_job(model_id, symbol)
        started.append(symbol)
    return _json_response(
        {"started": started, "model_id": model_id, "status": _cache_status_payload()}
    )


@app.post("/api/signal-cache/clear")
def api_signal_cache_clear():
    payload = request.get_json(silent=True) or {}
    model_id = str(payload.get("model_id") or request.args.get("model_id") or DEFAULT_MODEL).strip()
    symbol_arg = payload.get("symbol") or request.args.get("symbol")
    symbols = [str(symbol_arg).upper()] if symbol_arg else _load_symbols()
    removed = []
    for symbol in symbols:
        key = _cache_key(model_id, symbol)
        signal_results.pop(key, None)
        with jobs_lock:
            jobs.pop(key, None)
        cache_path = _signal_cache_path(symbol, model_id)
        if cache_path.exists():
            cache_path.unlink()
            removed.append(str(cache_path))
    return _json_response(
        {
            "removed": removed,
            "model_id": model_id,
            "symbols": symbols,
            "status": _cache_status_payload(),
        }
    )


if __name__ == "__main__":
    print("Serving derivatives30m standalone on http://127.0.0.1:5013")
    app.run(host="127.0.0.1", port=5013, debug=False, threaded=True)
