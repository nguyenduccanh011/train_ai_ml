from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIZ_DIR = ROOT / "visualization"
BASE_DATA_DIR = VIZ_DIR / "data"
MODEL_DIR = VIZ_DIR / "data_v2_rf_v22"
VERSION_KEY = "v2_rf_v22"
MODEL_COLOR = "#2196F3"

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


def _json_response(payload: Any):
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _symbol_file_name(symbol: str) -> str:
    return f"{symbol.upper()}.json"


def _signal_cache_path(symbol: str) -> Path:
    return MODEL_DIR / _symbol_file_name(symbol)


def _base_ohlcv_path(symbol: str) -> Path:
    return BASE_DATA_DIR / _symbol_file_name(symbol)


def _load_manifest_symbols() -> list[str]:
    manifest_path = VIZ_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = _read_json(manifest_path)
    return [str(sym).upper() for sym in manifest.get("base_symbols", [])]


def _load_model_index() -> dict[str, dict[str, Any]]:
    index_path = MODEL_DIR / "index.json"
    if not index_path.exists():
        return {}
    index = _read_json(index_path)
    return {
        str(row.get("symbol", "")).upper(): row
        for row in index.get("symbols", [])
        if row.get("symbol")
    }


def _load_base_symbols() -> list[str]:
    symbols = {
        path.stem.upper() for path in BASE_DATA_DIR.glob("*.json") if path.stem.lower() != "index"
    }
    symbols.update(_load_manifest_symbols())
    symbols.update(_load_model_index())
    return sorted(symbols)


def _load_ohlcv(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper()
    path = _base_ohlcv_path(symbol)
    if path.exists():
        return _read_json(path)

    from src.config_loader import get_pipeline_config, resolve_data_dir
    from src.data.loader import DataLoader

    pipeline = get_pipeline_config()
    data_dir = resolve_data_dir(
        pipeline.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    )
    df = DataLoader(data_dir).load_symbol(symbol)
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


def _build_v2_rf_v22_config():
    from src.pipeline import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(ROOT / "config" / "experiments" / "champions" / "v22.yaml")
    cfg.name = VERSION_KEY
    cfg.signals.entry_model.type = "random_forest"
    cfg.components.entry_model.type = "random_forest"
    return cfg


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


def _build_signal_payload(symbol: str, trades_df: Any) -> dict[str, Any]:
    from src.export.unified_export import compute_stats, make_markers, select_fields

    symbol = symbol.upper()
    records = _records_for_symbol(trades_df, symbol)
    trades = select_fields(pd.DataFrame(records)) if records else []
    for trade in trades:
        trade["symbol"] = symbol
    return {
        "symbol": symbol,
        f"{VERSION_KEY}_markers": make_markers(trades, VERSION_KEY, MODEL_COLOR, "arrowUp"),
        f"{VERSION_KEY}_trades": trades,
        f"{VERSION_KEY}_stats": compute_stats(trades, VERSION_KEY),
        "source": "live_on_demand",
    }


def _build_live_prediction_cache(cfg: Any, symbol: str) -> list[dict[str, Any]]:
    import numpy as np
    import src.data.target as target_module
    import src.features.engine as feature_engine_module
    from src.cache.feature_cache import FeatureCacheManager
    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.config_loader import load_config, resolve_data_dir
    from src.data.loader import DataLoader
    from src.data.target import TargetGenerator
    from src.env import get_results_dir
    from src.features.engine import FeatureEngine
    from src.signal_adapter import canonicalize_predictions

    symbol = symbol.upper()
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

    loader = DataLoader(abs_data_dir)
    feature_set = cfg.feature_set()
    engine = FeatureEngine(feature_set=feature_set)
    target_gen = TargetGenerator.from_config(legacy_split)

    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__]
    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=[symbol],
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=legacy_split.get("target", {}),
        code_paths=code_paths,
    )
    if df is None:
        print(f"    Live feature cache: MISS ({feature_set}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=[symbol])
        df = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=[symbol],
            timeframe=loader.timeframe,
            feature_set=feature_set,
            target_config=legacy_split.get("target", {}),
            code_paths=code_paths,
        )
        print(
            f"    Live feature cache: STORED ({feature_set}) key={saved_key[:8]} format={saved_fmt}"
        )
    else:
        print(f"    Live feature cache: HIT ({feature_set}) key={cache_key[:8]}")

    df = target_gen.generate_for_all_symbols(df)
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        df = TargetGenerator.generate_exit_labels(
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
    sym_df = df[df["symbol"].astype(str).str.upper() == symbol].reset_index(drop=True)
    if len(sym_df) < 20:
        return []

    X = np.nan_to_num(sym_df[feature_cols].values)
    y = sym_df["target"].values.astype(int)
    model = get_model(cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras)
    model.fit(X, y)
    y_pred = canonicalize_predictions(model.predict(X), legacy_split.get("target", {}))

    sell_model = None
    if has_exit:
        exit_model_cfg = cfg.signals.exit_model
        sell_model = get_exit_model(exit_model_cfg.type, device="cpu", **exit_model_cfg.extras)
        sell_model.fit(X, sym_df["target_sell"].values.astype(int))

    y_proba = None
    classes = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            final_est = model.steps[-1][1] if hasattr(model, "steps") else model
            classes = list(final_est.classes_)
    except Exception:
        y_proba = None

    return [
        {
            "symbol": symbol,
            "y_pred": y_pred,
            "y_pred_exit": sell_model.predict(X).astype(int) if sell_model is not None else None,
            "y_proba": y_proba,
            "classes": classes,
            "returns": sym_df["return_1d"].values,
            "sym_test_df": sym_df,
            "feature_cols": feature_cols,
        }
    ]


def _generate_signal(symbol: str) -> None:
    symbol = symbol.upper()
    try:
        from src.pipeline import Pipeline

        _load_ohlcv(symbol)
        cfg = _build_v2_rf_v22_config()
        live_cache = _build_live_prediction_cache(cfg, symbol)
        result = Pipeline(cfg, symbols=[symbol], device="cpu", prediction_cache=live_cache).run()
        payload = _build_signal_payload(symbol, result.trades_df)
        with jobs_lock:
            signal_results[symbol] = payload
            state = jobs[symbol]
            state.status = "done"
            state.finished_at = time.time()
    except Exception as exc:
        with jobs_lock:
            state = jobs.setdefault(symbol, JobState())
            state.status = "error"
            state.error = str(exc)
            state.finished_at = time.time()


def _ensure_job(symbol: str) -> JobState:
    symbol = symbol.upper()
    with jobs_lock:
        state = jobs.get(symbol)
        if state and state.status == "running":
            return state
        signal_results.pop(symbol, None)
        state = JobState()
        jobs[symbol] = state
    threading.Thread(target=_generate_signal, args=(symbol,), daemon=True).start()
    return state


@app.get("/")
def root():
    return send_from_directory(VIZ_DIR, "v2_rf_v22.html")


@app.get("/v2_rf_v22.html")
def page():
    return send_from_directory(VIZ_DIR, "v2_rf_v22.html")


@app.get("/api/symbols")
def api_symbols():
    index = _load_model_index()
    rows = []
    for symbol in _load_base_symbols():
        row = dict(index.get(symbol, {"symbol": symbol}))
        row.setdefault(f"{VERSION_KEY}_trades", 0)
        row.setdefault(f"{VERSION_KEY}_pnl", 0.0)
        row.setdefault(f"{VERSION_KEY}_wr", 0.0)
        row["cached"] = False
        row["has_historical_export"] = _signal_cache_path(symbol).exists()
        rows.append(row)
    rows.sort(
        key=lambda row: (
            not row["has_historical_export"],
            -(row.get(f"{VERSION_KEY}_pnl", 0) or 0),
            row["symbol"],
        )
    )
    return _json_response(rows)


@app.get("/api/ohlcv/<symbol>")
def api_ohlcv(symbol: str):
    try:
        return _json_response(_load_ohlcv(symbol))
    except Exception as exc:
        return _json_response({"error": str(exc), "symbol": symbol.upper()}), 404


@app.get("/api/signal/<symbol>")
def api_signal(symbol: str):
    symbol = symbol.upper()
    with jobs_lock:
        payload = signal_results.get(symbol)
        if payload is not None:
            return _json_response(payload)

    state = _ensure_job(symbol)
    return _json_response(
        {"status": state.status, "job_id": symbol, "symbol": symbol, "source": "on_demand"}
    ), 202


@app.get("/api/signal/<symbol>/status")
def api_signal_status(symbol: str):
    symbol = symbol.upper()
    with jobs_lock:
        if symbol in signal_results:
            return _json_response(
                {"status": "done", "job_id": symbol, "symbol": symbol, "source": "on_demand"}
            )
        state = jobs.get(symbol)
        if not state:
            return _json_response({"status": "not_started", "job_id": symbol, "symbol": symbol})
        return _json_response(
            {
                "status": state.status,
                "job_id": symbol,
                "symbol": symbol,
                "error": state.error,
                "elapsed_seconds": round((state.finished_at or time.time()) - state.started_at, 1),
            }
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, threaded=True)
