from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIZ_DIR = ROOT / "visualization"
BASE_DATA_DIR = VIZ_DIR / "data"
MODEL_DIR = VIZ_DIR / "data_top1_model"
VERSION_KEY = "top1_model"
MODEL_COLOR = "#FFD700"  # Gold color for top 1

# Path to the top 1 model config
TOP1_CONFIG_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "v22_exit_ablation_round42"
    / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
    / "config.resolved.yaml"
)

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


def _build_top1_config():
    from src.pipeline import ExperimentConfig

    if not TOP1_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Top 1 model config not found: {TOP1_CONFIG_PATH}")

    cfg = ExperimentConfig.from_yaml(TOP1_CONFIG_PATH)
    cfg.name = VERSION_KEY
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


def _build_signal_payload(
    symbol: str,
    trades_df: Any,
    realtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.export.unified_export import compute_stats, make_markers, select_fields

    symbol = symbol.upper()
    records = _records_for_symbol(trades_df, symbol)
    trades = select_fields(pd.DataFrame(records)) if records else []
    for trade in trades:
        trade["symbol"] = symbol
    trades, open_position = _normalize_open_position(symbol, trades, realtime=realtime)
    payload = {
        "symbol": symbol,
        f"{VERSION_KEY}_markers": make_markers(trades, VERSION_KEY, MODEL_COLOR, "arrowUp"),
        f"{VERSION_KEY}_trades": trades,
        f"{VERSION_KEY}_stats": compute_stats(trades, VERSION_KEY),
        "has_open_position": open_position is not None,
        "open_position": open_position,
        "source": "live_on_demand",
    }
    if realtime is not None:
        payload["realtime"] = realtime
    return payload


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
    if realtime:
        latest_bar_date = str(realtime.get("latest_bar_date", "") or "")

    last_idx = max(range(len(trades)), key=lambda i: _trade_date_key(trades[i]))
    updated = [dict(t) for t in trades]
    t = updated[last_idx]

    exit_reason = str(t.get("exit_reason", "") or "").lower()
    exit_date = str(t.get("exit_date", "") or "")[:10]
    should_mark_open = bool(
        latest_bar_date and exit_reason == "end" and exit_date == latest_bar_date
    )

    if not should_mark_open:
        return updated, None

    t["exit_date"] = ""
    t["exit_reason"] = "open"
    t["is_open_position"] = True
    updated[last_idx] = t

    open_position = {
        "symbol": symbol,
        "entry_date": str(t.get("entry_date", "") or "")[:10],
        "latest_bar_date": latest_bar_date,
        "holding_days": t.get("holding_days", 0),
        "unrealized_pnl_pct": t.get("pnl_pct", 0.0),
        "position_size": t.get("position_size", 1.0),
        "entry_trend": t.get("entry_trend"),
    }
    return updated, open_position


def _build_realtime_summary(cache_item: dict[str, Any]) -> dict[str, Any]:
    df = cache_item.get("sym_test_df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"mode": "realtime", "error": "No inference rows"}

    y_pred = cache_item.get("y_pred")
    y_pred_exit = cache_item.get("y_pred_exit")
    y_proba = cache_item.get("y_proba")
    classes = cache_item.get("classes")
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

    summary = {
        "mode": "realtime",
        "latest_bar_date": latest_bar_date,
        "latest_close": latest_close,
        "entry_signal_for_next_bar": int(y_pred[-1]),
        "entry_signal_used_for_latest_bar": int(y_pred[-2]) if len(y_pred) >= 2 else None,
        "exit_signal_for_next_bar": (
            int(y_pred_exit[-1]) if y_pred_exit is not None and len(y_pred_exit) > 0 else None
        ),
        "buy_proba_for_next_bar": buy_proba,
        "train_rows": int(cache_item.get("n_train_rows", 0)),
        "inference_rows": int(len(df)),
        "train_end_date": str(cache_item.get("train_end_date", "")),
    }
    return summary


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
            "n_train_rows": len(train_sym_df),
            "train_end_date": train_end_date,
        }
    ]


def _generate_signal(symbol: str) -> None:
    symbol = symbol.upper()
    try:
        from src.pipeline import Pipeline

        with jobs_lock:
            jobs.setdefault(symbol, JobState())

        _load_ohlcv(symbol)
        cfg = _build_top1_config()
        live_cache = _build_live_prediction_cache(cfg, symbol)
        if not live_cache:
            raise ValueError(f"Not enough data to generate realtime signal for {symbol}")
        result = Pipeline(cfg, symbols=[symbol], device="cpu", prediction_cache=live_cache).run()
        payload = _build_signal_payload(
            symbol,
            result.trades_df,
            realtime=_build_realtime_summary(live_cache[0]),
        )
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
    return send_from_directory(VIZ_DIR, "top1_model.html")


@app.get("/top1_model.html")
def page():
    return send_from_directory(VIZ_DIR, "top1_model.html")


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


@app.get("/api/open-positions")
def api_open_positions():
    symbols_query = str(request.args.get("symbols", "") or "").strip()
    max_symbols_raw = str(request.args.get("max_symbols", "0") or "0").strip()
    refresh = str(request.args.get("refresh", "0") or "0").strip().lower() in {"1", "true", "yes"}

    if symbols_query:
        symbols = [s.strip().upper() for s in symbols_query.split(",") if s.strip()]
    else:
        symbols = _load_base_symbols()

    try:
        max_symbols = max(int(max_symbols_raw), 0)
    except ValueError:
        max_symbols = 0
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    open_positions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for symbol in symbols:
        payload = None
        with jobs_lock:
            payload = signal_results.get(symbol)

        if payload is None or refresh:
            _generate_signal(symbol)
            with jobs_lock:
                payload = signal_results.get(symbol)
                state = jobs.get(symbol)
            if payload is None and state and state.error:
                errors.append({"symbol": symbol, "error": state.error})
                continue

        if payload is None:
            continue
        op = payload.get("open_position")
        if isinstance(op, dict) and op:
            open_positions.append(op)

    open_positions.sort(
        key=lambda row: (str(row.get("entry_date", "")), str(row.get("symbol", ""))),
        reverse=True,
    )
    return _json_response(
        {
            "count": len(open_positions),
            "symbols_scanned": len(symbols),
            "open_positions": open_positions,
            "errors": errors,
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TOP 1 MODEL SERVER")
    print("=" * 60)
    print("Model: v22_exit_ablation_round42")
    print(f"Config: {TOP1_CONFIG_PATH.name}")
    print("Leaderboard PnL: 15,768.62 | WR: 74.64% | Trades: 1,262")
    print("Server: http://127.0.0.1:5002")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5002, debug=True, threaded=True)
