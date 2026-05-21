from __future__ import annotations

import contextlib
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOP1_CONFIG_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "v22_exit_ablation_round42"
    / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
    / "config.resolved.yaml"
)


@dataclass
class DaySnapshot:
    cutoff: str
    open_positions: list[dict[str, Any]]
    new_entries: list[dict[str, Any]]
    next_session_predictions: list[dict[str, Any]]
    watchlist_top_buy_proba: list[dict[str, Any]]
    all_predictions: list[dict[str, Any]]
    stats: dict[str, Any]
    errors: list[dict[str, Any]]


def load_symbols(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if raw:
        path = Path(raw)
        if path.exists():
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                values = payload.get("symbols", payload) if isinstance(payload, dict) else payload
            else:
                values = path.read_text(encoding="utf-8").splitlines()
            return sorted({str(sym).strip().upper() for sym in values if str(sym).strip()})
        return sorted({part.strip().upper() for part in raw.split(",") if part.strip()})

    from src.data.loader import DataLoader
    from src.env import resolve_data_dir
    from src.market_profile import load_market_profile

    profile = load_market_profile("vn_stock")
    data_dir = resolve_data_dir(profile.data.data_dir)
    return DataLoader(str(data_dir)).symbols


def load_top1_config():
    from src.pipeline import ExperimentConfig

    if not TOP1_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Top1 config not found: {TOP1_CONFIG_PATH}")
    cfg = ExperimentConfig.from_yaml(TOP1_CONFIG_PATH)
    cfg.name = "top1_main_replay"
    return cfg


def load_dataset(symbols: list[str]) -> pd.DataFrame:
    from src.data.loader import DataLoader
    from src.env import resolve_data_dir
    from src.market_profile import load_market_profile

    profile = load_market_profile("vn_stock")
    data_dir = resolve_data_dir(profile.data.data_dir)
    loader = DataLoader(
        str(data_dir),
        timeframe=profile.data.default_timeframe,
        timestamp_column=profile.data.timestamp_column,
        timezone=profile.data.timezone,
        required_columns=profile.data.required_columns,
        optional_columns=profile.data.optional_columns,
    )
    df = loader.load_all(symbols=symbols, show_progress=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def load_cutoff_dates(df: pd.DataFrame, days: int, as_of: str) -> list[str]:
    series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna().dt.normalize()
    dates = sorted({d.date().isoformat() for d in series})
    if as_of:
        dates = [d for d in dates if d <= as_of]
    if days > 0:
        dates = dates[-days:]
    return dates


def _safe_date_str(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        return ts.date().isoformat()
    except Exception:
        return str(value)[:10]


def _build_day_cache_item(
    *,
    cfg: Any,
    symbol: str,
    sym_df: pd.DataFrame,
    feature_engine: Any,
) -> dict[str, Any] | None:
    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.data.target import TargetGenerator
    from src.signal_adapter import canonicalize_predictions

    if sym_df.empty:
        return None

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
    target_gen = TargetGenerator.from_config(legacy_split)

    feature_df = feature_engine.compute_for_all_symbols(sym_df.copy())
    train_df = target_gen.generate_for_all_symbols(feature_df.copy())

    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        train_df = TargetGenerator.generate_exit_labels(
            train_df,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    feature_cols = feature_engine.get_feature_columns(feature_df)
    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in train_df.columns
    if has_exit:
        drop_cols.append("target_sell")
    train_df = train_df.dropna(subset=drop_cols)
    train_sym_df = train_df[train_df["symbol"].astype(str).str.upper() == symbol].reset_index(
        drop=True
    )
    if len(train_sym_df) < 20:
        return None

    infer_df = feature_df[feature_df["symbol"].astype(str).str.upper() == symbol].reset_index(
        drop=True
    )
    infer_df = infer_df.dropna(subset=feature_cols).reset_index(drop=True)
    if infer_df.empty:
        return None
    if "return_1d" not in infer_df.columns:
        infer_df["return_1d"] = infer_df["close"].pct_change().fillna(0.0)

    X_train = np.nan_to_num(train_sym_df[feature_cols].values)
    y_train = train_sym_df["target"].values.astype(int)

    model = get_model(cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras)
    model.fit(X_train, y_train)

    X_infer = np.nan_to_num(infer_df[feature_cols].values)
    y_pred = canonicalize_predictions(model.predict(X_infer), legacy_split["target"])

    y_pred_exit = None
    if has_exit:
        exit_model_cfg = cfg.signals.exit_model
        exit_model = get_exit_model(exit_model_cfg.type, device="cpu", **exit_model_cfg.extras)
        exit_model.fit(X_train, train_sym_df["target_sell"].values.astype(int))
        y_pred_exit = exit_model.predict(X_infer).astype(int)

    y_proba = None
    classes = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_infer)
            final_est = model.steps[-1][1] if hasattr(model, "steps") else model
            classes = list(final_est.classes_)
    except Exception:
        y_proba = None

    return {
        "symbol": symbol,
        "y_pred": y_pred,
        "y_pred_exit": y_pred_exit,
        "y_proba": y_proba,
        "classes": classes,
        "returns": infer_df["return_1d"].values,
        "sym_test_df": infer_df,
        "feature_cols": feature_cols,
        "train_rows": int(len(train_sym_df)),
        "train_end_date": _safe_date_str(train_sym_df.iloc[-1].get("timestamp"))
        if not train_sym_df.empty
        else "",
    }


def _normalize_open_position(
    symbol: str, trades: list[dict[str, Any]], latest_bar_date: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not trades:
        return trades, None

    updated = [dict(t) for t in trades]
    last_idx = max(
        range(len(updated)),
        key=lambda i: (
            pd.to_datetime(
                updated[i].get("entry_date") or updated[i].get("exit_date"), errors="coerce"
            )
            if (updated[i].get("entry_date") or updated[i].get("exit_date"))
            else pd.Timestamp.min
        ),
    )
    t = updated[last_idx]
    exit_reason = str(t.get("exit_reason", "") or "").lower()
    exit_date = str(t.get("exit_date", "") or "")[:10]
    should_mark_open = bool(
        latest_bar_date and exit_reason == "end" and exit_date == latest_bar_date
    )
    if not should_mark_open:
        return updated, None

    entry_price = None
    try:
        raw_entry = t.get("entry_price")
        entry_price = float(raw_entry) if raw_entry is not None else None
    except Exception:
        entry_price = None
    latest_close = None
    try:
        latest_close = float(t.get("latest_close")) if t.get("latest_close") is not None else None
    except Exception:
        latest_close = None

    unrealized = t.get("pnl_pct", 0.0)
    if latest_close is not None and entry_price is not None and entry_price > 0:
        with contextlib.suppress(Exception):
            unrealized = round((latest_close / entry_price - 1.0) * 100.0, 2)

    t["exit_date"] = ""
    t["exit_reason"] = "open"
    t["is_open_position"] = True
    t["pnl_pct"] = unrealized
    updated[last_idx] = t

    return updated, {
        "symbol": symbol,
        "entry_date": str(t.get("entry_date", "") or "")[:10],
        "latest_bar_date": latest_bar_date,
        "holding_days": t.get("holding_days", 0),
        "unrealized_pnl_pct": unrealized,
        "position_size": t.get("position_size", 1.0),
        "entry_trend": t.get("entry_trend"),
    }


def _records_for_symbol(trades: Any, symbol: str) -> list[dict[str, Any]]:
    if isinstance(trades, pd.DataFrame):
        records = trades.to_dict("records")
    elif isinstance(trades, list):
        records = trades
    else:
        return []

    symbol = symbol.upper()
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("symbol", symbol)).upper() == symbol:
            rows.append(dict(record))
    return rows


def _build_day_snapshot(
    *,
    cfg: Any,
    cutoff: str,
    raw_df: pd.DataFrame,
    feature_engine: Any,
    symbols: list[str],
    watchlist_top: int,
    min_history: int,
) -> DaySnapshot:
    from src.evaluation.scoring import (
        calc_mdd_per_symbol,
        calc_metrics,
        calc_yearly_consistency,
        composite_score,
    )
    from src.pipeline import Pipeline

    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    cutoff_df = raw_df[raw_df["timestamp"] <= cutoff_ts].copy()
    if cutoff_df.empty:
        return DaySnapshot(cutoff, [], [], [], [], [], {"error": "empty_cutoff"}, [])

    all_predictions: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    new_entries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for idx, symbol in enumerate(symbols, start=1):
        sym_df = cutoff_df[cutoff_df["symbol"].astype(str).str.upper() == symbol].copy()
        if len(sym_df) < min_history:
            continue
        try:
            cache_item = _build_day_cache_item(
                cfg=cfg,
                symbol=symbol,
                sym_df=sym_df,
                feature_engine=feature_engine,
            )
            if cache_item is None:
                continue

            pred_row = {
                "symbol": symbol,
                "latest_bar_date": cutoff,
                "model_entry_signal_for_next_bar": int(cache_item["y_pred"][-1]),
                "entry_signal_for_next_bar": int(cache_item["y_pred"][-1]),
                "model_exit_signal_for_next_bar": (
                    int(cache_item["y_pred_exit"][-1])
                    if cache_item.get("y_pred_exit") is not None
                    and len(cache_item["y_pred_exit"]) > 0
                    else None
                ),
                "exit_signal_for_next_bar": (
                    int(cache_item["y_pred_exit"][-1])
                    if cache_item.get("y_pred_exit") is not None
                    and len(cache_item["y_pred_exit"]) > 0
                    else None
                ),
                "buy_proba_for_next_bar": None,
                "history_rows": int(len(sym_df)),
                "final_entry_signal_for_cutoff": 0,
                "final_exit_signal_for_cutoff": 0,
            }
            y_proba = cache_item.get("y_proba")
            classes = cache_item.get("classes")
            if (
                y_proba is not None
                and classes is not None
                and len(y_proba) > 0
                and len(classes) > 0
            ):
                try:
                    class_list = list(classes)
                    if 1 in class_list:
                        pred_row["buy_proba_for_next_bar"] = float(y_proba[-1][class_list.index(1)])
                except Exception:
                    pass
            all_predictions.append(pred_row)

            result = Pipeline(
                cfg, symbols=[symbol], device="cpu", prediction_cache=[cache_item]
            ).run()
            symbol_trades = (
                result.trades_df if isinstance(result.trades_df, pd.DataFrame) else pd.DataFrame()
            )
            if not symbol_trades.empty and "symbol" in symbol_trades.columns:
                trade_rows.extend(symbol_trades.to_dict("records"))
                recs = _records_for_symbol(symbol_trades, symbol)
                entry_dates = {
                    str(row.get("entry_date", ""))[:10]
                    for row in recs
                    if str(row.get("entry_date", ""))[:10]
                }
                exit_dates = {
                    str(row.get("exit_date", ""))[:10]
                    for row in recs
                    if str(row.get("exit_date", ""))[:10]
                }
                recs, open_pos = _normalize_open_position(symbol, recs, cutoff)
                if open_pos is not None:
                    open_positions.append(open_pos)
                final_entry_signal = int(cutoff in entry_dates)
                final_exit_signal = int(cutoff in exit_dates)
                pred_row["final_entry_signal_for_cutoff"] = final_entry_signal
                pred_row["final_exit_signal_for_cutoff"] = final_exit_signal
                if final_entry_signal == 1:
                    new_entries.append(
                        {
                            "symbol": symbol,
                            "entry_date": cutoff,
                            "entry_trend": open_pos.get("entry_trend") if open_pos else None,
                            "position_size": open_pos.get("position_size") if open_pos else None,
                            "final_entry_signal_for_cutoff": final_entry_signal,
                        }
                    )
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    all_predictions.sort(
        key=lambda row: (
            -int(row["entry_signal_for_next_bar"] == 1),
            -(row["buy_proba_for_next_bar"] or 0.0),
            row["symbol"],
        )
    )
    watchlist = sorted(
        all_predictions,
        key=lambda row: (-(row["buy_proba_for_next_bar"] or 0.0), row["symbol"]),
    )[:watchlist_top]

    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty and "pnl_pct" in trades_df.columns:
        pnl = pd.to_numeric(trades_df["pnl_pct"], errors="coerce").dropna()
        gross_profit = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
        gross_loss = float(-pnl[pnl < 0].sum()) if not pnl.empty else 0.0
        base_metrics = calc_metrics(trade_rows) if trade_rows else {}
        stats = {
            "symbol_count": len(symbols),
            "predicted_symbols": len(all_predictions),
            "open_positions": len(open_positions),
            "new_entries": len(new_entries),
            "final_entry_signal_count": sum(
                1
                for row in all_predictions
                if int(row.get("final_entry_signal_for_cutoff") or 0) == 1
            ),
            "final_exit_signal_count": sum(
                1
                for row in all_predictions
                if int(row.get("final_exit_signal_for_cutoff") or 0) == 1
            ),
            "entry_signal_for_next_bar": sum(
                1 for row in all_predictions if row["entry_signal_for_next_bar"] == 1
            ),
            "exit_signal_for_next_bar": sum(
                1 for row in all_predictions if row["exit_signal_for_next_bar"] == 1
            ),
            "trades": int(len(trades_df)),
            "wr": round(float((pnl > 0).mean() * 100), 2) if not pnl.empty else None,
            "total_pnl": round(float(pnl.sum()), 2) if not pnl.empty else None,
            "pf": round(gross_profit / gross_loss, 4) if gross_loss else None,
            "mdd_per_symbol": round(float(calc_mdd_per_symbol(trade_rows)), 2)
            if trade_rows
            else None,
            "yearly_consistency": round(float(calc_yearly_consistency(trade_rows)), 4)
            if trade_rows
            else None,
            "composite_score": round(float(composite_score(base_metrics, trade_rows)), 2)
            if trade_rows
            else None,
            "errors": len(errors),
        }
    else:
        stats = {
            "symbol_count": len(symbols),
            "predicted_symbols": len(all_predictions),
            "open_positions": len(open_positions),
            "new_entries": len(new_entries),
            "final_entry_signal_count": sum(
                1
                for row in all_predictions
                if int(row.get("final_entry_signal_for_cutoff") or 0) == 1
            ),
            "final_exit_signal_count": sum(
                1
                for row in all_predictions
                if int(row.get("final_exit_signal_for_cutoff") or 0) == 1
            ),
            "entry_signal_for_next_bar": sum(
                1 for row in all_predictions if row["entry_signal_for_next_bar"] == 1
            ),
            "exit_signal_for_next_bar": sum(
                1 for row in all_predictions if row["exit_signal_for_next_bar"] == 1
            ),
            "trades": int(len(trades_df)),
            "wr": None,
            "total_pnl": None,
            "pf": None,
            "mdd_per_symbol": None,
            "yearly_consistency": None,
            "composite_score": None,
            "errors": len(errors),
        }

    return DaySnapshot(
        cutoff=cutoff,
        open_positions=open_positions,
        new_entries=new_entries,
        next_session_predictions=all_predictions[:watchlist_top]
        if watchlist_top > 0
        else all_predictions,
        watchlist_top_buy_proba=watchlist,
        all_predictions=all_predictions,
        stats=stats,
        errors=errors,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
