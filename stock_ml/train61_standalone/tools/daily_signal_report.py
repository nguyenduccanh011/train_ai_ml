from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tools.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import serve_train61_model as server  # noqa: E402
from model_registry import DEFAULT_MODEL, get_model_cfg  # noqa: E402
from src.features.engine import FeatureEngine  # noqa: E402
from src.signal_adapter import canonicalize_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a daily report of open buy positions, new entries, next-session "
            "predictions, and per-day stats without using bars after each cutoff date."
        )
    )
    parser.add_argument("--model-id", default=os.getenv("TRAIN61_DEFAULT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--symbols", default="", help="Comma-separated symbols or JSON/TXT file")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument(
        "--as-of", default="", help="Last cutoff date YYYY-MM-DD, default DB latest"
    )
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "daily_signals"))
    parser.add_argument("--min-history", type=int, default=260)
    parser.add_argument(
        "--top", type=int, default=0, help="Limit predictions in summary, 0 keeps all"
    )
    parser.add_argument("--watchlist-top", type=int, default=10)
    return parser.parse_args()


def connect_db():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(database_url, row_factory=dict_row)


def load_symbols(raw: str, model_cfg: dict[str, Any]) -> list[str]:
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
    return server._load_symbols_for_model(model_cfg)


def trading_dates(conn, *, days: int, as_of: str, timeframe: str, provider: str) -> list[str]:
    params: list[Any] = [timeframe, provider]
    where_as_of = ""
    if as_of:
        where_as_of = "and timestamp::date <= %s::date"
        params.append(as_of)
    rows = conn.execute(
        f"""
        select timestamp::date as trade_date
        from market_bars
        where timeframe = %s and provider = %s
        {where_as_of}
        group by timestamp::date
        order by timestamp::date desc
        limit %s
        """,
        [*params, days],
    ).fetchall()
    return [str(row["trade_date"]) for row in reversed(rows)]


def load_bars_until(conn, symbol: str, cutoff: str, timeframe: str, provider: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        select symbol, timestamp, open, high, low, close, volume
        from market_bars
        where symbol = %s
          and timeframe = %s
          and provider = %s
          and timestamp::date <= %s::date
        order by timestamp
        """,
        (symbol, timeframe, provider, cutoff),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(row) for row in rows])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df


def load_context_until(cutoff: str) -> dict[str, pd.DataFrame]:
    loader = server.DataLoader(str(server.STANDALONE_DATASET_DIR))
    context = loader.load_all_context()
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    truncated: dict[str, pd.DataFrame] = {}
    for name, df in context.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
        tmp = tmp[tmp["timestamp"] <= cutoff_ts].copy()
        if not tmp.empty:
            truncated[name] = tmp
    return truncated


def predict_latest(
    *,
    df: pd.DataFrame,
    cutoff: str,
    artifact: dict[str, Any],
    feature_engine: FeatureEngine,
    context_data: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:
    if df.empty:
        return None
    feature_df = feature_engine.compute_for_all_symbols(df)
    if context_data:
        feature_df = feature_engine.add_market_context(feature_df, context_data)

    feature_cols = list(artifact["feature_cols"])
    missing = [col for col in feature_cols if col not in feature_df.columns]
    if missing:
        for col in missing:
            feature_df[col] = np.nan

    feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)
    row = feature_df.iloc[-1]
    X = np.nan_to_num(feature_df[feature_cols].tail(1).values)
    entry_pred = canonicalize_predictions(
        artifact["entry_model"].predict(X),
        artifact.get("target_config", {}),
    )
    entry_signal = int(entry_pred[-1])

    exit_signal = None
    exit_model = artifact.get("exit_model")
    if exit_model is not None:
        try:
            exit_signal = int(exit_model.predict(X)[-1])
        except Exception:
            exit_signal = None

    buy_proba = None
    try:
        proba = artifact["entry_model"].predict_proba(X)
        final_est = (
            artifact["entry_model"]._pipeline.steps[-1][1]
            if hasattr(artifact["entry_model"], "_pipeline")
            else artifact["entry_model"]
        )
        classes = list(getattr(final_est, "classes_", []))
        if 1 in classes:
            buy_proba = float(proba[-1][classes.index(1)])
    except Exception:
        buy_proba = None

    return {
        "symbol": str(row["symbol"]).upper(),
        "cutoff_date": cutoff,
        "latest_bar_date": str(pd.Timestamp(row["timestamp"]).date()),
        "latest_close": float(row["close"]),
        "entry_signal_for_next_bar": entry_signal,
        "exit_signal_for_next_bar": exit_signal,
        "buy_proba_for_next_bar": buy_proba,
        "history_rows": int(len(df)),
    }


def load_signal_cache(model_id: str, symbol: str) -> dict[str, Any] | None:
    path = server._signal_cache_path(symbol, model_id)
    if not path.exists():
        return None
    try:
        return server._normalize_cached_payload_for_model(model_id, server._read_json(path))
    except Exception:
        return None


def trades_as_of(
    payload: dict[str, Any] | None, model_id: str, cutoff: str
) -> list[dict[str, Any]]:
    if not payload:
        return []
    trades = payload.get(f"{model_id}_trades", [])
    if not isinstance(trades, list):
        return []
    cutoff_ts = pd.Timestamp(cutoff)
    result = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        entry_ts = pd.to_datetime(trade.get("entry_date"), errors="coerce")
        if pd.isna(entry_ts) or entry_ts.date() > cutoff_ts.date():
            continue
        row = dict(trade)
        exit_ts = pd.to_datetime(row.get("exit_date"), errors="coerce")
        row["is_open_as_of"] = bool(pd.isna(exit_ts) or exit_ts.date() > cutoff_ts.date())
        result.append(row)
    return result


def position_rows(
    payloads: dict[str, dict[str, Any] | None], model_id: str, cutoff: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    open_positions = []
    new_entries = []
    for symbol, payload in payloads.items():
        for trade in trades_as_of(payload, model_id, cutoff):
            entry_date = str(trade.get("entry_date", ""))[:10]
            if trade.get("is_open_as_of"):
                open_positions.append(
                    {
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "entry_price": trade.get("entry_price"),
                        "position_size": trade.get("position_size"),
                        "holding_days_as_of": max(
                            0,
                            (pd.Timestamp(cutoff) - pd.Timestamp(entry_date)).days,
                        )
                        if entry_date
                        else None,
                        "entry_trend": trade.get("entry_trend"),
                    }
                )
            if entry_date == cutoff:
                new_entries.append(
                    {
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "entry_price": trade.get("entry_price"),
                        "position_size": trade.get("position_size"),
                        "entry_trend": trade.get("entry_trend"),
                    }
                )
    open_positions.sort(key=lambda row: row["symbol"])
    new_entries.sort(key=lambda row: row["symbol"])
    return open_positions, new_entries


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


def main() -> int:
    args = parse_args()
    model_cfg = get_model_cfg(args.model_id)
    artifact = server._load_model_artifact(args.model_id, model_cfg=model_cfg)
    feature_engine = FeatureEngine(feature_set=str(artifact.get("feature_set") or "leading"))
    symbols = load_symbols(args.symbols, model_cfg)
    timeframe = server._default_timeframe()
    provider = server._default_data_provider()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {symbol: load_signal_cache(args.model_id, symbol) for symbol in symbols}

    report_days = []
    all_predictions: list[dict[str, Any]] = []
    with connect_db() as conn:
        dates = trading_dates(
            conn,
            days=args.days,
            as_of=args.as_of,
            timeframe=timeframe,
            provider=provider,
        )
        if not dates:
            raise SystemExit("No trading dates found in DB")

        for cutoff in dates:
            context_data = load_context_until(cutoff)
            predictions = []
            errors = []
            for symbol in symbols:
                try:
                    bars = load_bars_until(conn, symbol, cutoff, timeframe, provider)
                    if len(bars) < args.min_history:
                        errors.append(
                            {
                                "symbol": symbol,
                                "error": f"not_enough_history:{len(bars)}",
                            }
                        )
                        continue
                    pred = predict_latest(
                        df=bars,
                        cutoff=cutoff,
                        artifact=artifact,
                        feature_engine=feature_engine,
                        context_data=context_data,
                    )
                    if pred:
                        predictions.append(pred)
                except Exception as exc:
                    errors.append({"symbol": symbol, "error": str(exc)})

            open_positions, new_entries = position_rows(payloads, args.model_id, cutoff)
            predictions.sort(
                key=lambda row: (
                    -int(row["entry_signal_for_next_bar"] == 1),
                    -(row["buy_proba_for_next_bar"] or 0.0),
                    row["symbol"],
                )
            )
            selected_predictions = predictions[: args.top] if args.top > 0 else predictions
            watchlist = sorted(
                predictions,
                key=lambda row: (-(row["buy_proba_for_next_bar"] or 0.0), row["symbol"]),
            )[: args.watchlist_top]
            all_predictions.extend(predictions)
            report_days.append(
                {
                    "date": cutoff,
                    "open_positions": open_positions,
                    "new_entries": new_entries,
                    "next_session_predictions": selected_predictions,
                    "watchlist_top_buy_proba": watchlist,
                    "stats": {
                        "symbol_count": len(symbols),
                        "predicted_symbols": len(predictions),
                        "open_positions": len(open_positions),
                        "new_entries": len(new_entries),
                        "entry_signal_for_next_bar": sum(
                            1 for row in predictions if row["entry_signal_for_next_bar"] == 1
                        ),
                        "exit_signal_for_next_bar": sum(
                            1 for row in predictions if row["exit_signal_for_next_bar"] == 1
                        ),
                        "errors": len(errors),
                    },
                    "errors": errors,
                }
            )
            print(
                f"{cutoff}: open={len(open_positions)} new_entries={len(new_entries)} "
                f"entry_next={report_days[-1]['stats']['entry_signal_for_next_bar']} "
                f"errors={len(errors)}"
            )

    latest_date = report_days[-1]["date"]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "timeframe": timeframe,
        "provider": provider,
        "days": report_days,
    }
    json_path = output_dir / f"daily_signal_report_{latest_date}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    latest = report_days[-1]
    write_csv(output_dir / f"open_positions_{latest_date}.csv", latest["open_positions"])
    write_csv(output_dir / f"new_entries_{latest_date}.csv", latest["new_entries"])
    write_csv(
        output_dir / f"next_session_predictions_{latest_date}.csv",
        latest["next_session_predictions"],
    )
    write_csv(
        output_dir / f"watchlist_top_buy_proba_{latest_date}.csv", latest["watchlist_top_buy_proba"]
    )
    write_csv(output_dir / f"all_predictions_{latest_date}.csv", all_predictions)

    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
