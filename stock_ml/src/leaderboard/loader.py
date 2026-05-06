from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.backtest.defaults import DEFAULT_TRADING_COST
from src.evaluation.scoring import (
    calc_max_drawdown,
    calc_mdd_per_symbol,
    calc_metrics,
    calc_sharpe,
    calc_yearly_consistency,
    composite_score,
)
from src.leaderboard.schema import CostProfile, LeaderboardRow, TargetConfig

MISSING_TRADES_WARNING = "trades.csv missing → metrics from cache"
COST_PROFILE_UNKNOWN_WARNING = "cost_profile=unknown"


def run_dir_to_row(run_dir: str | Path, *, bundle: str | None = None) -> LeaderboardRow:
    run_path = Path(run_dir)
    predictions_meta = _read_json(run_path / "predictions_meta.json")
    ranking_row = _read_json(run_path / "ranking_row.json")
    metrics_cache = _read_json(run_path / "metrics.json")
    resolved_config = _read_yaml(run_path / "config.resolved.yaml")
    warnings: list[str] = []

    trades_path = run_path / "trades.csv"
    trades = _read_trades(trades_path) if trades_path.exists() else []
    if trades:
        metrics = calc_metrics(trades)
        sharpe = calc_sharpe(trades)
        max_drawdown = calc_max_drawdown(trades)
        mdd_per_symbol = calc_mdd_per_symbol(trades)
        yearly_consistency = calc_yearly_consistency(trades)
        composite = composite_score(metrics, trades)
    else:
        warnings.append(MISSING_TRADES_WARNING)
        metrics = _metrics_from_cache(metrics_cache, ranking_row)
        sharpe = float(metrics_cache.get("sharpe", 0.0))
        max_drawdown = float(
            metrics_cache.get("max_drawdown", ranking_row.get("max_drawdown", 0.0))
        )
        mdd_per_symbol = float(
            metrics_cache.get("mdd_per_symbol", ranking_row.get("mdd_per_symbol", 0.0))
        )
        yearly_consistency = float(
            metrics_cache.get(
                "yearly_consistency",
                ranking_row.get("yearly_consistency", ranking_row.get("per_year_consistency", 0.0)),
            )
        )
        composite = float(
            metrics_cache.get("composite_score", ranking_row.get("composite_score", 0.0))
        )

    target = _target_config(resolved_config)
    cost_profile = _cost_profile(resolved_config, warnings)
    first_year, last_year = _test_window(resolved_config, predictions_meta, trades)
    symbols = sorted({str(t["symbol"]) for t in trades})
    n_symbols = len(symbols) or int(
        metrics_cache.get("symbol_coverage", {}).get(
            "symbol_count",
            ranking_row.get("per_symbol_coverage", {}).get("symbol_count", 0),
        )
    )

    run_name = str(resolved_config.get("name") or ranking_row.get("name") or run_path.name)
    bundle_name = str(bundle or predictions_meta.get("matrix_name") or run_path.parent.name)
    config_hash = str(
        predictions_meta.get("config_hash") or ranking_row.get("config_hash") or "unknown"
    )
    generated_at = str(
        predictions_meta.get("created_at")
        or _mtime_iso(trades_path if trades_path.exists() else run_path)
    )

    return LeaderboardRow(
        run_id=f"{bundle_name}/{run_name}#{config_hash[:8]}",
        bundle=bundle_name,
        run_name=run_name,
        config_hash=config_hash,
        generated_at=generated_at,
        superseded=False,
        market=str(
            predictions_meta.get("market")
            or resolved_config.get("market")
            or ranking_row.get("market")
            or "unknown"
        ),
        currency=str(
            predictions_meta.get("currency")
            or resolved_config.get("execution", {}).get("currency")
            or ranking_row.get("currency")
            or "unknown"
        ),
        pnl_mode=str(
            predictions_meta.get("pnl_mode")
            or resolved_config.get("execution", {}).get("pnl_mode")
            or ranking_row.get("pnl_mode")
            or "unknown"
        ),
        schema=str(predictions_meta.get("schema") or ranking_row.get("schema") or "unknown"),
        timeframe=str(
            predictions_meta.get("timeframe") or ranking_row.get("timeframe") or "unknown"
        ),
        strategy=str(resolved_config.get("strategy") or run_name),
        feature_set=str(
            predictions_meta.get("feature_set")
            or ranking_row.get("feature_set")
            or resolved_config.get("signals", {}).get("features")
            or "unknown"
        ),
        entry_model=str(
            predictions_meta.get("entry_model") or ranking_row.get("entry_model") or "unknown"
        ),
        exit_model_type=str(
            predictions_meta.get("exit_model_type")
            or ranking_row.get("exit_model_type")
            or "unknown"
        ),
        exit_model_enabled=bool(
            predictions_meta.get("exit_model_enabled", ranking_row.get("exit_model_enabled", False))
        ),
        target=target,
        trades=int(metrics["trades"]),
        wr=float(metrics["wr"]),
        avg_pnl=float(metrics["avg_pnl"]),
        total_pnl=float(metrics["total_pnl"]),
        pf=float(metrics["pf"]),
        avg_hold=float(metrics["avg_hold"]),
        sharpe=round(float(sharpe), 4),
        max_drawdown=round(float(max_drawdown), 4),
        mdd_per_symbol=round(float(mdd_per_symbol), 4),
        yearly_consistency=round(float(yearly_consistency), 4),
        composite_score=float(composite),
        score_mode="live",
        n_symbols=n_symbols,
        first_test_year=first_year,
        last_test_year=last_year,
        cost_profile=cost_profile,
        fairness_group_key=_fairness_group_key(
            symbols,
            first_year,
            last_year,
            cost_profile,
            target,
            str(
                predictions_meta.get("market")
                or resolved_config.get("market")
                or ranking_row.get("market")
                or "unknown"
            ),
            str(
                predictions_meta.get("currency")
                or resolved_config.get("execution", {}).get("currency")
                or ranking_row.get("currency")
                or "unknown"
            ),
            str(predictions_meta.get("schema") or ranking_row.get("schema") or "unknown"),
        ),
        warnings=warnings,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_trades(path: Path) -> list[dict[str, Any]]:
    trades = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            entry_date = row.get("entry_date", "")
            exit_date = row.get("exit_date", "")
            trades.append(
                {
                    "symbol": row.get("symbol", "_"),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "pnl_pct": float(row.get("pnl_pct") or 0.0),
                    "pnl": float(row.get("pnl") or 0.0),
                    "holding_days": _holding_days(entry_date, exit_date),
                }
            )
    return trades


def _holding_days(entry_date: str, exit_date: str) -> int:
    try:
        return (datetime.fromisoformat(exit_date) - datetime.fromisoformat(entry_date)).days
    except ValueError:
        return 0


def _metrics_from_cache(
    metrics_cache: dict[str, Any], ranking_row: dict[str, Any]
) -> dict[str, Any]:
    return {
        "trades": int(metrics_cache.get("trades", ranking_row.get("trade_count", 0))),
        "wr": float(metrics_cache.get("wr", ranking_row.get("win_rate", 0.0))),
        "avg_pnl": float(metrics_cache.get("avg_pnl", 0.0)),
        "total_pnl": float(metrics_cache.get("total_pnl", ranking_row.get("total_pnl", 0.0))),
        "pf": float(metrics_cache.get("pf", 0.0)),
        "max_loss": float(metrics_cache.get("max_loss", 0.0)),
        "avg_hold": float(metrics_cache.get("avg_hold", ranking_row.get("avg_holding_days", 0.0))),
    }


def _target_config(config: dict[str, Any]) -> TargetConfig:
    target = config.get("signals", {}).get("target", {})
    return TargetConfig(
        type=str(target.get("type", "unknown")),
        forward_window=int(target.get("forward_window", 0)),
        gain_threshold=target.get("gain_threshold"),
        loss_threshold=target.get("loss_threshold"),
    )


def _cost_profile(config: dict[str, Any], warnings: list[str]) -> CostProfile:
    values = dict(DEFAULT_TRADING_COST)
    values.update(_known_cost_values(config.get("evaluation", {})))
    values.update(_known_cost_values(config.get("execution", {})))
    if any(value == "unknown" for value in values.values()):
        warnings.append(COST_PROFILE_UNKNOWN_WARNING)
    return CostProfile(**values)


def _known_cost_values(config_section: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config_section[key]
        for key in DEFAULT_TRADING_COST
        if key in config_section and config_section[key] != "unknown"
    }


def _test_window(
    config: dict[str, Any], meta: dict[str, Any], trades: list[dict[str, Any]]
) -> tuple[int, int]:
    split = config.get("split", {}) or meta.get("split", {})
    first_year = split.get("first_test_year")
    last_year = split.get("last_test_year")
    years = sorted(
        {
            int(str(t["entry_date"])[:4])
            for t in trades
            if str(t.get("entry_date", ""))[:4].isdigit()
        }
    )
    if first_year is None:
        first_year = years[0] if years else 0
    if last_year is None:
        last_year = years[-1] if years else 0
    return int(first_year), int(last_year)


def _fairness_group_key(
    symbols: list[str],
    first_year: int,
    last_year: int,
    cost_profile: CostProfile,
    target: TargetConfig,
    market: str,
    currency: str,
    schema: str,
) -> str:
    key_obj = {
        "market": market,
        "currency": currency,
        "schema": schema,
        "symbols": symbols,
        "window": [first_year, last_year],
        "cost_profile": cost_profile.model_dump(),
        "target": target.model_dump(),
    }
    payload = json.dumps(key_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(tzinfo=None).isoformat()
