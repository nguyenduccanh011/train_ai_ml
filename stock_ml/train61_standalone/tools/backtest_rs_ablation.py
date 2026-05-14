from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = (
    ROOT
    / "results"
    / "experiments"
    / "v22_exit_ablation_round42"
    / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
    / "config.resolved.yaml"
)
DEFAULT_SYMBOLS = ROOT / "config" / "liquidity_1b_symbols.json"
DEFAULT_OUT_DIR = ROOT / "results" / "rs_ablation"
DEFAULT_DATA_DIR = ROOT / "data" / "vn_stock_ai_dataset"


def _load_symbols(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = [str(symbol).upper() for symbol in payload.get("symbols", []) if str(symbol).strip()]
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    return sorted(set(symbols))


def _set_feature_set(cfg: Any, feature_set: str, name_suffix: str) -> Any:
    clone = cfg.model_copy(deep=True)
    clone.name = f"{clone.name}__{name_suffix}"
    clone.signals.features = feature_set
    clone.components.features = feature_set
    return clone


def _latest_data_year(symbols: list[str], data_dir: Path) -> int | None:
    from src.data.loader import DataLoader

    loader = DataLoader(str(data_dir))
    years: list[int] = []
    for symbol in symbols:
        try:
            df = loader.load_symbol(symbol)
        except FileNotFoundError:
            continue
        ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce").dropna()
        if not ts.empty:
            years.append(int(ts.max().year))
    return max(years) if years else None


def _align_last_test_year(cfg: Any, symbols: list[str], data_dir: Path) -> Any:
    latest_year = _latest_data_year(symbols, data_dir)
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


def _metric_summary(
    result: Any, feature_set: str, symbols: list[str], runtime_sec: float
) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    feature_cols: list[str] = []
    if result.prediction_cache:
        feature_cols = list(result.prediction_cache[0].get("feature_cols", []))
    rs_feature_cols = [
        col
        for col in feature_cols
        if str(col).startswith("rs_") or str(col).startswith("relative_strength")
    ]
    pnl = pd.Series(dtype=float)
    if isinstance(result.trades_df, pd.DataFrame) and "pnl_pct" in result.trades_df.columns:
        pnl = pd.to_numeric(result.trades_df["pnl_pct"], errors="coerce").dropna()
    wins = int((pnl > 0).sum()) if not pnl.empty else 0
    losses = int((pnl < 0).sum()) if not pnl.empty else 0
    gross_profit = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if not pnl.empty else 0.0
    return {
        "feature_set": feature_set,
        "n_symbols": len(symbols),
        "feature_count": len(feature_cols),
        "rs_feature_cols": rs_feature_cols,
        "runtime_sec": round(runtime_sec, 2),
        "trades": int(metrics.get("total_trades", metrics.get("trade_count", len(result.trades)))),
        "win_rate": round(wins / len(pnl) * 100, 2) if not pnl.empty else metrics.get("win_rate"),
        "total_pnl": round(float(pnl.sum()), 4) if not pnl.empty else metrics.get("total_pnl"),
        "avg_pnl": round(float(pnl.mean()), 4) if not pnl.empty else metrics.get("avg_pnl"),
        "pf": round(gross_profit / gross_loss, 4) if gross_loss else metrics.get("pf"),
        "max_loss": round(float(pnl.min()), 4) if not pnl.empty else metrics.get("max_loss"),
        "max_drawdown": metrics.get("max_drawdown"),
        "mdd_per_symbol": metrics.get("mdd_per_symbol"),
        "yearly_consistency": metrics.get("yearly_consistency"),
        "composite_score": metrics.get("composite_score"),
        "wins": wins,
        "losses": losses,
    }


def _run_one(label: str, cfg: Any, symbols: list[str], out_dir: Path) -> dict[str, Any]:
    from src.pipeline import Pipeline
    from src.pipeline.cache import PredictionCacheManager

    print(f"\n=== Running {label}: features={cfg.feature_set()} symbols={len(symbols)} ===")
    started = time.time()
    cache_mgr = PredictionCacheManager(out_dir / "prediction_cache")
    result = Pipeline(cfg, symbols=symbols, device="cpu", cache_manager=cache_mgr).run()
    runtime_sec = time.time() - started

    trades_path = out_dir / f"{label}_trades.csv"
    if not result.trades_df.empty:
        result.trades_df.to_csv(trades_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)

    summary = _metric_summary(result, cfg.feature_set(), symbols, runtime_sec)
    summary["prediction_cache"] = cache_mgr.stats()
    summary["trades_path"] = str(trades_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest Top1 model with no RS, old RS, and weighted-rank RS."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    os.environ["STOCK_DATA_DIR"] = str(args.data_dir.resolve())
    os.environ["STOCK_RESULTS_DIR"] = str((ROOT / "results").resolve())

    from src.pipeline import ExperimentConfig

    symbols = _load_symbols(args.symbols)
    base_cfg = ExperimentConfig.from_yaml(args.config)
    no_rs_cfg = _align_last_test_year(
        _set_feature_set(base_cfg, "leading", "no_rs"),
        symbols,
        args.data_dir,
    )
    rs_cfg = _align_last_test_year(
        _set_feature_set(base_cfg, "leading_rs", "with_rs"),
        symbols,
        args.data_dir,
    )
    weighted_rs_cfg = _align_last_test_year(
        _set_feature_set(base_cfg, "leading_rs_weighted", "weighted_rs"),
        symbols,
        args.data_dir,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        _run_one("no_rs", no_rs_cfg, symbols, args.out_dir),
        _run_one("with_rs", rs_cfg, symbols, args.out_dir),
        _run_one("weighted_rs", weighted_rs_cfg, symbols, args.out_dir),
    ]

    by_label = {summary["feature_set"]: summary for summary in summaries}
    delta = {}
    no_rs = by_label.get("leading", {})
    with_rs = by_label.get("leading_rs", {})
    weighted_rs = by_label.get("leading_rs_weighted", {})
    for key in [
        "trades",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "pf",
        "max_drawdown",
        "mdd_per_symbol",
        "yearly_consistency",
        "composite_score",
    ]:
        left = with_rs.get(key)
        right = no_rs.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta[f"{key}_with_rs_minus_no_rs"] = round(float(left) - float(right), 6)
        weighted_left = weighted_rs.get(key)
        old_right = with_rs.get(key)
        if isinstance(weighted_left, (int, float)) and isinstance(old_right, (int, float)):
            delta[f"{key}_weighted_rs_minus_old_rs"] = round(
                float(weighted_left) - float(old_right),
                6,
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args.config),
        "symbols": str(args.symbols),
        "data_dir": str(args.data_dir),
        "summaries": summaries,
        "delta": delta,
    }
    report_path = args.out_dir / "rs_ablation_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== Summary ===")
    for summary in summaries:
        print(
            f"{summary['feature_set']}: trades={summary['trades']} "
            f"wr={summary['win_rate']} pnl={summary['total_pnl']} "
            f"pf={summary['pf']} score={summary['composite_score']} "
            f"runtime={summary['runtime_sec']}s"
        )
    print("delta:", delta)
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
