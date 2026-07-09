"""
Batch retrain 315 leaky models với pivot fix + gap_days=25.

Workflow:
1. Load leaderboard.csv, filter leaky models (leading_v2/v3/v4/deriv/leading)
2. For each model:
   - Load config.resolved.yaml
   - Override gap_days=25
   - Run Pipeline
   - Save artifacts (trades.csv, metrics.json, ranking_row.json)
3. Rebuild leaderboard từ artifacts mới

Usage:
    python batch_retrain_leaky_models.py --verify-only  # Chỉ verify 5 models đầu
    python batch_retrain_leaky_models.py --batch-size 50  # Retrain 50 models
    python batch_retrain_leaky_models.py --all  # Retrain toàn bộ 315 models
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.scoring import (
    calc_max_drawdown,
    calc_mdd_per_symbol,
    calc_symbol_coverage,
    calc_yearly_consistency,
    composite_score,
)
from src.leaderboard.fairness import backtest_window_key, load_config, resolve_market_family
from src.pipeline import ExperimentConfig, Pipeline


def _stable_config_hash(config_dict: dict) -> str:
    """Generate stable hash from config dict."""
    import hashlib
    s = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_artifacts(result, cfg, run_dir: Path, config_dict: dict, run_meta: dict = None):
    """Save experiment artifacts (trades.csv, metrics.json, ranking_row.json)."""
    config_hash = _stable_config_hash(config_dict)
    run_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = cfg.split.model_dump()
    market_family = resolve_market_family(
        cfg.market,
        str(result.metadata.get("timeframe", "unknown")),
        load_config(),
    )
    window_key = backtest_window_key(
        int(split_cfg.get("first_test_year", 0)),
        int(split_cfg.get("last_test_year", 0)),
    )
    execution_cfg = {
        **(config_dict.get("execution", {}) or {}),
        **result.metadata.get("execution", {}),
    }
    for key in ("currency", "pnl_mode"):
        if key in result.metadata:
            execution_cfg[key] = result.metadata[key]

    if not result.trades_df.empty:
        result.trades_df.to_csv(run_dir / "trades.csv", index=False)

    trades = result.trades_df.to_dict("records") if not result.trades_df.empty else []
    symbol_coverage = calc_symbol_coverage(trades)
    metrics = dict(result.metrics)
    if trades and "composite_score" not in metrics:
        metrics["composite_score"] = composite_score(metrics, trades)
    metrics["max_drawdown"] = round(calc_max_drawdown(trades), 2)
    metrics.setdefault("mdd_per_symbol", round(calc_mdd_per_symbol(trades), 2))
    metrics.setdefault("yearly_consistency", round(calc_yearly_consistency(trades), 4))
    metrics["symbol_coverage"] = symbol_coverage

    predictions_meta = {
        "config_hash": config_hash,
        "market": cfg.market,
        "market_family": market_family,
        "currency": execution_cfg.get("currency", "unknown"),
        "pnl_mode": execution_cfg.get("pnl_mode", "equity_spot"),
        "schema": result.metadata.get("schema", "unknown"),
        "timeframe": result.metadata.get("timeframe", "unknown"),
        "entry_model": cfg.entry_model_type(),
        "exit_model_type": cfg.signals.exit_model.type,
        "exit_model_enabled": cfg.signals.exit_model.enabled,
        "feature_set": cfg.feature_set(),
        "split": split_cfg,
        "backtest_window_key": window_key,
        "cache_stats": result.metadata.get("cache_stats", {}),
        "created_at": datetime.now().isoformat(),
    }
    if run_meta:
        predictions_meta.update(run_meta)

    yearly_consistency = metrics.get("yearly_consistency", 0.0)
    ranking_row = {
        "name": result.name,
        "composite_score": metrics.get("composite_score", 0.0),
        "total_pnl": metrics.get("total_pnl", 0.0),
        "win_rate": metrics.get("wr", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "mdd_per_symbol": metrics.get("mdd_per_symbol", 0.0),
        "yearly_consistency": yearly_consistency,
        "trade_count": metrics.get("trades", 0),
        "avg_holding_days": metrics.get("avg_hold", 0.0),
        "per_year_consistency": yearly_consistency,
        "feature_set": predictions_meta["feature_set"],
        "entry_model": predictions_meta["entry_model"],
        "exit_model_type": predictions_meta["exit_model_type"],
        "exit_model_enabled": predictions_meta["exit_model_enabled"],
        "per_symbol_coverage": symbol_coverage,
        "config_hash": config_hash,
        "market": predictions_meta["market"],
        "market_family": predictions_meta["market_family"],
        "currency": predictions_meta["currency"],
        "pnl_mode": predictions_meta["pnl_mode"],
        "schema": predictions_meta["schema"],
        "timeframe": predictions_meta["timeframe"],
        "backtest_window_key": predictions_meta["backtest_window_key"],
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (run_dir / "ranking_row.json").write_text(
        json.dumps(ranking_row, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (run_dir / "predictions_meta.json").write_text(
        json.dumps(predictions_meta, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def retrain_model(bundle: str, run_name: str, symbols: list[str], device: str = "cpu"):
    """Retrain single model với leakage fix."""
    experiments_dir = ROOT / "results/experiments"
    run_dir = experiments_dir / bundle / run_name
    config_path = run_dir / "config.resolved.yaml"

    if not config_path.exists():
        print(f"  [SKIP] Config not found: {config_path}")
        return None

    print(f"\n{'='*80}")
    print(f"Retraining: {bundle}/{run_name}")
    print(f"{'='*80}")

    # Load config
    cfg = ExperimentConfig.from_yaml(config_path)
    config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Override gap_days=25 (leakage fix)
    original_gap = cfg.split.gap_days
    cfg.split.gap_days = 25
    config_dict["split"]["gap_days"] = 25

    print(f"  Original gap_days: {original_gap}")
    print(f"  Fixed gap_days: {cfg.split.gap_days}")
    print(f"  Feature set: {cfg.feature_set()}")
    print(f"  Test years: {cfg.split.first_test_year} - {cfg.split.last_test_year}")
    print(f"  Symbols: {len(symbols)}")

    # Run pipeline
    start_time = time.time()
    try:
        result = Pipeline(cfg, symbols=symbols, device=device).run()
        elapsed = time.time() - start_time

        # Save artifacts
        save_artifacts(result, cfg, run_dir, config_dict)

        print(f"\n  [OK] Retrain completed in {elapsed:.1f}s")
        print(f"  Trades: {len(result.trades_df)}")
        print(f"  WR: {result.metrics.get('wr', 0):.2f}%")
        print(f"  PF: {result.metrics.get('pf', 0):.2f}")
        print(f"  Total PnL: {result.metrics.get('total_pnl', 0):.2f}")
        print(f"  Composite score: {result.metrics.get('composite_score', 0):.2f}")

        return {
            "bundle": bundle,
            "run_name": run_name,
            "status": "success",
            "elapsed": elapsed,
            "metrics": result.metrics,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  [ERROR] Retrain failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            "bundle": bundle,
            "run_name": run_name,
            "status": "error",
            "elapsed": elapsed,
            "error": str(e),
        }


def _append_log(results, batch_size, skipped):
    """Append progress to incremental log file."""
    log_path = ROOT / "results/leakage_check/batch_retrain_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    total_time = sum(r["elapsed"] for r in results)
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "completed": len(results),
            "skipped": skipped,
            "success": success,
            "failed": failed,
            "total_time_minutes": round(total_time / 60, 1),
            "results": results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch retrain leaky models")
    parser.add_argument("--verify-only", action="store_true", help="Verify with 5 models")
    parser.add_argument("--batch-size", type=int, help="Number of models to retrain")
    parser.add_argument("--all", action="store_true", help="Retrain all leaky models")
    parser.add_argument("--device", default="cpu", help="Training device (cpu/cuda)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models with ranking_row.json modified after --skip-cutoff",
    )
    parser.add_argument(
        "--skip-cutoff",
        default="2026-05-18T08:00:00",
        help="ISO timestamp - skip models with ranking_row.json modified after this (default: 2026-05-18T08:00:00 = batch start)",
    )
    args = parser.parse_args()

    # Load leaderboard
    leaderboard_path = ROOT / "results/leaderboard/leaderboard.csv"
    df = pd.read_csv(leaderboard_path)

    # Filter leaky models
    leaky_sets = ["leading_v2", "leading_v3", "leading_v4", "leading_deriv", "leading"]
    leaky = df[df["feature_set"].isin(leaky_sets)].copy()

    print(f"Total models in leaderboard: {len(df)}")
    print(f"Leaky models: {len(leaky)}")
    print("\nLeaky feature sets:")
    print(leaky["feature_set"].value_counts().to_string())

    # Load symbols
    manifest_path = ROOT / "visualization/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols = [str(s).upper() for s in manifest.get("base_symbols", [])]
    print(f"\nSymbols: {len(symbols)}")

    # Determine batch size
    if args.verify_only:
        batch_size = 5
        print(f"\n[VERIFY MODE] Retrain first {batch_size} models")
    elif args.batch_size:
        batch_size = args.batch_size
        print(f"\n[BATCH MODE] Retrain {batch_size} models")
    elif args.all:
        batch_size = len(leaky)
        print(f"\n[FULL MODE] Retrain all {batch_size} models")
    else:
        print("\nERROR: Must choose --verify-only, --batch-size N, or --all")
        return 1

    # Skip existing logic
    skipped = 0
    cutoff_ts = None
    if args.skip_existing:
        from datetime import datetime as _dt
        cutoff_ts = _dt.fromisoformat(args.skip_cutoff).timestamp()
        print(f"\n[SKIP EXISTING] Cutoff: {args.skip_cutoff}")

    # Retrain
    results = []
    target_models = leaky.head(batch_size)
    for idx, (_, row) in enumerate(target_models.iterrows(), 1):
        # Check skip
        if args.skip_existing:
            ranking_path = ROOT / "results/experiments" / row["bundle"] / row["run_name"] / "ranking_row.json"
            if ranking_path.exists() and ranking_path.stat().st_mtime >= cutoff_ts:
                skipped += 1
                if skipped <= 5 or skipped % 20 == 0:
                    print(f"[{idx}/{batch_size}] [SKIP] Already retrained: {row['bundle']}/{row['run_name'][:60]}...")
                continue

        print(f"\n[{idx}/{batch_size}] Processing: {row['bundle']}/{row['run_name']}")
        result = retrain_model(row["bundle"], row["run_name"], symbols, device=args.device)
        if result:
            results.append(result)
            # Append to incremental log after each model
            _append_log(results, batch_size, skipped)

    # Summary
    print(f"\n{'='*80}")
    print("BATCH RETRAIN SUMMARY")
    print(f"{'='*80}")
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    total_time = sum(r["elapsed"] for r in results)

    print(f"Total target: {batch_size}")
    print(f"Skipped (already retrained): {skipped}")
    print(f"Processed: {len(results)}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    if results:
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Avg time per model: {total_time/len(results):.1f}s")

    if failed > 0:
        print("\nFailed models:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['bundle']}/{r['run_name']}: {r['error']}")

    # Save results log
    log_path = ROOT / "results/leakage_check/batch_retrain_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "results": results,
            "summary": {
                "success": success,
                "failed": failed,
                "total_time": total_time,
            }
        }, f, indent=2)
    print(f"\nLog saved: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
