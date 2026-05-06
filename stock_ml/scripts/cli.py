"""CLI for stock_ml pipeline v2.

Usage:
    python -m stock_ml run champions/v22
    python -m stock_ml run champions/v22 --device gpu
    python -m stock_ml run-matrix matrix/model_selection
    python -m stock_ml validate champions/v22
    python -m stock_ml list-components
    python -m stock_ml list-components --type fusion
    python -m stock_ml list-experiments
    python -m stock_ml compare champions/v22 champions/v34
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config" / "experiments"


def _resolve_yaml(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.suffix == ".yaml" or p.suffix == ".yml":
        return p if p.is_absolute() else REPO_ROOT / "config" / "experiments" / p
    return REPO_ROOT / "config" / "experiments" / f"{path_arg}.yaml"


def _load_symbols(market: str | None = None, limit: int | None = None) -> list[str]:
    from src.config_loader import get_pipeline_symbols

    symbols = get_pipeline_symbols(market=market)
    if limit is not None:
        return symbols[:limit]
    return symbols


def _print_result_summary(result_name: str, n_trades: int, trades_df: Any) -> None:
    print(f"\nResult: {n_trades} trades for {result_name}")
    if not hasattr(trades_df, "empty") or trades_df.empty:
        return
    df = trades_df
    if "pnl_pct" in df.columns:
        print(f"  Win rate: {(df['pnl_pct'] > 0).mean():.1%}")
        print(f"  Avg PnL:  {df['pnl_pct'].mean():.2f}%")


def _save_result_csv(result_name: str, trades_df: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _stable_config_hash(config_dict: dict[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _artifact_run_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _make_matrix_short_name(exp_name: str, row: dict[str, Any], matrix_prefix: str) -> str:
    feature_map = {"leading_v2": "v2", "leading_v3": "v3", "leading_v4": "v4"}
    entry_map = {
        "random_forest": "rf",
        "gru": "gru",
        "lightgbm": "lgb",
        "xgboost": "xgb",
        "catboost": "cat",
    }

    feature_set = str(row.get("feature_set") or "")
    entry_model = str(row.get("entry_model") or "")
    feature = feature_map.get(feature_set, feature_set)
    entry = entry_map.get(entry_model, entry_model)

    match = re.search(r"-strategy-([A-Za-z0-9_]+)$", exp_name)
    strategy = match.group(1) if match else ""

    if feature and entry and strategy:
        return f"{feature}_{entry}_{strategy}"

    stripped = exp_name.replace(f"{matrix_prefix}_", "").replace(matrix_prefix, "")
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", stripped).strip("_")
    return slug[:40] or "matrix_exp"


def _get_matrix_artifact_dir(matrix_name: str, run_name: str) -> Path:
    from src.env import get_results_dir

    return Path(get_results_dir()) / "experiments" / matrix_name / _artifact_run_name(run_name)


def _get_experiment_artifact_dir(run_name: str) -> Path:
    from src.env import get_results_dir

    return Path(get_results_dir()) / "experiments" / _artifact_run_name(run_name)


def _default_leaderboard_dir() -> Path:
    from src.env import get_results_dir

    return Path(get_results_dir()) / "leaderboard"


def _update_leaderboard(run_dir: Path, *, skip_leaderboard: bool = False) -> None:
    if skip_leaderboard:
        return
    from src.leaderboard.hooks import append_or_update

    row = append_or_update(run_dir, _default_leaderboard_dir())
    print(f"  Leaderboard: {row.run_id}")


def _save_experiment_artifacts(cfg: Any, result: Any, *, skip_leaderboard: bool = False) -> None:
    _save_artifacts(_get_experiment_artifact_dir(result.name), cfg, result)
    _update_leaderboard(
        _get_experiment_artifact_dir(result.name), skip_leaderboard=skip_leaderboard
    )


def _save_matrix_artifacts(
    matrix_name: str,
    cfg: Any,
    result: Any,
    run_meta: dict[str, Any] | None = None,
    *,
    skip_leaderboard: bool = False,
) -> Path:
    run_dir = _get_matrix_artifact_dir(matrix_name, result.name)
    _save_artifacts(run_dir, cfg, result, run_meta)
    _update_leaderboard(run_dir, skip_leaderboard=skip_leaderboard)
    return run_dir


def _save_artifacts(
    run_dir: Path, cfg: Any, result: Any, run_meta: dict[str, Any] | None = None
) -> None:
    from src.evaluation.scoring import (
        calc_max_drawdown,
        calc_mdd_per_symbol,
        calc_symbol_coverage,
        calc_yearly_consistency,
        composite_score,
    )

    config_dict = cfg.model_dump()
    config_hash = _stable_config_hash(config_dict)
    run_dir.mkdir(parents=True, exist_ok=True)
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
        "currency": execution_cfg.get("currency", "unknown"),
        "pnl_mode": execution_cfg.get("pnl_mode", "equity_spot"),
        "schema": result.metadata.get("schema", "unknown"),
        "timeframe": result.metadata.get("timeframe", "unknown"),
        "entry_model": cfg.entry_model_type(),
        "exit_model_type": cfg.signals.exit_model.type,
        "exit_model_enabled": cfg.signals.exit_model.enabled,
        "feature_set": cfg.feature_set(),
        "split": cfg.split.model_dump(),
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
        "currency": predictions_meta["currency"],
        "pnl_mode": predictions_meta["pnl_mode"],
        "schema": predictions_meta["schema"],
        "timeframe": predictions_meta["timeframe"],
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
    print(f"  Artifacts: {run_dir}")


def cmd_run(args: argparse.Namespace) -> int:
    device = args.device or "cpu"

    from src.pipeline import ExperimentConfig, Pipeline

    yaml_path = _resolve_yaml(args.experiment)
    if not yaml_path.exists():
        print(f"Error: config not found: {yaml_path}", file=sys.stderr)
        return 1

    cfg = ExperimentConfig.from_yaml(yaml_path)
    symbols = _load_symbols(cfg.market)
    pipeline = Pipeline(cfg, symbols=symbols, device=device)
    result = pipeline.run()

    _print_result_summary(result.name, result.n_trades, result.trades_df)

    if args.output:
        _save_result_csv(result.name, result.trades_df, Path(args.output))
    elif getattr(args, "save_results", False):
        if not result.trades_df.empty:
            results_dir = REPO_ROOT / "results"
            out = results_dir / f"trades_{cfg.strategy}.csv"
            _save_result_csv(result.name, result.trades_df, out)
        _save_experiment_artifacts(cfg, result, skip_leaderboard=args.skip_leaderboard)

    if getattr(args, "export", False) and not result.trades_df.empty:
        _export_to_dashboard(cfg.strategy, result.trades_df)

    return 0


def _run_matrix_worker(payload: dict[str, Any]) -> dict[str, Any] | None:
    from src.pipeline import Pipeline
    from src.pipeline.cache import PredictionCacheManager

    cfg = payload["cfg"]
    matrix_name = payload["matrix_name"]
    symbols_limit = payload.get("symbols_limit")
    symbols = _load_symbols(cfg.market, symbols_limit)
    if not symbols:
        raise ValueError(f"No symbols resolved for market='{cfg.market}' in run '{cfg.name}'")
    device = payload["device"]
    save_results = payload["save_results"]
    run_meta = payload.get("run_meta") or {}
    cache_root = Path(payload["cache_root"])

    cache_manager = PredictionCacheManager(cache_root)
    start = time.perf_counter()
    pipeline = Pipeline(cfg, symbols=symbols, device=device, cache_manager=cache_manager)
    result = pipeline.run()
    elapsed_seconds = round(time.perf_counter() - start, 3)
    worker_run_meta = dict(run_meta)
    worker_run_meta["elapsed_seconds"] = elapsed_seconds
    worker_run_meta["symbols_count"] = len(symbols)
    worker_run_meta["market"] = cfg.market
    run_dir = None
    if save_results:
        run_dir = _save_matrix_artifacts(
            matrix_name, cfg, result, worker_run_meta, skip_leaderboard=True
        )
    return {
        "name": cfg.name,
        "n_trades": result.n_trades,
        "elapsed_seconds": elapsed_seconds,
        "cache_stats": cache_manager.stats(),
        "run_dir": str(run_dir) if run_dir is not None else None,
    }


def _print_cache_stats(cache_stats: dict[str, int]) -> None:
    print(
        "[run-matrix] Cache stats - "
        f"hits: {cache_stats.get('hits', 0)}  "
        f"misses: {cache_stats.get('misses', 0)}  "
        f"stored: {cache_stats.get('stored', 0)}"
    )


def _merge_cache_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key in ("hits", "misses", "stored"):
        target[key] = target.get(key, 0) + int(source.get(key, 0))


def _run_matrix_configs_parallel(
    *,
    matrix_name: str,
    configs: list[Any],
    symbols_limit: int | None,
    device: str,
    save_results: bool,
    run_meta: dict[str, Any] | None,
    jobs: int,
    cache_root: Path,
    skip_leaderboard: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cache_stats = {"hits": 0, "misses": 0, "stored": 0}
    payloads = [
        {
            "matrix_name": matrix_name,
            "cfg": cfg,
            "symbols_limit": symbols_limit,
            "device": device,
            "save_results": save_results,
            "run_meta": run_meta,
            "cache_root": str(cache_root),
        }
        for cfg in configs
    ]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_matrix_worker, payload): payload["cfg"].name
            for payload in payloads
        }
        for index, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            result = future.result()
            if result is None:
                continue
            results.append(result)
            run_dir = result.get("run_dir")
            if save_results and run_dir:
                _update_leaderboard(Path(run_dir), skip_leaderboard=skip_leaderboard)
            _merge_cache_stats(cache_stats, result.get("cache_stats", {}))
            print(
                f"[run-matrix] {index}/{len(configs)} {name} done in "
                f"{result['elapsed_seconds']:.1f}s -> {result['n_trades']} trades"
            )
    _print_cache_stats(cache_stats)
    return results


def _run_matrix_configs(
    *,
    matrix_name: str,
    configs: list[Any],
    symbols_limit: int | None,
    device: str,
    dry_run: bool,
    save_results: bool,
    resume: bool,
    run_meta: dict[str, Any] | None = None,
    jobs: int = 1,
    skip_leaderboard: bool = False,
) -> list[Any]:
    from src.env import get_results_dir
    from src.pipeline import Pipeline
    from src.pipeline.cache import PredictionCacheManager

    if resume and not save_results:
        print("  [WARN] --resume ignored because --no-save-results was set")
    if jobs < 1:
        print("  [WARN] --jobs must be >= 1; using --jobs 1")
        jobs = 1

    cache_root = Path(get_results_dir()) / "cache" / "predictions"
    runnable_configs: list[Any] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg.name}")
        print(
            f"  market={cfg.market} strategy={cfg.strategy} features={cfg.feature_set()} "
            f"entry={cfg.entry_model_type()} exit={cfg.signals.exit_model.type} "
            f"exit_enabled={cfg.signals.exit_model.enabled}"
        )
        if dry_run:
            continue
        if resume and save_results:
            run_dir = _get_matrix_artifact_dir(matrix_name, cfg.name)
            if (run_dir / "ranking_row.json").exists():
                print("  [SKIP] already done")
                continue
        runnable_configs.append(cfg)

    if dry_run or not runnable_configs:
        return []

    if jobs > 1 and device != "cpu":
        print("  [WARN] --jobs > 1 is CPU-only; running serial for non-cpu device")
        jobs = 1
    if jobs > 1:
        return _run_matrix_configs_parallel(
            matrix_name=matrix_name,
            configs=runnable_configs,
            symbols_limit=symbols_limit,
            device=device,
            save_results=save_results,
            run_meta=run_meta,
            jobs=jobs,
            cache_root=cache_root,
            skip_leaderboard=skip_leaderboard,
        )

    cache_manager = PredictionCacheManager(cache_root)
    results = []
    for i, cfg in enumerate(runnable_configs, 1):
        symbols = _load_symbols(cfg.market, symbols_limit)
        if not symbols:
            raise ValueError(f"No symbols resolved for market='{cfg.market}' in run '{cfg.name}'")
        start = time.perf_counter()
        pipeline = Pipeline(cfg, symbols=symbols, device=device, cache_manager=cache_manager)
        result = pipeline.run()
        elapsed_seconds = round(time.perf_counter() - start, 3)
        results.append(result)
        print(f"  -> {result.n_trades} trades")
        print(f"[run-matrix] {i}/{len(runnable_configs)} {cfg.name} done in {elapsed_seconds:.1f}s")
        if save_results:
            save_run_meta = dict(run_meta or {})
            save_run_meta["elapsed_seconds"] = elapsed_seconds
            save_run_meta["symbols_count"] = len(symbols)
            save_run_meta["market"] = cfg.market
            _save_matrix_artifacts(
                matrix_name, cfg, result, save_run_meta, skip_leaderboard=skip_leaderboard
            )
    _print_cache_stats(cache_manager.stats())
    return results


def _write_matrix_ranking(bundle_dir: Path) -> None:
    import pandas as pd

    rows = []
    for row_path in sorted(bundle_dir.glob("*/ranking_row.json")):
        rows.append(json.loads(row_path.read_text(encoding="utf-8")))
    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    if "rank" in df.columns:
        df = df.drop(columns=["rank"])
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(bundle_dir / "ranking.csv", index=False)
    df.to_json(bundle_dir / "ranking.json", orient="records", indent=2, force_ascii=False)
    print(
        f"[matrix] ranking saved -> {bundle_dir / 'ranking.csv'} and {bundle_dir / 'ranking.json'} ({len(df)} rows)"
    )


def _select_top_k_matrix_configs(matrix_name: str, configs: list[Any], k: int) -> list[Any]:
    rows = []
    by_name = {cfg.name: cfg for cfg in configs}
    for cfg in configs:
        row_path = _get_matrix_artifact_dir(matrix_name, cfg.name) / "ranking_row.json"
        if row_path.exists():
            row = json.loads(row_path.read_text(encoding="utf-8"))
            rows.append((cfg.name, float(row.get("composite_score", 0.0))))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [by_name[name] for name, _score in rows[:k] if name in by_name]


def cmd_run_matrix(args: argparse.Namespace) -> int:
    from src.pipeline.matrix_expander import expand_matrix

    yaml_path = _resolve_yaml(args.matrix)
    if not yaml_path.exists():
        print(f"Error: matrix config not found: {yaml_path}", file=sys.stderr)
        return 1

    from src.env import get_results_dir

    configs = expand_matrix(yaml_path, limit=args.limit)
    from src.pipeline.validate import validate_config

    for cfg in configs:
        errors, warnings = validate_config(cfg, strict=args.strict_validate)
        for warning in warnings:
            print(f"WARNING [{cfg.name}]: {warning}", file=sys.stderr)
        if errors:
            for err in errors:
                print(f"Validation error [{cfg.name}]: {err}", file=sys.stderr)
            return 1

    device = args.device or "cpu"
    matrix_bundle_dir = Path(get_results_dir()) / "experiments" / yaml_path.stem

    print(f"Matrix: {len(configs)} experiments from {yaml_path.name}")
    if args.top_k_preview is not None:
        if args.top_k_preview <= 0:
            print("Error: --top-k-preview must be > 0", file=sys.stderr)
            return 1
        preview_limit = args.symbols_limit or 10
        preview_matrix_name = f"{yaml_path.stem}_preview"
        preview_bundle_dir = Path(get_results_dir()) / "experiments" / preview_matrix_name
        print(f"Top-k preview: running {len(configs)} experiments (symbols_limit={preview_limit})")
        _run_matrix_configs(
            matrix_name=preview_matrix_name,
            configs=configs,
            symbols_limit=preview_limit,
            device=device,
            dry_run=args.dry_run,
            save_results=args.save_results,
            resume=args.resume,
            run_meta={
                "run_scope": "preview",
                "symbols_limit": preview_limit,
                "matrix_name": preview_matrix_name,
                "source_matrix_yaml": str(yaml_path),
                "device": device,
            },
            jobs=args.jobs,
            skip_leaderboard=args.skip_leaderboard,
        )
        if args.dry_run:
            return 0
        if args.save_results:
            _write_matrix_ranking(preview_bundle_dir)
        top_configs = _select_top_k_matrix_configs(preview_matrix_name, configs, args.top_k_preview)
        if not top_configs:
            print("Error: no preview ranking rows found", file=sys.stderr)
            return 1
        print(f"\nTop-k full run: {len(top_configs)} experiments")
        _run_matrix_configs(
            matrix_name=yaml_path.stem,
            configs=top_configs,
            symbols_limit=args.symbols_limit,
            device=device,
            dry_run=False,
            save_results=args.save_results,
            resume=args.resume,
            run_meta={
                "run_scope": "full",
                "symbols_limit": args.symbols_limit,
                "matrix_name": yaml_path.stem,
                "source_matrix_yaml": str(yaml_path),
                "device": device,
            },
            jobs=args.jobs,
            skip_leaderboard=args.skip_leaderboard,
        )
        if args.save_results:
            _write_matrix_ranking(matrix_bundle_dir)
        return 0

    _run_matrix_configs(
        matrix_name=yaml_path.stem,
        configs=configs,
        symbols_limit=args.symbols_limit,
        device=device,
        dry_run=args.dry_run,
        save_results=args.save_results,
        resume=args.resume,
        run_meta={
            "run_scope": "full",
            "symbols_limit": args.symbols_limit,
            "matrix_name": yaml_path.stem,
            "source_matrix_yaml": str(yaml_path),
            "device": device,
        },
        jobs=args.jobs,
        skip_leaderboard=args.skip_leaderboard,
    )
    if args.save_results and not args.dry_run:
        _write_matrix_ranking(matrix_bundle_dir)
    return 0


def _load_ranking_rows(bundle_dir: Path) -> list[dict[str, Any]]:
    ranking_path = bundle_dir / "ranking.json"
    if ranking_path.exists():
        return json.loads(ranking_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row_path in sorted(bundle_dir.glob("*/ranking_row.json")):
        rows.append(json.loads(row_path.read_text(encoding="utf-8")))
    return rows


def _ranking_flags(row: dict[str, Any]) -> list[str]:
    flags = []
    coverage = row.get("per_symbol_coverage", {}) or {}
    if row.get("trade_count", 0) < 20:
        flags.append("LOW_TRADES")
    if coverage.get("top_symbol_pnl_ratio", 0.0) > 0.5:
        flags.append("SYMBOL_CONCENTRATION")
    if row.get("per_year_consistency", 0.0) > 1.5:
        flags.append("YEAR_INCONSISTENT")
    if row.get("mdd_per_symbol", 0.0) > 20:
        flags.append("HIGH_DRAWDOWN")
    return flags


def _print_ranking_table(
    rows: list[dict[str, Any]],
    top: int,
    *,
    sort_by: str = "composite_score",
    min_trades: int | None = None,
    max_mdd: float | None = None,
) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if "mdd_per_symbol" not in df.columns:
        df["mdd_per_symbol"] = df.get("max_drawdown", 0.0)
    if "per_year_consistency" not in df.columns:
        df["per_year_consistency"] = df.get("yearly_consistency", 0.0)
    if "yearly_consistency" not in df.columns:
        df["yearly_consistency"] = df["per_year_consistency"]

    if min_trades is not None:
        df = df[df["trade_count"] >= min_trades]
    if max_mdd is not None:
        df = df[df["mdd_per_symbol"] <= max_mdd]
    if df.empty:
        print("No ranking rows match the filters")
        return

    ascending = sort_by in {"mdd_per_symbol", "yearly_consistency", "per_year_consistency"}
    df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    if "rank" in df.columns:
        df = df.drop(columns=["rank"])
    df.insert(0, "rank", range(1, len(df) + 1))

    rule_rows = df[
        (df.get("entry_model", "") == "rule")
        | df["name"].astype(str).str.contains("rule", case=False, na=False)
    ]
    rule_row = rule_rows.iloc[0].to_dict() if not rule_rows.empty else None
    if rule_row:
        df["score_delta_rule"] = df["composite_score"] - float(rule_row.get("composite_score", 0.0))
        df["pnl_delta_rule"] = df["total_pnl"] - float(rule_row.get("total_pnl", 0.0))
        df["mdd_delta_rule"] = df["mdd_per_symbol"] - float(rule_row.get("mdd_per_symbol", 0.0))
        df["wr_delta_rule"] = df["win_rate"] - float(rule_row.get("win_rate", 0.0))

    records = df.to_dict("records")
    df["flags"] = [",".join(_ranking_flags(row)) for row in records]
    display_cols = [
        "rank",
        "name",
        "score",
        "total_pnl",
        "win_rate",
        "mdd",
        "trades",
        "hold_days",
        "yr_cv",
        "flags",
    ]
    display = df.assign(
        score=df["composite_score"],
        mdd=df["mdd_per_symbol"],
        trades=df["trade_count"],
        hold_days=df["avg_holding_days"],
        yr_cv=df["per_year_consistency"],
    )
    if rule_row:
        display_cols.extend(
            ["score_delta_rule", "pnl_delta_rule", "mdd_delta_rule", "wr_delta_rule"]
        )
    print(display.head(top)[display_cols].to_string(index=False))


def cmd_compare_matrix(args: argparse.Namespace) -> int:
    matrix_dir = Path(args.results_dir)
    if not matrix_dir.is_absolute():
        matrix_dir = REPO_ROOT / matrix_dir
    if not matrix_dir.exists():
        print(f"Error: matrix results dir not found: {matrix_dir}", file=sys.stderr)
        return 1

    rows = _load_ranking_rows(matrix_dir)
    if not rows:
        print(f"Error: no ranking rows found under {matrix_dir}", file=sys.stderr)
        return 1

    meta_paths = sorted(matrix_dir.glob("*/predictions_meta.json"))
    if meta_paths:
        sample = json.loads(meta_paths[0].read_text(encoding="utf-8"))
        if sample.get("run_scope") == "preview":
            print(
                "[WARNING] This bundle is a PREVIEW run (limited symbols). Results may not represent full performance.",
                file=sys.stderr,
            )

    _print_ranking_table(
        rows,
        args.top,
        sort_by=args.sort,
        min_trades=args.min_trades,
        max_mdd=args.max_mdd,
    )
    return 0


def cmd_compare_champions(args: argparse.Namespace) -> int:
    from src.env import get_results_dir
    from src.market_profile import resolve_market_name
    from src.pipeline import ExperimentConfig, Pipeline
    from src.pipeline.cache import PredictionCacheManager

    champions_dir = CONFIG_DIR / "champions"
    if args.champions:
        names = [n.strip() for n in args.champions.split(",") if n.strip()]
        yaml_paths = [champions_dir / f"{n}.yaml" for n in names]
        missing = [p for p in yaml_paths if not p.exists()]
        if missing:
            print(
                f"Error: champion configs not found: {[str(p) for p in missing]}", file=sys.stderr
            )
            return 1
        configs_with_paths = [(ExperimentConfig.from_yaml(path), path) for path in yaml_paths]
    else:
        target_market = resolve_market_name(None)
        configs_with_paths = [
            (cfg, path)
            for path in sorted(champions_dir.glob("*.yaml"))
            if (cfg := ExperimentConfig.from_yaml(path)).market == target_market
        ]

    if not configs_with_paths:
        print(f"Error: no champion configs under {champions_dir}", file=sys.stderr)
        return 1

    bundle_name = args.bundle_name or f"champions_{args.first_test_year}_{args.last_test_year}"
    bundle_dir = Path(get_results_dir()) / "experiments" / bundle_name

    device = args.device or "cpu"

    cache_root = Path(get_results_dir()) / "cache" / "predictions"
    cache_manager = PredictionCacheManager(cache_root)

    print(
        f"Bundle: {bundle_name} ({len(configs_with_paths)} champions, split={args.first_test_year}-{args.last_test_year})"
    )
    print(f"Output: {bundle_dir}")

    for i, (cfg, yaml_path) in enumerate(configs_with_paths, 1):
        cfg.split.first_test_year = args.first_test_year
        cfg.split.last_test_year = args.last_test_year
        if cfg.execution is not None:
            cfg.execution.split = cfg.split

        print(f"\n[{i}/{len(yaml_paths)}] {cfg.name} (strategy={cfg.strategy})")
        print(
            f"  features={cfg.feature_set()} entry={cfg.entry_model_type()} "
            f"exit={cfg.signals.exit_model.type} exit_enabled={cfg.signals.exit_model.enabled}"
        )

        run_dir = bundle_dir / _artifact_run_name(cfg.name)
        if args.resume and args.save_results and (run_dir / "ranking_row.json").exists():
            print("  [SKIP] already done")
            continue
        if args.dry_run:
            continue

        symbols = _load_symbols(cfg.market, args.symbols_limit)
        if not symbols:
            print(f"  [SKIP] no symbols resolved for market={cfg.market}")
            continue

        pipeline = Pipeline(cfg, symbols=symbols, device=device, cache_manager=cache_manager)
        result = pipeline.run()
        print(f"  -> {result.n_trades} trades")
        if args.save_results:
            _save_artifacts(
                run_dir,
                cfg,
                result,
                {
                    "run_scope": "full",
                    "market": cfg.market,
                    "symbols_count": len(symbols),
                    "symbols_limit": args.symbols_limit,
                    "matrix_name": bundle_name,
                    "source_matrix_yaml": str(yaml_path),
                    "device": device,
                },
            )

    if args.save_results and not args.dry_run:
        _write_matrix_ranking(bundle_dir)

    if not args.dry_run:
        rows = _load_ranking_rows(bundle_dir)
        if rows:
            print()
            _print_ranking_table(rows, args.top)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from src.pipeline import ExperimentConfig
    from src.pipeline.validate import validate_config

    yaml_path = _resolve_yaml(args.experiment)
    if not yaml_path.exists():
        print(f"Error: config not found: {yaml_path}", file=sys.stderr)
        return 1

    try:
        if args.experiment.startswith(("matrix/", "matrix\\")):
            from src.pipeline.matrix_expander import expand_matrix

            configs = expand_matrix(yaml_path)
        else:
            configs = [ExperimentConfig.from_yaml(yaml_path)]
    except Exception as e:
        print(f"Schema error: {e}", file=sys.stderr)
        return 1

    for cfg in configs:
        errors, warnings = validate_config(cfg, strict=args.strict)
        for warning in warnings:
            print(f"WARNING [{cfg.name}]: {warning}", file=sys.stderr)
        if errors:
            for err in errors:
                print(f"Validation error [{cfg.name}]: {err}", file=sys.stderr)
            return 1

    if len(configs) == 1:
        cfg = configs[0]
        print(f"OK: {cfg.name} (strategy={cfg.strategy}, features={cfg.feature_set()})")
    else:
        print(f"OK: {yaml_path.name} ({len(configs)} experiments)")
    return 0


def cmd_list_components(args: argparse.Namespace) -> int:
    from src.components.exit_models.registry import list_exit_models
    from src.components.features.registry import _BLOCK_REGISTRY
    from src.components.fusion.registry import list_strategies
    from src.components.models.registry import list_models
    from src.components.runners.runner_registry import list_runners
    from src.components.targets.registry import list_targets

    component_type = args.type

    if component_type in (None, "features"):
        print("=== Feature Blocks ===")
        for name in sorted(_BLOCK_REGISTRY):
            print(f"  {name}")

    if component_type in (None, "models"):
        print("=== Entry Models ===")
        for name in list_models():
            print(f"  {name}")

    if component_type in (None, "exit_models"):
        print("=== Exit Models ===")
        for name in list_exit_models():
            print(f"  {name}")

    if component_type in (None, "targets"):
        print("=== Target Generators ===")
        for name in list_targets():
            print(f"  {name}")

    if component_type in (None, "fusion"):
        import src.components.fusion.strategies  # noqa: F401 — trigger strategy registration

        print("=== Fusion Strategies ===")
        for name in list_strategies():
            print(f"  {name}")

    if component_type in (None, "runners"):
        print("=== Strategy Runners ===")
        for name, source in sorted(list_runners().items()):
            print(f"  {name} ({source})")

    return 0


def _export_to_dashboard(version_key: str, trades_df: Any) -> None:
    """Export trades_df to visualization JSON for dashboard consumption."""
    import pandas as pd
    from src.config_loader import get_model_config
    from src.export.unified_export import export_version, generate_manifest

    base_dir = REPO_ROOT
    results_dir = base_dir / "results"
    viz_dir = base_dir / "visualization"

    out_csv = results_dir / f"trades_{version_key}.csv"
    results_dir.mkdir(exist_ok=True)
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        trades_df.to_csv(out_csv, index=False)

    try:
        model_cfg = get_model_config(version_key)
    except KeyError:
        model_cfg = {"name": version_key, "color": "#888888", "order": 99}

    print(f"\n  [Export] Exporting {version_key} → dashboard...")
    result = export_version(version_key, model_cfg, str(results_dir), str(viz_dir))
    if result:
        generate_manifest([result], str(viz_dir))
    else:
        print(f"  [Export] ⚠ Export failed for {version_key}")


def cmd_export(args: argparse.Namespace) -> int:
    """Export champion trades CSV files to visualization JSON for dashboard."""
    import subprocess

    cmd = [sys.executable, "-m", "src.export.unified_export"]
    if args.versions:
        cmd += ["--versions", args.versions]
    if args.include_retired:
        cmd += ["--include-retired"]
    if args.base_data_dir:
        cmd += ["--base-data-dir", args.base_data_dir]

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def cmd_export_matrix(args: argparse.Namespace) -> int:
    from src.env import get_results_dir
    from src.export.unified_export import export_version, generate_manifest

    bundle_dir = Path(args.bundle_dir).expanduser()
    if not bundle_dir.is_dir():
        bundle_dir = Path(get_results_dir()) / "experiments" / args.bundle_dir
    if not bundle_dir.is_dir():
        print(f"Error: matrix results dir not found: {bundle_dir}", file=sys.stderr)
        return 1

    rows = _load_ranking_rows(bundle_dir)
    if not rows:
        print(f"Error: no ranking rows found under {bundle_dir}", file=sys.stderr)
        return 1

    sort_key = args.sort or "composite_score"
    reverse = sort_key not in {"mdd_per_symbol", "yearly_consistency"}
    rows = sorted(rows, key=lambda row: row.get(sort_key, 0) or 0, reverse=reverse)

    seen_fingerprints: set[tuple[float, float, int]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        fingerprint = (
            round(float(row.get("composite_score", 0) or 0), 4),
            round(float(row.get("total_pnl", 0) or 0), 4),
            int(row.get("trade_count", 0) or 0),
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        deduped.append(row)
    skipped = len(rows) - len(deduped)
    if skipped:
        print(f"  Dedup: skipped {skipped} row(s) with identical metrics")
    top_rows = deduped[: args.top_k]

    colors = [
        "#2196F3",
        "#FF9800",
        "#9C27B0",
        "#F44336",
        "#4CAF50",
        "#00BCD4",
        "#FF5722",
        "#3F51B5",
        "#795548",
        "#E91E63",
    ]
    tmp_results = Path(get_results_dir())
    tmp_results.mkdir(parents=True, exist_ok=True)
    viz_dir = REPO_ROOT / "visualization"
    exported: list[dict[str, Any]] = []
    tmp_files: list[Path] = []

    try:
        for index, row in enumerate(top_rows, 1):
            exp_name = row.get("name")
            if not exp_name:
                print(f"  [SKIP] ranking row missing name at rank {index}")
                continue

            artifact_dir = bundle_dir / _artifact_run_name(str(exp_name))
            trades_csv = artifact_dir / "trades.csv"
            if not trades_csv.exists():
                print(f"  [SKIP] no trades.csv: {exp_name}")
                continue

            short_name = _make_matrix_short_name(str(exp_name), row, bundle_dir.name)
            existing = {item["version_key"] for item in exported}
            base_name = short_name
            suffix = 2
            while short_name in existing:
                short_name = f"{base_name}_{suffix}"
                suffix += 1

            tmp_csv = tmp_results / f"trades_{short_name}.csv"
            shutil.copy2(trades_csv, tmp_csv)
            tmp_files.append(tmp_csv)

            model_cfg = {
                "name": short_name,
                "color": colors[(index - 1) % len(colors)],
                "marker_shape": "arrowUp",
                "active": True,
                "order": index - 1,
            }
            result = export_version(short_name, model_cfg, str(tmp_results), str(viz_dir))
            if result:
                result["matrix_bundle"] = bundle_dir.name
                result["composite_score"] = row.get("composite_score")
                exported.append(result)
                print(f"  [{index}/{len(top_rows)}] {short_name} <- {exp_name}")
    finally:
        for tmp_file in tmp_files:
            with contextlib.suppress(OSError):
                tmp_file.unlink()

    if not exported:
        print("Error: no experiments exported", file=sys.stderr)
        return 1

    generate_manifest(exported, str(viz_dir))
    print(f"\nDone: exported {len(exported)} matrix experiments to {viz_dir}")
    return 0


def cmd_score_models(args: argparse.Namespace) -> int:
    from src.export.unified_export import backfill_scores_from_viz

    viz_dir = REPO_ROOT / "visualization"
    scores = backfill_scores_from_viz(str(viz_dir), force=args.force)

    manifest_path = viz_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = sorted(
        manifest.get("models", []),
        key=lambda row: row.get("composite_score", 0) or 0,
        reverse=True,
    )
    print("version_key                      score     total_pnl")
    print("---------------------------------------------------")
    for row in rows:
        version_key = row.get("version_key", "")
        score = row.get("composite_score", scores.get(version_key, 0))
        total_pnl = row.get("total_pnl", 0)
        print(f"{version_key:<28} {score:>7.1f} {total_pnl:>11.1f}")
    print(f"\nUpdated {manifest_path}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from scripts.benchmark import _print_table, run_benchmark

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    try:
        results = run_benchmark(versions, device=args.device, symbols_limit=args.symbols_limit)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    _print_table(results)
    if args.output:
        import json

        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.output}")
    return 0


def cmd_list_experiments(args: argparse.Namespace) -> int:
    del args
    from src.market_profile import resolve_market_name
    from src.pipeline import ExperimentConfig

    champions_dir = CONFIG_DIR / "champions"
    matrix_dir = CONFIG_DIR / "matrix"
    target_market = resolve_market_name(None)

    print("=== Champion Experiments ===")
    for yaml_path in sorted(champions_dir.glob("*.yaml")):
        cfg = ExperimentConfig.from_yaml(yaml_path)
        if cfg.market == target_market:
            print(f"  champions/{yaml_path.stem}")

    print("\n=== Matrix Experiments ===")
    for yaml_path in sorted(matrix_dir.glob("*.yaml")):
        print(f"  matrix/{yaml_path.stem}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    frames: dict[str, pd.DataFrame] = {}
    for exp_arg in args.experiments:
        from src.pipeline import ExperimentConfig, Pipeline

        yaml_path = _resolve_yaml(exp_arg)
        if not yaml_path.exists():
            print(f"Error: config not found: {yaml_path}", file=sys.stderr)
            return 1
        cfg = ExperimentConfig.from_yaml(yaml_path)
        symbols = _load_symbols(cfg.market)
        pipeline = Pipeline(cfg, symbols=symbols, device=args.device or "cpu")
        result = pipeline.run()
        frames[cfg.name] = result.trades_df

    rows = []
    for name, df in frames.items():
        row = {"experiment": name, "n_trades": len(df)}
        if "pnl_pct" in df.columns:
            row["win_rate"] = f"{(df['pnl_pct'] > 0).mean():.1%}"
            row["avg_pnl"] = f"{df['pnl_pct'].mean():.2f}%"
            row["total_pnl"] = f"{df['pnl_pct'].sum():.1f}%"
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock_ml", description="Stock ML pipeline v2")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("experiment", help="Path to champion YAML (e.g. champions/v22)")
    p_run.add_argument("--device", default="cpu", help="cpu or gpu")
    p_run.add_argument("--output", help="Save trades CSV to this path")
    p_run.add_argument(
        "--save-results",
        action="store_true",
        dest="save_results",
        help="Auto-save trades to results/trades_{strategy}.csv",
    )
    p_run.add_argument(
        "--skip-leaderboard",
        action="store_true",
        dest="skip_leaderboard",
        help="Skip updating results/leaderboard after saving artifacts",
    )
    p_run.add_argument(
        "--export",
        action="store_true",
        help="After run, export trades to dashboard visualization JSON",
    )
    p_run.set_defaults(func=cmd_run)

    # run-matrix
    p_mat = sub.add_parser("run-matrix", help="Run all experiments in a matrix YAML")
    p_mat.add_argument("matrix", help="Path to matrix YAML (e.g. matrix/model_selection)")
    p_mat.add_argument("--device", default="cpu", help="cpu or gpu")
    p_mat.add_argument(
        "--dry-run", action="store_true", help="Print expanded experiments without running"
    )
    p_mat.add_argument("--limit", type=int, help="Run only the first N expanded experiments")
    p_mat.add_argument(
        "--symbols-limit", type=int, help="Run each experiment on the first N symbols"
    )
    p_mat.add_argument(
        "--resume", action="store_true", help="Skip experiments that already have ranking_row.json"
    )
    p_mat.add_argument(
        "--strict-validate",
        action="store_true",
        dest="strict_validate",
        help="Run strict validation before executing the matrix",
    )
    p_mat.add_argument(
        "--top-k-preview",
        type=int,
        metavar="K",
        dest="top_k_preview",
        help="Run all experiments with limited symbols first, then re-run top K",
    )
    p_mat.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers (CPU only)",
    )
    p_mat.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="save_results",
        help="Save matrix artifacts under results/experiments/<matrix_name>/",
    )
    p_mat.add_argument(
        "--skip-leaderboard",
        action="store_true",
        dest="skip_leaderboard",
        help="Skip updating results/leaderboard after saving artifacts",
    )
    p_mat.set_defaults(func=cmd_run_matrix)

    # compare-matrix
    p_cmp_mat = sub.add_parser("compare-matrix", help="Rank saved matrix experiment artifacts")
    p_cmp_mat.add_argument("results_dir", help="Path to results/experiments/<matrix_name>")
    p_cmp_mat.add_argument("--top", type=int, default=10, help="Show only the top N rows")
    p_cmp_mat.add_argument(
        "--sort",
        choices=["composite_score", "mdd_per_symbol", "yearly_consistency", "total_pnl"],
        default="composite_score",
        help="Metric to sort by; MDD and consistency sort ascending",
    )
    p_cmp_mat.add_argument(
        "--min-trades", type=int, dest="min_trades", help="Filter rows below this trade count"
    )
    p_cmp_mat.add_argument(
        "--max-mdd", type=float, dest="max_mdd", help="Filter rows above this MDD per symbol"
    )
    p_cmp_mat.set_defaults(func=cmd_compare_matrix)

    # compare-champions
    p_cc = sub.add_parser(
        "compare-champions",
        help="Run all champion configs with a shared split and produce a unified ranking",
    )
    p_cc.add_argument("--first-test-year", type=int, required=True, dest="first_test_year")
    p_cc.add_argument("--last-test-year", type=int, required=True, dest="last_test_year")
    p_cc.add_argument(
        "--champions",
        default="",
        help="Comma-separated champion names (default: all *.yaml in config/experiments/champions)",
    )
    p_cc.add_argument(
        "--bundle-name",
        default="",
        dest="bundle_name",
        help="Output dir name under results/experiments/ (default: champions_<first>_<last>)",
    )
    p_cc.add_argument("--device", default="cpu", help="cpu or gpu")
    p_cc.add_argument(
        "--symbols-limit", type=int, dest="symbols_limit", help="Run on first N symbols only"
    )
    p_cc.add_argument(
        "--resume",
        action="store_true",
        help="Skip champions that already have ranking_row.json in bundle dir",
    )
    p_cc.add_argument(
        "--dry-run", action="store_true", dest="dry_run", help="List configs without running"
    )
    p_cc.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="save_results",
        help="Save artifacts under results/experiments/<bundle_name>/",
    )
    p_cc.add_argument(
        "--skip-leaderboard",
        action="store_true",
        dest="skip_leaderboard",
        help="Skip updating results/leaderboard after saving artifacts",
    )
    p_cc.add_argument("--top", type=int, default=15, help="Show top N rows in final table")
    p_cc.set_defaults(func=cmd_compare_champions)

    # validate
    p_val = sub.add_parser("validate", help="Validate experiment config")
    p_val.add_argument("experiment", help="Path to champion YAML")
    p_val.add_argument("--strict", action="store_true", help="Enable research strict validation")
    p_val.set_defaults(func=cmd_validate)

    # list-components
    p_lc = sub.add_parser("list-components", help="List registered components")
    p_lc.add_argument(
        "--type",
        choices=["features", "models", "exit_models", "targets", "fusion", "runners"],
    )
    p_lc.set_defaults(func=cmd_list_components)

    # list-experiments
    p_le = sub.add_parser("list-experiments", help="List available experiment configs")
    p_le.set_defaults(func=cmd_list_experiments)

    # compare
    p_cmp = sub.add_parser("compare", help="Run and compare multiple experiments")
    p_cmp.add_argument("experiments", nargs="+", help="Experiment YAML paths")
    p_cmp.add_argument("--device", default="cpu")
    p_cmp.set_defaults(func=cmd_compare)

    # export
    p_exp = sub.add_parser(
        "export",
        help="Export trades CSVs to dashboard visualization JSON (runs unified_export)",
    )
    p_exp.add_argument(
        "--versions",
        default="",
        help="Comma-separated version keys to export (default: all active)",
    )
    p_exp.add_argument(
        "--include-retired",
        action="store_true",
        dest="include_retired",
        help="Include retired models",
    )
    p_exp.add_argument(
        "--base-data-dir",
        default="data",
        dest="base_data_dir",
        help="Base OHLCV data directory name (default: data)",
    )
    p_exp.set_defaults(func=cmd_export)

    # export-matrix
    p_exmat = sub.add_parser(
        "export-matrix",
        help="Export top-K matrix experiments to dashboard visualization JSON",
    )
    p_exmat.add_argument(
        "bundle_dir",
        help="Path to matrix bundle dir or name (e.g. model_selection)",
    )
    p_exmat.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of top experiments to export (default: 5)",
    )
    p_exmat.add_argument(
        "--sort",
        choices=["composite_score", "mdd_per_symbol", "yearly_consistency", "total_pnl"],
        default="composite_score",
        dest="sort",
        help="Metric to select top-K (default: composite_score)",
    )
    p_exmat.set_defaults(func=cmd_export_matrix)

    # score-models
    p_score = sub.add_parser(
        "score-models",
        help="Backfill composite_score in visualization manifest from existing dashboard JSON",
    )
    p_score.add_argument(
        "--force",
        action="store_true",
        help="Recompute score even for models that already have composite_score",
    )
    p_score.set_defaults(func=cmd_score_models)

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Benchmark pipeline runtime")
    p_bm.add_argument(
        "--versions",
        default="v22,v34,v37a,v32",
        help="Comma-separated versions to benchmark (default: v22,v34,v37a,v32)",
    )
    p_bm.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p_bm.add_argument(
        "--symbols-limit",
        type=int,
        default=None,
        dest="symbols_limit",
        help="Limit symbols per version (for quick test)",
    )
    p_bm.add_argument("--output", default=None, help="Save results JSON to this path")
    p_bm.set_defaults(func=cmd_benchmark)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
