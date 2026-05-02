"""CLI for stock_ml pipeline v2.

Usage:
    python -m stock_ml run champions/v22
    python -m stock_ml run champions/v22 --device gpu
    python -m stock_ml run-matrix matrix/test_2x2
    python -m stock_ml validate champions/v22
    python -m stock_ml list-components
    python -m stock_ml list-components --type fusion
    python -m stock_ml list-experiments
    python -m stock_ml compare champions/v22 champions/v34
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def _load_symbols() -> list[str]:
    from src.config_loader import get_pipeline_symbols

    return get_pipeline_symbols()


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


def _get_matrix_artifact_dir(matrix_name: str, run_name: str) -> Path:
    from src.env import get_results_dir

    return Path(get_results_dir()) / "experiments" / matrix_name / _artifact_run_name(run_name)


def _save_matrix_artifacts(matrix_name: str, cfg: Any, result: Any) -> None:
    from src.evaluation.scoring import (
        calc_max_drawdown,
        calc_mdd_per_symbol,
        calc_symbol_coverage,
        calc_yearly_consistency,
    )

    config_dict = cfg.model_dump()
    config_hash = _stable_config_hash(config_dict)
    run_dir = _get_matrix_artifact_dir(matrix_name, result.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not result.trades_df.empty:
        result.trades_df.to_csv(run_dir / "trades.csv", index=False)

    trades = result.trades_df.to_dict("records") if not result.trades_df.empty else []
    symbol_coverage = calc_symbol_coverage(trades)
    metrics = dict(result.metrics)
    metrics["max_drawdown"] = round(calc_max_drawdown(trades), 2)
    metrics.setdefault("mdd_per_symbol", round(calc_mdd_per_symbol(trades), 2))
    metrics.setdefault("yearly_consistency", round(calc_yearly_consistency(trades), 4))
    metrics["symbol_coverage"] = symbol_coverage

    ranking_row = {
        "name": result.name,
        "composite_score": metrics.get("composite_score", 0.0),
        "total_pnl": metrics.get("total_pnl", 0.0),
        "win_rate": metrics.get("wr", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "mdd_per_symbol": metrics.get("mdd_per_symbol", 0.0),
        "trade_count": metrics.get("trades", 0),
        "avg_holding_days": metrics.get("avg_hold", 0.0),
        "per_year_consistency": metrics.get("yearly_consistency", 0.0),
        "per_symbol_coverage": symbol_coverage,
        "config_hash": config_hash,
    }
    predictions_meta = {
        "config_hash": config_hash,
        "entry_model": cfg.entry_model_type(),
        "exit_model_type": cfg.components.exit_model.type,
        "exit_model_enabled": cfg.components.exit_model.enabled,
        "feature_set": cfg.feature_set(),
        "split": cfg.split.model_dump(),
        "cache_stats": result.metadata.get("cache_stats", {}),
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
    symbols = _load_symbols()

    from src.pipeline import ExperimentConfig, Pipeline

    yaml_path = _resolve_yaml(args.experiment)
    if not yaml_path.exists():
        print(f"Error: config not found: {yaml_path}", file=sys.stderr)
        return 1

    cfg = ExperimentConfig.from_yaml(yaml_path)
    pipeline = Pipeline(cfg, symbols=symbols, device=device)
    result = pipeline.run()

    _print_result_summary(result.name, result.n_trades, result.trades_df)

    if args.output:
        _save_result_csv(result.name, result.trades_df, Path(args.output))
    elif getattr(args, "save_results", False) and not result.trades_df.empty:
        results_dir = REPO_ROOT / "results"
        out = results_dir / f"trades_{cfg.strategy}.csv"
        _save_result_csv(result.name, result.trades_df, out)

    if getattr(args, "export", False) and not result.trades_df.empty:
        _export_to_dashboard(cfg.strategy, result.trades_df)

    return 0


def _run_matrix_configs(
    *,
    matrix_name: str,
    configs: list[Any],
    symbols: list[str],
    device: str,
    dry_run: bool,
    save_results: bool,
    resume: bool,
) -> list[Any]:
    from src.env import get_results_dir
    from src.pipeline import Pipeline
    from src.pipeline.cache import PredictionCacheManager

    if resume and not save_results:
        print("  [WARN] --resume ignored because --no-save-results was set")

    cache_root = Path(get_results_dir()) / "cache" / "predictions"
    cache_manager = PredictionCacheManager(cache_root)
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg.name}")
        print(
            f"  strategy={cfg.strategy} features={cfg.feature_set()} "
            f"entry={cfg.entry_model_type()} exit={cfg.components.exit_model.type} "
            f"exit_enabled={cfg.components.exit_model.enabled}"
        )
        if dry_run:
            continue
        if resume and save_results:
            run_dir = _get_matrix_artifact_dir(matrix_name, cfg.name)
            if (run_dir / "ranking_row.json").exists():
                print("  [SKIP] already done")
                continue
        pipeline = Pipeline(cfg, symbols=symbols, device=device, cache_manager=cache_manager)
        result = pipeline.run()
        results.append(result)
        print(f"  -> {result.n_trades} trades")
        if save_results:
            _save_matrix_artifacts(matrix_name, cfg, result)
    return results


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

    configs = expand_matrix(yaml_path, limit=args.limit)
    all_symbols = _load_symbols()
    symbols = all_symbols[: args.symbols_limit] if args.symbols_limit is not None else all_symbols
    device = args.device or "cpu"

    print(f"Matrix: {len(configs)} experiments from {yaml_path.name}")
    if args.top_k_preview is not None:
        if args.top_k_preview <= 0:
            print("Error: --top-k-preview must be > 0", file=sys.stderr)
            return 1
        preview_limit = args.symbols_limit or 10
        preview_symbols = all_symbols[:preview_limit]
        preview_matrix_name = f"{yaml_path.stem}_preview"
        print(
            f"Top-k preview: running {len(configs)} experiments on {len(preview_symbols)} symbols"
        )
        _run_matrix_configs(
            matrix_name=preview_matrix_name,
            configs=configs,
            symbols=preview_symbols,
            device=device,
            dry_run=args.dry_run,
            save_results=args.save_results,
            resume=args.resume,
        )
        if args.dry_run:
            return 0
        top_configs = _select_top_k_matrix_configs(preview_matrix_name, configs, args.top_k_preview)
        if not top_configs:
            print("Error: no preview ranking rows found", file=sys.stderr)
            return 1
        print(f"\nTop-k full run: {len(top_configs)} experiments")
        _run_matrix_configs(
            matrix_name=yaml_path.stem,
            configs=top_configs,
            symbols=symbols,
            device=device,
            dry_run=False,
            save_results=args.save_results,
            resume=args.resume,
        )
        return 0

    _run_matrix_configs(
        matrix_name=yaml_path.stem,
        configs=configs,
        symbols=symbols,
        device=device,
        dry_run=args.dry_run,
        save_results=args.save_results,
        resume=args.resume,
    )
    return 0


def cmd_compare_matrix(args: argparse.Namespace) -> int:
    import pandas as pd

    matrix_dir = Path(args.results_dir)
    if not matrix_dir.is_absolute():
        matrix_dir = REPO_ROOT / matrix_dir
    if not matrix_dir.exists():
        print(f"Error: matrix results dir not found: {matrix_dir}", file=sys.stderr)
        return 1

    rows = []
    for row_path in sorted(matrix_dir.glob("*/ranking_row.json")):
        rows.append(json.loads(row_path.read_text(encoding="utf-8")))

    if not rows:
        print(f"Error: no ranking_row.json found under {matrix_dir}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    if "mdd_per_symbol" not in df.columns:
        df["mdd_per_symbol"] = df.get("max_drawdown", 0.0)
    display = df.assign(
        score=df["composite_score"],
        mdd=df["mdd_per_symbol"],
        trades=df["trade_count"],
        hold_days=df["avg_holding_days"],
        yr_cv=df["per_year_consistency"],
    )[["name", "score", "total_pnl", "win_rate", "mdd", "trades", "hold_days", "yr_cv"]]
    print(display.to_string(index=False))

    warnings = []
    for row in rows:
        flags = []
        coverage = row.get("per_symbol_coverage", {}) or {}
        if row.get("trade_count", 0) < 20:
            flags.append("LOW_TRADES")
        if coverage.get("top_symbol_pnl_ratio", 0.0) > 0.5:
            flags.append("SYMBOL_CONCENTRATION")
        if row.get("per_year_consistency", 0.0) > 1.5:
            flags.append("YEAR_INCONSISTENT")
        if flags:
            warnings.append(f"  {row.get('name')}: {', '.join(flags)}")

    if warnings:
        print("\nWinner guard warnings:")
        print("\n".join(warnings))
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
        errors = validate_config(cfg)
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
    champions_dir = CONFIG_DIR / "champions"
    matrix_dir = CONFIG_DIR / "matrix"

    print("=== Champion Experiments ===")
    for yaml_path in sorted(champions_dir.glob("*.yaml")):
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
        symbols = _load_symbols()
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
        "--export",
        action="store_true",
        help="After run, export trades to dashboard visualization JSON",
    )
    p_run.set_defaults(func=cmd_run)

    # run-matrix
    p_mat = sub.add_parser("run-matrix", help="Run all experiments in a matrix YAML")
    p_mat.add_argument("matrix", help="Path to matrix YAML (e.g. matrix/test_2x2)")
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
        "--top-k-preview",
        type=int,
        metavar="K",
        dest="top_k_preview",
        help="Run all experiments with limited symbols first, then re-run top K",
    )
    p_mat.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="save_results",
        help="Save matrix artifacts under results/experiments/<matrix_name>/",
    )
    p_mat.set_defaults(func=cmd_run_matrix)

    # compare-matrix
    p_cmp_mat = sub.add_parser("compare-matrix", help="Rank saved matrix experiment artifacts")
    p_cmp_mat.add_argument("results_dir", help="Path to results/experiments/<matrix_name>")
    p_cmp_mat.set_defaults(func=cmd_compare_matrix)

    # validate
    p_val = sub.add_parser("validate", help="Validate experiment config")
    p_val.add_argument("experiment", help="Path to champion YAML")
    p_val.set_defaults(func=cmd_validate)

    # list-components
    p_lc = sub.add_parser("list-components", help="List registered components")
    p_lc.add_argument("--type", choices=["features", "models", "exit_models", "targets", "fusion"])
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
