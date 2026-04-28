"""CLI for stock_ml pipeline v2.

Usage:
    python -m stock_ml run champions/v22
    python -m stock_ml run champions/v22 --device gpu
    python -m stock_ml run legacy/v25              # run any legacy version via adapter
    python -m stock_ml run-matrix matrix/test_2x2
    python -m stock_ml validate champions/v22
    python -m stock_ml list-components
    python -m stock_ml list-components --type fusion
    python -m stock_ml list-experiments
    python -m stock_ml list-legacy                  # list all legacy version keys
    python -m stock_ml compare champions/v22 champions/v34
    python -m stock_ml migrate-legacy v25           # convert legacy config → YAML
    python -m stock_ml migrate-legacy --all         # convert all legacy configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def _is_legacy_run(experiment_arg: str) -> bool:
    """Return True if the experiment arg refers to a legacy version (legacy/vXX)."""
    return experiment_arg.startswith("legacy/") or experiment_arg.startswith("legacy\\")


def _extract_legacy_key(experiment_arg: str) -> str:
    return Path(experiment_arg).stem


def _print_result_summary(result_name: str, n_trades: int, trades_df: Any) -> None:
    print(f"\nResult: {n_trades} trades for {result_name}")
    if not hasattr(trades_df, "empty") or trades_df.empty:
        return
    df = trades_df
    if "pnl_pct" in df.columns:
        print(f"  Win rate: {(df['pnl_pct'] > 0).mean():.1%}")
        print(f"  Avg PnL:  {df['pnl_pct'].mean():.2f}%")


def cmd_run(args: argparse.Namespace) -> int:
    device = args.device or "cpu"
    symbols = _load_symbols()

    if _is_legacy_run(args.experiment):
        version_key = _extract_legacy_key(args.experiment)
        from src.pipeline.legacy_adapter import LegacyVersionAdapter

        adapter = LegacyVersionAdapter(version_key)
        result = adapter.run(symbols, device=device)
        _print_result_summary(result.name, result.n_trades, result.trades_df)
        if args.output:
            out = Path(args.output)
            result.trades_df.to_csv(out, index=False)
            print(f"  Saved to: {out}")
        return 0

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
        out = Path(args.output)
        result.trades_df.to_csv(out, index=False)
        print(f"  Saved to: {out}")

    return 0


def cmd_run_matrix(args: argparse.Namespace) -> int:
    from src.pipeline import Pipeline
    from src.pipeline.matrix_expander import expand_matrix

    yaml_path = _resolve_yaml(args.matrix)
    if not yaml_path.exists():
        print(f"Error: matrix config not found: {yaml_path}", file=sys.stderr)
        return 1

    configs = expand_matrix(yaml_path)
    symbols = _load_symbols()
    device = args.device or "cpu"

    print(f"Matrix: {len(configs)} experiments from {yaml_path.name}")
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg.name}")
        pipeline = Pipeline(cfg, symbols=symbols, device=device)
        result = pipeline.run()
        print(f"  → {result.n_trades} trades")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from src.pipeline import ExperimentConfig
    from src.pipeline.validate import validate_config

    yaml_path = _resolve_yaml(args.experiment)
    if not yaml_path.exists():
        print(f"Error: config not found: {yaml_path}", file=sys.stderr)
        return 1

    try:
        cfg = ExperimentConfig.from_yaml(yaml_path)
    except Exception as e:
        print(f"Schema error: {e}", file=sys.stderr)
        return 1

    errors = validate_config(cfg)
    if errors:
        for err in errors:
            print(f"Validation error: {err}", file=sys.stderr)
        return 1

    print(f"OK: {cfg.name} (strategy={cfg.strategy}, features={cfg.feature_set()})")
    return 0


def cmd_list_components(args: argparse.Namespace) -> int:
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


def cmd_list_legacy(args: argparse.Namespace) -> int:
    del args
    from src.pipeline.legacy_adapter import CHAMPION_VERSIONS, LEGACY_STRATEGY_MAP

    print("=== Legacy Versions (non-champion) ===")
    for key in sorted(k for k in LEGACY_STRATEGY_MAP if k not in CHAMPION_VERSIONS):
        print(f"  {key}")

    print("\n=== Champion Aliases in Legacy Map ===")
    for key in sorted(k for k in LEGACY_STRATEGY_MAP if k in CHAMPION_VERSIONS):
        print(f"  {key}  (has component runner — use champions/{key} instead)")

    return 0


def cmd_migrate_legacy(args: argparse.Namespace) -> int:
    from scripts.migrate_legacy import migrate_all, migrate_version

    if args.all:
        paths = migrate_all(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            skip_champions=not args.include_champions,
        )
        print(f"\nMigrated {len(paths)} version(s).")
    else:
        output_path = Path(args.output) if args.output else None
        migrate_version(args.version, output_path=output_path, dry_run=args.dry_run)

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
    p_run.set_defaults(func=cmd_run)

    # run-matrix
    p_mat = sub.add_parser("run-matrix", help="Run all experiments in a matrix YAML")
    p_mat.add_argument("matrix", help="Path to matrix YAML (e.g. matrix/test_2x2)")
    p_mat.add_argument("--device", default="cpu", help="cpu or gpu")
    p_mat.set_defaults(func=cmd_run_matrix)

    # validate
    p_val = sub.add_parser("validate", help="Validate experiment config")
    p_val.add_argument("experiment", help="Path to champion YAML")
    p_val.set_defaults(func=cmd_validate)

    # list-components
    p_lc = sub.add_parser("list-components", help="List registered components")
    p_lc.add_argument("--type", choices=["features", "models", "targets", "fusion"])
    p_lc.set_defaults(func=cmd_list_components)

    # list-experiments
    p_le = sub.add_parser("list-experiments", help="List available experiment configs")
    p_le.set_defaults(func=cmd_list_experiments)

    # list-legacy
    p_ll = sub.add_parser("list-legacy", help="List available legacy version keys")
    p_ll.set_defaults(func=cmd_list_legacy)

    # migrate-legacy
    p_ml = sub.add_parser("migrate-legacy", help="Convert legacy models.yaml entries to YAML")
    ml_group = p_ml.add_mutually_exclusive_group(required=True)
    ml_group.add_argument("version", nargs="?", help="Single version key (e.g. v25)")
    ml_group.add_argument("--all", action="store_true", help="Migrate all versions")
    p_ml.add_argument("--output", help="Output file path (single-version mode)")
    p_ml.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Output directory (--all mode)",
    )
    p_ml.add_argument(
        "--include-champions",
        action="store_true",
        dest="include_champions",
        help="Also migrate champion versions",
    )
    p_ml.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_ml.set_defaults(func=cmd_migrate_legacy)

    # compare
    p_cmp = sub.add_parser("compare", help="Run and compare multiple experiments")
    p_cmp.add_argument("experiments", nargs="+", help="Experiment YAML paths")
    p_cmp.add_argument("--device", default="cpu")
    p_cmp.set_defaults(func=cmd_compare)

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Benchmark new pipeline vs legacy runtime")
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
