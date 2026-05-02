"""Benchmark champion pipeline runtime.

Usage:
    python -m stock_ml benchmark [--versions v22,v34] [--output benchmark.json]
    python stock_ml/scripts/benchmark.py  (direct run)

Reports wall-clock time for prediction-cache build + backtest per champion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "stock_ml" / "tests" / "regression" / "golden"

BENCHMARK_CHAMPIONS = [
    "v22",
    "v34",
    "v37a",
    "v37d",
    "v32",
    "v35b",
    "v19_3",
    "rule",
]


def _load_meta(version: str) -> dict[str, Any]:
    path = GOLDEN_DIR / f"trades_{version}.meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Golden meta not found: {path}")
    return json.loads(path.read_text())


def _bench_new_pipeline(version: str, symbols: list[str], *, device: str = "cpu") -> float:
    """Return elapsed seconds for Pipeline.run() including prediction cache build."""
    from src.pipeline.config import ExperimentConfig
    from src.pipeline.orchestrator import Pipeline

    if version == "rule":
        yaml_path = REPO_ROOT / "stock_ml" / "config" / "experiments" / "champions" / "v22.yaml"
        cfg = ExperimentConfig.from_yaml(yaml_path)
        cfg = cfg.model_copy(update={"strategy": "rule", "name": "rule"})
    else:
        yaml_path = (
            REPO_ROOT / "stock_ml" / "config" / "experiments" / "champions" / f"{version}.yaml"
        )
        if not yaml_path.exists():
            cfg = ExperimentConfig.model_validate({"strategy": version, "name": version})
        else:
            cfg = ExperimentConfig.from_yaml(yaml_path)

    t0 = time.perf_counter()
    pipeline = Pipeline(cfg, symbols=symbols, device=device)
    pipeline.run()
    return time.perf_counter() - t0


def run_benchmark(
    versions: list[str],
    *,
    device: str = "cpu",
    symbols_limit: int | None = None,
) -> list[dict[str, Any]]:
    from src.env import resolve_data_dir

    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")
    if not Path(data_dir).is_dir():
        raise RuntimeError(f"Data directory not found: {data_dir}")

    results = []
    for version in versions:
        print(f"Benchmarking {version}...", flush=True)
        try:
            meta = _load_meta(version)
        except FileNotFoundError:
            print("  SKIP (no golden meta)")
            continue

        symbols = meta["symbols"]
        if symbols_limit:
            symbols = symbols[:symbols_limit]

        try:
            t_new = _bench_new_pipeline(version, symbols, device=device)
        except Exception as e:
            t_new = -1.0
            print(f"  NEW pipeline ERROR: {e}")

        row = {
            "version": version,
            "pipeline_s": round(t_new, 2),
            "status": "OK" if t_new >= 0 else "ERROR",
            "n_symbols": len(symbols),
        }
        results.append(row)
        print(f"  pipeline={t_new:.1f}s  [{row['status']}]")

    return results


def _print_table(results: list[dict[str, Any]]) -> None:
    header = f"{'Version':<12} {'Pipeline(s)':>11} {'Status':<8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['version']:<12} {r['pipeline_s']:>11.1f}  {r['status']:<8}")
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark new pipeline vs legacy.")
    parser.add_argument(
        "--versions",
        default=",".join(BENCHMARK_CHAMPIONS),
        help="Comma-separated list of versions to benchmark",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--symbols-limit",
        type=int,
        default=None,
        help="Limit number of symbols per version (for quick testing)",
    )
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    results = run_benchmark(versions, device=args.device, symbols_limit=args.symbols_limit)
    _print_table(results)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
