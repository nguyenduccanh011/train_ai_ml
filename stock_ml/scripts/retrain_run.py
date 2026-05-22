"""Retrain a single leaderboard run in place from its resolved config.

Re-runs the pipeline for an existing run directory, overwriting its artifacts
(trades.csv, metrics.json, ranking_row.json, predictions_meta.json with fresh
cache_keys) and rebuilding the leaderboard. The lifecycle.json sidecar (pin/retire
state) is preserved.

Usage:
    python -m stock_ml.scripts.retrain_run <run_dir> [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain a single run in place")
    parser.add_argument("run_dir", type=str, help="Path to the run directory")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or gpu")
    args = parser.parse_args()

    import src.safe_io  # noqa: F401
    from scripts.cli import _load_symbols, _save_matrix_artifacts
    from src.env import get_results_dir
    from src.pipeline import ExperimentConfig, Pipeline
    from src.pipeline.cache import PredictionCacheManager

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.exists():
        print(f"ERROR: no config.resolved.yaml in {run_dir}")
        return 1

    matrix_name = run_dir.parent.name
    cfg = ExperimentConfig.from_yaml(config_path)
    symbols = _load_symbols(cfg.market)
    if not symbols:
        print(f"ERROR: no symbols resolved for market={cfg.market!r}")
        return 1

    print(f"[retrain] {cfg.name} (market={cfg.market}, {len(symbols)} symbols)")
    cache_root = Path(get_results_dir()) / "cache" / "predictions"
    cache_manager = PredictionCacheManager(cache_root)
    pipeline = Pipeline(cfg, symbols=symbols, device=args.device, cache_manager=cache_manager)
    result = pipeline.run()

    run_meta = {"symbols_count": len(symbols), "market": cfg.market, "retrained": True}
    saved_dir = _save_matrix_artifacts(matrix_name, cfg, result, run_meta, skip_leaderboard=False)
    print(f"[retrain] done: {result.n_trades} trades -> {saved_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
