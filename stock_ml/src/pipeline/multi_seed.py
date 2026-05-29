"""Multi-seed experiment runner (Phase 1.5.5).

Runs same experiment with different seeds (data sampling, model init).
Reports mean ± std of all metrics across seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.experiment import ExperimentConfig, run_experiment
from src.seed import set_global_seed


def run_experiment_multi_seed(
    cfg: ExperimentConfig,
    data_root: str,
    symbols: list[str],
    out_dir: str,
    n_seeds: int | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Run experiment multiple times with different seeds (Phase 1.5.5).

    Args:
        cfg: ExperimentConfig
        data_root: path to OHLCV data directory
        symbols: list of symbols to use
        out_dir: output directory for results
        n_seeds: number of seeds to run (if seeds not provided)
                 if None, uses cfg.validation.n_seeds (default 1)
        seeds: explicit list of seeds (if provided, overrides n_seeds)

    Returns:
        summary dict with aggregated results (mean ± std for metrics)
    """
    if seeds is None:
        if n_seeds is None:
            if cfg.validation and "n_seeds" in cfg.validation:
                n_seeds = cfg.validation["n_seeds"]
            else:
                n_seeds = 1

        seeds = list(range(n_seeds))

    if n_seeds == 1 and len(seeds) == 1:
        set_global_seed(seeds[0])
        result = run_experiment(cfg, data_root, symbols, out_dir)
        result["n_seeds"] = 1
        return result

    all_results = []
    aggregated_metrics = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n[{cfg.name}] Running seed {seed_idx + 1}/{len(seeds)} (seed={seed})")
        set_global_seed(seed)

        cfg_copy = ExperimentConfig(
            name=cfg.name,
            strategy=cfg.strategy,
            market=cfg.market,
            feature_set=cfg.feature_set,
            target=cfg.target.copy(),
            entry_model=cfg.entry_model.copy(),
            exit_model=cfg.exit_model.copy(),
            split=cfg.split.copy(),
            engine=cfg.engine.copy(),
            seed=seed,
            signal_threshold=cfg.signal_threshold,
            hypothesis=cfg.hypothesis,
            validation=cfg.validation,
            strict_audit=cfg.strict_audit,
        )

        seed_out_dir = Path(out_dir) / f"seed_{seed}"
        seed_result = run_experiment(
            cfg_copy,
            data_root,
            symbols,
            str(seed_out_dir),
        )

        if seed_result.get("ok") is False:
            print(f"  Seed {seed} failed: {seed_result.get('reason')}")
            continue

        all_results.append(seed_result)

        if "aggregate" in seed_result:
            agg = seed_result["aggregate"]
            for key, value in agg.items():
                if isinstance(value, (int, float)):
                    if key not in aggregated_metrics:
                        aggregated_metrics[key] = []
                    aggregated_metrics[key].append(value)

    if not all_results:
        return {
            "name": cfg.name,
            "ok": False,
            "reason": "all_seeds_failed",
            "n_seeds": len(seeds),
        }

    aggregated = {}
    for key, values in aggregated_metrics.items():
        if len(values) > 1:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            aggregated[key] = float(values[0])

    return {
        "name": cfg.name,
        "ok": True,
        "n_seeds": len(seeds),
        "n_seeds_completed": len(all_results),
        "aggregate_all_seeds": aggregated,
        "seed_results": [r.get("aggregate") for r in all_results],
        "aggregate": all_results[0].get("aggregate") if all_results else {},
    }
