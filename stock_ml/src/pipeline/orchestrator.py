"""Pipeline orchestrator — wires ExperimentConfig into a runnable experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from inspect import signature
from typing import Any

import pandas as pd

from src.evaluation.scoring import (
    calc_mdd_per_symbol,
    calc_metrics,
    calc_yearly_consistency,
    composite_score,
)
from src.pipeline.cache import PredictionCacheManager
from src.pipeline.config import ExperimentConfig
from src.pipeline.trainer import build_prediction_cache

CHAMPION_RUNNER_MAP: dict[str, str] = {
    "rule": "src.components.runners.rule_runner",
    "v34": "src.components.runners.v34_runner",
}

CHAMPION_RUNNER_FUNCTION_MAP: dict[str, str] = {
    "rule": "run_rule_baseline",
}

CHAMPION_DF_CONVERTER_MAP: dict[str, str] = {
    "rule": "trades_to_dataframe",
    "v34": "trades_to_v34_dataframe",
}


@dataclass
class PipelineResult:
    name: str
    trades: list[Any] = field(default_factory=list)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_cache: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)


class Pipeline:
    """Orchestrate a full experiment run from ExperimentConfig.

    Usage:
        cfg = ExperimentConfig.from_yaml("config/experiments/champions/v22.yaml")
        pipeline = Pipeline(cfg, symbols=symbols, device="cpu")
        result = pipeline.run()
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        symbols: list[str],
        device: str = "cpu",
        prediction_cache: list[dict[str, Any]] | None = None,
        cache_manager: PredictionCacheManager | None = None,
        use_cache: bool = True,
    ) -> None:
        self.cfg = cfg
        self.symbols = symbols
        self.device = device
        self._prediction_cache = prediction_cache
        self._cache_manager = cache_manager
        self._use_cache = use_cache

    def run(self) -> PipelineResult:
        print(f"  [Pipeline] Running experiment: {self.cfg.name}")

        cache = self._prediction_cache or self._build_cache()

        runner_fn, df_converter = self._resolve_runner()

        from src.config_loader import load_config

        pipeline_cfg = load_config().get("pipeline", {})
        data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
        from src.env import resolve_data_dir

        abs_data_dir = resolve_data_dir(data_dir)

        runner_kwargs: dict[str, Any] = {
            "prediction_cache": cache,
            "device": self.device,
            "mods": self.cfg.mods or None,
            "params": self.cfg.params or None,
            "strategy_v3": self.cfg.strategy_v3,
        }
        if self.cfg.components.exit_model.enabled:
            runner_kwargs["enable_exit_model"] = True

        allowed_kwargs = set(signature(runner_fn).parameters)
        runner_kwargs = {k: v for k, v in runner_kwargs.items() if k in allowed_kwargs}
        trades = runner_fn(self.symbols, abs_data_dir, **runner_kwargs)

        trades_df = df_converter(trades) if trades else pd.DataFrame()
        metrics: dict[str, Any] = {}
        if not trades_df.empty:
            trades_list = trades_df.to_dict("records")
            metrics = calc_metrics(trades_list)
            metrics["mdd_per_symbol"] = round(calc_mdd_per_symbol(trades_list), 2)
            metrics["yearly_consistency"] = round(calc_yearly_consistency(trades_list), 4)
            metrics["composite_score"] = composite_score(metrics, trades_list)

        metadata = {
            "strategy": self.cfg.strategy,
            "feature_set": self.cfg.feature_set(),
            "n_symbols": len(self.symbols),
            "device": self.device,
        }
        if self._cache_manager is not None:
            metadata["cache_stats"] = self._cache_manager.stats()

        return PipelineResult(
            name=self.cfg.name,
            trades=trades,
            trades_df=trades_df,
            prediction_cache=cache,
            metadata=metadata,
            metrics=metrics,
        )

    def _build_cache(self) -> list[dict[str, Any]]:
        mgr = self._cache_manager
        if mgr is not None and self._use_cache:
            cached, key = mgr.load(self.cfg, self.symbols)
            if cached is not None:
                print(f"  [Pipeline] Prediction cache HIT key={key[:8]}")
                return cached
            print(f"  [Pipeline] Prediction cache MISS key={key[:8]}, training...")
        else:
            print(f"  [Pipeline] Building prediction cache for {self.cfg.feature_set()}...")

        result = build_prediction_cache(self.cfg, self.symbols, device=self.device)

        if mgr is not None and self._use_cache:
            saved_key = mgr.save(result, self.cfg, self.symbols)
            print(f"  [Pipeline] Prediction cache STORED key={saved_key[:8]}")

        return result

    def _resolve_runner(self):
        import importlib

        from src.components.runners._lineage_v34 import run_lineage
        from src.components.runners.generic_fusion import (
            FUSION_RUNNER_DEFS,
            run_fusion,
            trades_to_v19_3_dataframe,
            trades_to_v22_dataframe,
        )
        from src.components.runners.runner_registry import RUNNER_DEFS
        from src.components.runners.v34_runner import trades_to_v34_dataframe

        strategy = self.cfg.strategy
        if strategy in FUSION_RUNNER_DEFS:
            defn = FUSION_RUNNER_DEFS[strategy]
            converter = (
                trades_to_v19_3_dataframe if strategy == "v19_3" else trades_to_v22_dataframe
            )
            return partial(run_fusion, defn), converter
        if strategy in RUNNER_DEFS:
            return partial(run_lineage, RUNNER_DEFS[strategy]), trades_to_v34_dataframe
        if strategy not in CHAMPION_RUNNER_MAP:
            available = [*CHAMPION_RUNNER_MAP, *FUSION_RUNNER_DEFS, *RUNNER_DEFS]
            raise ValueError(
                f"No component runner registered for strategy '{strategy}'. Available: {available}"
            )

        module = importlib.import_module(CHAMPION_RUNNER_MAP[strategy])
        runner_name = CHAMPION_RUNNER_FUNCTION_MAP.get(strategy, f"run_{strategy}")
        df_converter_name = CHAMPION_DF_CONVERTER_MAP[strategy]

        runner_fn = getattr(module, runner_name)
        df_converter = getattr(module, df_converter_name)
        return runner_fn, df_converter
