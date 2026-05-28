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

STRATEGIES_WITHOUT_PREDICTION_CACHE = {"rule"}


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
        self._cache_keys: dict[str, str] = {"features": "", "predictions": ""}

    def run(self) -> PipelineResult:
        print(f"  [Pipeline] Running experiment: {self.cfg.name}")

        runner_fn, df_converter = self._resolve_runner()

        from src.env import resolve_data_dir
        from src.market_profile import resolve_run_context

        run_context = resolve_run_context(self.cfg)
        if run_context.resolved_data_dir is None:
            raise ValueError(f"Market {run_context.market!r} does not define data.data_dir")
        abs_data_dir = resolve_data_dir(run_context.resolved_data_dir)

        cache = (
            []
            if self.cfg.strategy in STRATEGIES_WITHOUT_PREDICTION_CACHE
            else self._prediction_cache or self._build_cache(run_context)
        )

        runner_kwargs: dict[str, Any] = {
            "prediction_cache": cache,
            "device": self.device,
            "mods": self.cfg.mods or None,
            "params": self.cfg.params or None,
            "strategy_v3": self.cfg.strategy_v3,
        }
        if self.cfg.signals.exit_model.enabled:
            runner_kwargs["enable_exit_model"] = True

        execution_keys = [
            "commission",
            "tax",
            "slippage",
            "initial_capital",
            "pnl_mode",
            "contract_multiplier",
            "funding_rate_column",
            "leverage",
            "maintenance_margin_rate",
            "liquidation_fee",
            "short_enabled",
            "roll_cost_rate",
            "expiry_date_column",
            "roll_rule",
            "roll_days_before_expiry",
            "next_volume_column",
            "next_oi_column",
        ]
        runner_kwargs.update(
            {
                k: run_context.execution_costs[k]
                for k in execution_keys
                if k in run_context.execution_costs
            }
        )

        allowed_kwargs = set(signature(runner_fn).parameters)
        runner_kwargs = {k: v for k, v in runner_kwargs.items() if k in allowed_kwargs}
        trades = runner_fn(self.symbols, abs_data_dir, **runner_kwargs)

        trades_df = df_converter(trades) if trades else pd.DataFrame()
        if isinstance(trades_df, list):
            trades_df = pd.DataFrame(trades_df)
        metrics: dict[str, Any] = {}
        if not trades_df.empty:
            trades_list = trades_df.to_dict("records")
            metrics = calc_metrics(trades_list)
            metrics["mdd_per_symbol"] = round(calc_mdd_per_symbol(trades_list), 2)
            metrics["yearly_consistency"] = round(calc_yearly_consistency(trades_list), 4)
            metrics["composite_score"] = composite_score(metrics, trades_list)

        resolved_feature_set = run_context.feature_set
        feature_set_label = (
            "|".join(resolved_feature_set)
            if isinstance(resolved_feature_set, list)
            else (resolved_feature_set or self.cfg.feature_set())
        )
        metadata = {
            "strategy": self.cfg.strategy,
            "market": run_context.market,
            "currency": run_context.execution_costs.get("currency", "unknown"),
            "pnl_mode": run_context.execution_costs.get("pnl_mode", "unknown"),
            "execution": run_context.execution_costs,
            "schema": run_context.schema or "unknown",
            "timeframe": run_context.timeframe,
            "feature_set": feature_set_label,
            "n_symbols": len(self.symbols),
            "device": self.device,
        }
        if self._cache_manager is not None:
            metadata["cache_stats"] = self._cache_manager.stats()
        metadata["cache_keys"] = dict(self._cache_keys)

        return PipelineResult(
            name=self.cfg.name,
            trades=trades,
            trades_df=trades_df,
            prediction_cache=cache,
            metadata=metadata,
            metrics=metrics,
        )

    def _build_cache(self, run_context=None) -> list[dict[str, Any]]:
        mgr = self._cache_manager
        if mgr is not None and self._use_cache:
            cached, key = mgr.load(self.cfg, self.symbols, run_context)
            self._cache_keys["predictions"] = key
            if cached is not None:
                print(f"  [Pipeline] Prediction cache HIT key={key[:8]}")
                return cached
            print(f"  [Pipeline] Prediction cache MISS key={key[:8]}, training...")
        else:
            print(f"  [Pipeline] Building prediction cache for {self.cfg.feature_set()}...")

        train_meta: dict[str, Any] = {}
        result = build_prediction_cache(
            self.cfg, self.symbols, device=self.device, out_meta=train_meta
        )
        self._cache_keys["features"] = train_meta.get("feature_cache_key", "")

        if mgr is not None and self._use_cache:
            saved_key = mgr.save(result, self.cfg, self.symbols, run_context)
            self._cache_keys["predictions"] = saved_key
            print(f"  [Pipeline] Prediction cache STORED key={saved_key[:8]}")

        return result

    def _resolve_runner(self):
        """Resolve strategy runners from fusion, lineage, then champion registries."""
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
