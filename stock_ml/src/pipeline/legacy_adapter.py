"""LegacyVersionAdapter — run any legacy backtest_vXX version via Pipeline interface.

Usage:
    from src.pipeline.legacy_adapter import LegacyVersionAdapter

    adapter = LegacyVersionAdapter("v25")
    result = adapter.run(symbols=symbols, device="cpu")

    # Or via CLI:
    python -m stock_ml run legacy/v25
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# All legacy strategy keys available via run_pipeline.get_backtest_function
LEGACY_STRATEGY_MAP: dict[str, tuple[str, str]] = {
    "v37a_exit": ("experiments.run_v37a", "backtest_v37a"),
    "v42_a": ("experiments.run_v42", "backtest_v42"),
    "v42_base": ("experiments.run_v42", "backtest_v42"),
    "v39g": ("experiments.run_v39g", "backtest_v39g"),
    "v39f": ("experiments.run_v39f", "backtest_v39f"),
    "v39d": ("experiments.run_v39d", "backtest_v39d"),
    "v39e": ("experiments.run_v39e", "backtest_v39e"),
    "v39b": ("experiments.run_v39b", "backtest_v39b"),
    "v39a2": ("experiments.run_v39a2", "backtest_v39a2"),
    "v39a": ("experiments.run_v39a", "backtest_v39a"),
    "v38b": ("experiments.run_v38b", "backtest_v38b"),
    "v38c": ("experiments.run_v38c", "backtest_v38c"),
    "v38d": ("experiments.run_v38d", "backtest_v38d"),
    "v38b2": ("experiments.run_v38b2", "backtest_v38b2"),
    "v38b3": ("experiments.run_v38b3", "backtest_v38b3"),
    "v38e": ("experiments.run_v38e", "backtest_v38e"),
    "v38c2": ("experiments.run_v38c2", "backtest_v38c2"),
    "v38d2": ("experiments.run_v38d2", "backtest_v38d2"),
    "v38bc": ("experiments.run_v38_combos", "backtest_v38bc"),
    "v38bd": ("experiments.run_v38_combos", "backtest_v38bd"),
    "v38cd": ("experiments.run_v38_combos", "backtest_v38cd"),
    "v38bcd": ("experiments.run_v38_combos", "backtest_v38bcd"),
    "v37a": ("experiments.run_v37a", "backtest_v37a"),
    "v37b": ("experiments.run_v32_final", "backtest_v32"),
    "v37c": ("experiments.run_v32_final", "backtest_v32"),
    "v37d": ("experiments.run_v37d", "backtest_v37d"),
    "v36a": ("experiments.run_v32_final", "backtest_v32"),
    "v36b": ("experiments.run_v32_final", "backtest_v32"),
    "v36c": ("experiments.run_v32_final", "backtest_v32"),
    "v35a": ("experiments.run_v32_final", "backtest_v32"),
    "v35b": ("experiments.run_v34_final", "backtest_v35b"),
    "v35c": ("experiments.run_v32_final", "backtest_v32"),
    "v34": ("experiments.run_v34_final", "backtest_v34"),
    "v33": ("experiments.run_v33_final", "backtest_v33"),
    "v32": ("experiments.run_v32_final", "backtest_v32"),
    "v31": ("experiments.run_v31_final", "backtest_v31"),
    "v30": ("experiments.run_v30", "backtest_v30"),
    "v29": ("experiments.run_v29", "backtest_v29"),
    "v28": ("experiments.run_v28", "backtest_v28"),
    "v27": ("experiments.run_v27", "backtest_v27"),
    "v26": ("experiments.run_v26", "backtest_v26"),
    "v25": ("experiments.run_v25", "backtest_v25"),
    "v24": ("experiments.run_v24", "backtest_v24"),
    "v23": ("experiments.run_v23_optimal", "backtest_v23"),
    "v22": ("experiments.run_v22_final", "backtest_v22"),
    "v21": ("src.strategies.legacy", "backtest_v21"),
    "v20": ("src.strategies.legacy", "backtest_v20"),
    "v19_4": ("src.strategies.legacy", "backtest_v19_4"),
    "v19_3": ("src.strategies.legacy", "backtest_v19_3"),
    "v19_2": ("src.strategies.legacy", "backtest_v19_2"),
    "v19_1": ("src.strategies.legacy", "backtest_v19_1"),
    "v19": ("src.strategies.legacy", "backtest_v19"),
    "v18": ("src.strategies.legacy", "backtest_v18"),
    "v17": ("src.strategies.legacy", "backtest_v17"),
    "v16": ("src.strategies.legacy", "backtest_v16"),
    "v15": ("src.strategies.legacy", "backtest_v15"),
    "v14": ("src.strategies.legacy", "backtest_v14"),
    "v13": ("src.strategies.legacy", "backtest_v13"),
    "v12": ("src.strategies.legacy", "backtest_v12"),
    "v11": ("src.strategies.legacy", "backtest_v11"),
}

# Champion versions that have dedicated component runners — use Pipeline, not adapter
CHAMPION_VERSIONS = frozenset(
    ["rule", "v19_3", "v22", "v32", "v34", "v35b", "v37a", "v37a_exit", "v37d", "v39d", "v42_a"]
)


@dataclass
class LegacyRunResult:
    name: str
    trades: list[dict[str, Any]] = field(default_factory=list)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_cache: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)


class LegacyVersionAdapter:
    """Wrap a legacy backtest_vXX function to run via the new pipeline interface.

    Reads config from config/models.yaml (feature_set, mods, params, target, exit_model).
    Builds prediction cache via trainer.build_prediction_cache if not provided.
    Not an exact-parity runner — use for approximate historical comparison.
    """

    def __init__(self, version_key: str) -> None:
        if version_key not in LEGACY_STRATEGY_MAP:
            raise ValueError(
                f"Unknown legacy version '{version_key}'. Available: {sorted(LEGACY_STRATEGY_MAP)}"
            )
        if version_key in CHAMPION_VERSIONS:
            import warnings

            warnings.warn(
                f"'{version_key}' is a champion with a dedicated component runner. "
                "Use Pipeline with the champion YAML instead of LegacyVersionAdapter.",
                stacklevel=2,
            )
        self.version_key = version_key
        self._model_cfg = self._load_model_cfg()

    # ── public API ──────────────────────────────────────────────────────

    def run(
        self,
        symbols: list[str],
        *,
        device: str = "cpu",
        prediction_cache: list[dict[str, Any]] | None = None,
    ) -> LegacyRunResult:
        """Train (or reuse) prediction cache, run backtest, return result."""
        print(f"  [LegacyAdapter] Running {self.version_key}")

        cache = prediction_cache or self._build_cache(symbols, device=device)
        trades = self._run_backtest(cache)
        trades_df = _trades_to_dataframe(trades)

        return LegacyRunResult(
            name=self.version_key,
            trades=trades,
            trades_df=trades_df,
            prediction_cache=cache,
            metadata={
                "strategy": self.version_key,
                "feature_set": self._model_cfg.get("feature_set", "leading_v2"),
                "n_symbols": len(symbols),
                "device": device,
                "source": "legacy_adapter",
            },
        )

    def build_experiment_config(self) -> dict[str, Any]:
        """Return a dict suitable for ExperimentConfig.model_validate().

        Useful for migrate_legacy — produces the new YAML schema from legacy config.
        """
        cfg = self._model_cfg
        feature_set = cfg.get("feature_set", "leading_v2")
        model_type = cfg.get("model_type", "lightgbm")
        target_raw = cfg.get("target", {})
        exit_raw = cfg.get("exit_model", {})

        target = {
            "type": target_raw.get("type", "trend_regime"),
            "forward_window": target_raw.get("forward_window", 8),
            "short_window": target_raw.get("short_window", 8),
            "long_window": target_raw.get("long_window", 20),
            "gain_threshold": target_raw.get("gain_threshold", 0.06),
            "loss_threshold": target_raw.get("loss_threshold", 0.03),
            "classes": target_raw.get("classes", 3),
        }

        exit_model = {
            "enabled": exit_raw.get("enabled", False),
            "forward_window": exit_raw.get("forward_window", 15),
            "loss_threshold": exit_raw.get("loss_threshold", 0.05),
        }

        return {
            "name": self.version_key,
            "strategy": self.version_key,
            "components": {
                "features": feature_set,
                "target": target,
                "entry_model": {"type": model_type},
                "exit_model": exit_model,
            },
            "mods": cfg.get("mods", {}),
            "params": cfg.get("params", {}),
        }

    # ── private ─────────────────────────────────────────────────────────

    def _load_model_cfg(self) -> dict[str, Any]:
        from src.config_loader import get_model_config

        cfg = get_model_config(self.version_key)
        if not cfg:
            raise ValueError(f"No config found for '{self.version_key}' in models.yaml")
        return cfg

    def _build_cache(self, symbols: list[str], *, device: str) -> list[dict[str, Any]]:
        from src.pipeline.config import (
            ComponentsConfig,
            EntryModelConfig,
            ExitModelConfig,
            ExperimentConfig,
            SplitConfig,
            TargetConfig,
        )
        from src.pipeline.trainer import build_prediction_cache

        cfg_dict = self.build_experiment_config()
        comp = cfg_dict["components"]

        target_cfg = TargetConfig(**comp["target"])
        entry_model_cfg = EntryModelConfig(**comp["entry_model"])
        exit_cfg_raw = comp["exit_model"]
        exit_model_cfg = ExitModelConfig(**exit_cfg_raw)

        exp_cfg = ExperimentConfig(
            name=self.version_key,
            strategy=self.version_key,
            components=ComponentsConfig(
                features=comp["features"],
                target=target_cfg,
                entry_model=entry_model_cfg,
                exit_model=exit_model_cfg,
            ),
            split=SplitConfig(),
            mods=cfg_dict.get("mods", {}),
            params=cfg_dict.get("params", {}),
        )

        return build_prediction_cache(exp_cfg, symbols, device=device)

    def _run_backtest(self, prediction_cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import importlib

        cfg = self._model_cfg
        strategy = cfg.get("strategy", self.version_key)
        module_name, func_name = LEGACY_STRATEGY_MAP[strategy]
        module = importlib.import_module(module_name)
        backtest_fn = getattr(module, func_name)

        mods = cfg.get("mods", {})
        params = cfg.get("params", {})
        proba_thresholds = cfg.get("proba_thresholds")

        sig_params = set(inspect.signature(backtest_fn).parameters.keys())

        if params:
            base_fn = backtest_fn

            def backtest_fn(y_pred, returns, df_test, feature_cols, **kwargs):
                return base_fn(y_pred, returns, df_test, feature_cols, **{**kwargs, **params})

        all_mod_kwargs = {
            "mod_a": mods.get("a", True),
            "mod_b": mods.get("b", True),
            "mod_c": mods.get("c", False),
            "mod_d": mods.get("d", False),
            "mod_e": mods.get("e", True),
            "mod_f": mods.get("f", True),
            "mod_g": mods.get("g", True),
            "mod_h": mods.get("h", True),
            "mod_i": mods.get("i", True),
            "mod_j": mods.get("j", True),
        }
        mod_kwargs = {k: v for k, v in all_mod_kwargs.items() if k in sig_params}

        all_trades: list[dict[str, Any]] = []
        for item in prediction_cache:
            y_pred_eff = (
                _apply_proba_thresholds(item, proba_thresholds)
                if proba_thresholds
                else item["y_pred"]
            )
            extra: dict[str, Any] = {}
            y_pred_exit = item.get("y_pred_exit")
            if y_pred_exit is not None and "y_pred_exit" in sig_params:
                extra["y_pred_exit"] = y_pred_exit

            r = backtest_fn(
                y_pred_eff,
                item["returns"],
                item["sym_test_df"],
                item["feature_cols"],
                **mod_kwargs,
                **extra,
            )
            for t in r["trades"]:
                t["symbol"] = item["symbol"]
            all_trades.extend(r["trades"])

        return all_trades


def _apply_proba_thresholds(item: dict[str, Any], proba_thresholds: dict[str, float] | None) -> Any:
    """Apply per-class probability thresholds to override raw predictions."""
    if proba_thresholds is None or item.get("y_proba") is None:
        return item["y_pred"]

    y_proba = item["y_proba"]
    class_to_idx = {int(cls): idx for idx, cls in enumerate(item.get("classes", [-1, 0, 1]))}
    y_pred = item["y_pred"].copy()

    for cls_str, threshold in proba_thresholds.items():
        cls = int(cls_str)
        idx = class_to_idx.get(cls)
        if idx is None:
            continue
        mask = y_proba[:, idx] >= threshold
        y_pred[mask] = cls

    return y_pred


def _trades_to_dataframe(trades: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert list of trade dicts to standardized DataFrame."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    col_order = [
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "pnl",
        "exit_reason",
    ]
    existing = [c for c in col_order if c in df.columns]
    rest = [c for c in df.columns if c not in col_order]
    return df[existing + rest]


def list_legacy_versions() -> list[str]:
    """Return all available legacy version keys (excluding champions)."""
    return sorted(k for k in LEGACY_STRATEGY_MAP if k not in CHAMPION_VERSIONS)


def list_all_legacy_versions() -> list[str]:
    """Return all legacy version keys including champion aliases."""
    return sorted(LEGACY_STRATEGY_MAP)
