"""Legacy pipeline wrapper (backward compatibility wrapper over run_experiment).

Delegates to run_experiment() for unified pipeline logic.
Maintains legacy output structure for backward compatibility with scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.backtest.engine import CostModel, EngineConfig
from src.data.loader import DataLoader
from src.pipeline.experiment import ExperimentConfig, run_experiment
from src.targets.forward import ForwardReturnTarget
from src.tracking.mlflow_logger import data_fingerprint, get_git_commit


@dataclass
class RunConfig:
    """Legacy config format (year-based splits, hardcoded models/features)."""

    data_root: str
    symbols: list[str]
    out_dir: str
    name: str = "baseline"
    train_years: int = 4
    test_years: int = 1
    gap_days: int = 25
    first_test_year: int = 2020
    last_test_year: int = 2025
    target: ForwardReturnTarget = field(default_factory=ForwardReturnTarget)
    engine: EngineConfig = field(default_factory=EngineConfig)
    seed: int = 42

    @property
    def required_gap_days(self) -> int:
        return max(self.target.horizon * 2 + 5, 7)


def run(cfg: RunConfig) -> dict:
    """Run pipeline via unified run_experiment() (delegates to single source of truth).

    Converts RunConfig → ExperimentConfig → run_experiment().
    Creates run_id directory structure for backward compatibility.
    """
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + f"-{get_git_commit()[:8]}"
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = ExperimentConfig(
        name=cfg.name,
        strategy="entry_exit_ensemble",
        market="vn_stock",
        feature_set="basic_v1",
        target={
            "type": "forward_return",
            "horizon": cfg.target.horizon,
            "gain_threshold": cfg.target.gain_threshold,
            "loss_threshold": cfg.target.loss_threshold,
        },
        entry_model={"type": "lightgbm", "params": {}},
        exit_model={"type": "none", "enabled": False, "params": {}},
        split={
            "type": "walk_forward_year",
            "train_years": cfg.train_years,
            "test_years": cfg.test_years,
            "gap_days": cfg.gap_days,
            "first_test_year": cfg.first_test_year,
            "last_test_year": cfg.last_test_year,
        },
        engine={
            "max_hold_bars": cfg.engine.max_hold_bars,
            "min_hold_bars": cfg.engine.min_hold_bars,
            "hard_stop_pct": cfg.engine.hard_stop_pct,
            "commission": cfg.engine.cost.commission,
            "tax": cfg.engine.cost.tax,
            "slippage": cfg.engine.cost.slippage,
        },
        seed=cfg.seed,
    )

    summary = run_experiment(
        exp_cfg,
        data_root=cfg.data_root,
        symbols=cfg.symbols,
        out_dir=str(out),
        run_id=run_id,
    )

    if summary.get("ok") is False:
        return summary

    loader = DataLoader(cfg.data_root)
    available = set(loader.list_symbols())
    requested = [s for s in cfg.symbols if s in available]

    data_fp = data_fingerprint(
        requested,
        "",
        "",
    )

    trades_src = out / f"trades_{cfg.name}.csv"
    signals_src = out / f"signals_{cfg.name}.csv"
    daily_src = out / f"daily_stats_{cfg.name}.csv"
    yearly_src = out / f"yearly_stats_{cfg.name}.csv"
    symbol_src = out / f"symbol_stats_{cfg.name}.csv"
    summary_src = out / f"summary_{cfg.name}.json"

    trades_dst = run_dir / "trades.csv"
    signals_dst = run_dir / "signals.csv"
    daily_dst = run_dir / "daily_stats.csv"
    yearly_dst = run_dir / "yearly_stats.csv"
    symbol_dst = run_dir / "symbol_stats.csv"
    summary_dst = run_dir / "summary.json"
    fingerprint_dst = run_dir / "data_fingerprint.txt"

    if trades_src.exists():
        trades_src.rename(trades_dst)
    if signals_src.exists():
        signals_src.rename(signals_dst)
    if daily_src.exists():
        daily_src.rename(daily_dst)
    if yearly_src.exists():
        yearly_src.rename(yearly_dst)
    if symbol_src.exists():
        symbol_src.rename(symbol_dst)

    if summary_src.exists():
        summary_src.unlink()

    fingerprint_dst.write_text(f"{data_fp}\n{get_git_commit()}\n", encoding="utf-8")

    legacy_summary = {
        "run_id": run_id,
        "name": cfg.name,
        "n_symbols": summary.get("n_symbols", 0),
        "n_trades": summary.get("n_trades", 0),
        "n_signals_buy": summary.get("n_signals_buy", 0),
        "n_signals_sell": summary.get("n_signals_sell", 0),
        "data_fingerprint": data_fp,
        "git_commit": get_git_commit(),
        "mlflow_run_id": summary.get("name", "unknown"),
        "aggregate": summary.get("aggregate", {}),
        "audit": summary.get("audit", {}),
        "outputs": {
            "trades": str(trades_dst),
            "signals": str(signals_dst),
            "daily_stats": str(daily_dst),
            "yearly_stats": str(yearly_dst),
            "symbol_stats": str(symbol_dst),
            "fingerprint": str(fingerprint_dst),
        },
        "config": summary.get("config", {}),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    summary_dst.write_text(json.dumps(legacy_summary, indent=2, default=str), encoding="utf-8")

    return legacy_summary


def build_default_config(
    data_root: str,
    symbols: list[str],
    out_dir: str,
    name: str = "baseline",
    **overrides,
) -> RunConfig:
    """Build legacy config from kwargs (for backward compatibility)."""
    target_kw = {
        k: overrides.pop(k)
        for k in list(overrides)
        if k in {"horizon", "gain_threshold", "loss_threshold"}
    }
    cost_kw = {
        k: overrides.pop(k) for k in list(overrides) if k in {"commission", "tax", "slippage"}
    }
    engine_kw = {
        k: overrides.pop(k)
        for k in list(overrides)
        if k in {"max_hold_bars", "min_hold_bars", "hard_stop_pct", "signal_exit_enabled"}
    }
    cfg = RunConfig(
        data_root=data_root,
        symbols=symbols,
        out_dir=out_dir,
        name=name,
        target=ForwardReturnTarget(**target_kw) if target_kw else ForwardReturnTarget(),
        engine=EngineConfig(cost=CostModel(**cost_kw), **engine_kw),
        **overrides,
    )
    return cfg
