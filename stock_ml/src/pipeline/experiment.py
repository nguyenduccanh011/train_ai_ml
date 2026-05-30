"""Experiment pipeline — research-grade end-to-end training and backtesting.

Supports independent entry/exit models and YAML-driven configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.backtest.engine import CostModel, EngineConfig, run_backtest, trades_to_dataframe
from src.backtest.integrity import audit_report, print_report
from src.backtest.stats import (
    aggregate_stats,
    per_day_stats,
    per_symbol_stats,
    per_year_stats,
)
from src.data.loader import DataLoader
from src.data.splitter import PurgedKFoldSplitter, YearSplitter
from src.features.registry import apply_features, get_feature_cols
from src.models.registry import build_entry_model, build_exit_model, build_regression_model
from src.signals.core import (
    generate_signals_from_predictions,
    generate_signals_from_technical_rules,
)
from src.targets.registry import build_target


@dataclass
class ExperimentConfig:
    """Configuration for an experiment — maps cleanly from YAML.

    Phase 1b.6: Canonical nested schema validation.
    Supports full YAML structure with components, split, engine, validation.
    """

    name: str
    strategy: str
    market: str
    feature_set: str
    target: dict
    entry_model: dict
    exit_model: dict
    split: dict
    engine: dict
    seed: int = 42
    signal_threshold: float = 0.0
    hypothesis: str = ""
    validation: dict | None = None
    strict_audit: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load experiment config from YAML file with strict validation.

        Args:
            path: path to YAML file

        Returns:
            ExperimentConfig instance

        Raises:
            FileNotFoundError: if file not found
            ValueError: if required fields missing or schema invalid
        """
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        required_top = ["name", "strategy", "market", "components", "split", "engine"]
        missing = [k for k in required_top if k not in raw]
        if missing:
            raise ValueError(f"YAML missing required keys: {missing}")

        comp = raw.get("components", {})
        required_comp = ["features", "target", "entry_model"]
        missing_comp = [k for k in required_comp if k not in comp]
        if missing_comp:
            raise ValueError(f"components missing required keys: {missing_comp}")

        target_cfg = comp["target"]
        if "type" not in target_cfg:
            raise ValueError("components.target missing required 'type' field")
        if target_cfg["type"] not in [
            "forward_return",
            "forward_return_regression",
            "trend_regime",
        ]:
            raise ValueError(f"Unknown target type: {target_cfg['type']}")

        entry_model_cfg = comp["entry_model"]
        if "type" not in entry_model_cfg:
            raise ValueError("components.entry_model missing required 'type' field")

        split_cfg = raw.get("split", {})
        if "type" not in split_cfg:
            raise ValueError("split missing required 'type' field")
        valid_split_types = ["walk_forward_year", "purged_kfold"]
        if split_cfg["type"] not in valid_split_types:
            raise ValueError(f"Unknown split type: {split_cfg['type']}")

        engine_cfg = raw.get("engine", {})
        if not isinstance(engine_cfg, dict):
            raise ValueError("engine must be a dict")

        validation_cfg = raw.get("validation")
        if validation_cfg is not None and not isinstance(validation_cfg, dict):
            raise ValueError("validation must be a dict or null")

        return cls(
            name=raw["name"],
            strategy=raw["strategy"],
            market=raw["market"],
            feature_set=comp.get("features", "basic_v1"),
            target=target_cfg,
            entry_model=entry_model_cfg,
            exit_model=comp.get("exit_model", {"type": "none", "enabled": False, "params": {}}),
            split=split_cfg,
            engine=engine_cfg,
            seed=raw.get("seed", 42),
            signal_threshold=raw.get("signal_threshold", 0.0),
            hypothesis=raw.get("hypothesis", ""),
            validation=validation_cfg,
            strict_audit=raw.get("strict_audit", True),
        )


def train_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    cfg: ExperimentConfig,
) -> tuple[Any, Any | None, pd.DataFrame]:
    """Train entry/exit models on fold and generate test signals.

    **PHASE 1b.1 DECISION: Regression approach (professional standard)**

    Supports both regression (float targets) and classification ({-1, 0, 1} targets).
    Auto-detects based on target dtype.

    **Regression (RECOMMENDED for Phase 1b+):**
    - Single entry model predicts forward return as float ∈ ℝ
    - Signal via threshold (cfg.signal_threshold, default 0):
      * if pred_return > +threshold → buy (1)
      * if pred_return < -threshold → sell (-1)
      * else → hold (0)
    - No separate exit model (exit signal from return magnitude)
    - Uses full training data (no row dropping, no semantic mixing)
    - Aligns with professional quant methodology (Two Sigma, Man AHL, de Prado)
    - Enables Phase 3 sizing via return magnitude (Kelly fractional ready)

    **Classification (Legacy, not recommended):**
    - Separate entry/exit models for binary classification
    - Entry: buy (1) vs not-buy (0)
    - Exit: sell (1) vs not-sell (0)
    - Note: Phase 1b.1 blocker was semantic issue (negative class mixin {neutral, sell})

    Args:
        train_df: training DataFrame with [symbol, date, target, *feat_cols]
                  target dtype determines mode: float64 → regression, int8 → classification
        test_df: test DataFrame with [symbol, date, *feat_cols]
        feat_cols: list of feature column names to use for training
        cfg: ExperimentConfig with model types, params, and signal_threshold

    Returns:
        (entry_model, exit_model_or_none, signals_df)
        signals_df has [symbol, date, signal, score] where:
          - signal ∈ {-1, 0, 1}
          - score = predicted return (regression) or predict_proba (classification)
    """
    train_clean = train_df.dropna(subset=["target", *feat_cols])
    if train_clean.empty:
        raise ValueError("No clean training data after dropna")

    X_train = train_clean[feat_cols].to_numpy(dtype=np.float32)
    y_full = train_clean["target"].to_numpy()

    test_use = test_df.dropna(subset=feat_cols).copy()
    if test_use.empty:
        return None, None, pd.DataFrame(columns=["symbol", "date", "signal", "score"])

    X_test = test_use[feat_cols].to_numpy(dtype=np.float32)

    # Rule-based technical indicators (no ML model)
    if cfg.entry_model["type"] == "technical_rules":
        return None, None, generate_signals_from_technical_rules(test_use)

    is_regression = y_full.dtype.kind == "f"

    if is_regression:
        entry_model = build_regression_model(
            cfg.entry_model["type"], cfg.entry_model.get("params", {})
        )
        entry_model.fit(X_train, y_full)
        exit_model = None

        pred_returns = entry_model.predict(X_test)

        signals_df = generate_signals_from_predictions(
            predictions=pred_returns,
            test_df=test_use,
            signal_threshold=cfg.signal_threshold,
            is_regression=True,
        )

    else:
        y_full = y_full.astype(np.int8)
        y_entry = (y_full == 1).astype(np.int8)
        entry_model = build_entry_model(cfg.entry_model["type"], cfg.entry_model.get("params", {}))
        entry_model.fit(X_train, y_entry)

        exit_model = None
        if cfg.exit_model.get("enabled", False):
            y_exit = (y_full == -1).astype(np.int8)
            exit_model = build_exit_model(cfg.exit_model["type"], cfg.exit_model.get("params", {}))
            exit_model.fit(X_train, y_exit)

        signals_df = generate_signals_from_predictions(
            predictions=None,
            test_df=test_use,
            signal_threshold=cfg.signal_threshold,
            is_regression=False,
            entry_model=entry_model,
            exit_model=exit_model,
            X_test=X_test,
        )

    return entry_model, exit_model, signals_df


def run_experiment(
    cfg: ExperimentConfig,
    data_root: str,
    symbols: list[str],
    out_dir: str,
    run_id: str | None = None,
) -> dict:
    """Run full experiment — load, train, backtest, report.

    Phase 1b.11: Resumable runs via fold checkpointing.

    Args:
        cfg: ExperimentConfig
        data_root: path to OHLCV data directory
        symbols: list of symbols to use
        out_dir: output directory for results CSVs + JSON
        run_id: optional run_id for fold checkpointing; if provided, saves folds to
                {out_dir}/{run_id}/folds/{fold_label}.parquet for resumability

    Returns:
        summary dict with metrics and file paths
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fold_cache_dir = None
    if run_id:
        fold_cache_dir = out / run_id / "folds"
        fold_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg.name}] loading {len(symbols)} symbols from {data_root}")

    loader = DataLoader(data_root)
    available = set(loader.list_symbols())
    requested = [s for s in symbols if s in available]
    missing = sorted(set(symbols) - available)
    if missing:
        print(
            f"  [warn] missing symbols skipped: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if not requested:
        raise ValueError("no symbols available in the dataset")

    raw = loader.load_many(requested)
    ohlcv = raw[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()

    print(f"[{cfg.name}] applying features: {cfg.feature_set}")
    feat = apply_features(raw, cfg.feature_set)
    feat_cols = get_feature_cols(cfg.feature_set)

    print(f"[{cfg.name}] applying target: {cfg.target['type']}")
    target = build_target(cfg.target)
    feat = target.apply(feat)

    print(f"[{cfg.name}] dataset: {len(ohlcv)} bars across {ohlcv['symbol'].nunique()} symbols")

    split_cfg = cfg.split.copy()
    split_type = split_cfg.get("type", "walk_forward_year")

    if split_type == "walk_forward_year":
        train_years = split_cfg.get("train_years", 4)
        test_years = split_cfg.get("test_years", 1)
        gap_days = split_cfg.get("gap_days", 25)

        if "first_test_year" in split_cfg and "last_test_year" in split_cfg:
            splitter = YearSplitter(
                train_years=train_years,
                test_years=test_years,
                gap_days=gap_days,
                first_test_year=split_cfg["first_test_year"],
                last_test_year=split_cfg["last_test_year"],
            )
            print(
                f"  [split] year range: {split_cfg['first_test_year']}-{split_cfg['last_test_year']} (explicit)"
            )
        else:
            splitter = YearSplitter.from_data(
                feat,
                train_years=train_years,
                test_years=test_years,
                gap_days=gap_days,
            )
            print(
                f"  [split] year range: {splitter.first_test_year}-{splitter.last_test_year} (auto-detected)"
            )
    elif split_type == "purged_kfold":
        n_splits = split_cfg.get("n_splits", 5)
        embargo_days = split_cfg.get("embargo_days", 0)
        label_horizon = split_cfg.get("label_horizon", 5)

        splitter = PurgedKFoldSplitter(
            n_splits=n_splits,
            embargo_days=embargo_days,
            label_horizon=label_horizon,
        )
        print(
            f"  [split] purged_kfold: {n_splits} splits, embargo_days={embargo_days}, label_horizon={label_horizon}"
        )
        windows = None  # Will be collected during split loop
    else:
        raise NotImplementedError(f"split type '{split_type}' not yet implemented")

    if split_type == "walk_forward_year":
        windows = splitter.windows()
    else:
        windows = None  # For purged_kfold, windows are collected during split()

    signal_frames: list[pd.DataFrame] = []
    windows_list = []  # Collect windows during split for purged_kfold
    for w, train_df, test_df in splitter.split(feat):
        windows_list.append(w)
        if train_df.empty or test_df.empty:
            print(f"  [fold {w.label}] empty — skipped")
            continue

        fold_cache_path = fold_cache_dir / f"{w.label}.parquet" if fold_cache_dir else None

        if fold_cache_path and fold_cache_path.exists():
            signals = pd.read_parquet(fold_cache_path)
            n_buys = (signals["signal"] > 0).sum()
            n_sells = (signals["signal"] < 0).sum()
            print(
                f"  [fold {w.label}] restored from checkpoint  buys={n_buys:>5}  sells={n_sells:>5}"
            )
            signal_frames.append(signals)
            continue

        try:
            entry_model, exit_model, signals = train_fold(train_df, test_df, feat_cols, cfg)
            if signals.empty:
                print(f"  [fold {w.label}] no signals generated — skipped")
                continue

            n_buys = (signals["signal"] > 0).sum()
            n_sells = (signals["signal"] < 0).sum()
            print(
                f"  [fold {w.label}] train={len(train_df.dropna(subset=['target', *feat_cols])):>6}  "
                f"test={len(signals):>6}  buys={n_buys:>5}  sells={n_sells:>5}"
            )

            if fold_cache_path:
                signals.to_parquet(fold_cache_path, index=False, compression="snappy")

            signal_frames.append(signals)
        except Exception as e:
            print(f"  [fold {w.label}] error: {e} — skipped")
            continue

    if not signal_frames:
        print("[!] no signals produced — abort")
        return {"name": cfg.name, "ok": False, "reason": "no_signals"}

    signals_all = pd.concat(signal_frames, ignore_index=True)
    signals_all = signals_all.sort_values(["symbol", "date"]).reset_index(drop=True)

    engine_cfg = cfg.engine.copy()
    # Handle nested costs dict from YAML (costs: {commission: 0.0025, tax: 0.001, ...})
    costs_dict = engine_cfg.pop("costs", {})
    # Also handle flat keys for backward compatibility
    cost_kw = costs_dict.copy() if costs_dict else {}
    # Remove slippage_model if present (not used by CostModel, but may be in YAML for doc)
    cost_kw.pop("slippage_model", None)
    # Also allow flat-level cost keys
    for k in ["commission", "tax", "slippage"]:
        if k in engine_cfg:
            cost_kw[k] = engine_cfg.pop(k)
    # Remove slippage_model from engine_cfg if present
    engine_cfg.pop("slippage_model", None)
    cost = CostModel(**cost_kw) if cost_kw else CostModel()
    engine = EngineConfig(cost=cost, **engine_cfg)

    print(f"[{cfg.name}] backtesting {len(signals_all)} signals")
    trades = run_backtest(signals_all, ohlcv, engine)
    trades_df = trades_to_dataframe(trades)

    agg = aggregate_stats(trades_df)
    yearly = per_year_stats(trades_df)
    daily = per_day_stats(trades_df)
    by_sym = per_symbol_stats(trades_df)

    required_gap = max(cfg.target.get("horizon", 5) * 2 + 5, 7)
    # Use collected windows for purged_kfold, pre-computed for walk_forward_year
    audit_windows = (
        windows if windows is not None else windows_list if "windows_list" in locals() else None
    )
    report = audit_report(trades_df, signals_all, windows=audit_windows, min_gap_days=required_gap)
    print_report(report)

    if cfg.strict_audit and report["overall"] == "FAIL":
        raise ValueError(f"[{cfg.name}] Strict audit failed: {report['n_fail']} check(s) failed")

    trades_path = out / f"trades_{cfg.name}.csv"
    signals_path = out / f"signals_{cfg.name}.csv"
    daily_path = out / f"daily_stats_{cfg.name}.csv"
    yearly_path = out / f"yearly_stats_{cfg.name}.csv"
    symbol_path = out / f"symbol_stats_{cfg.name}.csv"
    summary_path = out / f"summary_{cfg.name}.json"

    trades_df.to_csv(trades_path, index=False)
    signals_all.to_csv(signals_path, index=False)
    daily.to_csv(daily_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    by_sym.to_csv(symbol_path, index=False)

    summary = {
        "name": cfg.name,
        "strategy": cfg.strategy,
        "market": cfg.market,
        "feature_set": cfg.feature_set,
        "n_symbols": int(ohlcv["symbol"].nunique()),
        "n_trades": int(len(trades_df)),
        "n_signals_buy": int((signals_all["signal"] > 0).sum()),
        "n_signals_sell": int((signals_all["signal"] < 0).sum()),
        "entry_model": cfg.entry_model["type"],
        "exit_model_type": cfg.exit_model["type"],
        "exit_model_enabled": cfg.exit_model.get("enabled", False),
        "aggregate": agg,
        "audit": report,
        "outputs": {
            "trades": str(trades_path.relative_to(out.parent)),
            "signals": str(signals_path.relative_to(out.parent)),
            "daily_stats": str(daily_path.relative_to(out.parent)),
            "yearly_stats": str(yearly_path.relative_to(out.parent)),
            "symbol_stats": str(symbol_path.relative_to(out.parent)),
        },
        "config": {
            "feature_set": cfg.feature_set,
            "target": cfg.target,
            "entry_model": cfg.entry_model,
            "exit_model": cfg.exit_model,
            "split": cfg.split,
            "engine": {
                "max_hold_bars": engine.max_hold_bars,
                "min_hold_bars": engine.min_hold_bars,
                "hard_stop_pct": engine.hard_stop_pct,
                "commission": engine.cost.commission,
                "tax": engine.cost.tax,
                "slippage": engine.cost.slippage,
            },
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[{cfg.name}] wrote outputs to {out}")

    return summary
