"""End-to-end runnable pipeline.

  load OHLCV  ->  add features  ->  add forward-return target
              ->  walk-forward year split (with gap)
              ->  train baseline classifier per fold
              ->  predict signals on test fold
              ->  backtest with next-bar fill + costs
              ->  stats (aggregate / year / day) + integrity audit
              ->  write CSV + JSON to results dir

Designed to be invoked from the CLI script `stock_ml/scripts/run_v2.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.engine import CostModel, EngineConfig, run_backtest, trades_to_dataframe
from src.backtest.integrity import audit_report, print_report
from src.backtest.stats import aggregate_stats, per_day_stats, per_symbol_stats, per_year_stats
from src.data.loader import DataLoader
from src.data.splitter import YearSplitter
from src.features.basic import FEATURE_COLS, add_features
from src.models.baseline import BaselineModel
from src.seed import set_global_seed
from src.targets.forward import ForwardReturnTarget
from src.tracking.mlflow_logger import MLFlowLogger, data_fingerprint, get_git_commit


@dataclass
class RunConfig:
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
        # forward_window is in bars; for 1D timeframe + weekends, ~1.5× horizon suffices
        return max(self.target.horizon * 2 + 5, 7)


def _prep_dataset(cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(cfg.data_root)
    available = set(loader.list_symbols())
    requested = [s for s in cfg.symbols if s in available]
    missing = sorted(set(cfg.symbols) - available)
    if missing:
        logger.warning(
            f"missing symbols skipped: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if not requested:
        raise ValueError("no symbols available in the dataset")
    raw = loader.load_many(requested)
    ohlcv = raw[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    feat = add_features(raw)
    feat = cfg.target.apply(feat)
    return ohlcv, feat


def _train_and_predict(feat: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, list]:
    splitter = YearSplitter(
        train_years=cfg.train_years,
        test_years=cfg.test_years,
        gap_days=cfg.gap_days,
        first_test_year=cfg.first_test_year,
        last_test_year=cfg.last_test_year,
    )
    windows = splitter.windows()
    signal_frames: list[pd.DataFrame] = []

    for w, train_df, test_df in splitter.split(feat):
        if train_df.empty or test_df.empty:
            logger.debug(f"fold {w.label}: empty — skipped")
            continue
        train_clean = train_df.dropna(subset=["target", *FEATURE_COLS])
        if train_clean.empty:
            logger.debug(f"fold {w.label}: no usable train rows — skipped")
            continue
        X_tr = train_clean[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_tr = train_clean["target"].to_numpy(dtype=np.float64)
        model = BaselineModel(seed=cfg.seed).fit(X_tr, y_tr)

        test_use = test_df.dropna(subset=FEATURE_COLS).copy()
        if test_use.empty:
            continue
        X_te = test_use[FEATURE_COLS].to_numpy(dtype=np.float32)
        preds = model.predict(X_te)
        test_use = test_use.assign(signal=preds)
        signal_frames.append(test_use[["symbol", "date", "signal"]])
        logger.info(
            f"fold {w.label}: train={len(train_clean):>6} test={len(test_use):>6} "
            f"buys={(preds > 0).sum():>5} sells={(preds < 0).sum():>5}"
        )

    if not signal_frames:
        return pd.DataFrame(columns=["symbol", "date", "signal"]), windows
    signals = pd.concat(signal_frames, ignore_index=True)
    signals = signals.sort_values(["symbol", "date"]).reset_index(drop=True)
    return signals, windows


def run(cfg: RunConfig) -> dict:
    set_global_seed(cfg.seed)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + f"-{get_git_commit()[:8]}"
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"starting run {cfg.name} / {run_id}")
    logger.info(f"loading {len(cfg.symbols)} symbols from {cfg.data_root}")

    with MLFlowLogger(tracking_uri=out / "mlruns", experiment_name=cfg.name) as mlflow:
        ohlcv, feat = _prep_dataset(cfg)
        logger.info(f"dataset: {len(ohlcv)} bars across {ohlcv['symbol'].nunique()} symbols")

        signals, windows = _train_and_predict(feat, cfg)
        if signals.empty:
            logger.error("no signals produced — abort")
            return {"name": cfg.name, "ok": False, "reason": "no_signals"}

        trades = run_backtest(signals, ohlcv, cfg.engine)
        trades_df = trades_to_dataframe(trades)
        agg = aggregate_stats(trades_df)
        yearly = per_year_stats(trades_df)
        daily = per_day_stats(trades_df)
        by_sym = per_symbol_stats(trades_df)
        report = audit_report(
            trades_df, signals, windows=windows, min_gap_days=cfg.required_gap_days
        )
        print_report(report)

        data_fp = data_fingerprint(
            cfg.symbols,
            ohlcv["date"].min().strftime("%Y-%m-%d"),
            ohlcv["date"].max().strftime("%Y-%m-%d"),
        )

        mlflow.log_config(cfg)
        mlflow.log_reproducibility(
            cfg.symbols,
            ohlcv["date"].min().strftime("%Y-%m-%d"),
            ohlcv["date"].max().strftime("%Y-%m-%d"),
            cfg.seed,
        )
        mlflow.log_metrics(agg)

        trades_path = run_dir / "trades.csv"
        signals_path = run_dir / "signals.csv"
        daily_path = run_dir / "daily_stats.csv"
        yearly_path = run_dir / "yearly_stats.csv"
        symbol_path = run_dir / "symbol_stats.csv"
        fingerprint_path = run_dir / "data_fingerprint.txt"
        summary_path = run_dir / "summary.json"

        trades_df.to_csv(trades_path, index=False)
        signals.to_csv(signals_path, index=False)
        daily.to_csv(daily_path, index=False)
        yearly.to_csv(yearly_path, index=False)
        by_sym.to_csv(symbol_path, index=False)
        fingerprint_path.write_text(f"{data_fp}\n{get_git_commit()}\n", encoding="utf-8")

        summary = {
            "run_id": run_id,
            "name": cfg.name,
            "n_symbols": int(ohlcv["symbol"].nunique()),
            "n_trades": int(len(trades_df)),
            "n_signals_buy": int((signals["signal"] > 0).sum()),
            "n_signals_sell": int((signals["signal"] < 0).sum()),
            "data_fingerprint": data_fp,
            "git_commit": get_git_commit(),
            "mlflow_run_id": mlflow.get_run_id(),
            "aggregate": agg,
            "audit": report,
            "outputs": {
                "trades": str(trades_path),
                "signals": str(signals_path),
                "daily_stats": str(daily_path),
                "yearly_stats": str(yearly_path),
                "symbol_stats": str(symbol_path),
                "fingerprint": str(fingerprint_path),
            },
            "config": {
                "train_years": cfg.train_years,
                "test_years": cfg.test_years,
                "gap_days": cfg.gap_days,
                "first_test_year": cfg.first_test_year,
                "last_test_year": cfg.last_test_year,
                "target": {
                    "horizon": cfg.target.horizon,
                    "gain_threshold": cfg.target.gain_threshold,
                    "loss_threshold": cfg.target.loss_threshold,
                },
                "cost": {
                    "commission": cfg.engine.cost.commission,
                    "tax": cfg.engine.cost.tax,
                    "slippage": cfg.engine.cost.slippage,
                },
                "engine": {
                    "max_hold_bars": cfg.engine.max_hold_bars,
                    "min_hold_bars": cfg.engine.min_hold_bars,
                    "hard_stop_pct": cfg.engine.hard_stop_pct,
                },
            },
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info(f"wrote outputs to {run_dir}")

        mlflow.log_artifact(trades_path)
        mlflow.log_artifact(signals_path)
        mlflow.log_artifact(summary_path)

        return summary


def build_default_config(
    data_root: str,
    symbols: list[str],
    out_dir: str,
    name: str = "baseline",
    **overrides,
) -> RunConfig:
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
