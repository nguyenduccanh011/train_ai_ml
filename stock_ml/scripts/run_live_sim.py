"""CLI entry: stock_ml/scripts/run_live_sim.py

Run live simulator on historical data with day-by-day walk-forward execution.

Example:
    python -m stock_ml.scripts.run_live_sim \\
        --symbols AAA,SSI,VND \\
        --sim-start 2025-01-01 \\
        --sim-end 2025-12-31 \\
        --out results/live_sim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.backtest.engine import CostModel, EngineConfig  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.live_sim.config import LiveSimConfig  # noqa: E402
from src.live_sim.loop import LiveSimEngine  # noqa: E402
from src.targets.forward import ForwardReturnTarget  # noqa: E402

DEFAULT_DATA_ROOT = str(REPO_ROOT / "portable_data" / "vn_stock_ai_dataset_cleaned")


def parse_symbols(raw: str, data_root: str, max_symbols: int) -> list[str]:
    if raw.strip().upper() == "ALL":
        syms = DataLoader(data_root).list_symbols()
    else:
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if max_symbols and len(syms) > max_symbols:
        syms = syms[:max_symbols]
    return syms


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run live simulator: day-by-day walk-forward with signal freezing"
    )
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--symbols", default="AAA,SSI,VND,HPG,FPT")
    p.add_argument("--max-symbols", type=int, default=0, help="0 = no limit")
    p.add_argument("--sim-start", default="2025-01-02", help="First sim trading date (YYYY-MM-DD)")
    p.add_argument("--sim-end", default="2025-12-31", help="Last sim trading date (YYYY-MM-DD)")
    p.add_argument("--out", default=str(REPO_ROOT / "stock_ml" / "results" / "live_sim"))
    p.add_argument("--train-years", type=int, default=4)
    p.add_argument("--gap-days", type=int, default=25)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--gain-threshold", type=float, default=0.04)
    p.add_argument("--loss-threshold", type=float, default=0.04)
    p.add_argument("--max-hold-bars", type=int, default=20)
    p.add_argument("--min-hold-bars", type=int, default=1)
    p.add_argument("--hard-stop-pct", type=float, default=-0.08)
    p.add_argument(
        "--signal-exit-enabled",
        type=lambda x: x.lower() in ("true", "1"),
        default=True,
        help="Use model sell signal (-1) as exit trigger",
    )
    p.add_argument("--commission", type=float, default=0.0015)
    p.add_argument("--tax", type=float, default=0.0010)
    p.add_argument("--slippage", type=float, default=0.0010)
    p.add_argument("--min-volume-filter", type=float, default=0.0)
    args = p.parse_args(argv)

    symbols = parse_symbols(args.symbols, args.data_root, args.max_symbols)

    cfg = LiveSimConfig(
        data_root=args.data_root,
        symbols=symbols,
        out_dir=args.out,
        sim_start=args.sim_start,
        sim_end=args.sim_end,
        train_years=args.train_years,
        gap_days=args.gap_days,
        target=ForwardReturnTarget(
            horizon=args.horizon,
            gain_threshold=args.gain_threshold,
            loss_threshold=args.loss_threshold,
        ),
        engine=EngineConfig(
            max_hold_bars=args.max_hold_bars,
            min_hold_bars=args.min_hold_bars,
            hard_stop_pct=args.hard_stop_pct,
            signal_exit_enabled=args.signal_exit_enabled,
            cost=CostModel(
                commission=args.commission,
                tax=args.tax,
                slippage=args.slippage,
            ),
        ),
        min_volume_filter=args.min_volume_filter,
    )

    engine = LiveSimEngine(cfg)
    summary = engine.run()

    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
