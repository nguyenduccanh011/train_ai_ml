"""CLI entry: stock_ml/scripts/run_v2.py

Example:
    python -m stock_ml.scripts.run_v2 --symbols AAA,SSI,VND --out results/baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.data.loader import DataLoader  # noqa: E402
from src.pipeline.run import build_default_config, run  # noqa: E402

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
    p = argparse.ArgumentParser(description="Run leakage-safe walk-forward backtest")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--symbols", default="AAA,SSI,VND,HPG,FPT")
    p.add_argument("--max-symbols", type=int, default=0, help="0 = no limit")
    p.add_argument("--out", default=str(REPO_ROOT / "stock_ml" / "results" / "v2_baseline"))
    p.add_argument("--name", default="baseline")
    p.add_argument("--train-years", type=int, default=4)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--gap-days", type=int, default=25)
    p.add_argument("--first-test-year", type=int, default=2020)
    p.add_argument("--last-test-year", type=int, default=2024)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--gain-threshold", type=float, default=0.04)
    p.add_argument("--loss-threshold", type=float, default=0.04)
    p.add_argument("--max-hold-bars", type=int, default=20)
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
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args(argv)

    symbols = parse_symbols(args.symbols, args.data_root, args.max_symbols)
    cfg = build_default_config(
        data_root=args.data_root,
        symbols=symbols,
        out_dir=args.out,
        name=args.name,
        train_years=args.train_years,
        test_years=args.test_years,
        gap_days=args.gap_days,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        horizon=args.horizon,
        gain_threshold=args.gain_threshold,
        loss_threshold=args.loss_threshold,
        max_hold_bars=args.max_hold_bars,
        hard_stop_pct=args.hard_stop_pct,
        signal_exit_enabled=args.signal_exit_enabled,
        commission=args.commission,
        tax=args.tax,
        slippage=args.slippage,
        seed=args.seed,
    )
    summary = run(cfg)
    return 0 if summary.get("audit", {}).get("overall") != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
