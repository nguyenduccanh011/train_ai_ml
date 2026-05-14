from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_rs_ablation import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    _align_last_test_year,
    _load_symbols,
    _run_one,
    _set_feature_set,
)

DEFAULT_OUT_DIR = ROOT / "results" / "weighted_rs_universe_compare"

UNIVERSES = [
    ("universe61_initial", ROOT / "config" / "train61_symbols.json"),
    ("universe65_liquidity_prev", ROOT / "reports" / "liquidity_1b_symbols_prev65.json"),
    ("universe298_liquidity_1b", ROOT / "config" / "liquidity_1b_symbols.json"),
]


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n_symbols",
        "trades",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "pf",
        "max_loss",
        "max_drawdown",
        "mdd_per_symbol",
        "yearly_consistency",
        "composite_score",
        "wins",
        "losses",
        "runtime_sec",
        "trades_path",
    ]
    return {key: summary.get(key) for key in keys}


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in [
        "n_symbols",
        "trades",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "pf",
        "mdd_per_symbol",
        "yearly_consistency",
        "composite_score",
    ]:
        lval = left.get(key)
        rval = right.get(key)
        if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
            result[key] = round(float(lval) - float(rval), 6)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest leading_rs_weighted on 61, 65, and 298-symbol universes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    os.environ["STOCK_DATA_DIR"] = str(args.data_dir.resolve())
    os.environ["STOCK_RESULTS_DIR"] = str((ROOT / "results").resolve())

    from src.pipeline import ExperimentConfig

    base_cfg = ExperimentConfig.from_yaml(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    for label, symbols_path in UNIVERSES:
        symbols = _load_symbols(symbols_path)
        cfg = _align_last_test_year(
            _set_feature_set(base_cfg, "leading_rs_weighted", f"weighted_rs_{label}"),
            symbols,
            args.data_dir,
        )
        run_dir = args.out_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_one(label, cfg, symbols, run_dir)
        summary["symbols_path"] = str(symbols_path)
        summaries[label] = summary

    compact = {label: _compact(summary) for label, summary in summaries.items()}
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "feature_set": "leading_rs_weighted",
        "summaries": compact,
        "delta_vs_61": {
            "universe65_liquidity_prev": _delta(
                compact["universe65_liquidity_prev"],
                compact["universe61_initial"],
            ),
            "universe298_liquidity_1b": _delta(
                compact["universe298_liquidity_1b"],
                compact["universe61_initial"],
            ),
        },
    }
    report_path = args.out_dir / "weighted_rs_universe_compare_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Weighted RS Universe Compare ===")
    for label, summary in compact.items():
        print(
            f"{label}: symbols={summary['n_symbols']} trades={summary['trades']} "
            f"wr={summary['win_rate']} pnl={summary['total_pnl']} "
            f"pf={summary['pf']} score={summary['composite_score']} "
            f"runtime={summary['runtime_sec']}s"
        )
    print("delta_vs_61:", report["delta_vs_61"])
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
