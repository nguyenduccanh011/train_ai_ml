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
    DEFAULT_SYMBOLS,
    _align_last_test_year,
    _load_symbols,
    _run_one,
    _set_feature_set,
)

DEFAULT_OUT_DIR = ROOT / "results" / "rs_ablation"


def _set_mods(cfg: Any, *, g: bool, j: bool, name_suffix: str) -> Any:
    clone = cfg.model_copy(deep=True)
    clone.name = f"{clone.name}__{name_suffix}"
    clone.mods["g"] = g
    clone.mods["j"] = j
    if clone.strategy_v3 is not None:
        clone.strategy_v3.mods["g"] = g
        clone.strategy_v3.mods["j"] = j
    return clone


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest weighted RS UI variants with bear/chop filters relaxed."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-prefix", default="weighted_rs")
    parser.add_argument(
        "--only",
        choices=["all", "baseline", "no_bear", "no_chop", "no_bear_no_chop"],
        default="all",
        help="Run all variants or only one.",
    )
    args = parser.parse_args()

    os.environ["STOCK_DATA_DIR"] = str(args.data_dir.resolve())
    os.environ["STOCK_RESULTS_DIR"] = str((ROOT / "results").resolve())

    from src.pipeline import ExperimentConfig

    symbols = _load_symbols(args.symbols)
    base_cfg = ExperimentConfig.from_yaml(args.config)
    weighted_cfg = _align_last_test_year(
        _set_feature_set(base_cfg, "leading_rs_weighted", "weighted_rs"),
        symbols,
        args.data_dir,
    )
    variants = [
        (
            f"{args.label_prefix}_baseline",
            "baseline",
            _set_mods(weighted_cfg, g=True, j=True, name_suffix="baseline"),
        ),
        (
            f"{args.label_prefix}_relaxed_no_bear",
            "no_bear",
            _set_mods(weighted_cfg, g=False, j=True, name_suffix="relaxed_no_bear"),
        ),
        (
            f"{args.label_prefix}_relaxed_no_chop",
            "no_chop",
            _set_mods(weighted_cfg, g=True, j=False, name_suffix="relaxed_no_chop"),
        ),
        (
            f"{args.label_prefix}_relaxed_no_bear_no_chop",
            "no_bear_no_chop",
            _set_mods(weighted_cfg, g=False, j=False, name_suffix="relaxed_no_bear_no_chop"),
        ),
    ]
    if args.only != "all":
        variants = [variant for variant in variants if variant[1] == args.only]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "weighted_rs_ui_variants_summary.json"
    existing_variants: dict[str, Any] = {}
    if report_path.exists() and args.only != "all":
        try:
            existing_payload = json.loads(report_path.read_text(encoding="utf-8"))
            existing_variants = dict(existing_payload.get("variants", {}))
        except Exception:
            existing_variants = {}
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args.config),
        "symbols": str(args.symbols),
        "data_dir": str(args.data_dir),
        "variants": existing_variants,
    }
    for label, _, cfg in variants:
        summary = _run_one(label, cfg, symbols, args.out_dir)
        summary["variant"] = label
        summary["mods"] = {"g": bool(cfg.mods["g"]), "j": bool(cfg.mods["j"])}
        report["variants"][f"{label}_trades.csv"] = summary

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== UI Variant Summary ===")
    for filename, summary in report["variants"].items():
        print(
            f"{filename}: trades={summary['trades']} wr={summary['win_rate']} "
            f"pnl={summary['total_pnl']} pf={summary['pf']} "
            f"score={summary['composite_score']} runtime={summary['runtime_sec']}s"
        )
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
