from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from paths import CONFIG_PATH, ROOT, STOCK_ML_ROOT

DEFAULT_RUN_DIR = (
    STOCK_ML_ROOT
    / "results"
    / "experiments"
    / "derivatives_vn30f1m_phase259_30m_exit_model_algo_micro"
    / "deriv_p259_30m_exit_model_algo_micro_signals_features-all_features-signals_entry_model_type-xgboost-signals_target-tr_dual_5_27_c3-exit_model-x27l0188_cat-fusion-exit_only-params-baseline"
)
DEFAULT_OUT = ROOT / "config" / "model_config.realtime_top1.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_execution(base_cfg: dict[str, Any], ranking_row: dict[str, Any]) -> dict[str, Any]:
    execution = dict(base_cfg.get("execution") or {})
    execution["backtester"] = execution.get("backtester", "run_v22")
    execution["pnl_mode"] = ranking_row.get("pnl_mode", "futures_contract")
    execution["currency"] = ranking_row.get("currency", execution.get("currency", "VND"))
    execution["capital"] = execution.get("capital", 100000000)
    execution["commission"] = 0.0
    execution["tax"] = 0.0
    execution["slippage"] = 0.0
    return execution


def build_runtime_config(run_dir: Path, base_cfg_path: Path) -> dict[str, Any]:
    run_cfg_path = run_dir / "config.resolved.yaml"
    ranking_row_path = run_dir / "ranking_row.json"
    metrics_path = run_dir / "metrics.json"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_cfg_path}")
    if not ranking_row_path.exists():
        raise FileNotFoundError(f"Missing ranking row: {ranking_row_path}")

    run_cfg = _read_yaml(run_cfg_path)
    base_cfg = _read_yaml(base_cfg_path)
    ranking_row = _read_json(ranking_row_path)
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}

    out: dict[str, Any] = {}
    for key in (
        "name",
        "strategy",
        "market",
        "runner",
        "components",
        "split",
        "mods",
        "params",
        "fusion",
        "signals",
        "strategy_v3",
    ):
        if key in run_cfg:
            out[key] = run_cfg[key]

    out["execution"] = _runtime_execution(base_cfg, ranking_row)
    out["realtime_profile"] = {
        "mode": "live_inference",
        "source_run_dir": str(run_dir),
        "source_config_hash": ranking_row.get("config_hash", ""),
        "source_metrics": {
            "trades": metrics.get("trades"),
            "wr": metrics.get("wr"),
            "total_pnl": metrics.get("total_pnl"),
            "pf": metrics.get("pf"),
            "composite_score": metrics.get("composite_score"),
        },
        "notes": "Use this config for realtime data inference. Backtest summary fields above are for reference only.",
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract top1 run config into a realtime runtime config."
    )
    parser.add_argument(
        "--run-dir", default=str(DEFAULT_RUN_DIR), help="Path to top1 run directory"
    )
    parser.add_argument(
        "--base-config", default=str(CONFIG_PATH), help="Base standalone config path"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output realtime config path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_cfg_path = Path(args.base_config)
    output = Path(args.output)

    payload = build_runtime_config(run_dir=run_dir, base_cfg_path=base_cfg_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Wrote realtime config: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
