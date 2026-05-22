"""Regenerate champion golden artifacts after leakage fix.

Iterates over the 11 champion versions, replicates each parity test's runner call,
and writes fresh CSV + meta.json + updated checksums.txt.

Usage:
    python -m stock_ml.scripts.regen_champion_goldens
    python -m stock_ml.scripts.regen_champion_goldens --only v22,v32
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_DIR = ROOT / "tests" / "regression" / "golden"
CHECKSUMS = GOLDEN_DIR / "checksums.txt"


CHAMPION_SPECS = {
    "v22": {
        "runner": "run_v22",
        "df_converter": "trades_to_v22_dataframe",
        "enable_exit_model": True,
    },
    "v19_3": {
        "runner": "run_v19_3",
        "df_converter": "trades_to_v19_3_dataframe",
        "enable_exit_model": False,
    },
    "v32": {
        "runner": "run_v32",
        "df_converter": "trades_to_v32_dataframe",
        "enable_exit_model": False,
    },
    "v34": {
        "runner": "run_v34",
        "df_converter": "trades_to_v34_dataframe",
        "enable_exit_model": False,
    },
    "v35b": {
        "runner": "run_v35b",
        "df_converter": "trades_to_v35b_dataframe",
        "enable_exit_model": False,
    },
    "v37a": {
        "runner": "run_v37a",
        "df_converter": "trades_to_v37a_dataframe",
        "enable_exit_model": False,
    },
    "v37a_exit": {
        "runner": "run_v37a_exit",
        "df_converter": "trades_to_v37a_exit_dataframe",
        "enable_exit_model": False,
    },
    "v37d": {
        "runner": "run_v37d",
        "df_converter": "trades_to_v37d_dataframe",
        "enable_exit_model": False,
    },
    "v39d": {
        "runner": "run_v39d",
        "df_converter": "trades_to_v39d_dataframe",
        "enable_exit_model": False,
    },
    "v42_a": {
        "runner": "run_v42_a",
        "df_converter": "trades_to_v42_a_dataframe",
        "enable_exit_model": False,
    },
}


def regen_rule() -> tuple[str, int]:
    """Rule baseline uses run_rule_baseline directly, no prediction cache."""
    from src.components.runners import run_rule_baseline, trades_to_dataframe
    from src.env import resolve_data_dir

    meta_path = GOLDEN_DIR / "trades_rule.meta.json"
    csv_path = GOLDEN_DIR / "trades_rule.csv"
    meta = json.loads(meta_path.read_text())
    symbols = meta["symbols"]
    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")

    print(f"[rule] running rule baseline on {len(symbols)} symbols...")
    trades = run_rule_baseline(symbols=symbols, data_dir=str(data_dir), first_test_year=2020)
    df = trades_to_dataframe(trades).reset_index(drop=True)
    csv_text = df.to_csv(index=False)
    csv_path.write_bytes(csv_text.encode("utf-8"))
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    n_trades = len(df)

    meta["n_trades"] = n_trades
    meta["generated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta["generator"] = "scripts/regen_champion_goldens.py"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[rule] trades={n_trades} sha={sha[:12]}...")
    return sha, n_trades


def regen_champion(version: str, spec: dict) -> tuple[str, int]:
    """Regenerate one champion using its direct runner (matches parity test)."""
    from src.components import runners
    from src.env import resolve_data_dir
    from src.pipeline.build_predictions import _build_predictions

    meta_path = GOLDEN_DIR / f"trades_{version}.meta.json"
    csv_path = GOLDEN_DIR / f"trades_{version}.csv"
    meta = json.loads(meta_path.read_text())
    symbols = meta["symbols"]
    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")

    print(f"[{version}] building predictions ({len(symbols)} symbols)...")
    prediction_cache = _build_predictions(
        symbols,
        meta["feature_set"],
        meta["target_config"],
        "cpu",
        model_type=meta["model_type"],
        exit_model_cfg=meta.get("exit_model_config"),
    )

    runner_fn = getattr(runners, spec["runner"])
    df_converter = getattr(runners, spec["df_converter"])
    runner_kwargs = {
        "symbols": symbols,
        "data_dir": str(data_dir),
        "prediction_cache": prediction_cache,
    }
    if spec["enable_exit_model"]:
        runner_kwargs["enable_exit_model"] = True

    print(f"[{version}] running {spec['runner']}...")
    trades = runner_fn(**runner_kwargs)
    df_raw = df_converter(trades).reset_index(drop=True)
    # Round-trip via CSV (matches parity test normalization)
    csv_text = df_raw.to_csv(index=False)
    df_round = pd.read_csv(io.StringIO(csv_text)).reset_index(drop=True)
    n_trades = len(df_round)

    csv_path.write_bytes(csv_text.encode("utf-8"))
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    meta["n_trades"] = n_trades
    meta["generated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta["generator"] = "scripts/regen_champion_goldens.py"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[{version}] trades={n_trades} sha={sha[:12]}...")
    return sha, n_trades


def update_checksums(new_hashes: dict[str, str]) -> None:
    lines = CHECKSUMS.read_text().splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            out.append(line)
            continue
        old_hash, name = parts
        fname = name.lstrip("*")
        if fname in new_hashes:
            out.append(f"{new_hashes[fname]} *{fname}")
            seen.add(fname)
        else:
            out.append(line)
    # Append any new entries not seen
    for fname, h in new_hashes.items():
        if fname not in seen:
            out.append(f"{h} *{fname}")
    CHECKSUMS.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[checksums] updated {len(new_hashes)} entries")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated versions to regen (default: all 11 champions)",
    )
    args = parser.parse_args()

    versions = (
        [v.strip() for v in args.only.split(",") if v.strip()]
        if args.only
        else ["rule", *CHAMPION_SPECS.keys()]
    )

    new_hashes: dict[str, str] = {}
    for v in versions:
        try:
            if v == "rule":
                sha, _ = regen_rule()
            elif v in CHAMPION_SPECS:
                sha, _ = regen_champion(v, CHAMPION_SPECS[v])
            else:
                print(f"[skip] unknown version: {v}")
                continue
            new_hashes[f"trades_{v}.csv"] = sha
        except Exception as exc:
            print(f"[ERROR] {v}: {exc}")
            raise

    if new_hashes:
        update_checksums(new_hashes)

    print("\nAll done. Re-run regression tests to verify:")
    print("  python -m pytest stock_ml/tests/regression/ -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
