"""Regenerate v22 golden artifacts after leakage fix changes trade behavior.

Writes:
  tests/regression/golden/trades_v22.csv  (overwritten with fresh pipeline output)
  tests/regression/golden/trades_v22.meta.json  (updated n_trades, generated_at)
  tests/regression/golden/checksums.txt  (updated v22 SHA256)

Usage:
    python -m stock_ml.scripts.regen_v22_golden
"""

from __future__ import annotations

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
GOLDEN_CSV = GOLDEN_DIR / "trades_v22.csv"
GOLDEN_META = GOLDEN_DIR / "trades_v22.meta.json"
CHECKSUMS = GOLDEN_DIR / "checksums.txt"
CHAMPION_YAML = ROOT / "config" / "experiments" / "champions" / "v22.yaml"


def main() -> int:
    from src.env import resolve_data_dir
    from src.pipeline import ExperimentConfig, Pipeline
    from src.pipeline.build_predictions import _build_predictions

    meta = json.loads(GOLDEN_META.read_text())
    symbols: list[str] = meta["symbols"]
    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")
    if not Path(data_dir).is_dir():
        print(f"ERROR: data dir not found: {data_dir}")
        return 1

    print(f"[regen] Building predictions for {len(symbols)} symbols...")
    prediction_cache = _build_predictions(
        symbols,
        meta["feature_set"],
        meta["target_config"],
        "cpu",
        model_type=meta["model_type"],
        exit_model_cfg=meta["exit_model_config"],
    )

    print("[regen] Running pipeline...")
    cfg = ExperimentConfig.from_yaml(CHAMPION_YAML)
    pipeline = Pipeline(cfg, symbols=symbols, device="cpu", prediction_cache=prediction_cache)
    result = pipeline.run()

    from src.components.runners import trades_to_v22_dataframe

    df = trades_to_v22_dataframe(result.trades).reset_index(drop=True)
    csv_text = df.to_csv(index=False)
    df_round = pd.read_csv(io.StringIO(csv_text)).reset_index(drop=True)
    n_trades = len(df_round)

    print(f"[regen] Pipeline produced {n_trades} trades (was {meta['n_trades']})")

    # Write CSV
    GOLDEN_CSV.write_text(csv_text, encoding="utf-8")
    sha = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    print(f"[regen] Wrote {GOLDEN_CSV.name}  sha256={sha[:12]}...")

    # Update meta
    meta["n_trades"] = n_trades
    meta["generated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta["generator"] = "scripts/regen_v22_golden.py"
    GOLDEN_META.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[regen] Wrote {GOLDEN_META.name}")

    # Update checksums.txt
    lines = CHECKSUMS.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.endswith(" *trades_v22.csv"):
            new_lines.append(f"{sha} *trades_v22.csv")
        else:
            new_lines.append(line)
    CHECKSUMS.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[regen] Wrote {CHECKSUMS.name}")
    print("[regen] Done. Re-run regression tests to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
