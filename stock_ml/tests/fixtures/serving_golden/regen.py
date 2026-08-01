"""Regenerate the serving-golden fixture (ENGINE_UPGRADE §11.5.3 / Phase 2b). DEV-ONLY reproducer.

The committed fixture is what ``tests/test_serving_golden.py`` checks; this script documents exactly how
it was produced so an intentional re-baseline (Phase 4) can refresh it reproducibly. It is NOT run by the
test suite (the test only reads the committed artifacts — no switch, no auto-regen).

Needs the train Postgres (template config) + the back-adjusted DuckDB (OHLCV). Run from the repo root:

    DATABASE_URL=postgresql+asyncpg://stockml:stockml_dev@localhost:5433/stockml \
    STOCK_DATA_DIR=$PWD/market_data/market.duckdb \
    python stock_ml/tests/fixtures/serving_golden/regen.py

The fixture strategy uses per-symbol features only (``leading_v2``), so the served chain reads nothing but
the slice — the committed fixture is self-contained (no duckdb at test time). The DuckDB above is needed
only here, to source the training rows + the OHLCV slice.

Provenance of the current fixture:
  - strategy/template : template 112 ``sumEX_thr15``, strategy ``regression_dual_ml_recombine`` — entry
                        forward-return + exit forward-drawdown heads on per-symbol ``leading_v2`` features
                        (no ensemble / breadth / xsec → no market-context read at serve time).
  - symbols           : FPT, SSI, HPG, MWG, VCI, VND (6 liquid names).
  - bundle            : single-fit on data <= 2024-01-01 (no prediction_history → deterministic).
  - slice             : 2022-06-01..2024-06-30 (>= 260 warmup bars before the 2024 signal window).
  - golden            : discrete signal (buy/sell/hold) from generate_signals_from_bundle on the slice.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

FX = Path(__file__).resolve().parent
DUCKDB = REPO_ROOT / "market_data" / "market.duckdb"
TEMPLATE_ID = 112
SYMBOLS = ["FPT", "SSI", "HPG", "MWG", "VCI", "VND"]
CUTOFF = "2024-01-01"
SLICE_START, SLICE_END = "2022-06-01", "2024-06-30"


def main() -> None:
    from stock_ml.src.data.loader import get_loader
    from stock_ml.src.serving.bundle import load_bundle
    from stock_ml.src.serving.inference import generate_signals_from_bundle

    tmp_out = FX / "_tmp_export"
    shutil.rmtree(tmp_out, ignore_errors=True)
    shutil.rmtree(FX / "bundle", ignore_errors=True)

    # 1) mini single-fit bundle via the production export path
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "stock_ml/scripts/ops/export_bundle.py"),
            "--template-id",
            str(TEMPLATE_ID),
            "--symbols",
            ",".join(SYMBOLS),
            "--duckdb",
            str(DUCKDB),
            "--cutoff",
            CUTOFF,
            "--retrain-schedule",
            "yearly",
            "--out",
            str(tmp_out),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    built = next(tmp_out.glob("bundle_*"))
    shutil.move(str(built), str(FX / "bundle"))
    shutil.rmtree(tmp_out, ignore_errors=True)

    # 2) committed OHLCV slice (warm for the z-window)
    raw = get_loader(str(DUCKDB)).load_many(SYMBOLS)
    raw["date"] = pd.to_datetime(raw["date"])
    sl = raw[(raw.date >= pd.Timestamp(SLICE_START)) & (raw.date <= pd.Timestamp(SLICE_END))]
    sl = sl[["symbol", "date", "open", "high", "low", "close", "volume"]]
    sl = sl.sort_values(["symbol", "date"]).reset_index(drop=True)
    sl.to_parquet(FX / "ohlcv_slice.parquet", index=False)

    # 3) golden discrete signals via the production chain
    sig = generate_signals_from_bundle(load_bundle(FX / "bundle"), sl)[["symbol", "date", "signal"]]
    sig = sig.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.strftime("%Y-%m-%d")
    sig.sort_values(["symbol", "date"]).reset_index(drop=True).to_csv(
        FX / "golden_signals.csv", index=False
    )
    print(f"[regen] wrote bundle + slice ({len(sl)} rows) + golden ({len(sig)} rows)")


if __name__ == "__main__":
    main()
