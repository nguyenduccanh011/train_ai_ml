"""§11.5.3 — Two-way equivalence attestation. Flip a bundle's ``parity.json`` to PASS iff the PRODUCTION
signal path reproduces the xưởng backtest on the FRESH serve window (§11.9: attest on the un-seeded part).

It runs the bundle through the exact production chain (``generate_signals_from_bundle`` = build_feature_frame
→ predict_slot_signals → recombine_signals) on the same OHLCV the backtest saw, then compares the discrete
``signal`` (∈ {-1,0,1}) to the backtest's ``run_signals`` (Postgres) on the overlapping (symbol, date) at or
after the attest window. Signal-level, bit-exact (§14.3 — no tolerance). 100% match → parity.status=PASS
with the compared counts; any mismatch → stays PENDING and prints the diff (production is wrong, fix it).

The bundle model must be the one that OWNS the attest window — export with ``--replicate-last-fold`` so the
bundle == the backtest's last walk-forward fold; then the serve window reproduces exactly (single-fit trains
on a wider window and will legitimately differ).

Usage:
  python stock_ml/scripts/ops/attest_bundle.py --bundle <dir> --run-id <backtest run_id> \
      --duckdb market_data/market.duckdb [--window 2025-01-01]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stock_ml"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402

from stock_ml.src.serving.bundle import load_bundle  # noqa: E402
from stock_ml.src.serving.inference import generate_signals_from_bundle  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")


def _backtest_signals(run_id: str, window: str, symbols: list[str]) -> pd.DataFrame:
    con = psycopg2.connect(**PG)
    try:
        df = pd.read_sql(
            "SELECT symbol, date, signal FROM run_signals "
            "WHERE run_id=%s AND date >= %s AND symbol = ANY(%s)",
            con,
            params=(run_id, window, symbols),
        )
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bundle", required=True)
    p.add_argument(
        "--run-id", required=True, help="backtest run_id whose run_signals are the xưởng side"
    )
    p.add_argument("--duckdb", default="market_data/market.duckdb")
    p.add_argument(
        "--window", default=None, help="attest from this date (default: bundle cutoff_date)"
    )
    a = p.parse_args()

    bundle_dir = Path(a.bundle)
    bundle = load_bundle(bundle_dir)
    # Feed the FEATURE-SCOPE universe: the backtest computed CSRank/breadth over the UNION of every fold's
    # symbols (build_feature_frame runs once on that union, then the splitter masks per fold). Serving must
    # feed the same union or the cross-sectional features — and thus the signals — diverge. Fall back to the
    # serve universe for legacy static bundles that have no universe_by_year.
    uby = bundle.manifest.get("universe_by_year") or {}
    if uby:
        universe = sorted({s for lst in uby.values() for s in lst})
    else:
        universe = list(bundle.manifest.get("universe", []))
    if not universe:
        raise SystemExit("attest: bundle manifest has no universe")
    window = a.window or bundle.manifest.get("cutoff_date")
    if not window:
        raise SystemExit("attest: no --window and no cutoff_date in manifest")

    # Production side: full history for warmup (z-window/trailing features), then keep the attest window.
    from src.data.loader import get_loader

    loader = get_loader(a.duckdb)
    available = set(loader.list_symbols())
    load_syms = [s for s in universe if s in available]
    ohlcv = loader.load_many(load_syms)[
        ["symbol", "date", "open", "high", "low", "close", "volume"]
    ]
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    prod = generate_signals_from_bundle(bundle, ohlcv)[["symbol", "date", "signal"]]
    prod["date"] = pd.to_datetime(prod["date"]).dt.normalize()
    prod = prod[prod["date"] >= pd.Timestamp(window)]

    # Xưởng side: the backtest's persisted signals on the same window/universe.
    bt = _backtest_signals(a.run_id, window, load_syms)

    merged = prod.merge(bt, on=["symbol", "date"], suffixes=("_prod", "_bt"), how="inner")
    n = len(merged)
    if n == 0:
        raise SystemExit(
            f"attest: no overlapping (symbol,date) between production and run_id={a.run_id} "
            f"on/after {window} — check the run_id/window/universe"
        )
    mism = merged[merged["signal_prod"] != merged["signal_bt"]]
    n_mis = len(mism)
    print(
        f"[attest] window>={window} | universe={len(load_syms)} | compared={n} | mismatch={n_mis} "
        f"({100 * (n - n_mis) / n:.4f}% match)"
    )

    parity_path = bundle_dir / "parity.json"
    parity = (
        json.loads(parity_path.read_text(encoding="utf-8"))
        if parity_path.is_file()
        else {"schema": "parity.json/1", "generated_by": "attest_bundle"}
    )
    if n_mis == 0:
        parity["status"] = "PASS"
        parity["attest"] = {
            "run_id": a.run_id,
            "window": str(window),
            "compared": n,
            "note": "production signal chain reproduced the backtest on the serve window",
        }
        parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[attest] PASS — {n} signals reproduce bit-for-bit. parity.json -> PASS.")
    else:
        ex = mism.head(5)[["symbol", "date", "signal_prod", "signal_bt"]].to_dict("records")
        print(
            f"[attest] FAIL — {n_mis}/{n} signals differ. parity stays PENDING. First diffs: {ex}"
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
