"""Verify a serving bundle reproduces the full signal chain on real data.

Loads a bundle, rebuilds features on OHLCV (NO targets — pure inference), runs the
predict-only path + the aggregate recombine, and reports the resulting buy/sell
signals. Also checks the chain is deterministic (same input -> same signals).

This is the serving-side smoke gate: it proves a bundle's models + config + feature
spec actually drive the champion signal chain (predict -> decoupled z-bands ->
downleg force-gate) end-to-end, independent of the research DB.

Usage:
  python stock_ml/scripts/verify_bundle.py \
      --bundle bundles/bundle_n2_pbfill_p0p03_w25_2026-01-01 \
      --duckdb market_data/market.duckdb [--since 2024-01-01]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from stock_ml.src.serving.bundle import load_bundle  # noqa: E402
from stock_ml.src.serving.inference import generate_signals_from_bundle  # noqa: E402


def _serving_chain(bundle, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Full serving signal chain (shared with the serving repo via core façade)."""
    return generate_signals_from_bundle(bundle, ohlcv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--duckdb", default="market_data/market.duckdb")
    p.add_argument("--since", default=None, help="only load OHLCV on/after this date (warmup)")
    args = p.parse_args()

    bundle = load_bundle(args.bundle)
    print(f"[verify] loaded bundle {Path(args.bundle).name}")
    print(
        f"[verify] strategy={bundle.manifest.get('strategy')} "
        f"cutoff={bundle.manifest.get('cutoff_date')} models={sorted(bundle.models)}"
    )

    from stock_ml.src.data.loader import get_loader

    symbols = bundle.manifest.get("universe") or []
    loader = get_loader(args.duckdb)
    available = set(loader.list_symbols())
    requested = [s for s in symbols if s in available]
    raw = loader.load_many(requested)
    ohlcv = raw[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    if args.since:
        ohlcv = ohlcv[pd.to_datetime(ohlcv["date"]) >= pd.Timestamp(args.since)].copy()
    print(
        f"[verify] {len(ohlcv)} bars, {ohlcv['symbol'].nunique()} symbols, "
        f"{pd.to_datetime(ohlcv['date']).min().date()} .. {pd.to_datetime(ohlcv['date']).max().date()}"
    )

    sig1 = _serving_chain(bundle, ohlcv)
    sig2 = _serving_chain(bundle, ohlcv)

    # determinism
    s1 = sig1.sort_values(["symbol", "date"])["signal"].to_numpy()
    s2 = sig2.sort_values(["symbol", "date"])["signal"].to_numpy()
    deterministic = bool((s1 == s2).all())

    cutoff = pd.Timestamp(bundle.manifest["cutoff_date"])
    post = sig1[pd.to_datetime(sig1["date"]) > cutoff]

    n_buy = int((sig1["signal"] > 0).sum())
    n_sell = int((sig1["signal"] < 0).sum())
    print(f"[verify] signals: buy={n_buy} sell={n_sell} neutral={int((sig1['signal'] == 0).sum())}")
    print(
        f"[verify] POST-cutoff (production-relevant): "
        f"buy={int((post['signal'] > 0).sum())} sell={int((post['signal'] < 0).sum())} "
        f"over {post['date'].nunique()} dates"
    )
    print(f"[verify] deterministic: {deterministic}")

    problems = []
    if not deterministic:
        problems.append("non-deterministic signals")
    if n_buy == 0 and n_sell == 0:
        problems.append("no signals produced at all")
    if problems:
        raise SystemExit("[verify] FAIL: " + "; ".join(problems))
    print("[verify] PASS — serving chain reproduces signals deterministically")


if __name__ == "__main__":
    main()
