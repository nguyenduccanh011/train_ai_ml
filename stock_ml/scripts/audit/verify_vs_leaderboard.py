"""Credibility check: how closely does a bundle reproduce the leaderboard backtest?

Compares the bundle's serving signals (single-fit + z-window primed by the bundle
model) against the leaderboard run's stored per-bar signals (walk-forward, each year
predicted by that year's fold). They are expected to AGREE strongly in the year the
bundle is responsible for (its cutoff year, where bundle ≈ the last fold's model) and
to DIVERGE in earlier years (the bundle re-predicts the past with the newer model).

This quantifies the production↔backtest gap per year so it is known, not assumed.

Usage:
  python stock_ml/scripts/verify_vs_leaderboard.py \
      --bundle bundles/bundle_n2_pbfill_p0p03_w25_2026-01-01 \
      --leaderboard-csv results/_lb_signals_958.csv \
      --duckdb market_data/market.duckdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

import pandas as pd  # noqa: E402

from src.serving.bundle import load_bundle  # noqa: E402
from src.serving.inference import generate_signals_from_bundle  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--leaderboard-csv", required=True)
    p.add_argument("--duckdb", default="market_data/market.duckdb")
    p.add_argument(
        "--since",
        default="2024-06-01",
        help="load OHLCV from this date (avoids old std==0 NaN bars; "
        "needs >=~1y warmup before the cutoff year)",
    )
    args = p.parse_args()

    bundle = load_bundle(args.bundle)
    cutoff_year = pd.Timestamp(bundle.manifest["cutoff_date"]).year
    symbols = bundle.manifest.get("universe") or []

    from src.data.loader import get_loader

    loader = get_loader(args.duckdb)
    available = set(loader.list_symbols())
    requested = [s for s in symbols if s in available]
    raw = loader.load_many(requested)
    ohlcv = raw[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    ohlcv = ohlcv[pd.to_datetime(ohlcv["date"]) >= pd.Timestamp(args.since)].copy()

    print(
        f"[verify-lb] generating bundle signals for {len(requested)} symbols "
        f"(since {args.since}) ..."
    )
    bsig = generate_signals_from_bundle(bundle, ohlcv)[["symbol", "date", "signal"]].copy()
    bsig["date"] = pd.to_datetime(bsig["date"]).dt.normalize()

    lb = pd.read_csv(args.leaderboard_csv)
    lb["date"] = pd.to_datetime(lb["date"]).dt.normalize()

    m = lb.merge(bsig, on=["symbol", "date"], suffixes=("_lb", "_bundle"), how="inner")
    if m.empty:
        raise SystemExit("[verify-lb] no overlapping (symbol,date) — check inputs")
    m["year"] = m["date"].dt.year
    m["agree"] = m["signal_lb"] == m["signal_bundle"]

    print(f"\n[verify-lb] overlapping bars: {len(m)}")
    print(
        f"{'year':<6}{'bars':>8}{'agree%':>9}{'lb_buy':>8}{'bd_buy':>8}{'lb_sell':>9}{'bd_sell':>9}"
        + "   (bundle responsible)"
    )
    for y, g in m.groupby("year"):
        mark = "  <== cutoff year" if y == cutoff_year else ""
        print(
            f"{y:<6}{len(g):>8}{g['agree'].mean() * 100:>8.1f}%"
            f"{int((g['signal_lb'] > 0).sum()):>8}{int((g['signal_bundle'] > 0).sum()):>8}"
            f"{int((g['signal_lb'] < 0).sum()):>9}{int((g['signal_bundle'] < 0).sum()):>9}{mark}"
        )

    cut = m[m["year"] == cutoff_year]
    overall = m["agree"].mean() * 100
    cut_agree = cut["agree"].mean() * 100 if len(cut) else float("nan")
    print(
        f"\n[verify-lb] overall agree {overall:.1f}% | cutoff-year {cutoff_year} agree {cut_agree:.1f}%"
    )
    # confusion on the cutoff year (the production-relevant comparison)
    if len(cut):
        print(f"[verify-lb] cutoff-year confusion (lb -> bundle):")
        ct = pd.crosstab(cut["signal_lb"], cut["signal_bundle"])
        print(ct.to_string())

        # Disagreement BY MONTH — clustering early in the year points at warmup (the
        # 252-bar z-window + path-dependent downleg leg-state aren't fully warm yet).
        cut = cut.copy()
        cut["month"] = cut["date"].dt.month
        print(f"\n[verify-lb] cutoff-year disagreement by month (warmup check):")
        print(f"{'month':<7}{'bars':>7}{'disagree':>10}{'disagree%':>11}")
        for mo, g in cut.groupby("month"):
            dis = (~g["agree"]).sum()
            print(f"{mo:<7}{len(g):>7}{dis:>10}{(dis / len(g) * 100):>10.1f}%")


if __name__ == "__main__":
    main()
