from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.realtime_top1_common import (
    _build_day_snapshot,
    load_cutoff_dates,
    load_dataset,
    load_symbols,
    load_top1_config,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build next-day signal handoff from top1 replay.")
    parser.add_argument("--days", type=int, default=1, help="Number of trading days to inspect.")
    parser.add_argument("--as-of", default="", help="Last cutoff date YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols or JSON/TXT file. Default: all dataset symbols.",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        default=0,
        help="Limit symbols for smoke runs. 0 = all.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "realtime_top1_replay"),
        help="Handoff output directory.",
    )
    parser.add_argument("--min-history", type=int, default=260)
    parser.add_argument("--watchlist-top", type=int, default=10)
    return parser.parse_args()


def _handoff_rows(snapshot) -> list[dict[str, Any]]:
    rows = []
    for rank, row in enumerate(snapshot.all_predictions, start=1):
        if int(row.get("entry_signal_for_next_bar") or 0) != 1:
            continue
        rows.append(
            {
                "signal_date": snapshot.cutoff,
                "signal_rank": rank,
                "symbol": row["symbol"],
                "buy_proba_for_next_bar": row.get("buy_proba_for_next_bar"),
                "entry_signal_for_next_bar": row.get("entry_signal_for_next_bar"),
                "exit_signal_for_next_bar": row.get("exit_signal_for_next_bar"),
                "history_rows": row.get("history_rows"),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_top1_config()
    symbols = load_symbols(args.symbols)
    if args.limit_symbols > 0:
        symbols = symbols[: args.limit_symbols]

    raw_df = load_dataset(symbols)
    cutoff_dates = load_cutoff_dates(raw_df, args.days, args.as_of)
    if not cutoff_dates:
        raise SystemExit("No cutoff dates found")

    from src.features.engine import FeatureEngine

    feature_engine = FeatureEngine(feature_set=cfg.feature_set())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff = cutoff_dates[-1]
    print(f"[handoff] {cutoff} symbols={len(symbols)}", flush=True)
    snapshot = _build_day_snapshot(
        cfg=cfg,
        cutoff=cutoff,
        raw_df=raw_df,
        feature_engine=feature_engine,
        symbols=symbols,
        watchlist_top=args.watchlist_top,
        min_history=args.min_history,
    )

    handoff_rows = _handoff_rows(snapshot)
    handoff_csv = output_dir / f"signal_handoff_{cutoff}.csv"
    write_csv(handoff_csv, handoff_rows)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": cutoff,
        "handoff_path": str(handoff_csv),
        "symbols": len(symbols),
        "candidate_count": len(handoff_rows),
        "entry_signal_count": snapshot.stats["entry_signal_for_next_bar"],
    }
    manifest_path = output_dir / f"signal_handoff_{cutoff}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"wrote {handoff_csv}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
