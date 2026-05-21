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
    TOP1_CONFIG_PATH,
    _build_day_snapshot,
    load_cutoff_dates,
    load_dataset,
    load_symbols,
    load_top1_config,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay top1 daily signals from main project.")
    parser.add_argument("--days", type=int, default=5, help="Number of trading days to replay.")
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
        help="Report output directory.",
    )
    parser.add_argument("--min-history", type=int, default=260)
    parser.add_argument("--watchlist-top", type=int, default=10)
    return parser.parse_args()


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

    days: list[dict[str, Any]] = []
    for cutoff in cutoff_dates:
        print(f"[replay] {cutoff} symbols={len(symbols)}", flush=True)
        snapshot = _build_day_snapshot(
            cfg=cfg,
            cutoff=cutoff,
            raw_df=raw_df,
            feature_engine=feature_engine,
            symbols=symbols,
            watchlist_top=args.watchlist_top,
            min_history=args.min_history,
        )
        days.append(
            {
                "date": snapshot.cutoff,
                "open_positions": snapshot.open_positions,
                "new_entries": snapshot.new_entries,
                "next_session_predictions": snapshot.next_session_predictions,
                "watchlist_top_buy_proba": snapshot.watchlist_top_buy_proba,
                "all_predictions_path": str(output_dir / f"all_predictions_{snapshot.cutoff}.csv"),
                "stats": snapshot.stats,
                "errors": snapshot.errors,
            }
        )
        write_csv(output_dir / f"all_predictions_{snapshot.cutoff}.csv", snapshot.all_predictions)
        print(
            f"  open={snapshot.stats['open_positions']} new={snapshot.stats['new_entries']} "
            f"entry_next={snapshot.stats['entry_signal_for_next_bar']} "
            f"exit_next={snapshot.stats['exit_signal_for_next_bar']} "
            f"errors={snapshot.stats['errors']}",
            flush=True,
        )

    latest = days[-1]
    latest_date = latest["date"]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(TOP1_CONFIG_PATH),
        "symbols": len(symbols),
        "days": days,
    }
    json_path = output_dir / f"daily_signal_report_{latest_date}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    write_csv(output_dir / f"open_positions_{latest_date}.csv", latest["open_positions"])
    write_csv(output_dir / f"new_entries_{latest_date}.csv", latest["new_entries"])
    write_csv(
        output_dir / f"next_session_predictions_{latest_date}.csv",
        latest["next_session_predictions"],
    )
    write_csv(
        output_dir / f"watchlist_top_buy_proba_{latest_date}.csv",
        latest["watchlist_top_buy_proba"],
    )

    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
