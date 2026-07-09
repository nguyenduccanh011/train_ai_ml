from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEADERBOARD_JSON = "leaderboard.json"
DEFAULT_EXPERIMENTS_DIR = ROOT / "results" / "experiments"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "leaderboard"


def cmd_rebuild(args: argparse.Namespace) -> int:
    from src.leaderboard.aggregator import rebuild_leaderboard

    rows = rebuild_leaderboard(args.experiments_dir, args.output_dir)

    # Print market breakdown
    from collections import defaultdict

    by_market = defaultdict(int)
    for row in rows:
        by_market[row.market] += 1

    print(f"rebuilt {len(rows)} rows -> {args.output_dir}")
    print(f"  by market: {dict(by_market)}")
    print(f"  per-market leaderboards: {args.output_dir}/by_market/<market>/")
    return 0


def cmd_append(args: argparse.Namespace) -> int:
    from src.leaderboard.aggregator import append_or_update

    row = append_or_update(args.run_dir, args.output_dir, bundle=args.bundle)
    print(f"appended {row.run_id} -> {args.output_dir}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from src.leaderboard.aggregator import validate_leaderboard

    rows = validate_leaderboard(args.leaderboard_json)
    print(f"validated {len(rows)} rows")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate backtest leaderboard artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rebuild = subparsers.add_parser(
        "rebuild", help="Rebuild leaderboard from all experiment ranking rows."
    )
    rebuild.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    rebuild.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    rebuild.set_defaults(func=cmd_rebuild)

    append = subparsers.add_parser("append", help="Append or replace one run in the leaderboard.")
    append.add_argument("run_dir", type=Path)
    append.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    append.add_argument("--bundle", default=None)
    append.set_defaults(func=cmd_append)

    validate = subparsers.add_parser("validate", help="Validate a leaderboard JSON file.")
    validate.add_argument(
        "leaderboard_json", type=Path, nargs="?", default=DEFAULT_OUTPUT_DIR / LEADERBOARD_JSON
    )
    validate.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
