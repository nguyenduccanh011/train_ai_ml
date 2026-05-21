from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train61 simple local scheduler")
    parser.add_argument("--model-id", default=os.getenv("TRAIN61_DEFAULT_MODEL", "train61_pooled"))
    parser.add_argument("--symbols", default=os.getenv("TRAIN61_SCHEDULER_SYMBOLS", ""))
    parser.add_argument(
        "--latest-limit", type=int, default=int(os.getenv("SIEUTINHIEU_LATEST_LIMIT", "10"))
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.getenv("SCHEDULER_INTERVAL_SECONDS", "86400")),
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--skip-ingest", action="store_true", help="Only compute signals")
    parser.add_argument("--force", action="store_true", help="Force signal regeneration")
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def run_cycle(args: argparse.Namespace) -> None:
    if not args.skip_ingest:
        ingest_cmd = [
            sys.executable,
            "tools/ingest_sieutinhieu.py",
            "--mode",
            "latest",
            "--latest-limit",
            str(args.latest_limit),
        ]
        if args.symbols:
            ingest_cmd.extend(["--symbols", args.symbols])
        run_command(ingest_cmd)

    worker_cmd = [
        sys.executable,
        "tools/worker.py",
        "run-once-after-ingest",
        "--model-id",
        args.model_id,
    ]
    if args.symbols:
        worker_cmd.extend(["--symbols", args.symbols])
    if args.force:
        worker_cmd.append("--force")
    run_command(worker_cmd)


def main() -> int:
    args = parse_args()
    while True:
        started = time.time()
        run_cycle(args)
        elapsed = round(time.time() - started, 1)
        print(f"scheduler cycle completed in {elapsed}s", flush=True)
        if args.once:
            return 0
        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
