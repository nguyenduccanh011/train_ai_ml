"""Batch experiment runner — processes YAML configs and runs experiments sequentially or in parallel.

Usage:
    python stock_ml/scripts/run_experiments.py \
        --pending config/experiments/pending \
        --done config/experiments/done \
        --failed config/experiments/failed \
        --data-root /path/to/data \
        --symbols AAA,SSI,VNM \
        --out results/ \
        [--parallel 4]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.data.loader import DataLoader
from src.pipeline.experiment import ExperimentConfig, run_experiment
from src.pipeline.multi_seed import run_experiment_multi_seed
from src.leaderboard.aggregator import rebuild_leaderboard


def try_acquire_lock(yaml_path: Path, timeout_sec: float = 5.0) -> bool:
    """Phase 1b.10: Try to acquire per-file lock with timeout.

    Uses atomic rename to avoid race conditions. If lock file already exists,
    another worker is processing this YAML — skip it.

    Args:
        yaml_path: path to YAML file
        timeout_sec: timeout in seconds (not strictly enforced; just skip if locked)

    Returns:
        True if lock acquired, False if already locked
    """
    lock_path = yaml_path.with_suffix(yaml_path.suffix + ".lock")
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def release_lock(yaml_path: Path) -> None:
    """Release per-file lock."""
    lock_path = yaml_path.with_suffix(yaml_path.suffix + ".lock")
    if lock_path.exists():
        lock_path.unlink()


def run_one_experiment(
    yaml_path: Path,
    done_dir: Path,
    failed_dir: Path,
    data_root: str,
    symbols: list[str],
    out_dir: str,
) -> tuple[Path, bool, str]:
    """Run a single experiment and move YAML to done/failed.

    Phase 1b.10: Atomic YAML queue with per-file locking.

    Args:
        yaml_path: path to YAML config file
        done_dir: directory for successful runs
        failed_dir: directory for failed runs
        data_root: OHLCV data root directory
        symbols: list of symbols
        out_dir: output directory for results

    Returns:
        (yaml_path, success: bool, message: str)
    """
    if not try_acquire_lock(yaml_path):
        msg = "Skipped — already being processed by another worker"
        print(f"\n[skipped] {yaml_path.name}: {msg}")
        return yaml_path, False, msg

    try:
        print(f"\n{'=' * 60}")
        print(f"Running: {yaml_path.name}")
        print(f"{'=' * 60}")

        cfg = ExperimentConfig.from_yaml(yaml_path)

        # Phase 1.5.5: Use multi-seed runner if validation.n_seeds > 1
        n_seeds = 1
        if cfg.validation and "n_seeds" in cfg.validation:
            n_seeds = cfg.validation["n_seeds"]

        if n_seeds > 1:
            print(f"  [multi-seed] Running {n_seeds} seeds for model stability analysis")
            summary = run_experiment_multi_seed(cfg, data_root, symbols, out_dir, n_seeds=n_seeds)
        else:
            summary = run_experiment(cfg, data_root, symbols, out_dir)

        done_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(yaml_path), str(done_dir / yaml_path.name))

        msg = f"OK — {summary.get('n_trades', 0)} trades, {summary.get('aggregate', {}).get('total_pnl', 0):.2%} PnL"
        print(f"\n[success] {yaml_path.name}: {msg}")
        return yaml_path, True, msg

    except Exception as e:
        failed_dir.mkdir(parents=True, exist_ok=True)
        error_log_path = failed_dir / f"{yaml_path.stem}_error.log"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"Error in {yaml_path.name}:\n")
            f.write(traceback.format_exc())

        shutil.move(str(yaml_path), str(failed_dir / yaml_path.name))
        msg = f"FAILED — see {error_log_path.name}"
        print(f"\n[failed] {yaml_path.name}: {msg}")
        return yaml_path, False, str(e)

    finally:
        release_lock(yaml_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments from YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pending",
        required=True,
        type=Path,
        help="Directory containing pending YAML configs",
    )
    parser.add_argument(
        "--done",
        required=True,
        type=Path,
        help="Directory to move successful YAMLs",
    )
    parser.add_argument(
        "--failed",
        required=True,
        type=Path,
        help="Directory to move failed YAMLs + error logs",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=str,
        help="Path to OHLCV data directory",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        type=str,
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for experiment results",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--rebuild-leaderboard",
        type=bool,
        default=True,
        help="Rebuild leaderboard.json after all runs complete (default: True)",
    )

    args = parser.parse_args()

    pending_dir = Path(args.pending)
    done_dir = Path(args.done)
    failed_dir = Path(args.failed)
    data_root = args.data_root

    # Support "ALL" keyword to load all available symbols
    if args.symbols.strip().upper() == "ALL":
        loader = DataLoader(data_root)
        symbols = loader.list_symbols()
        print(f"[symbols] Loading all {len(symbols)} available symbols")
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    out_dir = Path(args.out)

    if not pending_dir.exists():
        print(f"Error: pending directory not found: {pending_dir}")
        return 1

    yamls = sorted(pending_dir.glob("*.yaml"))
    if not yamls:
        print(f"No YAML files found in {pending_dir}")
        return 0

    print(f"Found {len(yamls)} experiment configs to run")
    print(f"Symbols: {symbols}")
    print(f"Parallel workers: {args.parallel}")

    results = []

    if args.parallel == 1:
        for yaml_path in yamls:
            result = run_one_experiment(
                yaml_path,
                done_dir,
                failed_dir,
                data_root,
                symbols,
                str(out_dir),
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    run_one_experiment,
                    yaml_path,
                    done_dir,
                    failed_dir,
                    data_root,
                    symbols,
                    str(out_dir),
                ): yaml_path
                for yaml_path in yamls
            }
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Worker error: {e}")

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)

    print(f"\n{'=' * 60}")
    print(f"Summary: {n_ok} passed, {n_fail} failed")
    print(f"Results written to: {out_dir}")

    if args.rebuild_leaderboard and n_ok > 0:
        print("Rebuilding leaderboard...")
        experiments_dir = out_dir / "experiments"
        leaderboard_dir = out_dir / "leaderboard"
        rebuild_leaderboard(str(experiments_dir), str(leaderboard_dir))
        print(f"Leaderboard rebuilt at: {leaderboard_dir}")

    print(f"{'=' * 60}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
