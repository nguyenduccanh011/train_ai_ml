"""
Quarantine old leaky experiment leaderboard entries.

This script does not delete model artifacts. It renames each affected
``ranking_row.json`` so ``rebuild_leaderboard`` will no longer discover the
old run. The rename is reversible because the original file remains in the
same run directory with a timestamped suffix.

Criteria:
- keep runs whose resolved config has split.gap_days >= --min-gap
- quarantine runs with gap_days below --min-gap, missing config, or parse errors

Usage:
    python cleanup_old_leaky_artifacts.py --dry-run
    python cleanup_old_leaky_artifacts.py --execute
    python cleanup_old_leaky_artifacts.py --dry-run --source artifacts
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.leaderboard import rebuild_leaderboard

LEADERBOARD_FILES = (
    "leaderboard.csv",
    "leaderboard.json",
    "index.json",
    "summary.json",
    "schema.json",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _classify_run(run_dir: Path, min_gap: int) -> tuple[bool, str, int | None]:
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.exists():
        return False, "missing_config", None
    try:
        cfg = _read_yaml(config_path)
    except Exception as exc:
        return False, f"config_parse_error:{exc}", None
    gap = (cfg.get("split") or {}).get("gap_days")
    try:
        gap_value = int(gap)
    except (TypeError, ValueError):
        return False, f"invalid_gap_days:{gap}", None
    if gap_value < min_gap:
        return False, f"gap_days_{gap_value}_below_{min_gap}", gap_value
    return True, "fixed_gap_days", gap_value


def _backup_leaderboard(output_dir: Path, stamp: str) -> list[str]:
    backup_dir = output_dir / f"backup_before_leakage_cleanup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in LEADERBOARD_FILES:
        src = output_dir / name
        if src.exists():
            dst = backup_dir / name
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(ROOT)))
    return copied


def _item_from_run_dir(run_dir: Path, min_gap: int, leaderboard_lookup: dict[tuple[str, str], Any]):
    ranking_path = run_dir / "ranking_row.json"
    bundle = run_dir.parent.name
    run_name = run_dir.name
    is_fixed, reason, gap = _classify_run(run_dir, min_gap)
    ranking = {}
    metrics = {}
    if ranking_path.exists():
        try:
            ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        except Exception:
            ranking = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    lb_row = leaderboard_lookup.get((bundle, run_name))
    item = {
        "bundle": bundle,
        "run_name": run_name,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "ranking_row": str(ranking_path.relative_to(ROOT)),
        "gap_days": gap,
        "reason": reason,
        "feature_set": (
            getattr(lb_row, "feature_set", None)
            if lb_row is not None
            else ranking.get("feature_set")
        ),
        "wr": getattr(lb_row, "wr", None) if lb_row is not None else metrics.get("wr"),
        "pf": getattr(lb_row, "pf", None) if lb_row is not None else metrics.get("pf"),
        "composite_score": (
            getattr(lb_row, "composite_score", None)
            if lb_row is not None
            else metrics.get("composite_score")
        ),
        "ranking_exists": ranking_path.exists(),
    }
    return is_fixed, item


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine old leaky leaderboard artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument(
        "--execute", action="store_true", help="Rename leaky ranking rows and rebuild"
    )
    parser.add_argument("--min-gap", type=int, default=25, help="Minimum safe split.gap_days")
    parser.add_argument(
        "--source",
        choices=("artifacts", "leaderboard"),
        default="artifacts",
        help="Scan all active ranking_row.json files, or only current leaderboard rows",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest output path. Defaults to results/leakage_check/cleanup_old_leaky_<timestamp>.json",
    )
    args = parser.parse_args()

    if args.dry_run == args.execute:
        print("ERROR: choose exactly one of --dry-run or --execute")
        return 1

    leaderboard_path = ROOT / "results/leaderboard/leaderboard.csv"
    experiments_dir = ROOT / "results/experiments"
    output_dir = ROOT / "results/leaderboard"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else ROOT / "results/leakage_check" / f"cleanup_old_leaky_{stamp}.json"
    )
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    df = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
    leaderboard_lookup = {
        (str(row["bundle"]), str(row["run_name"])): row for _, row in df.iterrows()
    }
    keep: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    missing_ranking = 0

    if args.source == "artifacts":
        run_dirs = sorted(path.parent for path in experiments_dir.glob("**/ranking_row.json"))
    else:
        run_dirs = [
            experiments_dir / str(row["bundle"]) / str(row["run_name"]) for _, row in df.iterrows()
        ]

    for run_dir in run_dirs:
        is_fixed, item = _item_from_run_dir(run_dir, args.min_gap, leaderboard_lookup)
        if is_fixed:
            keep.append(item)
        else:
            quarantine.append(item)
            if not item["ranking_exists"]:
                missing_ranking += 1

    print(f"Source: {args.source}")
    print(f"Current leaderboard rows: {len(df)}")
    print(f"Scanned active run dirs: {len(run_dirs)}")
    print(f"Keep fixed rows: {len(keep)}")
    print(f"Quarantine old/leaky rows: {len(quarantine)}")
    print(f"Missing ranking_row among quarantine rows: {missing_ranking}")
    if quarantine:
        print("\nQuarantine reasons:")
        reason_counts = pd.Series([item["reason"] for item in quarantine]).value_counts()
        print(reason_counts.to_string())
        print("\nFirst 10 quarantine candidates:")
        preview = pd.DataFrame(quarantine).head(10)
        print(
            preview[["bundle", "run_name", "gap_days", "reason", "wr", "pf"]].to_string(index=False)
        )

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "min_gap": args.min_gap,
        "leaderboard_path": str(leaderboard_path.relative_to(ROOT)),
        "total": len(df),
        "keep_count": len(keep),
        "quarantine_count": len(quarantine),
        "missing_ranking_count": missing_ranking,
        "keep": keep,
        "quarantine": quarantine,
        "renamed": [],
        "leaderboard_backup": [],
        "rebuilt_rows": None,
    }

    if args.execute:
        manifest["leaderboard_backup"] = _backup_leaderboard(output_dir, stamp)
        suffix = f".leaky_quarantine_{stamp}.json"
        for item in quarantine:
            ranking_path = ROOT / item["ranking_row"]
            if not ranking_path.exists():
                continue
            quarantined_path = ranking_path.with_name("ranking_row" + suffix)
            if quarantined_path.exists():
                raise FileExistsError(quarantined_path)
            ranking_path.rename(quarantined_path)
            manifest["renamed"].append(
                {
                    "from": str(ranking_path.relative_to(ROOT)),
                    "to": str(quarantined_path.relative_to(ROOT)),
                }
            )
        rows = rebuild_leaderboard(experiments_dir, output_dir)
        manifest["rebuilt_rows"] = len(rows)
        print(f"\nRenamed ranking_row files: {len(manifest['renamed'])}")
        print(f"Rebuilt leaderboard rows: {len(rows)}")
        print("Leaderboard backups:")
        for path in manifest["leaderboard_backup"]:
            print(f"  {path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
