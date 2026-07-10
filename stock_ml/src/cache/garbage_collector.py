"""Cache garbage collector for the model lifecycle.

Attributes cache files to runs via the cache_keys persisted in each run's
predictions_meta.json (written by the pipeline from M2 onward). Any cache file
whose key is not referenced by a current run is an orphan.

Orphans are quarantined (moved to results/cache/_trash/<timestamp>/) rather than
deleted, so they can be recovered for `older_than_days` before purge.

Cache layout handled:
  results/cache/features/<feature_set>/<key>.{parquet,pkl,json}
  results/cache/predictions/<key>.pkl

Legacy runs (pre-M2) have empty cache_keys, so their feature caches cannot be
attributed and will be quarantined on first sweep — this is intentional
(see project decision 2026-05-22).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

QUARANTINE_DIRNAME = "_trash"


@dataclass
class GCReport:
    referenced_features: set[str] = field(default_factory=set)
    referenced_predictions: set[str] = field(default_factory=set)
    orphan_files: list[Path] = field(default_factory=list)
    orphan_bytes: int = 0
    quarantined: list[Path] = field(default_factory=list)
    purged: list[Path] = field(default_factory=list)
    dry_run: bool = True

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_files)

    def summary(self) -> str:
        mb = self.orphan_bytes / (1024 * 1024)
        mode = "DRY-RUN" if self.dry_run else "APPLIED"
        lines = [
            f"[cache-gc] mode={mode}",
            f"  referenced: {len(self.referenced_features)} feature keys, "
            f"{len(self.referenced_predictions)} prediction keys",
            f"  orphans: {self.orphan_count} files, {mb:.1f} MB",
        ]
        if self.quarantined:
            lines.append(f"  quarantined: {len(self.quarantined)} files")
        if self.purged:
            lines.append(f"  purged from trash: {len(self.purged)} files")
        return "\n".join(lines)


def gather_referenced_keys(experiments_dir: str | Path) -> tuple[set[str], set[str]]:
    """Scan every run's predictions_meta.json and collect non-empty cache keys."""
    features: set[str] = set()
    predictions: set[str] = set()
    exp_dir = Path(experiments_dir)
    if not exp_dir.is_dir():
        return features, predictions
    for meta_path in exp_dir.glob("*/*/predictions_meta.json"):
        keys = _read_cache_keys(meta_path)
        if keys.get("features"):
            features.add(keys["features"])
        if keys.get("predictions"):
            predictions.add(keys["predictions"])
    # Also handle one-level layout (matrix_name/predictions_meta.json)
    for meta_path in exp_dir.glob("*/predictions_meta.json"):
        keys = _read_cache_keys(meta_path)
        if keys.get("features"):
            features.add(keys["features"])
        if keys.get("predictions"):
            predictions.add(keys["predictions"])
    return features, predictions


def _read_cache_keys(meta_path: Path) -> dict[str, str]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = meta.get("cache_keys") or {}
    return {
        "features": str(raw.get("features", "")),
        "predictions": str(raw.get("predictions", "")),
    }


def find_feature_cache_files(features_root: str | Path) -> dict[str, list[Path]]:
    """Map feature cache key -> list of its files (parquet/pkl/json). Key is filename stem."""
    root = Path(features_root)
    out: dict[str, list[Path]] = {}
    if not root.is_dir():
        return out
    for path in root.glob("*/*"):
        if path.is_file() and path.suffix in {".parquet", ".pkl", ".json"}:
            out.setdefault(path.stem, []).append(path)
    return out


def find_prediction_cache_files(predictions_root: str | Path) -> dict[str, list[Path]]:
    """Map prediction cache key -> list of its files (.pkl). Key is filename stem."""
    root = Path(predictions_root)
    out: dict[str, list[Path]] = {}
    if not root.is_dir():
        return out
    for path in root.glob("*.pkl"):
        if path.is_file():
            out.setdefault(path.stem, []).append(path)
    return out


def collect_orphans(grouped: dict[str, list[Path]], referenced: set[str]) -> list[Path]:
    orphans: list[Path] = []
    for key, files in grouped.items():
        if key not in referenced:
            orphans.extend(files)
    return orphans


def quarantine(orphans: list[Path], cache_root: str | Path) -> list[Path]:
    """Move orphan files into results/cache/_trash/<timestamp>/, preserving subpath."""
    root = Path(cache_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_dir = root / QUARANTINE_DIRNAME / stamp
    moved: list[Path] = []
    for src in orphans:
        try:
            rel = src.relative_to(root)
        except ValueError:
            rel = Path(src.name)
        dest = trash_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved.append(dest)
    return moved


def purge_trash(cache_root: str | Path, older_than_days: float) -> list[Path]:
    """Delete quarantine batches older than older_than_days. Returns removed batch dirs."""
    trash_root = Path(cache_root) / QUARANTINE_DIRNAME
    if not trash_root.is_dir():
        return []
    cutoff = datetime.now().timestamp() - older_than_days * 86400
    removed: list[Path] = []
    for batch in trash_root.iterdir():
        if batch.is_dir() and batch.stat().st_mtime < cutoff:
            shutil.rmtree(batch, ignore_errors=True)
            removed.append(batch)
    return removed


def sweep(
    results_dir: str | Path,
    *,
    dry_run: bool = True,
    purge_older_than_days: float | None = None,
) -> GCReport:
    """Run a full GC pass: attribute caches, find orphans, optionally quarantine/purge."""
    results = Path(results_dir)
    cache_root = results / "cache"
    experiments_dir = results / "experiments"

    ref_features, ref_predictions = gather_referenced_keys(experiments_dir)

    feature_files = find_feature_cache_files(cache_root / "features")
    prediction_files = find_prediction_cache_files(cache_root / "predictions")

    orphans = collect_orphans(feature_files, ref_features)
    orphans += collect_orphans(prediction_files, ref_predictions)

    orphan_bytes = sum(p.stat().st_size for p in orphans if p.exists())

    report = GCReport(
        referenced_features=ref_features,
        referenced_predictions=ref_predictions,
        orphan_files=orphans,
        orphan_bytes=orphan_bytes,
        dry_run=dry_run,
    )

    if not dry_run and orphans:
        report.quarantined = quarantine(orphans, cache_root)

    if purge_older_than_days is not None:
        report.purged = purge_trash(cache_root, purge_older_than_days)

    return report
