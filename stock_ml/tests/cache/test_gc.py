"""Tests for the cache garbage collector."""

from __future__ import annotations

import json
from pathlib import Path

from src.cache.garbage_collector import (
    QUARANTINE_DIRNAME,
    gather_referenced_keys,
    purge_trash,
    sweep,
)


def _make_run(experiments_dir: Path, name: str, feat_key: str, pred_key: str) -> None:
    run_dir = experiments_dir / "bundle" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions_meta.json").write_text(
        json.dumps({"cache_keys": {"features": feat_key, "predictions": pred_key}}),
        encoding="utf-8",
    )


def _make_feature_cache(cache_root: Path, feature_set: str, key: str) -> list[Path]:
    d = cache_root / "features" / feature_set
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in (".parquet", ".json"):
        p = d / f"{key}{ext}"
        p.write_text("x" * 100, encoding="utf-8")
        files.append(p)
    return files


def _make_prediction_cache(cache_root: Path, key: str) -> Path:
    d = cache_root / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.pkl"
    p.write_text("x" * 50, encoding="utf-8")
    return p


def test_gather_referenced_keys(tmp_path: Path) -> None:
    exp = tmp_path / "experiments"
    _make_run(exp, "run1", "feat_a", "pred_a")
    _make_run(exp, "run2", "feat_b", "")
    feats, preds = gather_referenced_keys(exp)
    assert feats == {"feat_a", "feat_b"}
    assert preds == {"pred_a"}


def test_sweep_dry_run_finds_orphans_but_keeps_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    exp = results / "experiments"
    cache = results / "cache"
    _make_run(exp, "run1", "feat_live", "pred_live")
    _make_feature_cache(cache, "leading", "feat_live")  # referenced
    orphan_files = _make_feature_cache(cache, "leading", "feat_orphan")  # orphan
    _make_prediction_cache(cache, "pred_live")  # referenced
    orphan_pred = _make_prediction_cache(cache, "pred_orphan")  # orphan

    report = sweep(results, dry_run=True)

    assert report.dry_run is True
    assert report.orphan_count == 3  # 2 feature files + 1 prediction file
    # Files still on disk
    assert all(p.exists() for p in orphan_files)
    assert orphan_pred.exists()
    assert not report.quarantined


def test_sweep_apply_quarantines_orphans(tmp_path: Path) -> None:
    results = tmp_path / "results"
    exp = results / "experiments"
    cache = results / "cache"
    _make_run(exp, "run1", "feat_live", "pred_live")
    live_files = _make_feature_cache(cache, "leading", "feat_live")
    orphan_files = _make_feature_cache(cache, "leading", "feat_orphan")

    report = sweep(results, dry_run=False)

    assert report.dry_run is False
    # Orphans moved out of original location
    assert all(not p.exists() for p in orphan_files)
    # Live files untouched
    assert all(p.exists() for p in live_files)
    # Quarantine dir created
    trash = cache / QUARANTINE_DIRNAME
    assert trash.is_dir()
    assert len(report.quarantined) == len(orphan_files)
    assert all(p.exists() for p in report.quarantined)


def test_purge_trash_removes_old_batches(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    batch = cache / QUARANTINE_DIRNAME / "20200101_000000"
    batch.mkdir(parents=True)
    (batch / "old.parquet").write_text("x", encoding="utf-8")
    # Set mtime far in the past
    import os

    old_time = 0  # 1970
    os.utime(batch, (old_time, old_time))

    removed = purge_trash(cache, older_than_days=1)
    assert batch in removed
    assert not batch.exists()


def test_purge_trash_keeps_recent_batches(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    batch = cache / QUARANTINE_DIRNAME / "recent"
    batch.mkdir(parents=True)
    (batch / "x.parquet").write_text("x", encoding="utf-8")
    removed = purge_trash(cache, older_than_days=1)
    assert removed == []
    assert batch.exists()
