"""load_bundle compatibility gates (ENGINE_UPGRADE §11.5.2).

Two independent gates:
  (A) strict_libs — a MAJOR-version skew of a pickle-critical lib (lightgbm/scikit-learn/numpy) raises,
      because it can silently corrupt an unpickled estimator.
  (B) ENGINE_WHEEL_PIN — the measuring-stick refuse: an exact ``stock_ml_core`` mismatch raises, so a
      serving host never silently changes engine version under stored curves. Opt-in; off by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_ml.src.serving.bundle import _MANIFEST_NAME, load_bundle, write_bundle


def _make_bundle(tmp: Path) -> Path:
    """A minimal, valid bundle (the 'entry' model is any picklable object — gates don't predict)."""
    return write_bundle(
        tmp / "b",
        models={"entry": {"dummy": 1}},
        config={"name": "t", "strategy": "s"},
        feature_spec={"entry_feat_cols": []},
        manifest_extra={"template_name": "t"},
    )


def _patch_manifest(bundle: Path, **lib_overrides: str) -> None:
    m = json.loads((bundle / _MANIFEST_NAME).read_text(encoding="utf-8"))
    m["lib_versions"] = {**m.get("lib_versions", {}), **lib_overrides}
    (bundle / _MANIFEST_NAME).write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def test_pickle_critical_major_skew_raises(tmp_path: Path) -> None:
    b = _make_bundle(tmp_path)
    # Force a scikit-learn MAJOR skew vs whatever the runtime has.
    _patch_manifest(b, **{"scikit-learn": "99.0.0"})
    with pytest.raises(ValueError, match="scikit-learn major version skew"):
        load_bundle(b)


def test_pickle_skew_ignored_when_strict_libs_false(tmp_path: Path) -> None:
    b = _make_bundle(tmp_path)
    _patch_manifest(b, **{"numpy": "99.0.0"})
    load_bundle(b, strict_libs=False)  # must not raise


def test_engine_pin_off_by_default(tmp_path: Path) -> None:
    b = _make_bundle(tmp_path)
    _patch_manifest(b, stock_ml_core="0.0.1-ancient")  # mismatch, but pin is OFF
    load_bundle(b)  # must not raise


def test_engine_pin_refuses_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    b = _make_bundle(tmp_path)
    _patch_manifest(b, stock_ml_core="0.0.1-ancient")
    monkeypatch.setenv("ENGINE_WHEEL_PIN", "1")
    with pytest.raises(ValueError, match="engine-wheel pin"):
        load_bundle(b)


def test_fold_models_roundtrip(tmp_path: Path) -> None:
    """§11.9: write_bundle ships per-fold models under fold_models/<year>/, load_bundle returns them
    keyed by year; a legacy bundle (no fold_models) loads with fold_models=None."""
    b = write_bundle(
        tmp_path / "fm",
        models={"entry": {"serve": 1}},  # last/serve fold
        config={"name": "t", "strategy": "s"},
        feature_spec={"entry_feat_cols": []},
        manifest_extra={"template_name": "t"},
        fold_models={2024: {"entry": {"y": 2024}}, 2025: {"entry": {"y": 2025}}},
    )
    loaded = load_bundle(b)
    assert loaded.manifest["fold_years"] == [2024, 2025]
    assert loaded.fold_models is not None
    assert loaded.fold_models[2024]["entry"] == {"y": 2024}
    assert loaded.fold_models[2025]["entry"] == {"y": 2025}
    # legacy bundle has no fold_models
    assert load_bundle(_make_bundle(tmp_path / "legacy")).fold_models is None


def test_engine_pin_refuses_when_wheel_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle with no baked stock_ml_core can't be attested → refuse under the pin (re-export it)."""
    b = _make_bundle(tmp_path)
    m = json.loads((b / _MANIFEST_NAME).read_text(encoding="utf-8"))
    m["lib_versions"] = {k: v for k, v in m.get("lib_versions", {}).items() if k != "stock_ml_core"}
    (b / _MANIFEST_NAME).write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv("ENGINE_WHEEL_PIN", "1")
    with pytest.raises(ValueError, match="engine-wheel pin"):
        load_bundle(b)
