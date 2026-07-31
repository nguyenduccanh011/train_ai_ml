"""Round-trip + fail-loud tests for the serving model bundle format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from stock_ml.src.serving.bundle import load_bundle, write_bundle  # noqa: E402


class _ToyModel:
    """Minimal picklable stand-in for a fitted regressor."""

    def __init__(self, w: float):
        self.w = w

    def predict(self, X):
        return np.asarray(X, dtype=float).sum(axis=1) * self.w


def _make(tmp_path) -> Path:
    return write_bundle(
        tmp_path / "bundle",
        models={"entry": _ToyModel(1.5), "exit": _ToyModel(-0.3)},
        config={"strategy": "regression_dual_ml_recombine", "signal_threshold": 5.0},
        feature_spec={"entry_features": "leading_v4", "feature_cols": ["atr", "adx"]},
        manifest_extra={
            "template_id": 958,
            "cutoff_date": "2026-01-01",
            "universe": ["VNM", "FPT"],
        },
    )


def test_roundtrip_preserves_payload(tmp_path):
    path = _make(tmp_path)
    loaded = load_bundle(path)

    assert sorted(loaded.models) == ["entry", "exit"]
    assert loaded.config["signal_threshold"] == 5.0
    assert loaded.feature_spec["feature_cols"] == ["atr", "adx"]
    assert loaded.manifest["template_id"] == 958
    # model is functional after the joblib round-trip
    pred = loaded.models["entry"].predict([[1.0, 1.0]])
    assert np.isclose(pred[0], 3.0)


def test_missing_entry_model_rejected(tmp_path):
    with pytest.raises(ValueError, match="entry"):
        write_bundle(
            tmp_path / "b",
            models={"exit": _ToyModel(1.0)},
            config={},
            feature_spec={},
            manifest_extra={},
        )


def test_refuses_overwrite(tmp_path):
    _make(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        _make(tmp_path)


def test_checksum_mismatch_fails_loud(tmp_path):
    path = _make(tmp_path)
    # tamper with config after writing
    cfg = path / "config.json"
    cfg.write_text(json.dumps({"strategy": "TAMPERED"}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bundle(path)


def test_missing_payload_file_fails_loud(tmp_path):
    path = _make(tmp_path)
    (path / "models" / "exit.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="missing payload"):
        load_bundle(path)


def test_format_version_major_mismatch_rejected(tmp_path):
    path = _make(tmp_path)
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    man["format_version"] = "99.0"
    (path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_bundle(path)


def test_lightgbm_major_skew_rejected(tmp_path):
    path = _make(tmp_path)
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    man.setdefault("lib_versions", {})["lightgbm"] = "1.0.0"  # fake old major
    (path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    # manifest edited -> its own checksum isn't tracked, but lib gate should trip
    # only if runtime has lightgbm; skip the assertion when lightgbm is absent.
    try:
        from importlib.metadata import version

        version("lightgbm")
    except Exception:
        pytest.skip("lightgbm not installed; lib-skew gate not exercised")
    with pytest.raises(ValueError, match="lightgbm major version skew"):
        load_bundle(path)
