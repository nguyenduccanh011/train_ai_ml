"""Smoke tests for Phase 1.4 entry model wrappers.

Tests: fit + predict + predict_proba with small synthetic data.
No equivalence check against legacy (legacy registry is flat builders, not wrappers).
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_data(n: int = 200, n_features: int = 20, n_classes: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    label_map = {0: -1, 1: 0, 2: 1}
    y = np.array([label_map[v] for v in y])
    return X, y


# ── LightGBM ──────────────────────────────────────────────────────────────────


class TestLightGBMEntryModel:
    def test_fit_predict(self):
        pytest.importorskip("lightgbm")
        from src.components.models.lightgbm_classifier import LightGBMEntryModel

        X, y = _make_data()
        m = LightGBMEntryModel(n_estimators=10)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({-1, 0, 1})

    def test_predict_proba(self):
        pytest.importorskip("lightgbm")
        from src.components.models.lightgbm_classifier import LightGBMEntryModel

        X, y = _make_data()
        m = LightGBMEntryModel(n_estimators=10)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes(self):
        pytest.importorskip("lightgbm")
        from src.components.models.lightgbm_classifier import LightGBMEntryModel

        X, y = _make_data()
        m = LightGBMEntryModel(n_estimators=10)
        m.fit(X, y)
        assert sorted(m.classes_) == [-1, 0, 1]

    def test_name(self):
        pytest.importorskip("lightgbm")
        from src.components.models.lightgbm_classifier import LightGBMEntryModel

        assert LightGBMEntryModel.name == "lightgbm"


# ── XGBoost ───────────────────────────────────────────────────────────────────


class TestXGBoostEntryModel:
    def test_fit_predict(self):
        pytest.importorskip("xgboost")
        from src.components.models.xgboost_classifier import XGBoostEntryModel

        X, y = _make_data()
        m = XGBoostEntryModel(n_estimators=10)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba(self):
        pytest.importorskip("xgboost")
        from src.components.models.xgboost_classifier import XGBoostEntryModel

        X, y = _make_data()
        m = XGBoostEntryModel(n_estimators=10)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba is not None
        assert proba.shape[1] == 3

    def test_name(self):
        pytest.importorskip("xgboost")
        from src.components.models.xgboost_classifier import XGBoostEntryModel

        assert XGBoostEntryModel.name == "xgboost"


# ── CatBoost ──────────────────────────────────────────────────────────────────


class TestCatBoostEntryModel:
    def test_fit_predict(self):
        pytest.importorskip("catboost")
        from src.components.models.catboost_classifier import CatBoostEntryModel

        X, y = _make_data()
        m = CatBoostEntryModel(iterations=10)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba(self):
        pytest.importorskip("catboost")
        from src.components.models.catboost_classifier import CatBoostEntryModel

        X, y = _make_data()
        m = CatBoostEntryModel(iterations=10)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba is not None
        assert proba.shape[1] == 3

    def test_name(self):
        pytest.importorskip("catboost")
        from src.components.models.catboost_classifier import CatBoostEntryModel

        assert CatBoostEntryModel.name == "catboost"


# ── RandomForest ──────────────────────────────────────────────────────────────


class TestRandomForestEntryModel:
    def test_fit_predict(self):
        from src.components.models.random_forest import RandomForestEntryModel

        X, y = _make_data()
        m = RandomForestEntryModel(n_estimators=10)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({-1, 0, 1})

    def test_predict_proba(self):
        from src.components.models.random_forest import RandomForestEntryModel

        X, y = _make_data()
        m = RandomForestEntryModel(n_estimators=10)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(X), 3)

    def test_classes(self):
        from src.components.models.random_forest import RandomForestEntryModel

        X, y = _make_data()
        m = RandomForestEntryModel(n_estimators=10)
        m.fit(X, y)
        assert sorted(m.classes_) == [-1, 0, 1]

    def test_name(self):
        from src.components.models.random_forest import RandomForestEntryModel

        assert RandomForestEntryModel.name == "random_forest"


# ── Ensemble ──────────────────────────────────────────────────────────────────


class TestEnsembleEntryModel:
    def test_soft_voting(self):
        pytest.importorskip("lightgbm")
        from src.components.models.ensemble import EnsembleEntryModel
        from src.components.models.lightgbm_classifier import LightGBMEntryModel
        from src.components.models.random_forest import RandomForestEntryModel

        X, y = _make_data()
        m = EnsembleEntryModel(
            models=[LightGBMEntryModel(n_estimators=10), RandomForestEntryModel(n_estimators=10)],
            voting="soft",
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_hard_voting(self):
        from src.components.models.ensemble import EnsembleEntryModel
        from src.components.models.random_forest import RandomForestEntryModel

        X, y = _make_data()
        m = EnsembleEntryModel(
            models=[RandomForestEntryModel(n_estimators=5), RandomForestEntryModel(n_estimators=5)],
            voting="hard",
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_empty_models_raises(self):
        from src.components.models.ensemble import EnsembleEntryModel

        with pytest.raises(ValueError):
            EnsembleEntryModel(models=[])

    def test_name(self):
        from src.components.models.ensemble import EnsembleEntryModel

        assert EnsembleEntryModel.name == "ensemble"


# ── Registry ──────────────────────────────────────────────────────────────────


class TestModelsRegistry:
    def test_list_models(self):
        from src.components.models.registry import list_models

        names = list_models()
        assert "lightgbm" in names
        assert "random_forest" in names
        assert "gru" in names

    def test_get_model_lightgbm(self):
        pytest.importorskip("lightgbm")
        from src.components.models.registry import get_model

        m = get_model("lightgbm", n_estimators=5)
        assert m.name == "lightgbm"

    def test_get_model_random_forest(self):
        from src.components.models.registry import get_model

        m = get_model("random_forest", n_estimators=5)
        assert m.name == "random_forest"

    def test_get_unknown_raises(self):
        from src.components.models.registry import get_model

        with pytest.raises(KeyError):
            get_model("unknown_model_xyz")
