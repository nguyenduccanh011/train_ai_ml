from __future__ import annotations

import numpy as np
import pytest


def _make_binary_data(n: int = 120, n_features: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(int)
    return X, y


class TestNullExitModel:
    def test_fit_predict(self):
        from src.components.exit_models.no_exit_model import NullExitModel

        X, y = _make_binary_data()
        m = NullExitModel()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds) == {0}
        assert m.predict_proba(X) is None


class TestMLExitModel:
    def test_lightgbm_fit_predict(self):
        pytest.importorskip("lightgbm")
        from src.components.exit_models.ml_exit import MLExitModel

        X, y = _make_binary_data()
        m = MLExitModel("lightgbm", n_estimators=10)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

    def test_lightgbm_predict_proba(self):
        pytest.importorskip("lightgbm")
        from src.components.exit_models.ml_exit import MLExitModel

        X, y = _make_binary_data()
        m = MLExitModel("lightgbm", n_estimators=10)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba is not None
        assert proba.shape[0] == len(X)


class TestExitModelRegistry:
    def test_get_exit_model(self):
        pytest.importorskip("lightgbm")
        from src.components.exit_models.registry import get_exit_model

        assert get_exit_model("null", device="cpu").name == "null"
        assert get_exit_model("lightgbm", device="cpu", n_estimators=10).name == "lightgbm"

    def test_list_exit_models(self):
        from src.components.exit_models.registry import list_exit_models

        models = list_exit_models()
        assert "null" in models
        assert "lightgbm" in models
