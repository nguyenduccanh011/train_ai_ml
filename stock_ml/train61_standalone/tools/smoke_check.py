from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_registry import DEFAULT_MODEL, MODELS, model_availability, model_requirements
from serve_train61_model import app


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    _check(DEFAULT_MODEL in MODELS, f"Default model not registered: {DEFAULT_MODEL}")
    _check(bool(MODELS), "Model registry is empty")

    for model_id in MODELS:
        requirements = model_requirements(model_id)
        availability = model_availability(model_id)
        print(f"{model_id}: requirements={requirements} available={availability['available']}")
        for missing_path in availability["missing"]:
            print(f"  missing: {missing_path}")
        _check(availability["available"], f"{model_id} has missing required artifacts")

    client = app.test_client()
    for endpoint in ("/api/models", "/api/model-info", "/api/symbols"):
        response = client.get(endpoint)
        _check(response.status_code == 200, f"{endpoint} returned {response.status_code}")
        _check(response.is_json, f"{endpoint} did not return JSON")

    models_payload = client.get("/api/models").get_json()
    _check(isinstance(models_payload, list), "/api/models payload is not a list")
    _check(len(models_payload) == len(MODELS), "/api/models count does not match registry")
    print("smoke_check: ok")


if __name__ == "__main__":
    main()
