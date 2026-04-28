"""MatrixExpander — expand a matrix YAML into a list of ExperimentConfigs.

Matrix YAML format:
    name_prefix: q3_2026
    axes:
      features: [leading_v2, leading_v4]
      model_type: [lightgbm, xgboost]
      strategy: [v22]
    base:                  # merged into every experiment
      split:
        first_test_year: 2023

Each axis combination becomes one ExperimentConfig. Experiments sharing the
same feature_set + target will reuse the same prediction cache (key-based).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.config import ExperimentConfig


def expand_matrix(matrix_path: str | Path) -> list[ExperimentConfig]:
    """Load a matrix YAML and return one ExperimentConfig per axis combination."""
    with open(matrix_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    name_prefix = spec.get("name_prefix", "exp")
    axes: dict[str, list[Any]] = spec.get("axes", {})
    base: dict[str, Any] = spec.get("base", {})

    if not axes:
        raise ValueError(f"Matrix YAML {matrix_path} has no 'axes' section.")

    axis_names = list(axes.keys())
    axis_values = [axes[k] for k in axis_names]

    configs: list[ExperimentConfig] = []
    for combo in itertools.product(*axis_values):
        overrides: dict[str, Any] = dict(zip(axis_names, combo))
        cfg_data = _deep_merge(base, _axes_to_cfg_data(overrides))
        name_parts = [f"{k}-{v}" for k, v in overrides.items()]
        cfg_data.setdefault("name", f"{name_prefix}_{'-'.join(name_parts)}")
        cfg_data.setdefault("strategy", overrides.get("strategy", name_prefix))

        configs.append(ExperimentConfig.model_validate(cfg_data))

    return configs


def _axes_to_cfg_data(overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert flat axis overrides into nested ExperimentConfig structure."""
    data: dict[str, Any] = {}
    components: dict[str, Any] = {}

    for key, value in overrides.items():
        if key == "features":
            components["features"] = value
        elif key == "model_type":
            components.setdefault("entry_model", {})["type"] = value
        elif key == "strategy":
            data["strategy"] = value
        elif key == "target_type":
            components.setdefault("target", {})["type"] = value
        else:
            data[key] = value

    if components:
        data["components"] = components
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
