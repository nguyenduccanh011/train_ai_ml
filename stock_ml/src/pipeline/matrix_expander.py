"""MatrixExpander — expand a matrix YAML into a list of ExperimentConfigs."""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.config import ExperimentConfig


def expand_matrix(matrix_path: str | Path, limit: int | None = None) -> list[ExperimentConfig]:
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
    for i, combo in enumerate(itertools.product(*axis_values)):
        if limit is not None and i >= limit:
            break
        overrides: dict[str, Any] = dict(zip(axis_names, combo))
        cfg_data = _deep_merge(base, _axes_to_cfg_data(overrides))
        cfg_data["name"] = _matrix_name(name_prefix, overrides)
        cfg_data.setdefault("strategy", overrides.get("strategy", name_prefix))

        configs.append(ExperimentConfig.model_validate(cfg_data))

    return configs


def _axes_to_cfg_data(overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert axis overrides into ExperimentConfig structure."""
    data: dict[str, Any] = {}

    for key, value in overrides.items():
        path = _axis_path(key)
        if isinstance(value, dict) and path in {"signals.entry_model", "signals.exit_model"}:
            current = deepcopy(value)
        else:
            current = value
        _set_nested(data, path, current)

    return data


def _axis_path(key: str) -> str:
    aliases = {
        "features": "signals.features",
        "model_type": "signals.entry_model.type",
        "target_type": "signals.target.type",
        "exit_model": "signals.exit_model",
    }
    path = aliases.get(key, key)
    allowed_roots = {
        "name",
        "strategy",
        "runner",
        "components",
        "split",
        "mods",
        "params",
        "fusion",
        "signals",
    }
    if path.split(".", 1)[0] not in allowed_roots:
        raise ValueError(f"Unknown matrix axis '{key}'")
    return path


def _set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override wins on conflicts."""
    result = deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _matrix_name(name_prefix: str, overrides: dict[str, Any]) -> str:
    parts = [f"{_slug(k)}-{_slug(v)}" for k, v in overrides.items()]
    return f"{name_prefix}_{'-'.join(parts)}"


def _slug(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        label = value.get("label") or value.get("type")
        if label:
            return _slug(label)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    if isinstance(value, list):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return str(value).replace(".", "_").replace("/", "_").replace(" ", "_")
