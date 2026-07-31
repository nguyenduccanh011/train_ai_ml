"""Model bundle format — write/read a self-contained, versioned model artifact.

A bundle is a directory (optionally tar.gz-packed by the caller) laid out as::

    <bundle>/
    ├── manifest.json      # template_id, run_id, cutoff, universe, versions, checksums
    ├── config.json        # resolved ExperimentConfig (as dict) — drives predict+recombine
    ├── feature_spec.json   # entry/exit feature-set names + resolved feature columns
    └── models/<name>.joblib # fitted model objects (entry, exit, entry2, exit2, ...)

Design principles (see memory ``project_production_bundle_deploy`` and
``feedback_fail_loud_no_silent_fallback``):
- **Fail loud**: a missing file, a checksum mismatch, an incompatible bundle
  format version, or a major-version lightgbm skew raises — never silently
  loads a wrong/partial artifact.
- **Deterministic & verifiable**: every payload file is SHA256'd into the
  manifest at write time and re-checked at load time.
- **Self-describing**: the manifest records the library versions and git sha the
  bundle was produced with, so a serving host can prove it matches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib

from stock_ml.src.serving import BUNDLE_FORMAT_VERSION

# Libraries whose version skew can silently corrupt unpickled models. lightgbm/scikit-learn/numpy carry
# pickled estimator internals, so a MAJOR-version mismatch between export host and serving host is
# rejected outright (§11.5.2, was lightgbm-only). ``stock_ml_core`` is the wheel that GENERATED the
# stored curves — tracked so §11.5.2's exact-wheel gate can refuse a silent measuring-stick change.
_TRACKED_LIBS = ["stock_ml_core", "lightgbm", "scikit-learn", "numpy", "pandas", "joblib", "xgboost"]

# Libs whose MAJOR-version skew breaks pickle compatibility (hard gate under strict_libs).
_PICKLE_CRITICAL_LIBS = ("lightgbm", "scikit-learn", "numpy")


def _engine_pin_required() -> bool:
    """True when production must REFUSE a bundle whose generating wheel (``stock_ml_core``) differs from
    the installed one — the ONLY 'refuse' still alive (§11.5.2): not because changed numbers are wrong,
    but because the measuring stick must never change silently. Default OFF (dev/backtest/tests upgrade
    the wheel freely); set env ``ENGINE_WHEEL_PIN=1`` in the serving container. Escape: rollback the wheel
    or run a logged re-baseline (§11.5.4). Mirrors the opt-in fail-loud of _market_context_required."""
    import os

    return os.environ.get("ENGINE_WHEEL_PIN", "").strip().lower() in ("1", "true", "yes")

_MODELS_SUBDIR = "models"
_FOLD_MODELS_SUBDIR = "fold_models"  # §11.9: per-fold models at fold_models/<year>/<head>.joblib
_MANIFEST_NAME = "manifest.json"
_CONFIG_NAME = "config.json"
_FEATURE_SPEC_NAME = "feature_spec.json"
# Self-sufficient export record (ENGINE_UPGRADE §11.5.1): 227 resolved engine nodes + recombine
# z-window + catalog fingerprint + wheel version + data declaration. Write-only in G1 (load ignores).
_RESOLVED_NAME = "resolved.json"
# Optional: walk-forward predictions (score/exit_score) for bars BEFORE the bundle's
# test window, so the serving z-window is primed by the model that actually owned each
# past segment (not re-predicted by the current model). See project_production_bundle_deploy.
_PRED_HISTORY_NAME = "prediction_history.parquet"


@dataclass(frozen=True)
class LoadedBundle:
    """Result of :func:`load_bundle` — everything a serving host needs to infer."""

    models: dict[str, Any]  # name -> fitted model object (entry, exit, entry2, ...) — the last/serve fold
    config: dict[str, Any]  # resolved ExperimentConfig as a dict
    feature_spec: dict[str, Any]  # feature-set names + resolved feature columns
    manifest: dict[str, Any]  # provenance + checksums
    path: Path  # bundle directory it was loaded from
    prediction_history: Any = None  # DataFrame[symbol,date,score,exit_score] or None (legacy seed)
    # §11.9: per-fold models {test_year: {head: model}} so serving re-runs walk-forward — each bar
    # scored by the model that OWNED its year, history reproduces the backtest by construction (no seed).
    # None on legacy single-model bundles (serving falls back to `models`).
    fold_models: dict[int, dict[str, Any]] | None = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lib_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in _TRACKED_LIBS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def _major(v: str) -> str:
    return v.split(".", 1)[0]


def write_bundle(
    out_dir: str | Path,
    *,
    models: dict[str, Any],
    config: dict[str, Any],
    feature_spec: dict[str, Any],
    manifest_extra: dict[str, Any],
    prediction_history: Any = None,
    resolved_config: dict[str, Any] | None = None,
    fold_models: dict[int, dict[str, Any]] | None = None,
) -> Path:
    """Serialize a model bundle to ``out_dir`` and return its path.

    Args:
        out_dir: target directory (created; must not already contain a bundle).
        models: name -> fitted model object. Must be non-empty; ``entry`` is
                required (every strategy has at least an entry model).
        config: resolved ExperimentConfig as a JSON-able dict (drives the serving
                predict + recombine chain).
        feature_spec: feature-set names + resolved feature columns.
        manifest_extra: caller-supplied provenance to merge into the manifest
                (e.g. template_id, run_id, cutoff_date, universe, seed,
                min_warmup_bars, retrain_schedule, git_sha).
        resolved_config: optional self-sufficient export record (§11.5.1) written as
                ``resolved.json`` and checksummed like config/feature_spec.

    Raises:
        ValueError: if models is empty or lacks an ``entry`` model, or the target
                directory already holds a bundle.
    """
    if not models:
        raise ValueError("write_bundle: models is empty — nothing to serialize")
    if "entry" not in models:
        raise ValueError(
            f"write_bundle: models must include an 'entry' model; got {sorted(models)}"
        )

    out = Path(out_dir)
    if (out / _MANIFEST_NAME).exists():
        raise ValueError(f"write_bundle: a bundle already exists at {out} — refusing to overwrite")
    (out / _MODELS_SUBDIR).mkdir(parents=True, exist_ok=True)

    checksums: dict[str, str] = {}

    # --- model files (the last/serve fold, at models/) ---
    for name, model in models.items():
        rel = f"{_MODELS_SUBDIR}/{name}.joblib"
        dest = out / rel
        joblib.dump(model, dest)
        checksums[rel] = _sha256_file(dest)

    # --- per-fold models (§11.9), at fold_models/<year>/<head>.joblib ---
    fold_years: list[int] = []
    if fold_models:
        for year in sorted(fold_models):
            fold_years.append(int(year))
            for name, model in fold_models[year].items():
                rel = f"{_FOLD_MODELS_SUBDIR}/{year}/{name}.joblib"
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, dest)
                checksums[rel] = _sha256_file(dest)

    # --- config + feature spec (+ optional resolved.json, §11.5.1) ---
    _payloads = [(_CONFIG_NAME, config), (_FEATURE_SPEC_NAME, feature_spec)]
    if resolved_config is not None:
        _payloads.append((_RESOLVED_NAME, resolved_config))
    for rel, payload in _payloads:
        dest = out / rel
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        checksums[rel] = _sha256_file(dest)

    # --- optional prediction history (z-window seed) ---
    if prediction_history is not None:
        dest = out / _PRED_HISTORY_NAME
        prediction_history.to_parquet(dest, index=False)
        checksums[_PRED_HISTORY_NAME] = _sha256_file(dest)

    manifest = {
        **manifest_extra,
        "format_version": BUNDLE_FORMAT_VERSION,
        "model_names": sorted(models),
        # §11.9: which walk-forward folds ship their own model (empty on legacy single-model bundles).
        "fold_years": fold_years,
        "fold_model_names": sorted(next(iter(fold_models.values()))) if fold_models else [],
        "lib_versions": _lib_versions(),
        "checksums": checksums,
    }
    (out / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return out


def load_bundle(bundle_dir: str | Path, *, strict_libs: bool = True) -> LoadedBundle:
    """Load and validate a bundle. Fail loud on any integrity/compatibility issue.

    Args:
        bundle_dir: path to a bundle directory written by :func:`write_bundle`.
        strict_libs: if True (default), raise on a MAJOR-version skew of a pickle-critical lib
                (lightgbm / scikit-learn / numpy) between the serving host and the bundle.

    The §11.5.2 measuring-stick gate (exact ``stock_ml_core`` match) is SEPARATE and opt-in via
    ``ENGINE_WHEEL_PIN`` — independent of ``strict_libs``.

    Raises:
        FileNotFoundError: manifest or any checksummed payload file missing.
        ValueError: format-version incompatibility, checksum mismatch, a pickle-critical major-version
                skew (strict_libs), or an engine-wheel mismatch (ENGINE_WHEEL_PIN).
    """
    path = Path(bundle_dir)
    manifest_path = path / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"load_bundle: no {_MANIFEST_NAME} at {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fmt = str(manifest.get("format_version", ""))
    if _major(fmt) != _major(BUNDLE_FORMAT_VERSION):
        raise ValueError(
            f"load_bundle: bundle format_version {fmt!r} incompatible with "
            f"runtime {BUNDLE_FORMAT_VERSION!r} (major mismatch)"
        )

    checksums = manifest.get("checksums")
    if not checksums:
        raise ValueError(f"load_bundle: manifest has no checksums — refusing to trust {path}")

    # Verify every payload file before deserializing anything.
    for rel, expected in checksums.items():
        f = path / rel
        if not f.exists():
            raise FileNotFoundError(f"load_bundle: missing payload file {rel} in {path}")
        actual = _sha256_file(f)
        if actual != expected:
            raise ValueError(
                f"load_bundle: checksum mismatch for {rel} "
                f"(expected {expected[:12]}…, got {actual[:12]}…) — bundle corrupted or tampered"
            )

    runtime = _lib_versions()
    baked = manifest.get("lib_versions", {})

    # (A) Pickle compatibility: a MAJOR-version skew of the libs that carry pickled estimator internals
    # (lightgbm/scikit-learn/numpy) can silently corrupt an unpickled model — hard gate under strict_libs.
    if strict_libs:
        for lib in _PICKLE_CRITICAL_LIBS:
            if lib in baked and lib in runtime and _major(baked[lib]) != _major(runtime[lib]):
                raise ValueError(
                    f"load_bundle: {lib} major version skew — bundle built with "
                    f"{baked[lib]}, runtime has {runtime[lib]}. Pickle may not load "
                    f"correctly. Re-export the bundle or pin {lib} to match."
                )

    # (B) §11.5.2 measuring-stick gate: refuse when the wheel that GENERATED this bundle's curves
    # (stock_ml_core) differs — EXACT, not just major. Opt-in (ENGINE_WHEEL_PIN in the serving container)
    # so dev/backtest/tests upgrade the wheel freely. A bundle with no recorded wheel can't be attested,
    # so under the pin that also refuses (re-export it). Escape: rollback the wheel, or re-baseline (§11.5.4).
    if _engine_pin_required():
        baked_wheel = baked.get("stock_ml_core")
        runtime_wheel = runtime.get("stock_ml_core")
        if baked_wheel is None or runtime_wheel is None or baked_wheel != runtime_wheel:
            raise ValueError(
                f"load_bundle: engine-wheel pin — bundle stock_ml_core {baked_wheel!r} != runtime "
                f"{runtime_wheel!r} (ENGINE_WHEEL_PIN set). Refusing to serve on a different measuring "
                "stick. Rollback the wheel or run a logged re-baseline (ENGINE_UPGRADE §11.5.2/§11.5.4)."
            )

    config = json.loads((path / _CONFIG_NAME).read_text(encoding="utf-8"))
    feature_spec = json.loads((path / _FEATURE_SPEC_NAME).read_text(encoding="utf-8"))

    models: dict[str, Any] = {}
    for name in manifest.get("model_names", []):
        models[name] = joblib.load(path / _MODELS_SUBDIR / f"{name}.joblib")
    if "entry" not in models:
        raise ValueError(f"load_bundle: bundle at {path} has no 'entry' model")

    prediction_history = None
    if _PRED_HISTORY_NAME in checksums:
        import pandas as pd

        prediction_history = pd.read_parquet(path / _PRED_HISTORY_NAME)

    # §11.9: per-fold models {test_year: {head: model}} — serving re-runs walk-forward from these.
    fold_models: dict[int, dict[str, Any]] | None = None
    fold_years = manifest.get("fold_years") or []
    if fold_years:
        fold_model_names = manifest.get("fold_model_names") or ["entry"]
        fold_models = {}
        for year in fold_years:
            fold_models[int(year)] = {
                name: joblib.load(path / _FOLD_MODELS_SUBDIR / str(year) / f"{name}.joblib")
                for name in fold_model_names
            }

    return LoadedBundle(
        models=models,
        config=config,
        feature_spec=feature_spec,
        manifest=manifest,
        path=path,
        prediction_history=prediction_history,
        fold_models=fold_models,
    )
