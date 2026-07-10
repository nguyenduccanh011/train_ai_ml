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

# Libraries whose version skew can silently corrupt unpickled models. lightgbm is
# the critical one (the champion models are LGBM); a major-version mismatch
# between export host and serving host is rejected outright.
_TRACKED_LIBS = ["lightgbm", "scikit-learn", "numpy", "pandas", "joblib", "xgboost"]

_MODELS_SUBDIR = "models"
_MANIFEST_NAME = "manifest.json"
_CONFIG_NAME = "config.json"
_FEATURE_SPEC_NAME = "feature_spec.json"
# Optional: walk-forward predictions (score/exit_score) for bars BEFORE the bundle's
# test window, so the serving z-window is primed by the model that actually owned each
# past segment (not re-predicted by the current model). See project_production_bundle_deploy.
_PRED_HISTORY_NAME = "prediction_history.parquet"


@dataclass(frozen=True)
class LoadedBundle:
    """Result of :func:`load_bundle` — everything a serving host needs to infer."""

    models: dict[str, Any]  # name -> fitted model object (entry, exit, entry2, ...)
    config: dict[str, Any]  # resolved ExperimentConfig as a dict
    feature_spec: dict[str, Any]  # feature-set names + resolved feature columns
    manifest: dict[str, Any]  # provenance + checksums
    path: Path  # bundle directory it was loaded from
    prediction_history: Any = None  # DataFrame[symbol,date,score,exit_score] or None


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

    # --- model files ---
    for name, model in models.items():
        rel = f"{_MODELS_SUBDIR}/{name}.joblib"
        dest = out / rel
        joblib.dump(model, dest)
        checksums[rel] = _sha256_file(dest)

    # --- config + feature spec ---
    for rel, payload in ((_CONFIG_NAME, config), (_FEATURE_SPEC_NAME, feature_spec)):
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
        strict_libs: if True (default), raise when the serving host's lightgbm
                MAJOR version differs from the bundle's (pickle compatibility).
                Other tracked libs only warn via the returned manifest.

    Raises:
        FileNotFoundError: manifest or any checksummed payload file missing.
        ValueError: format-version incompatibility, checksum mismatch, or (when
                strict_libs) a lightgbm major-version skew.
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

    # lightgbm pickle compatibility is the hard gate.
    if strict_libs:
        runtime = _lib_versions()
        baked = manifest.get("lib_versions", {})
        for lib in ("lightgbm",):
            if lib in baked and lib in runtime and _major(baked[lib]) != _major(runtime[lib]):
                raise ValueError(
                    f"load_bundle: {lib} major version skew — bundle built with "
                    f"{baked[lib]}, runtime has {runtime[lib]}. Pickle may not load "
                    f"correctly. Re-export the bundle or pin {lib} to match."
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

    return LoadedBundle(
        models=models,
        config=config,
        feature_spec=feature_spec,
        manifest=manifest,
        path=path,
        prediction_history=prediction_history,
    )
