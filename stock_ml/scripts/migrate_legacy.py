"""migrate_legacy.py — Convert legacy models.yaml entries to new ExperimentConfig YAML files.

Usage:
    python -m stock_ml migrate-legacy v25
    python -m stock_ml migrate-legacy v25 --output config/experiments/legacy/v25.yaml
    python -m stock_ml migrate-legacy --all
    python -m stock_ml migrate-legacy --all --output-dir config/experiments/legacy/

The generated YAML files are usable with:
    python -m stock_ml run legacy/v25      (once runner registered in orchestrator)
    python -m stock_ml validate legacy/v25
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def _load_model_cfg(version_key: str) -> dict[str, Any]:
    """Load raw config from models.yaml for a given version key."""
    from src.config_loader import get_model_config

    cfg = get_model_config(version_key)
    if not cfg:
        raise ValueError(f"No config found for '{version_key}' in models.yaml")
    return cfg


def _cfg_to_experiment_yaml(version_key: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy model config dict to new ExperimentConfig schema dict."""
    feature_set = model_cfg.get("feature_set", "leading_v2")
    model_type = model_cfg.get("model_type", "lightgbm")

    target_raw = model_cfg.get("target") or {}
    target = {
        "type": target_raw.get("type", "trend_regime"),
        "forward_window": target_raw.get("forward_window", 8),
        "short_window": target_raw.get("short_window", 8),
        "long_window": target_raw.get("long_window", 20),
        "gain_threshold": target_raw.get("gain_threshold", 0.06),
        "loss_threshold": target_raw.get("loss_threshold", 0.03),
        "classes": target_raw.get("classes", 3),
    }

    exit_raw = model_cfg.get("exit_model") or {}
    exit_model = {
        "enabled": exit_raw.get("enabled", False),
        "forward_window": exit_raw.get("forward_window", 15),
        "loss_threshold": exit_raw.get("loss_threshold", 0.05),
    }

    mods = model_cfg.get("mods") or {}
    params = model_cfg.get("params") or {}

    doc: dict[str, Any] = {
        "name": version_key,
        "strategy": version_key,
        "components": {
            "features": feature_set,
            "target": target,
            "entry_model": {"type": model_type},
            "exit_model": exit_model,
        },
    }
    if mods:
        doc["mods"] = mods
    if params:
        doc["params"] = params

    # Preserve metadata as YAML comments by adding a _meta key (stripped at load time)
    name_label = model_cfg.get("name", version_key)
    description = model_cfg.get("description", "")
    active = model_cfg.get("active", True)
    retired_reason = model_cfg.get("retired_reason")
    doc["_meta"] = {
        "label": name_label,
        "description": description,
        "active": active,
        **({"retired_reason": retired_reason} if retired_reason else {}),
    }

    return doc


def migrate_version(
    version_key: str,
    *,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Migrate one legacy version to a new YAML file.

    Returns the path where the file was (or would be) written.
    """
    model_cfg = _load_model_cfg(version_key)
    doc = _cfg_to_experiment_yaml(version_key, model_cfg)

    if output_path is None:
        output_path = Path("stock_ml/config/experiments/legacy") / f"{version_key}.yaml"

    if dry_run:
        print(f"[dry-run] Would write: {output_path}")
        print(yaml.dump(doc, allow_unicode=True, sort_keys=False))
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, allow_unicode=True, sort_keys=False)

    print(f"  Wrote: {output_path}")
    return output_path


def migrate_all(
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
    skip_champions: bool = True,
) -> list[Path]:
    """Migrate all versions in models.yaml to YAML files.

    Args:
        output_dir: Directory to write files. Defaults to config/experiments/legacy/.
        dry_run: Print output without writing.
        skip_champions: Skip versions with dedicated component runners.
    """
    from src.config_loader import load_config
    from src.pipeline.legacy_adapter import CHAMPION_VERSIONS

    cfg = load_config()
    models = cfg.get("models", {})

    if output_dir is None:
        output_dir = Path("stock_ml/config/experiments/legacy")

    written: list[Path] = []
    skipped: list[str] = []

    for version_key in sorted(models):
        if skip_champions and version_key in CHAMPION_VERSIONS:
            skipped.append(version_key)
            continue
        try:
            path = migrate_version(
                version_key,
                output_path=output_dir / f"{version_key}.yaml",
                dry_run=dry_run,
            )
            written.append(path)
        except Exception as e:
            print(f"  WARNING: skipping {version_key}: {e}", file=sys.stderr)

    if skipped:
        print(f"  Skipped champions (have component runners): {', '.join(skipped)}")

    return written


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate legacy models.yaml entries to ExperimentConfig YAML files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("version", nargs="?", help="Single version key to migrate (e.g. v25)")
    group.add_argument("--all", action="store_true", help="Migrate all versions")

    parser.add_argument("--output", type=Path, help="Output file path (single-version mode)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (--all mode). Default: config/experiments/legacy/",
    )
    parser.add_argument(
        "--include-champions",
        action="store_true",
        help="Also migrate champion versions (usually have dedicated runners)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print YAML without writing files")

    args = parser.parse_args(argv)

    if args.all:
        paths = migrate_all(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            skip_champions=not args.include_champions,
        )
        print(f"\nMigrated {len(paths)} version(s).")
    else:
        migrate_version(args.version, output_path=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
