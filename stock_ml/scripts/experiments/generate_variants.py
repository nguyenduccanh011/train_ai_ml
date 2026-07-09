#!/usr/bin/env python3
"""Generate experiment config variants from a base YAML and parameter grid.

Cartesian product across parameter dimensions → multiple YAML files ready to queue.

Usage:
    python scripts/generate_variants.py \\
        --base config/experiments/done/alpha_gate_v1.yaml \\
        --group "2026-06-02_param_tuning" \\
        --variant-type params \\
        --grid '{"entry_model.params.learning_rate": [0.01, 0.05, 0.1], "entry_model.params.max_depth": [6, 8, 10]}' \\
        [--out config/experiments/pending] \\
        [--dry-run]

Feature sets sweep:
    python scripts/generate_variants.py \\
        --base config/experiments/done/alpha_gate_v1.yaml \\
        --group "2026-06-02_feature_sweep" \\
        --variant-type features \\
        --grid '{"components.features": ["leading_v2", "leading_v3", "basic_v1"]}'
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Save YAML file with nice formatting."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def set_nested(obj: dict[str, Any], path: str, value: Any) -> None:
    """Set nested dict value using dot notation. E.g., 'entry_model.params.learning_rate' -> obj['entry_model']['params']['learning_rate'] = value"""
    keys = path.split(".")
    current = obj
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def parse_grid(grid_json: str) -> dict[str, list[Any]]:
    """Parse --grid JSON into dict of param_path -> list of values.

    Args:
        grid_json: JSON string like '{"lr": [0.01, 0.05], "depth": [6, 8]}'

    Returns:
        dict mapping param paths to value lists
    """
    return json.loads(grid_json)


def cartesian_variants(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate cartesian product of all grid dimensions.

    Args:
        base: base config dict
        grid: param_path -> list of values

    Returns:
        list of variant configs (copies of base with grid values substituted)
    """
    if not grid:
        return [base]

    param_paths = list(grid.keys())
    value_lists = [grid[p] for p in param_paths]

    variants = []
    for value_combo in itertools.product(*value_lists):
        variant = json.loads(json.dumps(base))  # deep copy
        for path, value in zip(param_paths, value_combo, strict=True):
            set_nested(variant, path, value)
        variants.append(variant)

    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment config variants from a base YAML and parameter grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Base YAML config file (template)",
    )
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="Experiment group name (appears in metadata.experiment_group)",
    )
    parser.add_argument(
        "--variant-type",
        type=str,
        required=True,
        choices=["params", "features", "target", "model"],
        help="Type of variant (for tracking in leaderboard)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        required=True,
        help='Parameter grid as JSON, e.g. \'{"lr": [0.01, 0.05], "depth": [6, 8]}\'',
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("config/experiments/pending"),
        help="Output directory for generated YAMLs (default: config/experiments/pending)",
    )
    parser.add_argument(
        "--parent-run-id",
        type=str,
        default="",
        help="Parent run ID (baseline model being optimized from)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Notes about the experiment (appears in metadata.notes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.base.exists():
        print(f"Error: base config not found: {args.base}")
        return 1

    try:
        grid = parse_grid(args.grid)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in --grid: {e}")
        return 1

    # Load base config
    base_cfg = load_yaml(args.base)
    if not base_cfg:
        print(f"Error: failed to load base config: {args.base}")
        return 1

    # Generate variants
    variants = cartesian_variants(base_cfg, grid)
    print(f"Generating {len(variants)} variant(s) from base: {args.base.name}")

    # Add metadata to each variant
    for i, variant in enumerate(variants):
        if "metadata" not in variant:
            variant["metadata"] = {}
        variant["metadata"]["experiment_group"] = args.group
        variant["metadata"]["variant_type"] = args.variant_type
        if args.parent_run_id:
            variant["metadata"]["parent_run_id"] = args.parent_run_id
        if args.notes:
            variant["metadata"]["notes"] = args.notes

    # Show or write output
    if args.dry_run:
        for i, variant in enumerate(variants):
            grid_values = " | ".join(
                f"{k.split('.')[-1]}={v}"
                for k, v in zip(grid.keys(), list(itertools.product(*grid.values()))[i], strict=True)
                if i < len(list(itertools.product(*grid.values())))
            )
            print(
                f"  [{i:03d}] {grid_values} | metadata={{group={args.group}, type={args.variant_type}}}"
            )
        print(f"\n[dry-run] Would generate {len(variants)} YAMLs to: {args.out}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for i, variant in enumerate(variants):
        # Output name: {group}_{idx:03d}.yaml
        out_name = f"{args.group}_{i:03d}.yaml"
        out_path = args.out / out_name

        save_yaml(out_path, variant)
        output_paths.append(out_path)
        print(f"  [{i:03d}] {out_path.name}")

    print(f"\n✓ Generated {len(output_paths)} YAML(s) in: {args.out}")
    print(
        f"Queue with: python scripts/run_experiments.py --pending {args.out} --done ... --failed ..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
