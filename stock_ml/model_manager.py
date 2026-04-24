"""
Model Manager CLI — manage model versions from the command line.

Usage:
    python model_manager.py list                    # List all models
    python model_manager.py list --active           # List active models only
    python model_manager.py retire v18              # Retire a model
    python model_manager.py activate v18            # Re-activate a model
    python model_manager.py compare                 # Compare all active models
    python model_manager.py compare --versions v25,v24,v23,rule  # Compare specific versions
    python model_manager.py add v25 --name "V25 New" --color "#FF5722" --strategy v25
"""
import sys
import os
import argparse

import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.config_loader import get_config_path, load_config
from src.env import get_results_dir, get_experiment_dir
from src.evaluation.scoring import composite_score


def cmd_list(args):
    """List all models."""
    cfg = load_config(force_reload=True)
    models = cfg.get("models", {})
    sorted_models = sorted(models.items(), key=lambda x: x[1].get("order", 99))

    print("=" * 90)
    print("MODEL REGISTRY")
    print("=" * 90)
    print(f"  {'Key':<10} {'Name':<20} {'Status':<10} {'Color':<10} {'Strategy':<10} {'Description'}")
    print("  " + "-" * 85)

    for key, m in sorted_models:
        active = m.get("active", True)
        if args.active and not active:
            continue
        status = "✓ Active" if active else "✗ Retired"
        color = m.get("color", "?")
        strategy = m.get("strategy", "?")
        desc = m.get("description", "")[:40]
        retired_reason = m.get("retired_reason", "")
        if retired_reason:
            desc = f"[{retired_reason}]"
        print(f"  {key:<10} {m.get('name','?'):<20} {status:<10} {color:<10} {strategy:<10} {desc}")

    # Check for trades CSV files
    experiment = getattr(args, "experiment", None) or ""
    results_dir = get_experiment_dir(experiment) if experiment else get_results_dir()
    label = f"{results_dir}" + (f" [experiment: {experiment}]" if experiment else "")
    print(f"\n  TRADES CSV FILES ({label}):")
    for key, m in sorted_models:
        csv_path = os.path.join(results_dir, f"trades_{key}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            total_pnl = df["pnl_pct"].sum() if "pnl_pct" in df.columns else 0
            n_trades = len(df)
            print(f"    ✓ trades_{key}.csv: {n_trades} trades, PnL={total_pnl:+.1f}%")
        else:
            print(f"    ✗ trades_{key}.csv: NOT FOUND")

    print("=" * 90)


def cmd_retire(args):
    """Retire a model."""
    path = get_config_path()
    cfg = load_config(force_reload=True)
    models = cfg.get("models", {})

    if args.version not in models:
        print(f"Error: Model '{args.version}' not found. Available: {list(models.keys())}")
        return

    models[args.version]["active"] = False
    if args.reason:
        models[args.version]["retired_reason"] = args.reason

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✓ Model '{args.version}' retired.")
    if args.reason:
        print(f"  Reason: {args.reason}")
    print(f"  Config updated: {path}")


def cmd_activate(args):
    """Re-activate a retired model."""
    path = get_config_path()
    cfg = load_config(force_reload=True)
    models = cfg.get("models", {})

    if args.version not in models:
        print(f"Error: Model '{args.version}' not found. Available: {list(models.keys())}")
        return

    models[args.version]["active"] = True
    if "retired_reason" in models[args.version]:
        del models[args.version]["retired_reason"]

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✓ Model '{args.version}' re-activated.")


def cmd_compare(args):
    """Compare models using their trades CSV files.

    Supports --versions flag to compare specific versions (not just active ones).
    Supports --experiment flag to compare within a specific experiment subfolder.
    When called from run_pipeline.py, args.versions may contain a comma-separated list.
    """
    cfg = load_config(force_reload=True)
    models = cfg.get("models", {})

    # Resolve results dir: experiment subfolder or flat
    experiment = getattr(args, "experiment", None) or ""
    results_dir = get_experiment_dir(experiment) if experiment else get_results_dir()

    # Determine which versions to compare
    specific_versions = getattr(args, "versions", None) or ""
    if specific_versions:
        # Compare specific versions (may include retired models)
        version_keys = [v.strip() for v in specific_versions.split(",")]
        selected = []
        for vk in version_keys:
            if vk in models:
                selected.append((vk, models[vk]))
            else:
                # Version not in registry — try to load CSV anyway
                selected.append((vk, {"name": vk.upper(), "order": 99}))
        sorted_models = sorted(selected, key=lambda x: x[1].get("order", 99))
        print("=" * 130)
        print(f"MODEL COMPARISON — Selected: {', '.join(version_keys)}")
        print("=" * 130)
    else:
        # Default: compare all active models
        active = {k: v for k, v in models.items() if v.get("active", True)}
        sorted_models = sorted(active.items(), key=lambda x: x[1].get("order", 99))
        print("=" * 130)
        print("MODEL COMPARISON — Active Models")
        print("=" * 130)

    metrics = {}
    trades_cache = {}
    for key, m in sorted_models:
        csv_path = os.path.join(results_dir, f"trades_{key}.csv")
        if not os.path.exists(csv_path):
            print(f"  ⚠ Skipping {key}: no trades CSV")
            continue
        df = pd.read_csv(csv_path)
        if "pnl_pct" not in df.columns:
            continue
        pnls = df["pnl_pct"].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        gp = wins.sum()
        gl = abs(losses.sum())
        metrics[key] = {
            "name": m.get("name", key),
            "trades": len(df),
            "wr": len(wins) / len(df) * 100 if len(df) > 0 else 0,
            "avg_pnl": np.mean(pnls),
            "total_pnl": np.sum(pnls),
            "pf": gp / gl if gl > 0 else 99,
            "max_loss": np.min(pnls) if len(pnls) > 0 else 0,
            "avg_hold": df["holding_days"].mean() if "holding_days" in df.columns else 0,
        }
        trades_cache[key] = df.to_dict("records")

    if not metrics:
        print("  No trades data found. Run backtest first.")
        return

    # Compute all extra metrics and composite scores
    from src.evaluation.scoring import (
        calc_sharpe, calc_mdd_per_symbol, calc_yearly_consistency,
    )
    for key, m in metrics.items():
        tl = trades_cache.get(key, [])
        m["sharpe"]  = calc_sharpe(tl)
        m["mdd_sym"] = calc_mdd_per_symbol(tl)
        m["cv"]      = calc_yearly_consistency(tl)
        m["score"]   = composite_score(m, trades=tl)

    # Check metadata consistency
    meta_warnings = []
    symbol_sets = {}
    for key in metrics:
        meta_path = os.path.join(results_dir, f"trades_{key}.meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            symbol_sets[key] = set(meta.get("symbols", []))
        else:
            meta_warnings.append(key)

    if meta_warnings:
        print(f"\n  ⚠ Missing metadata for: {', '.join(meta_warnings)}")
        print(f"    Re-run with --force to generate metadata for fair comparison.")

    if len(symbol_sets) >= 2:
        sets_list = list(symbol_sets.values())
        if not all(s == sets_list[0] for s in sets_list[1:]):
            print(f"\n  ⚠ WARNING: Symbol lists differ between models! Comparison may be unfair.")
            for key, syms in symbol_sets.items():
                print(f"    {key}: {len(syms)} symbols")

    # Summary table
    hdr = f"  {'Model':<20} | {'#':>4} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'Sharpe':>7} {'MDD/sym':>8} {'CV':>6} {'Score':>7}"
    print(f"\n{hdr}")
    print("  " + "-" * 105)
    best_score = max(m["score"] for m in metrics.values())
    for key, m in metrics.items():
        star = " ◀ BEST" if m["score"] == best_score else ""
        print(f"  {m['name']:<20} | {m['trades']:>4} {m['avg_pnl']:>+7.2f}% "
              f"{m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} "
              f"{m['sharpe']:>7.3f} {m['mdd_sym']:>7.1f}% {m['cv']:>5.2f} "
              f"{m['score']:>6.1f}{star}")

    # Scoring legend
    print(f"\n  Sharpe = avg_pnl/std(pnl)  |  MDD/sym = avg per-symbol drawdown  |  CV = std/mean of yearly PnL")

    # Recommendation
    print(f"\n  RECOMMENDATION (by composite score):")
    best_key = max(metrics, key=lambda k: metrics[k]["score"])
    worst_key = min(metrics, key=lambda k: metrics[k]["score"])
    worst = metrics[worst_key]
    print(f"    Best:  {best_key} ({metrics[best_key]['name']}) — Score={metrics[best_key]['score']:.1f}, "
          f"Sharpe={metrics[best_key]['sharpe']:.3f}, MDD/sym={metrics[best_key]['mdd_sym']:.1f}%")
    if worst["total_pnl"] < 0 or worst["score"] < 0:
        print(f"    Consider retiring '{worst_key}' ({worst['name']}): "
              f"Score={worst['score']:.1f}, TotalPnL={worst['total_pnl']:+.1f}%")

    print("=" * 130)


def cmd_experiments(args):
    """List all experiment subfolders in results/."""
    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        print(f"  No results directory found: {results_dir}")
        return

    experiments = []
    for name in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, name)
        if not os.path.isdir(exp_path):
            continue
        exp_json = os.path.join(exp_path, "experiment.json")
        if os.path.exists(exp_json):
            import json
            with open(exp_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            experiments.append((name, meta))
        else:
            # Subfolder without experiment.json — still show it
            experiments.append((name, {}))

    if not experiments:
        print("  No experiment subfolders found.")
        print(f"  Run with --feature-sets to create experiments: results/{results_dir}")
        return

    print("=" * 110)
    print("EXPERIMENTS")
    print("=" * 110)
    print(f"  {'Key':<35} {'feature_set':<15} {'ml_model':<12} {'versions':<25} {'#symbols':<10} {'generated_at'}")
    print("  " + "-" * 105)
    for name, meta in experiments:
        feat = meta.get("feature_set", "?")
        ml = meta.get("ml_model", "?")
        versions = ", ".join(meta.get("versions", []))
        n_sym = meta.get("n_symbols", "?")
        gen = meta.get("generated_at", "?")[:19]
        # Count CSVs if no experiment.json
        if not meta:
            exp_path = os.path.join(results_dir, name)
            csvs = [f for f in os.listdir(exp_path) if f.endswith(".csv")]
            feat = ml = "?"
            versions = ", ".join(c.replace("trades_", "").replace(".csv", "") for c in csvs)
            n_sym = gen = "?"
        print(f"  {name:<35} {feat:<15} {ml:<12} {versions:<25} {str(n_sym):<10} {gen}")

    print("=" * 110)
    print(f"\n  Usage: python model_manager.py compare --experiment <key> --versions v26,v27")


def cmd_add(args):
    """Add a new model version."""
    path = get_config_path()
    cfg = load_config(force_reload=True)
    models = cfg.get("models", {})

    if args.version in models:
        print(f"Error: Model '{args.version}' already exists. Use a different key.")
        return

    max_order = max((m.get("order", 0) for m in models.values()), default=0)
    new_model = {
        "name": args.name or f"V{args.version.upper()}",
        "description": args.description or "",
        "color": args.color or "#888888",
        "active": True,
        "strategy": args.strategy or args.version,
        "mods": {"a": True, "b": True, "c": False, "d": False,
                 "e": True, "f": True, "g": True, "h": True, "i": True, "j": True},
        "params": {},
        "marker_shape": "arrowUp",
        "order": max_order + 1,
    }
    models[args.version] = new_model

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✓ Model '{args.version}' added to registry.")
    print(f"  Name: {new_model['name']}")
    print(f"  Color: {new_model['color']}")
    print(f"  Strategy: {new_model['strategy']}")
    print(f"\n  Next steps:")
    print(f"    1. Create backtest function in experiments/run_{args.version}.py")
    print(f"    2. Run: python experiments/run_{args.version}.py")
    print(f"    3. Export: python -m src.export.unified_export --versions {args.version}")
    print(f"    4. Open dashboard.html — model appears automatically!")


def main():
    parser = argparse.ArgumentParser(description="Model Manager CLI")
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List all models")
    p_list.add_argument("--active", action="store_true", help="Show only active models")
    p_list.add_argument("--experiment", type=str, default="",
                        help="Show CSV files from experiment subfolder (e.g., leading_v2__lightgbm)")

    # retire
    p_retire = sub.add_parser("retire", help="Retire a model")
    p_retire.add_argument("version", help="Model version key (e.g., v18)")
    p_retire.add_argument("--reason", type=str, default="", help="Reason for retirement")

    # activate
    p_activate = sub.add_parser("activate", help="Re-activate a retired model")
    p_activate.add_argument("version", help="Model version key")

    # compare
    p_compare = sub.add_parser("compare", help="Compare all active models")
    p_compare.add_argument("--versions", type=str, default="",
                           help="Comma-separated versions to compare (e.g., v25,v24,v23,rule). Default: all active")
    p_compare.add_argument("--experiment", type=str, default="",
                           help="Read CSV files from experiment subfolder (e.g., leading_v2__lightgbm)")

    # experiments
    sub.add_parser("experiments", help="List all experiment subfolders in results/")

    # add
    p_add = sub.add_parser("add", help="Add a new model version")
    p_add.add_argument("version", help="Version key (e.g., v25)")
    p_add.add_argument("--name", type=str, help="Display name")
    p_add.add_argument("--color", type=str, help="Hex color (e.g., #FF5722)")
    p_add.add_argument("--strategy", type=str, help="Strategy key")
    p_add.add_argument("--description", type=str, help="Description")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "retire":
        cmd_retire(args)
    elif args.command == "activate":
        cmd_activate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "add":
        cmd_add(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
