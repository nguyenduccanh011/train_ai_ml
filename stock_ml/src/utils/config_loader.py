"""
Config loader — reads base.yaml and champion experiment YAML files.
"""

import os
from pathlib import Path

import yaml

from stock_ml.src.backtest.defaults import DEFAULT_TRADING_COST
from stock_ml.src.market_profile import resolve_market_name, resolve_run_context
from stock_ml.src.utils.env import resolve_data_dir

_CONFIG_CACHE = {}


def get_config_path():
    """Return absolute path to base.yaml."""
    return get_base_config_path()


def get_base_config_path():
    """Return absolute path to base.yaml."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "base.yaml")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _champions_dir() -> Path:
    return _repo_root() / "config" / "experiments" / "champions"


def _base_to_runtime_config(base: dict) -> dict:
    data = base.get("data", {})
    training = base.get("training", {})
    evaluation = base.get("evaluation", {})
    base_market = resolve_market_name(base.get("market"))
    context = resolve_run_context({"market": base_market})
    profile_data_dir = context.resolved_data_dir
    explicit_list = ",".join(context.resolved_symbols)
    pipeline = {
        "market": base_market,
        "data_dir": profile_data_dir or data.get("data_dir"),
        "feature_set": "leading_v2",
        "train_years": 4,
        "test_years": 1,
        "first_test_year": 2020,
        "last_test_year": 2025,
        "min_rows": 2000,
        "model_type": "lightgbm",
        "symbols": {
            "mode": "explicit",
            "min_rows": 2000,
            "explicit_list": explicit_list,
        },
        "target": {
            "type": "trend_regime",
            "trend_method": "dual_ma",
            "short_window": 5,
            "long_window": 20,
            "classes": 3,
        },
        **DEFAULT_TRADING_COST,
    }
    return {
        **base,
        "market": base_market,
        "pipeline": pipeline,
        "scoring": base.get(
            "scoring",
            {
                "mode": "live",
                "confidence_k": 120,
                "weights": {
                    "sharpe": 0.30,
                    "avg_pnl": 0.25,
                    "profit_factor": 0.22,
                    "mdd_per_symbol": 0.15,
                    "yr_consistency": 0.08,
                    "total_pnl_scale": 0.10,
                },
            },
        ),
        "visualization": {},
        "training": training,
        "evaluation": evaluation,
        "models": _load_champion_models(base_market),
        "symbol_profiles": {},
    }


def _load_champion_models(market: str | None = None) -> dict:
    """DEPRECATED: Load champion models from YAML files.

    Phase 0-3: Models are now managed in DB via StrategyTemplate instead.
    This function is kept for backward compatibility only and now returns empty dict.

    Use /api/templates endpoints instead to manage strategy templates.
    """
    # TODO: Phase 0-3.1 — Load from DB templates instead of YAML
    # For now, return empty dict to avoid YAML deps
    return {}


def load_config(force_reload=False):
    """Load and cache runtime config derived from base.yaml + champion YAML."""
    path = get_config_path()
    if path in _CONFIG_CACHE and not force_reload:
        return _CONFIG_CACHE[path]
    base = load_base_config(force_reload=force_reload)
    cfg = _base_to_runtime_config(base)
    _CONFIG_CACHE[path] = cfg
    return cfg


def load_base_config(force_reload=False):
    """Load and cache the base.yaml config."""
    path = get_base_config_path()
    if path in _CONFIG_CACHE and not force_reload:
        return _CONFIG_CACHE[path]
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _CONFIG_CACHE[path] = cfg
        return cfg
    except FileNotFoundError:
        return {}


def get_training_device():
    """Get the training device setting from base.yaml.

    Returns:
        str: "auto" | "gpu" | "cuda" | "cpu" (default: "cpu")
    """
    base = load_base_config()
    training = base.get("training", {})
    return training.get("device", "cpu")


def get_all_models(include_retired=False):
    """Return dict of model_key -> model_config, sorted by order."""
    cfg = load_config()
    models = cfg.get("models", {})
    if not include_retired:
        models = {k: v for k, v in models.items() if v.get("active", True)}
    return dict(sorted(models.items(), key=lambda x: x[1].get("order", 99)))


def get_active_models():
    """Return only active models."""
    return get_all_models(include_retired=False)


def get_model_config(version_key):
    """Get config for a specific model version."""
    cfg = load_config()
    models = cfg.get("models", {})
    if version_key not in models:
        raise KeyError(
            f"Model '{version_key}' not found in champion YAML configs. Available: {list(models.keys())}"
        )
    return models[version_key]


def get_pipeline_config():
    """Get pipeline defaults."""
    cfg = load_config()
    return cfg.get("pipeline", {})


def get_visualization_config():
    """Get visualization defaults."""
    cfg = load_config()
    return cfg.get("visualization", {})


def get_exit_abbreviations():
    """Get exit reason -> abbreviation mapping."""
    viz = get_visualization_config()
    return viz.get("exit_reason_abbreviations", {})


def get_model_color(version_key):
    """Get the color for a model version."""
    model = get_model_config(version_key)
    return model.get("color", "#888888")


def get_model_colors():
    """Get dict of version_key -> color for all active models."""
    return {k: v.get("color", "#888888") for k, v in get_active_models().items()}


def get_symbol_profiles():
    """Return {symbol -> profile_name} mapping from runtime config."""
    cfg = load_config()
    profiles = {}
    for profile_name, syms in cfg.get("symbol_profiles", {}).items():
        for sym in syms or []:
            profiles[str(sym)] = profile_name
    return profiles


def _resolve_universe_from_db(slug: str, available: set[str]) -> list[str] | None:
    """Resolve a DB universe slug to its in-data symbols (sorted), or None.

    Shared by the explicit ``mode=db`` path and the market-default fallback — the
    market YAML no longer hardcodes a symbol list (P5); the default set lives in
    universe_sets under slug ``"{market}_default"``.
    """
    # UniverseRepository is async-only; get_pipeline_symbols is sync, so query the
    # ORM tables directly through a sync session instead of the async repo.
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import sessionmaker

        from stock_ml.db.engine import sync_engine
        from stock_ml.db.models.universe import UniverseSetModel, UniverseSymbolModel

        SessionLocal = sessionmaker(bind=sync_engine)
        with SessionLocal() as session:
            universe_id = session.execute(
                select(UniverseSetModel.id).where(UniverseSetModel.slug == slug)
            ).scalar_one_or_none()
            if universe_id is None:
                return None
            db_symbols = (
                session.execute(
                    select(UniverseSymbolModel.symbol).where(
                        UniverseSymbolModel.universe_id == universe_id
                    )
                )
                .scalars()
                .all()
            )
            resolved = sorted(sym for sym in db_symbols if sym in available)
            return resolved or None
    except Exception as e:
        print(f"[warn] Failed to load universe from DB (slug={slug}): {e}")
        return None


def get_pipeline_symbols(
    symbols_arg="",
    min_rows_override=None,
    market: str | None = None,
    universe_config: dict | None = None,
):
    """Resolve the canonical symbol list for a pipeline run.

    Priority: CLI --symbols > universe config > market profile defaults > config explicit_list > auto-detect.
    Returns a sorted list of symbol strings.

    Args:
        symbols_arg: CLI --symbols argument (comma-separated)
        min_rows_override: override min data rows filter
        market: market name (overrides pipeline config)
        universe_config: universe config from ExperimentConfig (dict with mode + params)
    """
    pipeline = get_pipeline_config()
    run_context = resolve_run_context({"market": market or pipeline.get("market")})
    resolved_market = run_context.market
    profile = run_context.market_profile
    sym_cfg = pipeline.get("symbols", {})
    min_rows = min_rows_override or sym_cfg.get("min_rows", pipeline.get("min_rows", 2000))
    if run_context.resolved_data_dir is None:
        raise ValueError(f"Market {resolved_market!r} does not define data.data_dir")

    from stock_ml.src.data.loader import get_loader

    abs_data_dir = resolve_data_dir(run_context.resolved_data_dir)
    # Factory picks DuckDBLoader for a .duckdb path, CSV DataLoader otherwise.
    loader = get_loader(
        abs_data_dir,
        timeframe=run_context.timeframe,
    )
    available = set(loader.list_symbols())

    if symbols_arg and symbols_arg.strip():
        pick = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
        return sorted(s for s in pick if s in available)

    # Universe config from experiment YAML
    if universe_config:
        mode = universe_config.get("mode", "explicit")
        if mode == "explicit":
            explicit_list = universe_config.get("explicit_list", [])
            if explicit_list:
                pick = [
                    s.strip().upper()
                    for s in (explicit_list if isinstance(explicit_list, list) else [explicit_list])
                ]
                return sorted(s for s in pick if s in available)
        elif mode == "group":
            group_label = universe_config.get("group")
            if group_label and hasattr(profile.symbols, "groups"):
                syms = [s for s, lbl in profile.symbols.groups.items() if lbl == group_label]
                resolved = sorted(s for s in syms if s in available)
                if resolved:
                    return resolved
        elif mode == "file":
            file_path = universe_config.get("file")
            if file_path:
                import yaml

                try:
                    with open(file_path) as f:
                        universe_yaml = yaml.safe_load(f) or {}
                        symbols_list = universe_yaml.get("symbols", [])
                        if symbols_list:
                            pick = [s.strip().upper() for s in symbols_list]
                            return sorted(s for s in pick if s in available)
                except Exception as e:
                    print(f"[warn] Failed to load universe file {file_path}: {e}")
        elif mode == "db":
            slug = universe_config.get("slug")
            if slug:
                resolved = _resolve_universe_from_db(slug, available)
                if resolved:
                    return resolved

    # Market default symbol set now lives in the DB (universe_sets), not the market
    # YAML. Convention: slug "{market}_default" (seeded by migration 0021).
    resolved = _resolve_universe_from_db(f"{resolved_market}_default", available)
    if resolved:
        return resolved

    mode = sym_cfg.get("mode", "auto")
    if mode == "explicit":
        explicit = sym_cfg.get("explicit_list", "")
        if explicit and explicit.strip():
            pick = [s.strip().upper() for s in explicit.split(",") if s.strip()]
            return sorted(s for s in pick if s in available)

    viable = []
    for sym in loader.list_symbols():
        try:
            df = loader.load_symbol(sym)
            if len(df) >= min_rows:
                viable.append(sym)
        except (FileNotFoundError, Exception):
            continue
    return sorted(viable)
