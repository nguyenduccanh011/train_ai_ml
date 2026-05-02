"""
Config loader — reads base.yaml and champion experiment YAML files.
"""

import os
from pathlib import Path

import yaml

from src.env import resolve_data_dir

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
    pipeline = {
        "data_dir": data.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned"),
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
            "explicit_list": "ACB,AAS,AAV,ACV,BCG,BCM,BID,BSR,BVH,CTG,DCM,DGC,DIG,DPM,EIB,FPT,FRT,GAS,GEX,GMD,HCM,HDB,HDG,HPG,HSG,KBC,KDH,LPB,MBB,MSN,MWG,NKG,NLG,NT2,NVL,OCB,PC1,PDR,PLX,PNJ,POW,PVD,PVS,REE,SAB,SBT,SHB,SSI,STB,TCB,TPB,VCB,VCI,VDS,VHM,VIC,VJC,VND,VNM,VPB,VTP",
        },
        "target": {
            "type": "trend_regime",
            "trend_method": "dual_ma",
            "short_window": 5,
            "long_window": 20,
            "classes": 3,
        },
        "commission": 0.0015,
        "tax": 0.001,
    }
    return {
        **base,
        "pipeline": pipeline,
        "scoring": {
            "weights": {
                "total_pnl": 0.45,
                "profit_factor": 0.25,
                "mdd_per_symbol": 0.2,
                "sharpe": 0.1,
            }
        },
        "visualization": {},
        "training": training,
        "evaluation": evaluation,
        "models": _load_champion_models(),
        "symbol_profiles": {},
    }


def _load_champion_models() -> dict:
    models = {}
    for idx, path in enumerate(sorted(_champions_dir().glob("*.yaml"))):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        components = raw.get("components", {}) or {}
        entry_model = components.get("entry_model", {}) or {}
        exit_model = components.get("exit_model", {}) or {}
        strategy_v3 = raw.get("strategy_v3", {}) or {}
        params = {**(raw.get("params", {}) or {}), **(strategy_v3.get("params", {}) or {})}
        model_cfg = {
            "name": raw.get("name", path.stem),
            "strategy": raw.get("strategy", raw.get("name", path.stem)),
            "feature_set": components.get("features", raw.get("feature_set", "leading_v2")),
            "target": components.get("target", raw.get("target", {})),
            "model_type": entry_model.get("type", "lightgbm"),
            "entry_model": entry_model,
            "exit_model": exit_model,
            "mods": raw.get("mods", {}),
            "params": params,
            "active": True,
            "order": raw.get("order", idx),
            "color": raw.get("color", "#888888"),
        }
        models[path.stem] = model_cfg
    return models


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


def get_pipeline_symbols(symbols_arg="", min_rows_override=None):
    """Resolve the canonical symbol list for a pipeline run.

    Priority: CLI --symbols > config explicit_list > auto-detect viable symbols.
    Returns a sorted list of symbol strings.
    """
    pipeline = get_pipeline_config()
    sym_cfg = pipeline.get("symbols", {})
    min_rows = min_rows_override or sym_cfg.get("min_rows", pipeline.get("min_rows", 2000))
    data_dir = pipeline.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")

    from src.data.loader import DataLoader

    abs_data_dir = resolve_data_dir(data_dir)
    loader = DataLoader(abs_data_dir)

    if symbols_arg and symbols_arg.strip():
        pick = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
        return sorted(s for s in pick if s in loader.symbols)

    mode = sym_cfg.get("mode", "auto")
    if mode == "explicit":
        explicit = sym_cfg.get("explicit_list", "")
        if explicit and explicit.strip():
            pick = [s.strip().upper() for s in explicit.split(",") if s.strip()]
            return sorted(s for s in pick if s in loader.symbols)

    viable = []
    for sym in loader.symbols:
        try:
            df = loader.load_symbol(sym)
            if len(df) >= min_rows:
                viable.append(sym)
        except (FileNotFoundError, Exception):
            continue
    return sorted(viable)
