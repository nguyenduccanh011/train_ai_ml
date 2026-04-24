"""
Config loader — reads models.yaml and base.yaml, provides helper functions.
"""
import os
import yaml

from src.env import resolve_data_dir, get_results_dir


_CONFIG_CACHE = {}


def get_config_path():
    """Return absolute path to models.yaml."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "models.yaml")


def get_base_config_path():
    """Return absolute path to base.yaml."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "base.yaml")


def load_config(force_reload=False):
    """Load and cache the models.yaml config."""
    path = get_config_path()
    if path in _CONFIG_CACHE and not force_reload:
        return _CONFIG_CACHE[path]
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _CONFIG_CACHE[path] = cfg
    return cfg


def load_base_config(force_reload=False):
    """Load and cache the base.yaml config."""
    path = get_base_config_path()
    if path in _CONFIG_CACHE and not force_reload:
        return _CONFIG_CACHE[path]
    try:
        with open(path, "r", encoding="utf-8") as f:
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
        raise KeyError(f"Model '{version_key}' not found in models.yaml. "
                       f"Available: {list(models.keys())}")
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
    """Return {symbol -> profile_name} mapping from models.yaml symbol_profiles section."""
    cfg = load_config()
    profiles = {}
    for profile_name, syms in cfg.get("symbol_profiles", {}).items():
        for sym in (syms or []):
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
