"""
Model registry - unified interface for all ML models.

Supports GPU acceleration for:
  - LightGBM: device="gpu" (OpenCL-based, works with NVIDIA/AMD)
  - XGBoost:  device="cuda" (CUDA-based, NVIDIA only)
  - CatBoost: task_type="GPU" (CUDA-based, NVIDIA only)

Usage:
  build_model("lightgbm", device="gpu")    # GPU mode
  build_model("lightgbm", device="cpu")    # CPU mode (default)
  build_model("lightgbm", device="auto")   # Auto-detect GPU
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


# ── Model Definitions ────────────────────────────────────────────

MODEL_CATALOG = {
    # Tree-based
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20,
                   "n_jobs": -1, "random_state": 42, "class_weight": "balanced"},
        "needs_scaling": False,
    },
    "extra_trees": {
        "class": ExtraTreesClassifier,
        "params": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20,
                   "n_jobs": -1, "random_state": 42, "class_weight": "balanced"},
        "needs_scaling": False,
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
                   "subsample": 0.8, "random_state": 42},
        "needs_scaling": False,
    },
    "adaboost": {
        "class": AdaBoostClassifier,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        "needs_scaling": False,
    },
    # Linear
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {"max_iter": 1000, "C": 1.0, "class_weight": "balanced",
                   "random_state": 42, "n_jobs": -1},
        "needs_scaling": True,
    },
    "sgd": {
        "class": SGDClassifier,
        "params": {"loss": "modified_huber", "max_iter": 1000,
                   "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        "needs_scaling": True,
    },
    # Distance-based
    "knn": {
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 15, "weights": "distance", "n_jobs": -1},
        "needs_scaling": True,
    },
    # SVM
    "svm": {
        "class": SVC,
        "params": {"kernel": "rbf", "C": 1.0, "probability": True,
                   "class_weight": "balanced", "random_state": 42},
        "needs_scaling": True,
    },
    # Naive Bayes
    "naive_bayes": {
        "class": GaussianNB,
        "params": {},
        "needs_scaling": True,
    },
}

# Add XGBoost if available
if HAS_XGB:
    MODEL_CATALOG["xgboost"] = {
        "class": XGBClassifier,
        "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                   "subsample": 0.8, "colsample_bytree": 0.8,
                   "use_label_encoder": False, "eval_metric": "mlogloss",
                   "random_state": 42, "n_jobs": -1},
        "needs_scaling": False,
    }

if HAS_LGB:
    MODEL_CATALOG["lightgbm"] = {
        "class": LGBMClassifier,
        "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                   "subsample": 0.8, "colsample_bytree": 0.8,
                   "random_state": 42, "n_jobs": -1, "verbose": -1},
        "needs_scaling": False,
    }

if HAS_CAT:
    MODEL_CATALOG["catboost"] = {
        "class": CatBoostClassifier,
        "params": {"iterations": 300, "depth": 6, "learning_rate": 0.05,
                   "random_state": 42, "verbose": 0},
        "needs_scaling": False,
    }


def get_available_models() -> List[str]:
    return list(MODEL_CATALOG.keys())


# ── GPU Detection ────────────────────────────────────────────────

def _detect_gpu() -> bool:
    """Check if NVIDIA GPU is available for training."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'gpu' or 'cpu' based on hardware detection.
    
    Returns normalized device string: 'gpu', 'cuda', or 'cpu'.
    """
    if device == "auto":
        return "gpu" if _detect_gpu() else "cpu"
    return device.lower()


_GPU_DETECTED = None  # Lazy cache


def detect_device(device: str = "auto") -> str:
    """Public API: resolve device setting with caching.
    
    Args:
        device: "auto" | "gpu" | "cuda" | "cpu"
    
    Returns:
        Resolved device string
    """
    global _GPU_DETECTED
    if device == "auto":
        if _GPU_DETECTED is None:
            _GPU_DETECTED = _detect_gpu()
        return "gpu" if _GPU_DETECTED else "cpu"
    return device.lower()


def _apply_device_params(name: str, params: dict, device: str) -> dict:
    """Apply GPU/CPU device parameters to model-specific params.
    
    Each library has different parameter names for GPU:
      - LightGBM: device="gpu" or device="cpu"
      - XGBoost:  device="cuda" or device="cpu", tree_method="hist"
      - CatBoost: task_type="GPU" or task_type="CPU"
    """
    resolved = detect_device(device)
    params = params.copy()
    
    if resolved in ("gpu", "cuda"):
        if name == "lightgbm":
            params["device"] = "gpu"
            # GPU mode doesn't support n_jobs parallelism
            params.pop("n_jobs", None)
        elif name == "xgboost":
            params["device"] = "cuda"
            params["tree_method"] = "hist"
            # GPU mode doesn't support n_jobs parallelism
            params.pop("n_jobs", None)
        elif name == "catboost":
            params["task_type"] = "GPU"
            params["devices"] = "0"
    else:
        # Explicitly set CPU mode
        if name == "lightgbm":
            params["device"] = "cpu"
        elif name == "xgboost":
            params["device"] = "cpu"
        elif name == "catboost":
            params["task_type"] = "CPU"
    
    return params


# ── Build Functions ──────────────────────────────────────────────

def build_model(name: str, scaling: str = "robust", device: str = "cpu",
                **override_params) -> Pipeline:
    """Build a sklearn Pipeline with optional scaler + model.
    
    Args:
        name: Model name from MODEL_CATALOG (e.g., "lightgbm", "xgboost")
        scaling: "robust" or "standard" (only for models that need scaling)
        device: "auto" | "gpu" | "cuda" | "cpu" — GPU acceleration for 
                LightGBM/XGBoost/CatBoost. Other models ignore this param.
        **override_params: Additional params passed to the model constructor
    
    Returns:
        sklearn Pipeline with optional scaler + model
    
    Examples:
        build_model("lightgbm")                    # CPU (default)
        build_model("lightgbm", device="gpu")      # GPU mode
        build_model("lightgbm", device="auto")     # Auto-detect
    """
    if name not in MODEL_CATALOG:
        raise ValueError(f"Unknown model: {name}. Available: {get_available_models()}")

    spec = MODEL_CATALOG[name]
    params = {**spec["params"], **override_params}
    
    # Apply GPU/CPU device params for supported models
    if name in ("lightgbm", "xgboost", "catboost"):
        params = _apply_device_params(name, params, device)
    
    model = spec["class"](**params)

    steps = []
    if spec["needs_scaling"]:
        scaler = RobustScaler() if scaling == "robust" else StandardScaler()
        steps.append(("scaler", scaler))
    steps.append(("model", model))

    return Pipeline(steps)


def build_all_models(
    model_names: Optional[List[str]] = None,
    scaling: str = "robust",
    device: str = "cpu",
) -> Dict[str, Pipeline]:
    """Build all requested models (or all available)."""
    names = model_names or get_available_models()
    return {name: build_model(name, scaling, device=device) for name in names}
