"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric CV.

Phase 1.5.4: Quantifies likelihood that the best in-sample model
underperforms out-of-sample.

Reference: Bailey, Borwein, López de Prado, Zhu (2017),
"The Probability of Backtest Overfitting"
"""

from __future__ import annotations

import itertools

import numpy as np


def pbo(scores: np.ndarray, metric_fn=np.mean) -> dict:
    """Compute PBO via Combinatorially Symmetric CV.

    Phase 1.5.4: PBO answers: "What's the probability the best model is overfit?"

    Args:
        scores: shape (n_models, n_folds) — each row is a model, each column is OOS performance
                Best practice: rows = strategy variants, cols = time-ordered folds
        metric_fn: callable to aggregate fold scores into a model score (default mean)

    Returns:
        dict with keys:
        - pbo: probability of backtest overfitting (0-1, lower is better)
        - pbo_type1: Type I error (in-sample champion != OOS champion)
        - pbo_type2: Type II error (in-sample champion < OOS median)
        - n_models: number of models evaluated
        - n_folds: number of folds
        - champion_is: index of in-sample best model
        - champion_oos: index of OOS best model
    """
    scores = np.asarray(scores)
    n_models, n_folds = scores.shape

    if n_models < 2:
        raise ValueError("Need at least 2 models for PBO")
    if n_folds < 2:
        raise ValueError("Need at least 2 folds for PBO")

    n_combinations = 2**n_folds
    if n_combinations > 10000:
        raise ValueError(
            f"Too many combinations ({n_combinations}) for {n_folds} folds. "
            "Consider reducing n_folds."
        )

    pbo_count = 0

    for combo in itertools.product([0, 1], repeat=n_folds):
        is_folds = [i for i, in_sample in enumerate(combo) if in_sample == 1]
        oos_folds = [i for i, in_sample in enumerate(combo) if in_sample == 0]

        if not is_folds or not oos_folds:
            continue

        is_scores = scores[:, is_folds]
        oos_scores = scores[:, oos_folds]

        is_metrics = np.array([metric_fn(s) for s in is_scores])
        oos_metrics = np.array([metric_fn(s) for s in oos_scores])

        is_champion = np.argmax(is_metrics)
        oos_champion = np.argmax(oos_metrics)
        oos_median = np.median(oos_metrics)

        if is_champion != oos_champion or oos_metrics[is_champion] < oos_median:
            pbo_count += 1

    pbo_value = pbo_count / n_combinations

    is_metrics = np.array([metric_fn(s) for s in scores])
    champion_is = int(np.argmax(is_metrics))

    oos_metrics = np.mean(scores, axis=1)
    champion_oos = int(np.argmax(oos_metrics))

    type1_error = champion_is != champion_oos
    type2_error = oos_metrics[champion_is] < np.median(oos_metrics)

    return {
        "pbo": float(pbo_value),
        "pbo_type1": float(type1_error),
        "pbo_type2": float(type2_error),
        "n_models": int(n_models),
        "n_folds": int(n_folds),
        "champion_is_idx": champion_is,
        "champion_oos_idx": champion_oos,
        "champion_is_score": float(is_metrics[champion_is]),
        "champion_oos_score": float(oos_metrics[champion_oos]),
    }
