"""Serving inference: bundle + OHLCV -> signals, via the exact champion chain.

This is the single entry point a production service calls. It reproduces the
research signal chain end-to-end without the DB or training code:
    build_feature_frame(with_targets=False) -> predict_slot_signals -> recombine_signals
"""

from __future__ import annotations

import re

import pandas as pd

from stock_ml.src.pipeline.experiment import (
    ExperimentConfig,
    build_feature_frame,
    predict_slot_signals,
    recombine_signals,
)
from stock_ml.src.serving.bundle import LoadedBundle

_OHLCV_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]
_ENTRY_HEAD_RE = re.compile(r"entry(\d+)$")  # entry2, entry3, ... ensemble heads
_EXIT_HEAD_RE = re.compile(r"exit(\d+)$")    # exit2, ... ensemble exit heads


def generate_signals_from_bundle(bundle: LoadedBundle, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Produce final buy/sell/hold signals for ``ohlcv`` using a loaded bundle.

    Args:
        bundle: result of stock_ml.src.serving.bundle.load_bundle.
        ohlcv: [symbol, date, open, high, low, close, volume] with ENOUGH warmup
               history before the dates of interest (>= manifest['min_warmup_bars'])
               so trailing features and the 252-bar recombine z-window are warm.

    Returns:
        DataFrame [symbol, date, signal, score, ...] where signal ∈ {-1, 0, 1}.

    Raises:
        ValueError: if the resolved feature columns drift from the bundle's spec
                (would silently change signals) or required columns are missing.
    """
    missing = [c for c in _OHLCV_COLS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"generate_signals_from_bundle: ohlcv missing columns {missing}")

    cfg = ExperimentConfig(**bundle.config)
    (feat, entry_feat_cols, exit_feat_cols, _et, _xt,
     entry2_feat_cols, entry3_feat_cols,
     entry4_feat_cols, entry5_feat_cols, entry6_feat_cols) = build_feature_frame(
        ohlcv[_OHLCV_COLS].copy(),
        cfg,
        requested_symbols=sorted(ohlcv["symbol"].unique()),
        data_root="",
        with_targets=False,
    )
    # Per-head entry feature columns (None = shares the primary entry features). Indexed by
    # the head number k so entryK -> entryK_feat_cols, for any head count the bundle carries.
    _entry_head_feat = {
        2: entry2_feat_cols, 3: entry3_feat_cols,
        4: entry4_feat_cols, 5: entry5_feat_cols, 6: entry6_feat_cols,
    }

    # Guard against silent feature-pipeline drift between export host and serving.
    spec_entry = bundle.feature_spec.get("entry_feat_cols")
    if spec_entry is not None and list(entry_feat_cols) != list(spec_entry):
        raise ValueError(
            "generate_signals_from_bundle: resolved entry feature columns differ from "
            f"the bundle's feature_spec ({len(entry_feat_cols)} vs {len(spec_entry)}) — "
            "feature pipeline drift would silently change signals."
        )

    # Per-bar NaN isolation (bug #2): a handful of old bars have NaN features (e.g.
    # bb_pct_20 in a std==0 zone). Drop ONLY those rows, with a logged count, instead of
    # failing the whole universe — so the full history can be loaded for a warm z-window /
    # downleg leg-state. The model layer (predict_slot_signals) stays fail-loud; we hand it
    # clean rows. Dropped bars carry no signal (the backtest skips them too), so they don't
    # affect agreement. Recent live data is normally clean, making this a no-op there.
    feat_cols_all = sorted(set(entry_feat_cols) | set(exit_feat_cols))
    nan_rows = feat[feat_cols_all].isna().any(axis=1)
    if nan_rows.any():
        n = int(nan_rows.sum())
        syms = feat.loc[nan_rows, "symbol"].nunique()
        print(f"[serving] dropping {n} NaN-feature bar(s) across {syms} symbol(s) "
              f"(std==0 / data gaps) — they carry no signal")
        feat = feat[~nan_rows].copy()
        if feat.empty:
            raise ValueError("generate_signals_from_bundle: all bars NaN after isolation")

    # ENSEMBLE heads (N-head champions n2_3h_/4h_/5h_*): a bundle carries entry2..entryN and
    # optional exit2.. models. Build the head lists generically from whatever the bundle holds
    # (entryK -> scoreK, exitK -> exit_scoreK) so ANY head count serves identically to the
    # backtest — no per-head wiring. A 2-head bundle yields empty lists (unchanged behaviour).
    entry_ensemble = [
        (f"score{int(m.group(1))}", bundle.models[name], _entry_head_feat.get(int(m.group(1))))
        for name in sorted(bundle.models)
        if (m := _ENTRY_HEAD_RE.fullmatch(name))
    ]
    exit_ensemble = [
        (f"exit_score{int(m.group(1))}", bundle.models[name], exit_feat_cols)
        for name in sorted(bundle.models)
        if (m := _EXIT_HEAD_RE.fullmatch(name))
    ]
    raw = predict_slot_signals(
        bundle.models["entry"],
        bundle.models.get("exit"),
        feat,
        entry_feat_cols,
        cfg,
        exit_feat_cols=exit_feat_cols,
        entry_ensemble=entry_ensemble,
        exit_ensemble=exit_ensemble,
    )

    # z-window seed: for past bars, use the walk-forward predictions that the model which
    # actually owned that segment produced (carried in the bundle), instead of re-predicting
    # them with the current model. This makes the trailing 252-bar z-score — and therefore
    # the decoupled buy band — match the backtest. Only score/exit_score are overridden;
    # OHLCV (downleg gate) and the new (post-history) bars stay as the current model's.
    ph = bundle.prediction_history
    if ph is not None and not ph.empty:
        ph = ph.copy()
        ph["date"] = pd.to_datetime(ph["date"]).dt.normalize()
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        # Seed every prediction column the history carries AND the current frame has
        # (score/exit_score always; scoreK/exit_scoreK for N-head bundles whose pred-history
        # was exported with them). Past bars use the walk-forward predictions that owned that
        # segment so the trailing z-window matches the backtest, for ANY head count.
        seed_cols = [c for c in ph.columns
                     if c not in ("symbol", "date") and c in raw.columns]
        raw = raw.merge(
            ph[["symbol", "date", *seed_cols]],
            on=["symbol", "date"], how="left", suffixes=("", "_hist"),
        )
        seed = raw["score_hist"].notna()
        for c in seed_cols:
            raw.loc[seed, c] = raw.loc[seed, f"{c}_hist"].astype(raw[c].dtype)
        raw = raw.drop(columns=[f"{c}_hist" for c in seed_cols])

    return recombine_signals(raw, cfg)
