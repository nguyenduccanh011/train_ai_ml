"""Experiment pipeline — research-grade end-to-end training and backtesting.

Supports independent entry/exit models and YAML-driven configuration.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stock_ml.src.backtest.engine import (
    engine_config_from_dict,
    run_backtest,
    trades_to_dataframe,
)
from stock_ml.src.backtest.integrity import audit_report, print_report
from stock_ml.src.backtest.stats import (
    aggregate_stats,
    per_day_stats,
    per_symbol_stats,
    per_year_stats,
)
from stock_ml.src.data.splitter import PurgedKFoldSplitter, YearSplitter
from stock_ml.src.data.universe_resolver import parse_universe_policy_slug as _parse_policy_slug
from stock_ml.src.features.market import build_equal_weight_index
from stock_ml.src.features.resolver import FeatureResolver
from stock_ml.src.features.sectors import build_sector_map
from stock_ml.src.models.registry import build_entry_model, build_exit_model, build_regression_model
from stock_ml.src.signals.core import (
    generate_signals_from_predictions,
    generate_signals_from_technical_rules,
)
from stock_ml.src.targets.registry import build_target, target_forward_span


@dataclass
class ExperimentConfig:
    """Configuration for an experiment — maps cleanly from YAML.

    Phase 1b.6: Canonical nested schema validation.
    Supports full YAML structure with components, split, engine, validation.
    Phase 0.3: Direction support for long/short strategies.
    Phase 0.4: Per-slot feature sets and targets (optional overrides).
    """

    name: str
    strategy: str
    market: str
    feature_set: str
    target: dict
    entry_model: dict
    exit_model: dict
    split: dict
    engine: dict
    seed: int = 42
    signal_threshold: float = 0.0
    # Hysteresis band (Phase 1). None → derived from signal_threshold (+thr / -thr),
    # which reproduces the legacy symmetric rule exactly.
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    signal_mode: str = "entry_first"  # DEPRECATED: no-op in regression; use entry/exit_threshold
    model_mode: str = (
        "ml_only"  # ml_only | rule_only | hybrid_ml_entry_rule_exit | hybrid_rule_entry_ml_exit
    )
    direction: str = "long"  # long | short (flip signals if short)
    regime_model: dict = field(default_factory=lambda: {"type": "none", "enabled": False})
    size_model: dict = field(default_factory=lambda: {"type": "none", "enabled": False})
    yaml_schema_version: int = 2  # 1 = legacy, 2 = with signal_mode, model_mode, regime/size slots
    hypothesis: str = ""
    validation: dict | None = None
    strict_audit: bool = True
    metadata: dict | None = None
    data_source_dir: str | None = None
    universe: dict | None = None
    # Dynamic point-in-time universe (docs/UPGRADE_DYNAMIC_UNIVERSE.md). None → legacy
    # fixed-symbols behavior (byte-identical). When set, run_experiment resolves the
    # per-fold top-N universe causally and ignores the incoming symbols list.
    # e.g. {"mode": "dynamic_topn", "n": 900, "metric": "adv",
    #       "lookback": "prior_year", "min_sessions": 100}
    universe_policy: dict | None = None
    # Per-slot features/targets (Phase 0.4) — None means use global
    entry_features: str | None = None
    entry_target: dict | None = None
    exit_features: str | None = None
    exit_target: dict | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load experiment config from YAML file with strict validation.

        Args:
            path: path to YAML file

        Returns:
            ExperimentConfig instance

        Raises:
            FileNotFoundError: if file not found
            ValueError: if required fields missing or schema invalid
        """
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        required_top = ["name", "strategy", "market", "components", "split", "engine"]
        missing = [k for k in required_top if k not in raw]
        if missing:
            raise ValueError(f"YAML missing required keys: {missing}")

        comp = raw.get("components", {})
        required_comp = ["features", "target", "entry_model"]
        missing_comp = [k for k in required_comp if k not in comp]
        if missing_comp:
            raise ValueError(f"components missing required keys: {missing_comp}")

        target_cfg = comp["target"]
        if "type" not in target_cfg:
            raise ValueError("components.target missing required 'type' field")
        if target_cfg["type"] not in [
            "forward_return",
            "forward_return_regression",
            "trend_regime",
            "zigzag_pivot",
            "early_wave_v2",
        ]:
            raise ValueError(f"Unknown target type: {target_cfg['type']}")

        entry_model_cfg = comp["entry_model"]
        if "type" not in entry_model_cfg:
            raise ValueError("components.entry_model missing required 'type' field")

        split_cfg = raw.get("split", {})
        if "type" not in split_cfg:
            raise ValueError("split missing required 'type' field")
        valid_split_types = ["walk_forward_year", "purged_kfold"]
        if split_cfg["type"] not in valid_split_types:
            raise ValueError(f"Unknown split type: {split_cfg['type']}")

        engine_cfg = raw.get("engine", {})
        if not isinstance(engine_cfg, dict):
            raise ValueError("engine must be a dict")

        validation_cfg = raw.get("validation")
        if validation_cfg is not None and not isinstance(validation_cfg, dict):
            raise ValueError("validation must be a dict or null")

        metadata_cfg = raw.get("metadata")
        if metadata_cfg is not None and not isinstance(metadata_cfg, dict):
            raise ValueError("metadata must be a dict or null")

        # Validate direction
        direction = raw.get("direction", "long")
        if direction not in ["long", "short"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'long' or 'short'")

        # Auto-detect model_mode if not explicitly set
        model_mode = raw.get("model_mode")
        if not model_mode:
            entry_type = entry_model_cfg.get("type")
            exit_type = comp.get("exit_model", {}).get("type")
            exit_enabled = comp.get("exit_model", {}).get("enabled", False)

            if entry_type == "rule":
                if exit_enabled and exit_type == "rule":
                    model_mode = "rule_only"
                elif exit_enabled and exit_type != "rule":
                    model_mode = "hybrid_rule_entry_ml_exit"
                else:
                    model_mode = "rule_only"
            else:
                if exit_enabled and exit_type == "rule":
                    model_mode = "hybrid_ml_entry_rule_exit"
                else:
                    model_mode = "ml_only"

        # Extract per-slot features/targets (Phase 0.4)
        # Copy dicts to avoid mutating the original YAML
        entry_model_copy = entry_model_cfg.copy() if entry_model_cfg else {}
        exit_model_copy = comp.get("exit_model", {}).copy() if comp.get("exit_model") else {}

        entry_features = entry_model_copy.pop("features", None)
        entry_target_override = entry_model_copy.pop("target", None)
        exit_features = exit_model_copy.pop("features", None)
        exit_target_override = exit_model_copy.pop("target", None)

        return cls(
            name=raw["name"],
            strategy=raw["strategy"],
            market=raw["market"],
            feature_set=comp.get("features", "basic_v1"),
            target=target_cfg,
            entry_model=entry_model_copy,
            exit_model=exit_model_copy or {"type": "none", "enabled": False, "params": {}},
            split=split_cfg,
            engine=engine_cfg,
            seed=raw.get("seed", 42),
            signal_threshold=raw.get("signal_threshold", 0.0),
            entry_threshold=raw.get("entry_threshold"),
            exit_threshold=raw.get("exit_threshold"),
            signal_mode=comp.get("signal_mode", "entry_first"),
            model_mode=model_mode,
            direction=direction,
            regime_model=comp.get("regime_model", {"type": "none", "enabled": False}),
            size_model=comp.get("size_model", {"type": "none", "enabled": False}),
            yaml_schema_version=raw.get("schema_version", 2),
            hypothesis=raw.get("hypothesis", ""),
            validation=validation_cfg,
            strict_audit=raw.get("strict_audit", True),
            metadata=metadata_cfg or {},
            data_source_dir=raw.get("data", {}).get("source_dir"),
            universe=raw.get("universe"),
            universe_policy=raw.get("universe_policy"),
            entry_features=entry_features,
            entry_target=entry_target_override,
            exit_features=exit_features,
            exit_target=exit_target_override,
        )

    @classmethod
    async def from_template_id_async(cls, template_id: int, session) -> ExperimentConfig:
        """Load experiment config from DB strategy template (Phase 0-3: DB-First).

        Supports dual-component slots (ML + Rule per slot). Mode is inferred from
        null/non-null component IDs — no explicit mode field needed.

        Args:
            template_id: ID of StrategyTemplateModel in DB
            session: SQLAlchemy async session

        Returns:
            ExperimentConfig instance

        Raises:
            ValueError: if template not found or invalid
        """
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        from stock_ml.db.models.template import ComponentSlotModel, StrategyTemplateModel

        result = await session.execute(
            select(StrategyTemplateModel)
            .where(StrategyTemplateModel.id == template_id)
            .options(
                selectinload(StrategyTemplateModel.feature_set),
                selectinload(StrategyTemplateModel.target),
                selectinload(StrategyTemplateModel.component_slots).selectinload(
                    ComponentSlotModel.ml_component
                ),
                selectinload(StrategyTemplateModel.component_slots).selectinload(
                    ComponentSlotModel.rule_component
                ),
            )
        )
        template = result.scalar_one_or_none()

        if not template:
            raise ValueError(f"Template with ID {template_id} not found")

        # Helper to get slot by type from component_slots
        def _get_slot(slot_type):
            return next((s for s in template.component_slots if s.slot_type == slot_type), None)

        # Helper to build slot model dict from dual components (ML + Rule)
        def _build_slot_dict(ml_comp, rule_comp):
            if ml_comp:
                model_type = ml_comp.algorithm
            elif rule_comp:
                model_type = "rule"
            else:
                model_type = "none"
            result = {
                "type": model_type,
                "params": ml_comp.params.copy() if ml_comp else {},
            }
            if rule_comp:
                result["rule_conditions"] = rule_comp.params.get("conditions", [])
                result["rule_logic"] = rule_comp.params.get("logic", "AND")
            return result

        # Build entry_model dict from component_slots
        entry_slot = _get_slot("entry")
        entry_features = None
        entry_target = None
        if entry_slot:
            entry_model = _build_slot_dict(entry_slot.ml_component, entry_slot.rule_component)
            entry_features = entry_slot.feature_set_name
            entry_target = entry_slot.target_config
        else:
            entry_model = {"type": "none", "params": {}}

        # Build exit_model dict
        exit_slot = _get_slot("exit")
        exit_features = None
        exit_target = None
        if exit_slot:
            exit_model = _build_slot_dict(exit_slot.ml_component, exit_slot.rule_component)
            exit_model["enabled"] = True
            exit_features = exit_slot.feature_set_name
            exit_target = exit_slot.target_config
        else:
            exit_model = {
                "type": "none",
                "enabled": False,
                "params": {},
            }

        # Build regime_model dict
        regime_slot = _get_slot("regime")
        if regime_slot:
            regime_model = _build_slot_dict(regime_slot.ml_component, regime_slot.rule_component)
            regime_model["enabled"] = True
        else:
            regime_model = {
                "type": "none",
                "enabled": False,
            }

        # Build size_model dict
        size_slot = _get_slot("size")
        if size_slot:
            size_model = _build_slot_dict(size_slot.ml_component, size_slot.rule_component)
            size_model["enabled"] = True
        else:
            size_model = {
                "type": "none",
                "enabled": False,
            }

        # Build target dict
        target = {
            "type": template.target.type,
            **template.target.params.copy(),
        }

        # Auto-infer model_mode from component slots
        inferred_mode = template.infer_model_mode()
        db_mode = template.model_mode
        model_mode = inferred_mode if inferred_mode != "none" else db_mode

        return cls(
            name=template.name,
            strategy=template.strategy,
            market=template.market,
            feature_set=template.feature_set.name,
            target=target,
            entry_model=entry_model,
            exit_model=exit_model,
            split=template.split_config.copy(),
            engine=template.engine_config.copy(),
            seed=template.seed,
            signal_threshold=template.signal_threshold,
            # entry/exit_threshold columns arrive in migration 0017; tolerate their
            # absence so this loader works before and after the migration.
            entry_threshold=getattr(template, "entry_threshold", None),
            exit_threshold=getattr(template, "exit_threshold", None),
            signal_mode=template.signal_mode,
            model_mode=model_mode,
            direction=template.direction,
            regime_model=regime_model,
            size_model=size_model,
            yaml_schema_version=template.schema_version,
            hypothesis=template.hypothesis or "",
            validation=template.validation_config.copy() if template.validation_config else None,
            strict_audit=True,
            metadata={},
            data_source_dir=None,
            universe={"slug": template.universe_slug, "mode": "db"}
            if template.universe_slug
            else None,
            # Dynamic point-in-time universe: a 'dyn_topn:...' universe_slug IS the policy
            # (interim §7.1 persistence — reproducible from the DB template alone).
            universe_policy=_parse_policy_slug(template.universe_slug),
            entry_features=entry_features,
            entry_target=entry_target,
            exit_features=exit_features,
            exit_target=exit_target,
        )


def _normalize_rule_conditions(rule_conditions: Any, where: str) -> list[dict]:
    """Validate slot rule_conditions and return them in RuleModel's expected form.

    Fails loud: a rule slot with no usable conditions is a configuration error, not a
    reason to silently fall back to an "always true" rule.
    """
    if not rule_conditions:
        raise ValueError(
            f"{where}: type='rule' requires non-empty rule_conditions, got {rule_conditions!r}. "
            "Refusing to silently treat a condition-less rule as always-true."
        )
    if not isinstance(rule_conditions, list):
        raise ValueError(f"{where}: rule_conditions must be a list, got {type(rule_conditions).__name__}")
    normalized = []
    for i, cond in enumerate(rule_conditions):
        if not isinstance(cond, dict) or "feature" not in cond or "op" not in cond:
            raise ValueError(
                f"{where}: rule_conditions[{i}] must be a dict with 'feature' and 'op', got {cond!r}"
            )
        normalized.append({"feature": cond["feature"], "op": cond["op"], "value": cond.get("value", 0)})
    return normalized


def require_no_nan(df: pd.DataFrame, subset: list[str], *, stage: str) -> pd.DataFrame:
    """Fail loud instead of silently dropping rows with NaN in ``subset``.

    Policy (fail-loud, no silent fallback): any NaN in a required column aborts the
    run with a diagnostic, so the cause can be inspected and a handling decision made
    explicitly — rather than ``dropna`` quietly deleting the rows and shrinking the
    dataset. Note this includes structurally-expected NaN (feature warmup, forward-
    window target tail); those must be trimmed/handled upstream on purpose.

    Returns ``df`` unchanged when no NaN is present.
    """
    nan_cols = [c for c in subset if c in df.columns and df[c].isna().any()]
    if not nan_cols:
        return df

    mask = df[subset].isna().any(axis=1)
    n = int(mask.sum())
    col_counts = sorted(
        ((c, int(df[c].isna().sum())) for c in nan_cols), key=lambda kv: -kv[1]
    )
    detail = "\n".join(f"  {c}: {cnt} NaN" for c, cnt in col_counts)
    id_cols = [c for c in ("symbol", "date") if c in df.columns]
    sample = df.loc[mask, id_cols].head(10).to_dict("records") if id_cols else []
    raise ValueError(
        f"[{stage}] {n}/{len(df)} rows have NaN in required columns — refusing to "
        "silently drop them (fail-loud policy). Inspect the cause and handle it "
        "explicitly (e.g. trim feature warmup or the forward-window target tail, or "
        f"fix the feature computation).\nNaN by column:\n{detail}\n"
        f"Sample affected rows: {sample}"
    )


def assert_feature_integrity(
    feat: pd.DataFrame,
    feature_cols: list[str],
    *,
    warmup: int = 252,
    max_nan_frac: float = 0.05,
) -> None:
    """Fail loud when feature columns stay NaN well past their warmup window.

    A little NaN is legitimate (rolling/EWM warmup, divide-by-zero on flat bars). A
    feature that is still mostly NaN after `warmup` rows per symbol signals a real
    computation bug (e.g. index misalignment) — abort instead of letting the downstream
    dropna() silently delete the affected rows and shrink the universe.
    """
    missing = [c for c in feature_cols if c not in feat.columns]
    if missing:
        raise ValueError(f"feature integrity: missing feature columns {missing}")

    post = (
        feat.sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .apply(lambda g: g.iloc[warmup:])
    )
    if post.empty:
        raise ValueError(
            f"feature integrity: no rows remain after warmup={warmup} — symbols too short"
        )

    nan_frac = post[feature_cols].isna().mean()
    bad = nan_frac[nan_frac > max_nan_frac].sort_values(ascending=False)
    if not bad.empty:
        detail = "\n".join(f"  {c}: {frac:.1%} NaN" for c, frac in bad.items())
        raise ValueError(
            f"feature integrity check failed (warmup={warmup}, threshold={max_nan_frac:.0%}). "
            "These columns are mostly NaN after warmup — likely a feature bug; refusing to "
            f"silently drop the affected rows:\n{detail}"
        )


def trim_feature_warmup(
    feat: pd.DataFrame, feature_cols: list[str], *, name: str = ""
) -> pd.DataFrame:
    """Drop each symbol's leading rows that are NaN in any required feature column.

    Rolling/EWM/52-week features are structurally NaN for the first rows of every
    symbol (their warmup window). ``require_no_nan`` forbids silently dropping NaN
    downstream, so we trim that warmup head HERE — explicitly and logged — which is
    the upstream handling its docstring calls for. Only the leading *contiguous* NaN
    block is removed: any NaN that appears mid-series is left in place so the
    fail-loud guard still catches genuine feature bugs (e.g. index misalignment).
    """
    missing = [c for c in feature_cols if c not in feat.columns]
    if missing:
        raise ValueError(f"trim_feature_warmup: missing feature columns {missing}")

    def _trim(g: pd.DataFrame) -> pd.DataFrame:
        ok = g[feature_cols].notna().all(axis=1).to_numpy()
        if not ok.any():
            return g.iloc[0:0]  # whole symbol is too short — drop it (logged via total)
        return g.iloc[int(ok.argmax()):]  # first all-non-NaN row onward

    out = (
        feat.sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .apply(_trim)
    )
    dropped = len(feat) - len(out)
    kept_syms = out["symbol"].nunique() if not out.empty else 0
    print(
        f"[{name}] feature warmup trim: dropped {dropped} leading NaN rows "
        f"({len(out)} kept across {kept_syms} symbols)"
    )
    return out


def trim_target_tail(
    feat: pd.DataFrame, target_cols: list[str], *, name: str = ""
) -> pd.DataFrame:
    """Drop each symbol's trailing rows that are NaN in any label column.

    Forward-looking targets (triple_barrier, velocity_exit, ...) are structurally NaN
    for a symbol's LAST rows: there is no future window to resolve the label. For a
    fixed universe that lives to the backtest end this tail sits only in the final test
    fold and is handled elsewhere, but a symbol that DELISTS mid-history (e.g. a name
    the point-in-time/dynamic universe legitimately includes) has this NaN tail land
    inside a TRAIN fold, tripping the fail-loud ``require_no_nan``. Trimming the trailing
    contiguous NaN-label block here is the symmetric counterpart to ``trim_feature_warmup``
    (which trims the leading feature-warmup block). Only the trailing *contiguous* NaN
    block is removed per symbol: any NaN that appears mid-series is left in place so the
    fail-loud guard still catches genuine label bugs.
    """
    cols = [c for c in target_cols if c in feat.columns]
    if not cols:
        return feat

    # Only trim the NaN-label tail of symbols that END before the panel does — i.e. names
    # that DELIST mid-history. A symbol still trading at the panel end keeps its natural
    # forward-window tail (those rows are valid TEST inputs in the final fold; they carry
    # no label but are predicted on features), so trimming them would drop live test bars.
    panel_end = feat["date"].max()

    def _trim(g: pd.DataFrame) -> pd.DataFrame:
        if g["date"].max() >= panel_end:
            return g  # symbol alive to the end — leave its forward-window tail intact
        ok = g[cols].notna().all(axis=1).to_numpy()
        if not ok.any():
            return g.iloc[0:0]
        last_ok = len(ok) - 1 - int(ok[::-1].argmax())  # last all-non-NaN row index
        return g.iloc[: last_ok + 1]

    out = (
        feat.sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .apply(_trim)
    )
    dropped = len(feat) - len(out)
    if dropped:
        print(f"[{name}] target tail trim: dropped {dropped} trailing NaN-label rows "
              f"(delisted/short symbols' forward-window tail)")
    return out


# Forward-looking targets that are CLASSIFIERS (discrete {-1,0,1}) even though their NaN
# tail makes them float dtype. forward_return is intentionally excluded — it has always
# been consumed as a regression ordinal here, and changing it would move every existing run.
_CLASSIFICATION_TARGET_TYPES = frozenset(
    {"trend_regime", "early_wave_v2", "action_oracle", "amplitude_oracle"}
)


def train_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    cfg: ExperimentConfig,
    exit_feat_cols: list[str] | None = None,
    entry_target_col: str = "target",
    exit_target_col: str = "target",
    entry2_target_col: str | None = None,
    exit2_target_col: str | None = None,
    entry2_feat_cols: list[str] | None = None,
    entry3_target_col: str | None = None,
    entry3_feat_cols: list[str] | None = None,
    entry4_target_col: str | None = None,
    entry4_feat_cols: list[str] | None = None,
    entry5_target_col: str | None = None,
    entry5_feat_cols: list[str] | None = None,
    entry6_target_col: str | None = None,
    entry6_feat_cols: list[str] | None = None,
    out_models: dict[str, Any] | None = None,
) -> tuple[Any, Any | None, pd.DataFrame]:
    """Train entry/exit models on fold and generate test signals.

    **PHASE 1b.1 DECISION: Regression approach (professional standard)**

    Supports both regression (float targets) and classification ({-1, 0, 1} targets).
    Auto-detects based on target dtype.

    **Regression (RECOMMENDED for Phase 1b+):**
    - Single entry model predicts forward return as float ∈ ℝ
    - Signal via threshold (cfg.signal_threshold, default 0):
      * if pred_return > +threshold → buy (1)
      * if pred_return < -threshold → sell (-1)
      * else → hold (0)
    - No separate exit model (exit signal from return magnitude)
    - Uses full training data (no row dropping, no semantic mixing)
    - Aligns with professional quant methodology (Two Sigma, Man AHL, de Prado)
    - Enables Phase 3 sizing via return magnitude (Kelly fractional ready)

    **Classification (Legacy, not recommended):**
    - Separate entry/exit models for binary classification
    - Entry: buy (1) vs not-buy (0)
    - Exit: sell (1) vs not-sell (0)
    - Note: Phase 1b.1 blocker was semantic issue (negative class mixin {neutral, sell})

    **Phase 0.4: Per-slot features/targets:**
    - exit_feat_cols: None → use feat_cols (default); else use different feature set
    - entry_target_col: column name for entry target; default "target"
    - exit_target_col: column name for exit target; default "target"

    Args:
        train_df: training DataFrame with [symbol, date, target, target_entry, target_exit, *feat_cols]
                  target dtype determines mode: float64 → regression, int8 → classification
        test_df: test DataFrame with [symbol, date, *feat_cols]
        feat_cols: list of entry feature column names
        cfg: ExperimentConfig with model types, params, and signal_threshold
        exit_feat_cols: list of exit feature column names (None → use feat_cols)
        entry_target_col: column name for entry target label
        exit_target_col: column name for exit target label

    Returns:
        (entry_model, exit_model_or_none, signals_df)
        signals_df has [symbol, date, signal, score] where:
          - signal ∈ {-1, 0, 1}
          - score = predicted return (regression) or predict_proba (classification)
    """
    # Phase 0.4: Use per-slot target columns
    if exit_feat_cols is None:
        exit_feat_cols = feat_cols

    train_clean = require_no_nan(train_df, [entry_target_col, *feat_cols], stage="train")
    if train_clean.empty:
        raise ValueError("No training data in fold")

    X_train = train_clean[feat_cols].to_numpy(dtype=np.float32)
    y_full = train_clean[entry_target_col].to_numpy()

    test_use = require_no_nan(test_df, feat_cols + exit_feat_cols, stage="test").copy()
    if test_use.empty:
        raise ValueError(
            f"test fold is empty (test_df={len(test_df)} rows, "
            f"symbols={sorted(test_df['symbol'].unique())})."
        )

    X_test = test_use[feat_cols].to_numpy(dtype=np.float32)
    X_test_exit = (
        test_use[exit_feat_cols].to_numpy(dtype=np.float32)
        if exit_feat_cols != feat_cols
        else X_test
    )

    # P1-E2 xsec_rank_topk (append-only dispatch, EARLY RETURN — every path below is
    # byte-untouched): CROSS-SECTIONAL RANKING head. One LGBMRanker (objective=lambdarank,
    # group = trading DATE, relevance = per-date QUINTILE of the forward-return target)
    # learns the RELATIVE ordering of the universe per day — the objective P1-E1 showed is
    # the one carrying the signal (Rank-IC .036-.043 vs regression .025-.028 on the same
    # demeaned label). Per-date quintiles are demean-invariant (demeaning by date does not
    # change within-date ranks), so the plain forward_return_regression target reproduces
    # E1's fwd20_dm relevance exactly. Emits score-only rows (signal=0) + carried OHLCV;
    # the aggregate membership step (_xsec_rank_membership_signals) builds the buys/sells.
    if cfg.strategy == "xsec_rank_topk":
        import lightgbm as lgb

        params = dict(cfg.entry_model.get("params", {}))
        trs = train_clean.sort_values(["date", "symbol"], kind="mergesort")
        rel_pct = trs[entry_target_col].groupby(trs["date"].to_numpy()).rank(
            pct=True, method="average"
        )
        rel = np.clip((rel_pct * 5).astype(int).clip(upper=4), 0, 4).to_numpy()
        grp = trs.groupby("date", sort=True).size().to_numpy()
        ranker = lgb.LGBMRanker(
            objective="lambdarank", random_state=cfg.seed,
            label_gain=list(range(32)), **params,
        )
        ranker.fit(trs[feat_cols].to_numpy(dtype=np.float32), rel, group=grp)
        signals_df = test_use.copy()
        signals_df["signal"] = np.zeros(len(signals_df), dtype=np.int8)
        signals_df["score"] = ranker.predict(X_test).astype(np.float32)
        keep = ["symbol", "date", "signal", "score"]
        keep += [c for c in ("open", "close", "high", "low", "volume") if c in signals_df.columns]
        if out_models is not None:
            out_models["entry"] = ranker
        return ranker, None, signals_df[keep]

    # Legacy: technical_rules route to rule model
    if cfg.entry_model["type"] == "technical_rules":
        entry_rules = cfg.entry_model.get("rules", {})
        exit_rules = cfg.exit_model.get("rules", {}) if cfg.exit_model.get("enabled", False) else {}
        score_feat = cfg.entry_model.get("score_feature", "macd_line")
        return (
            None,
            None,
            generate_signals_from_technical_rules(
                test_use, entry_rules=entry_rules, exit_rules=exit_rules, score_feature=score_feat
            ),
        )

    # Determine regression vs classification. Forward-looking labels carry NaN tails
    # (dropped by the splitter) so a classifier target like early_wave_v2 arrives here as
    # float {-1,0,1}; dtype alone would mislabel it regression. Keyed on the target TYPE so
    # only the declared classification targets flip — forward_return stays regression as before.
    _entry_ttype = (cfg.entry_target or {}).get("type") or (cfg.target or {}).get("type")
    _is_clf_target = _entry_ttype in _CLASSIFICATION_TARGET_TYPES
    is_regression = np.issubdtype(y_full.dtype, np.floating) and not _is_clf_target
    if _is_clf_target and not np.issubdtype(y_full.dtype, np.integer):
        y_full = y_full.astype(np.int8)

    # Check for dual-component slots (new architecture: Phase 0-3.1)
    entry_has_ml = cfg.entry_model["type"] not in ("none", "rule")
    entry_has_rule = "rule_conditions" in cfg.entry_model
    is_dual_component = entry_has_ml and entry_has_rule

    # Dual-ML regression: an enabled ML exit slot (its own model + target) alongside
    # an ML entry. Distinct from ml_only, which mirrors the entry score for exits.
    exit_is_ml = cfg.exit_model.get("enabled", False) and cfg.exit_model.get("type") not in (
        "none",
        "rule",
    )

    # Dispatch logic
    if cfg.strategy in ("single_ml_action_classifier", "single_ml_action_classifier_exitout"):
        # SINGLE-MODEL ACTION CLASSIFIER (SMAC): ONE multiclass model decides the per-bar
        # action {0=OUT, 1=ENTER, 2=HOLD, 3=EXIT} from a single feature set + single oracle
        # target (action_oracle) — no second head, no recombine. Directly tests whether one
        # coherent model beats the decoupled 2-slot system, whose realized PnL couples the
        # entry/exit heads so neither can be improved in isolation (the "masking" problem).
        # LGBMClassifier auto-detects multiclass from the 4 label values, so the existing
        # entry-model wrapper is reused as-is. Intended for the rules-off sandbox so the
        # MODEL's decision — not the mechanical pullback/overext/trailing rules — drives it.
        ENTER_CLS, EXIT_CLS, OUT_CLS, CUT_CLS = 1, 3, 0, 4
        y_full = y_full.astype(np.int8)
        params = cfg.entry_model.get("params", {}).copy()
        # OUT/HOLD dominate the timeline; up-weight the rare ENTER/EXIT pivots so the model
        # doesn't collapse to "always HOLD/OUT".
        params.setdefault("class_weight", "balanced")
        _emtype = cfg.entry_model["type"]
        if _emtype in ("torch_gru", "torch_lstm"):
            # SEQUENCE path: a recurrent model sees a CAUSAL window of the last `window`
            # bars per symbol (vs the snapshot the LightGBM path sees). Build the 3D
            # windows from train_clean/test_use (which carry symbol+date) so the temporal
            # ordering is correct; the rest of the SMAC decision logic is unchanged.
            from stock_ml.src.models.torch_seq import TorchGRUClassifier, build_seq_windows

            _win = int(params.get("window", 24))
            X_train_seq = build_seq_windows(train_clean, feat_cols, _win)
            X_test_seq = build_seq_windows(test_use, feat_cols, _win)
            entry_model = TorchGRUClassifier(
                params, seed=cfg.seed, rnn=("lstm" if _emtype == "torch_lstm" else "gru")
            )
            entry_model.fit(X_train_seq, y_full)
            exit_model = None
            proba = entry_model.predict_proba(X_test_seq)
            clf = entry_model
        else:
            entry_model = build_entry_model(_emtype, params, seed=cfg.seed)
            entry_model.fit(X_train, y_full)
            exit_model = None
            proba = entry_model.predict_proba(X_test)
            clf = getattr(entry_model, "_clf", None)
        # predict_proba columns align to the model's sorted classes_, which may omit a class
        # absent from a fold's train set — map by class label, never by fixed index.
        classes = list(clf.classes_) if clf is not None else sorted(set(y_full.tolist()))

        def _pcls(c: int) -> np.ndarray:
            if c in classes:
                return proba[:, classes.index(c)]
            return np.zeros(len(proba), dtype=np.float64)

        p_enter = _pcls(ENTER_CLS)
        # Exit conviction = profit-take (EXIT) + loss-cut (CUT); CUT is present only in the
        # 5-class oracle (cut_drawdown>0), else P(CUT)=0 and this reduces to v1.
        p_exit = _pcls(EXIT_CLS) + _pcls(CUT_CLS)
        # entry_threshold / exit_threshold act as P(ENTER) / P(EXIT) cutoffs (the model-side
        # selectivity knob). When None, fall back to the model's own argmax decision.
        argmax_cls = np.asarray(classes)[proba.argmax(axis=1)]
        buy = (
            p_enter >= cfg.entry_threshold
            if cfg.entry_threshold is not None
            else (argmax_cls == ENTER_CLS)
        )
        # State-machine policy option (engine flag smac_exit_on_out): stay IN only while the
        # model says HOLD — exit the moment it LEAVES the HOLD state (predicts OUT or EXIT).
        # v1 sold ONLY on EXIT (an upside peak), so a FAILING trade — which the oracle labels
        # OUT along its down-leg, so the model predicts OUT there — was held until a late peak
        # formed (forensic: losers held 45d to avg -12.8% vs winners 19d/+7.5%). Acting on the
        # model's OWN out prediction cuts losers with the model's existing signal — no rule.
        # Selected by the strategy variant (kept off engine_config, whose strict EngineConfig
        # dataclass rejects unknown keys).
        exit_on_out = cfg.strategy == "single_ml_action_classifier_exitout"
        # EXIT (profit-take peak) and CUT (loss-cut, 5-class oracle) both close the position.
        if cfg.exit_threshold is not None:
            sell = p_exit >= cfg.exit_threshold
        elif exit_on_out:
            sell = np.isin(argmax_cls, [EXIT_CLS, CUT_CLS, OUT_CLS])
        else:
            sell = np.isin(argmax_cls, [EXIT_CLS, CUT_CLS])
        # entry_first priority: a buy outranks a concurrent sell (only matters when flat —
        # the engine ignores buys while a position is open).
        sig = np.where(buy, 1, np.where(sell, -1, 0)).astype(np.int8)
        if cfg.direction == "short":
            sig = (sig * -1).astype(np.int8)
        signals_df = test_use[["symbol", "date"]].copy()
        signals_df["signal"] = sig
        signals_df["score"] = p_enter.astype(np.float32)
        signals_df["exit_score"] = p_exit.astype(np.float32)

    elif isinstance(cfg.engine, dict) and cfg.engine.get("rule_only_no_ml"):
        # RULE-ONLY (no ML fit): reproduce the ml-bypassed champion with ZERO model training.
        # The edge is the recombine causal price gates (upleg entry-gate + downleg/nonbull exit
        # force-gates) plus the engine gates (market gate, pullback fill, overext/trailing) —
        # NOT the ML heads (the champion bypasses them via a very loose entry_threshold). Emit a
        # CONSTANT entry score so the recombine's RAW-entry path (entry_raw_threshold) buys every
        # bar, then its causal price gates select; the constant exit score leaves the dormant
        # z-exit off so exits come purely from the force/engine gates. Deterministic (no seed,
        # no fit). Carry OHLCV so the recombine price gates have their inputs.
        entry_model = None
        exit_model = None
        n = len(test_use)
        signals_df = test_use.copy()
        signals_df["signal"] = np.ones(n, dtype=np.int8)
        signals_df["score"] = np.ones(n, dtype=np.float32)
        signals_df["exit_score"] = np.zeros(n, dtype=np.float32)
        keep = ["symbol", "date", "signal", "score", "exit_score"]
        keep += [c for c in ("open", "close", "high", "low", "volume") if c in signals_df.columns]
        signals_df = signals_df[keep]

    elif is_dual_component and is_regression:
        # NEW: Dual-component mode (ML + Rule filter on ML predictions)
        entry_model = build_regression_model(
            cfg.entry_model["type"], cfg.entry_model.get("params", {}), seed=cfg.seed
        )
        entry_model.fit(X_train, y_full)
        exit_model = None

        pred_returns = entry_model.predict(X_test)

        # Use new generate_slot_signals for ML + rule filter
        from stock_ml.src.signals.core import generate_slot_signals

        signals_array = generate_slot_signals(
            X_test=test_use,
            ml_predictions=pred_returns,
            rule_conditions=cfg.entry_model.get("rule_conditions"),
            rule_logic=cfg.entry_model.get("rule_logic", "AND"),
            signal_threshold=cfg.signal_threshold,
            direction=cfg.direction,
            entry_threshold=cfg.entry_threshold,
            exit_threshold=cfg.exit_threshold,
        )

        signals_df = test_use.copy()
        signals_df["signal"] = signals_array
        signals_df["score"] = pred_returns.astype(np.float32)
        signals_df = signals_df[["symbol", "date", "signal", "score"]]

    elif cfg.model_mode == "ml_only" and is_regression and exit_is_ml:
        # NEW: Dual-ML regression — independent entry & exit regressors, each with its
        # own soft-label target (e.g. zigzag bottom proximity for entry, peak proximity
        # for exit). Entry score drives buys (+1), exit score drives sells (-1). No
        # mirrored threshold, no rule, no hard exit: exit is purely the exit model's call.
        entry_model = build_regression_model(
            cfg.entry_model["type"],
            _params_with_monotone(cfg.entry_model.get("params", {}), feat_cols),
            seed=cfg.seed,
        )
        entry_model.fit(X_train, y_full)

        y_exit_full = train_clean[exit_target_col].to_numpy()
        X_train_exit = X_train if exit_feat_cols == feat_cols else (
            train_clean[exit_feat_cols].to_numpy(dtype=np.float32)
        )
        exit_model = build_regression_model(
            cfg.exit_model["type"],
            _params_with_monotone(cfg.exit_model.get("params", {}), exit_feat_cols),
            seed=cfg.seed,
        )
        exit_model.fit(X_train_exit, y_exit_full)

        pred_entry = entry_model.predict(X_test)
        pred_exit = exit_model.predict(X_test_exit)

        # Optional entry-alpha sign flip (research: test whether the entry score is
        # inverted vs the true edge). Negating pred_entry flips zE in the decoupled
        # recombine, so the buy fires where the entry score was LOW. Keyed off a
        # component param; stripped from the LGBM kwargs by _params_with_monotone.
        if cfg.entry_model.get("params", {}).get("negate_score"):
            pred_entry = -pred_entry

        # entry_threshold / exit_threshold are POSITIVE proximity cutoffs here (both
        # targets are in [0,1]), not the mirrored ±return cutoffs ml_only uses.
        hi = cfg.entry_threshold if cfg.entry_threshold is not None else cfg.signal_threshold
        lo = cfg.exit_threshold if cfg.exit_threshold is not None else cfg.signal_threshold
        signals_array = np.where(
            pred_exit > lo, -1, np.where(pred_entry > hi, 1, 0)
        ).astype(np.int8)
        if cfg.direction == "short":
            signals_array = (signals_array * -1).astype(np.int8)

        signals_df = test_use.copy()
        signals_df["signal"] = signals_array
        signals_df["score"] = pred_entry.astype(np.float32)
        # Keep the exit model's prediction too, so the dual-ML exit chain is
        # chartable. score = entry alpha, exit_score = exit model (downside) pred.
        signals_df["exit_score"] = pred_exit.astype(np.float32)
        # ENSEMBLE: a 2nd entry head on a different target (e.g. reversal). Its z-score feeds
        # a UNION buy in the recombine (buy if EITHER head fires), keeping both entry styles'
        # winners. Trained on the same entry features; rows with a NaN 2nd target are dropped
        # for its fit only (the longer-horizon target has a deeper tail-NaN).
        if entry2_target_col is not None and entry2_target_col in train_clean.columns:
            y2 = train_clean[entry2_target_col].to_numpy()
            # The reversal head can use its OWN feature set (entry2_feat_cols); else share the
            # primary entry features. Mask out rows with a NaN 2nd target OR any NaN 2nd feature
            # (e.g. zigzag features before the first confirmed pivot) for its fit only.
            cols2 = entry2_feat_cols if entry2_feat_cols else feat_cols
            X_tr2 = train_clean[cols2].to_numpy(dtype=np.float32)
            ok2 = ~np.isnan(y2) & np.isfinite(X_tr2).all(axis=1)
            entry_model2 = build_regression_model(
                cfg.entry_model["type"],
                _params_with_monotone(cfg.entry_model.get("params", {}), cols2),
                seed=cfg.seed,
            )
            entry_model2.fit(X_tr2[ok2], y2[ok2])
            X_te2 = (X_test if cols2 == feat_cols
                     else np.nan_to_num(test_use[cols2].to_numpy(dtype=np.float32), nan=0.0))
            signals_df["score2"] = entry_model2.predict(X_te2).astype(np.float32)
            if out_models is not None:
                out_models["entry2"] = entry_model2
        # ENSEMBLE 3rd entry head (score3, e.g. continuation/breakout): a head orthogonal to
        # BOTH the momentum and reversal heads. Shares the primary entry features; fit only on
        # rows with a non-NaN 3rd target. Unioned into the buy via entry3_z_threshold downstream.
        if entry3_target_col is not None and entry3_target_col in train_clean.columns:
            y3 = train_clean[entry3_target_col].to_numpy()
            # The breakout head can use its OWN feature set (entry3_feat_cols); else share primary.
            cols3 = entry3_feat_cols if entry3_feat_cols else feat_cols
            X_tr3 = train_clean[cols3].to_numpy(dtype=np.float32)
            ok3 = ~np.isnan(y3) & np.isfinite(X_tr3).all(axis=1)
            entry_model3 = build_regression_model(
                cfg.entry_model["type"],
                _params_with_monotone(cfg.entry_model.get("params", {}), cols3),
                seed=cfg.seed,
            )
            entry_model3.fit(X_tr3[ok3], y3[ok3])
            X_te3 = (X_test if cols3 == feat_cols
                     else np.nan_to_num(test_use[cols3].to_numpy(dtype=np.float32), nan=0.0))
            signals_df["score3"] = entry_model3.predict(X_te3).astype(np.float32)
            if out_models is not None:
                out_models["entry3"] = entry_model3
        # ENSEMBLE 4th entry head (score4): yet another orthogonal target. Same pattern as score3.
        if entry4_target_col is not None and entry4_target_col in train_clean.columns:
            y4 = train_clean[entry4_target_col].to_numpy()
            cols4 = entry4_feat_cols if entry4_feat_cols else feat_cols
            X_tr4 = train_clean[cols4].to_numpy(dtype=np.float32)
            ok4 = ~np.isnan(y4) & np.isfinite(X_tr4).all(axis=1)
            entry_model4 = build_regression_model(
                cfg.entry_model["type"],
                _params_with_monotone(cfg.entry_model.get("params", {}), cols4),
                seed=cfg.seed,
            )
            entry_model4.fit(X_tr4[ok4], y4[ok4])
            X_te4 = (X_test if cols4 == feat_cols
                     else np.nan_to_num(test_use[cols4].to_numpy(dtype=np.float32), nan=0.0))
            signals_df["score4"] = entry_model4.predict(X_te4).astype(np.float32)
            if out_models is not None:
                out_models["entry4"] = entry_model4
        # ENSEMBLE 5th entry head (score5): another orthogonal winner target. Same pattern.
        if entry5_target_col is not None and entry5_target_col in train_clean.columns:
            y5 = train_clean[entry5_target_col].to_numpy()
            cols5 = entry5_feat_cols if entry5_feat_cols else feat_cols
            X_tr5 = train_clean[cols5].to_numpy(dtype=np.float32)
            ok5 = ~np.isnan(y5) & np.isfinite(X_tr5).all(axis=1)
            entry_model5 = build_regression_model(
                cfg.entry_model["type"],
                _params_with_monotone(cfg.entry_model.get("params", {}), cols5),
                seed=cfg.seed,
            )
            entry_model5.fit(X_tr5[ok5], y5[ok5])
            X_te5 = (X_test if cols5 == feat_cols
                     else np.nan_to_num(test_use[cols5].to_numpy(dtype=np.float32), nan=0.0))
            signals_df["score5"] = entry_model5.predict(X_te5).astype(np.float32)
            if out_models is not None:
                out_models["entry5"] = entry_model5
        # ENSEMBLE 6th entry head (score6): volume-region-balance winner head. Same pattern.
        if entry6_target_col is not None and entry6_target_col in train_clean.columns:
            y6 = train_clean[entry6_target_col].to_numpy()
            cols6 = entry6_feat_cols if entry6_feat_cols else feat_cols
            X_tr6 = train_clean[cols6].to_numpy(dtype=np.float32)
            ok6 = ~np.isnan(y6) & np.isfinite(X_tr6).all(axis=1)
            entry_model6 = build_regression_model(
                cfg.entry_model["type"],
                _params_with_monotone(cfg.entry_model.get("params", {}), cols6),
                seed=cfg.seed,
            )
            entry_model6.fit(X_tr6[ok6], y6[ok6])
            X_te6 = (X_test if cols6 == feat_cols
                     else np.nan_to_num(test_use[cols6].to_numpy(dtype=np.float32), nan=0.0))
            signals_df["score6"] = entry_model6.predict(X_te6).astype(np.float32)
            if out_models is not None:
                out_models["entry6"] = entry_model6
        # ENSEMBLE 2nd exit head (e.g. zigzag pre-peak), unioned into the SELL downstream.
        if exit2_target_col is not None and exit2_target_col in train_clean.columns:
            yx2 = train_clean[exit2_target_col].to_numpy()
            okx2 = ~np.isnan(yx2)
            exit_model2 = build_regression_model(
                cfg.exit_model["type"],
                _params_with_monotone(cfg.exit_model.get("params", {}), exit_feat_cols),
                seed=cfg.seed,
            )
            exit_model2.fit(X_train_exit[okx2], yx2[okx2])
            signals_df["exit_score2"] = exit_model2.predict(X_test_exit).astype(np.float32)
            if out_models is not None:
                out_models["exit2"] = exit_model2
        # Carry close so the aggregate recombine can apply a causal entry-regime gate.
        keep = ["symbol", "date", "signal", "score", "exit_score"]
        if "score2" in signals_df.columns:
            keep.append("score2")
        if "score3" in signals_df.columns:
            keep.append("score3")
        if "score4" in signals_df.columns:
            keep.append("score4")
        if "score5" in signals_df.columns:
            keep.append("score5")
        if "score6" in signals_df.columns:
            keep.append("score6")
        if "exit_score2" in signals_df.columns:
            keep.append("exit_score2")
        # Carry OHLCV (when present) so the aggregate recombine can apply causal price
        # gates — entry-regime gate, exit force-gate, and the reversal-confirm early entry.
        keep += [c for c in ("open", "close", "high", "low", "volume") if c in signals_df.columns]
        signals_df = signals_df[keep]

    elif cfg.model_mode == "ml_only" and is_regression:
        # EXISTING: ML-only regression mode
        entry_model = build_regression_model(
            cfg.entry_model["type"], cfg.entry_model.get("params", {}), seed=cfg.seed
        )
        entry_model.fit(X_train, y_full)
        exit_model = None

        pred_returns = entry_model.predict(X_test)

        signals_df = generate_signals_from_predictions(
            predictions=pred_returns,
            test_df=test_use,
            signal_threshold=cfg.signal_threshold,
            is_regression=True,
            signal_mode=cfg.signal_mode,
            direction=cfg.direction,
            entry_threshold=cfg.entry_threshold,
            exit_threshold=cfg.exit_threshold,
        )

    elif entry_has_rule and not entry_has_ml and is_regression and exit_is_ml:
        # NEW: RULE entry + ML downside-REGRESSION exit. The entry is a static rule
        # mask (no entry ML); the exit is a forward-downside regressor (its own
        # target/features) that SELLS when the predicted drop clears exit_threshold —
        # exactly t015's exit head, but gated by a price rule instead of an ML entry.
        # Combines the dual-ML exit (build_regression_model + exit_threshold) with the
        # rule-entry mask. Keyed on the derived flags (not model_mode) because a
        # rule-entry template infers model_mode='rule_only', which would otherwise be
        # swallowed by the classification branch below and corrupt the float target.
        entry_params = cfg.entry_model.get("params", {}).copy()
        entry_params["feature_cols"] = feat_cols
        entry_params["conditions"] = _normalize_rule_conditions(
            cfg.entry_model.get("rule_conditions"), where="entry_model"
        )
        entry_params["logic"] = cfg.entry_model.get("rule_logic", "AND")
        entry_model = build_entry_model("rule", entry_params, seed=cfg.seed)
        entry_model.fit(X_train, y_full)  # no-op for RuleModel
        entry_mask = entry_model.predict(X_test).astype(bool)

        # Exit regressor: train on the train fold ONLY (walk-forward, no leakage).
        y_exit_full = train_clean[exit_target_col].to_numpy()
        X_train_exit = X_train if exit_feat_cols == feat_cols else (
            train_clean[exit_feat_cols].to_numpy(dtype=np.float32)
        )
        exit_model = build_regression_model(
            cfg.exit_model["type"],
            _params_with_monotone(cfg.exit_model.get("params", {}), exit_feat_cols),
            seed=cfg.seed,
        )
        exit_model.fit(X_train_exit, y_exit_full)
        pred_exit = exit_model.predict(X_test_exit)

        # exit_threshold is the positive downside cutoff (predicted drop magnitude).
        lo = cfg.exit_threshold if cfg.exit_threshold is not None else cfg.signal_threshold
        signals_array = np.where(
            pred_exit > lo, -1, np.where(entry_mask, 1, 0)
        ).astype(np.int8)
        if cfg.direction == "short":
            signals_array = (signals_array * -1).astype(np.int8)

        signals_df = test_use.copy()
        signals_df["signal"] = signals_array
        # No entry alpha (rule entry); surface the exit (downside) prediction so the
        # sell chain is chartable on the model-details page.
        signals_df["score"] = pred_exit.astype(np.float32)
        signals_df["exit_score"] = pred_exit.astype(np.float32)
        signals_df = signals_df[["symbol", "date", "signal", "score", "exit_score"]]

    elif cfg.model_mode == "ml_only" and not is_regression and exit_is_ml:
        # NEW: Dual-ML CLASSIFICATION — independent ML entry & exit CLASSIFIERS on a
        # discrete forward target (e.g. early_wave_v2). The entry classifier learns the BUY
        # class (+1) → buys (+1); the exit classifier learns the SELL class (-1) of its own
        # target → sells (-1). Mirrors the dual-ML REGRESSION branch but for classifiers —
        # the dispatch the engine previously lacked (ml_only had no classification path).
        if exit_feat_cols != feat_cols:
            raise ValueError(
                "ml_only classification dual-ML requires entry and exit to share the same "
                "feature set (signal generation feeds a single X_test to both classifiers)."
            )
        y_full = y_full.astype(np.int8)
        y_entry = (y_full == 1).astype(np.int8)
        entry_model = build_entry_model(
            cfg.entry_model["type"], cfg.entry_model.get("params", {}).copy(), seed=cfg.seed
        )
        entry_model.fit(X_train, y_entry)

        y_exit_full = (
            train_clean[exit_target_col].to_numpy()
            if exit_target_col in train_clean
            else y_full
        )
        y_exit = (y_exit_full == -1).astype(np.int8)
        exit_model = build_exit_model(
            cfg.exit_model["type"], cfg.exit_model.get("params", {}).copy(), seed=cfg.seed
        )
        exit_model.fit(X_train, y_exit)

        # Signals: honor entry_threshold / exit_threshold as BUY / SELL probability cutoffs
        # — the model-side selectivity knob (analog of the entry cascade's alpha gate): a
        # higher entry cutoff trades fewer, higher-conviction names. When a threshold is None
        # it falls back to the model's own argmax (predict == 1), matching the hybrid path.
        # entry_first priority: a buy outranks a concurrent sell.
        def _proba1(model, X):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            return (model.predict(X) == 1).astype(np.float64)

        entry_p = _proba1(entry_model, X_test)
        exit_p = _proba1(exit_model, X_test)
        buy = entry_p >= cfg.entry_threshold if cfg.entry_threshold is not None else (
            entry_model.predict(X_test) == 1
        )
        sell = exit_p >= cfg.exit_threshold if cfg.exit_threshold is not None else (
            exit_model.predict(X_test) == 1
        )
        sig = np.where(buy, 1, np.where(sell, -1, 0)).astype(np.int8)
        if cfg.direction == "short":
            sig = (sig * -1).astype(np.int8)
        signals_df = test_use[["symbol", "date"]].copy()
        signals_df["signal"] = sig
        signals_df["score"] = entry_p.astype(np.float32)
        signals_df["exit_score"] = exit_p.astype(np.float32)

    elif cfg.model_mode == "rule_only" or (
        cfg.model_mode.startswith("hybrid") and not is_regression
    ):
        # EXISTING: Rule-only or hybrid classification mode
        y_full = y_full.astype(np.int8)
        y_entry = (y_full == 1).astype(np.int8)

        entry_params = cfg.entry_model.get("params", {}).copy()
        if cfg.entry_model["type"] == "rule":
            entry_params["feature_cols"] = feat_cols
            # Honor rule_conditions/rule_logic declared at the slot level. Without this
            # the RuleModel sees no conditions and degenerates to "always buy".
            entry_params["conditions"] = _normalize_rule_conditions(
                cfg.entry_model.get("rule_conditions"), where="entry_model"
            )
            entry_params["logic"] = cfg.entry_model.get("rule_logic", "AND")
        entry_model = build_entry_model(cfg.entry_model["type"], entry_params, seed=cfg.seed)
        entry_model.fit(X_train, y_entry)

        exit_model = None
        # rule_only must also honor an enabled rule exit slot: without this the exit
        # component is loaded from the DB but silently ignored (every position exits
        # only via engine max_hold/hard_stop). Hybrid still builds its (ML or rule) exit.
        exit_is_enabled_rule = (
            cfg.exit_model.get("enabled", False)
            and cfg.exit_model.get("type") == "rule"
            and bool(cfg.exit_model.get("rule_conditions"))
        )
        if cfg.model_mode.startswith("hybrid") or exit_is_enabled_rule:
            exit_type = cfg.exit_model.get("type", "lightgbm")
            if exit_type == "rule":
                exit_params = cfg.exit_model.get("params", {}).copy()
                exit_params["feature_cols"] = exit_feat_cols
                exit_params["conditions"] = _normalize_rule_conditions(
                    cfg.exit_model.get("rule_conditions"), where="exit_model"
                )
                exit_params["logic"] = cfg.exit_model.get("rule_logic", "AND")
            else:
                exit_params = cfg.exit_model.get("params", {}).copy()

            # Get exit target label from exit_target_col
            y_exit_full = (
                train_clean[exit_target_col].to_numpy()
                if exit_target_col in train_clean
                else y_full
            )
            y_exit = (y_exit_full == -1).astype(np.int8)

            # Use exit_feat_cols for exit model training if different
            if exit_feat_cols is not feat_cols:
                X_train_exit = train_clean[exit_feat_cols].to_numpy(dtype=np.float32)
            else:
                X_train_exit = X_train

            exit_model = build_exit_model(exit_type, exit_params, seed=cfg.seed)
            exit_model.fit(X_train_exit, y_exit)

        signals_df = generate_signals_from_predictions(
            predictions=None,
            test_df=test_use,
            signal_threshold=cfg.signal_threshold,
            is_regression=False,
            entry_model=entry_model,
            exit_model=exit_model,
            X_test=X_test,
            signal_mode=cfg.signal_mode,
            direction=cfg.direction,
        )

    else:
        raise ValueError(
            f"Invalid model_mode '{cfg.model_mode}' or unsupported combination with "
            f"target type '{cfg.target.get('type')}'. Supported: "
            f"ml_only (regression), rule_only (classification), hybrid_* (classification), "
            f"or dual-component (ML + rule filter on regression)"
        )

    return entry_model, exit_model, signals_df


def predict_slot_signals(
    entry_model: Any,
    exit_model: Any,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    cfg: ExperimentConfig,
    exit_feat_cols: list[str] | None = None,
    entry_ensemble: list[tuple] | None = None,
    exit_ensemble: list[tuple] | None = None,
) -> pd.DataFrame:
    """Predict-only dual-ML signal generation (mirrors ``train_fold``'s dual-ML branch).

    For SERVING: the models are already fitted (loaded from a bundle), so this skips
    fitting and only runs the predict path, producing the per-bar raw frame
    (``score`` = entry pred, ``exit_score`` = exit pred, plus carried OHLCV) that
    ``recombine_signals`` then turns into the final buy/sell decision. The raw
    ``signal`` column reproduces train_fold's threshold rule so the two paths match
    bit-for-bit before recombine (verified in test_predict_slot_matches_train_fold).

    ENSEMBLE heads (optional, mirror train_fold's): the N-head champions (n2_3h_* /
    n2_4h_* / n2_5h_*) carry extra entry heads (entry2..entryN -> score2..scoreN) and
    optional exit heads (exit2.. -> exit_score2..). Pass them generically as:
        entry_ensemble = [(score_col, model, feat_cols|None), ...]
        exit_ensemble  = [(exit_score_col, model, feat_cols|None), ...]
    so the recombine's union buys/sells fire exactly as in the backtest, for ANY head
    count (no per-head wiring). ``feat_cols=None`` shares the primary entry/exit
    features. NaN handling matches train_fold (nan_to_num on a divergent feature set).
    """
    if exit_feat_cols is None:
        exit_feat_cols = feat_cols

    test_use = require_no_nan(test_df, feat_cols + exit_feat_cols, stage="test").copy()
    if test_use.empty:
        raise ValueError("predict_slot_signals: test frame empty after NaN guard")

    X_test = test_use[feat_cols].to_numpy(dtype=np.float32)
    X_test_exit = (
        test_use[exit_feat_cols].to_numpy(dtype=np.float32)
        if exit_feat_cols != feat_cols
        else X_test
    )

    pred_entry = entry_model.predict(X_test)
    pred_exit = exit_model.predict(X_test_exit)
    if cfg.entry_model.get("params", {}).get("negate_score"):
        pred_entry = -pred_entry

    hi = cfg.entry_threshold if cfg.entry_threshold is not None else cfg.signal_threshold
    lo = cfg.exit_threshold if cfg.exit_threshold is not None else cfg.signal_threshold
    signals_array = np.where(pred_exit > lo, -1, np.where(pred_entry > hi, 1, 0)).astype(np.int8)
    if cfg.direction == "short":
        signals_array = (signals_array * -1).astype(np.int8)

    signals_df = test_use.copy()
    signals_df["signal"] = signals_array
    signals_df["score"] = pred_entry.astype(np.float32)
    signals_df["exit_score"] = pred_exit.astype(np.float32)

    # ENSEMBLE heads — same predict path (and NaN handling) as train_fold's dual-ML branch.
    # Generic over head count: each (score_col, model, cols) emits one scoreN column.
    def _X_for(cols: list[str] | None, base: np.ndarray, base_cols: list[str]) -> np.ndarray:
        if not cols or cols == base_cols:
            return base
        return np.nan_to_num(test_use[cols].to_numpy(dtype=np.float32), nan=0.0)

    ensemble_cols: list[str] = []
    for score_col, model, cols in (entry_ensemble or []):
        signals_df[score_col] = model.predict(_X_for(cols, X_test, feat_cols)).astype(np.float32)
        ensemble_cols.append(score_col)
    for score_col, model, cols in (exit_ensemble or []):
        signals_df[score_col] = model.predict(
            _X_for(cols, X_test_exit, exit_feat_cols)
        ).astype(np.float32)
        ensemble_cols.append(score_col)

    keep = ["symbol", "date", "signal", "score", "exit_score"]
    keep += [c for c in ensemble_cols if c in signals_df.columns]
    keep += [c for c in ("open", "close", "high", "low", "volume") if c in signals_df.columns]
    return signals_df[keep]


def _params_with_monotone(params: dict, feat_cols: list[str]) -> dict:
    """Translate a domain-prior ``monotone_map`` into LightGBM ``monotone_constraints``.

    ``params['monotone_map']`` is ``{feature_name: +1|-1}`` — a HARD shape prior forcing
    the head's score to rise (+1) or fall (-1) with that feature, monotonically, even
    where the (noisy) data alone wouldn't. Features absent from the map get 0
    (unconstrained). The resulting list is aligned to ``feat_cols`` (the model sees X as
    a bare array, so the alignment must happen here, where column order is known).

    Used to force the EXIT head to value weakening signals (lower MA slope / ma_align /
    rsi_div / macd_div / HA trend -> higher exit score) regardless of the IC ceiling.
    No-op when no monotone_map is present, so every other template is unchanged.
    """
    # 'negate_score' is a research dispatch flag (handled where pred_entry is computed),
    # not a LightGBM kwarg — always strip it so the model builder never sees it.
    params = {k: v for k, v in params.items() if k != "negate_score"}
    mm = params.get("monotone_map")
    if not mm:
        return params
    out = {k: v for k, v in params.items() if k != "monotone_map"}
    out["monotone_constraints"] = [int(mm.get(c, 0)) for c in feat_cols]
    return out


def _causal_zscore_by_symbol(
    s: pd.Series, group: pd.Series, window: int, min_periods: int
) -> pd.Series:
    """Per-symbol TRAILING rolling z-score (mean/std over the past `window` bars only).

    Causal by construction: ``rolling`` looks strictly backwards (no ``center``), so a
    prediction is normalised against its own symbol's prior predictions and never any
    future value. Used by the dual-ML recombine entry.
    """
    def _cz(x: pd.Series) -> pd.Series:
        m = x.rolling(window, min_periods=min_periods).mean()
        sd = x.rolling(window, min_periods=min_periods).std()
        return (x - m) / (sd + 1e-12)

    return s.groupby(group, sort=False).transform(_cz)


def _causal_leg_dir(close: np.ndarray, pct: float = 0.06) -> np.ndarray:
    """Per-bar CAUSAL running zigzag direction (+1 up-leg, -1 down-leg). Flips only after
    price reverses >=pct from the running extreme — a live zigzag, no look-ahead."""
    n = len(close)
    leg = np.zeros(n, dtype=np.int8)
    if n == 0:
        return leg
    direction = 0
    ext = close[0]
    for i in range(1, n):
        price = close[i]
        if direction >= 0 and price > ext:
            ext = price; direction = 1
        elif direction <= 0 and price < ext:
            ext = price; direction = -1
        elif direction == 1 and price <= ext * (1.0 - pct):
            direction = -1; ext = price
        elif direction == -1 and price >= ext * (1.0 + pct):
            direction = 1; ext = price
        leg[i] = direction
    return leg


def _causal_leg_dir_vol(close: np.ndarray, base_pct: float = 0.12, vmult: float = 0.4) -> np.ndarray:
    """Vol-conditional causal zigzag: the reversal threshold SHRINKS when realized vol is
    high (deep drops predicted -> flip to down-leg / sell EARLIER) and GROWS when vol is low
    (shallow reversals there are usually bounceable whipsaws -> wait). Exploits the one
    learnable exit signal (vol predicts drawdown MAGNITUDE, IC ~0.30) on a DIRECTION-confirmed
    leg, sidestepping the unpredictable-direction problem. pct_i = base*(1 - vmult*(2*rank-1)),
    clipped to [0.5,1.5]*base. rank = causal 120-bar percentile of 20-bar return-vol. Past-only."""
    n = len(close)
    leg = np.zeros(n, dtype=np.int8)
    if n < 25:
        return leg
    cs = pd.Series(close)
    rv = cs.pct_change().rolling(20, min_periods=10).std()
    rank = rv.rolling(120, min_periods=40).rank(pct=True).to_numpy()
    pct_arr = np.where(np.isnan(rank), base_pct,
                       base_pct * np.clip(1.0 - vmult * (2.0 * rank - 1.0), 0.5, 1.5))
    direction = 0
    ext = close[0]
    for i in range(1, n):
        price = close[i]; pct = pct_arr[i]
        if direction >= 0 and price > ext:
            ext = price; direction = 1
        elif direction <= 0 and price < ext:
            ext = price; direction = -1
        elif direction == 1 and price <= ext * (1.0 - pct):
            direction = -1; ext = price
        elif direction == -1 and price >= ext * (1.0 + pct):
            direction = 1; ext = price
        leg[i] = direction
    return leg


def _causal_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI (causal). Used by the entry-regime gate."""
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
    return 100.0 - 100.0 / (1.0 + ru / (rd + 1e-12))


def _entry_gate_mask(df: pd.DataFrame, gate: str) -> np.ndarray:
    """Boolean buy-allow mask from causal OHLCV context (conditional-IC concentrators).

    Tokens (combine with '_'): 'rsi50'/'rsi55' = RSI14 below that level (the real causal
    edge pocket: entry-head IC ~0.08 at RSI<50 vs ~0.03 at 50-70); 'upleg' = causal
    zigzag up-leg only (avoids buying in a confirmed down-leg); 'dd<pct>' (e.g. 'dd15',
    'dd20') = price is at least pct% below its trailing 120-bar high (deep-drawdown /
    V-bottom pocket: forward returns concentrate where dd_from_high is deep but the model
    rarely buys). All strictly past-only.
    """
    if "close" not in df.columns:
        raise ValueError("entry_gate needs a 'close' column carried into the signals frame")
    tokens = set(gate.split("_"))
    allow = np.ones(len(df), dtype=bool)
    for sym, idx in df.groupby("symbol", sort=False).indices.items():
        idx = np.asarray(idx)
        c = df["close"].to_numpy()[idx]
        m = np.ones(len(idx), dtype=bool)
        if "upleg" in tokens:
            m &= _causal_leg_dir(c) > 0
        if "rsi50" in tokens:
            m &= _causal_rsi(c) < 50.0
        if "rsi55" in tokens:
            m &= _causal_rsi(c) < 55.0
        if "frag" in tokens:
            # FRAGILE-SPIKE veto (entry_selection_diag/entry_extension_confirm): within the
            # near-high buy zone, the crash-prone subset is EXTENDED far above the 126d low AND
            # ran up hot (20d) AND vol is elevated. Block that 3-way 'taut spring' — cuts the
            # buy-zone crash rate ~11.6%->9.6% at no cost to fwd return (grinders kept). Causal.
            cs = pd.Series(c)
            rmin = cs.rolling(126, min_periods=60).min().to_numpy()
            ext = c >= rmin * 1.5
            ret20 = (cs / cs.shift(20) - 1.0).to_numpy()
            hot = ret20 >= 0.12
            rv = cs.pct_change().rolling(20, min_periods=20).std()
            rvpct = rv.rolling(120, min_periods=60).rank(pct=True).to_numpy()
            hivol = rvpct >= 0.6
            fragile = ext & hot & hivol
            fragile[np.isnan(rmin) | np.isnan(ret20) | np.isnan(rvpct)] = False  # unknown -> allow
            m &= ~fragile
        if "red" in tokens:
            # buy only into weakness: the entry bar's own day is down (ret1<0). Trade
            # forensics: red-bar entries WR 70% vs hot-thrust entries 54% — the head buys
            # the right names but mistimes the fill onto green spikes.
            cs = pd.Series(c)
            m &= (cs / cs.shift(1) - 1.0 < 0).fillna(False).to_numpy()
        for tok in tokens:
            if tok.startswith("noth") and tok[4:].isdigit():
                # no-thrust: block buying into a hot 5-bar thrust (ret5 > pct%). Removes the
                # low-WR chase pocket (near-high + spike) without the broad rsi/dd volume cut.
                pct = float(tok[4:]) / 100.0
                cs = pd.Series(c)
                ret5 = (cs / cs.shift(5) - 1.0).to_numpy()
                cond = ret5 <= pct
                cond[np.isnan(ret5)] = True
                m &= cond
            if tok.startswith("rsiok") and tok[5:].isdigit():
                # KNIFE-FILTER (research 2026-06-18, shape>point): skip a dip-buy when RSI14 is FALLING
                # sharply over 5 bars (rsi - rsi[5] < -thr) = the net-negative quick-loss knife cohort
                # (q0-4 hold, -133u/fold). rsi_slope_5 (botIC -0.077) separates a knife (rsi still falling)
                # from a bottoming dip (rsi turning up) — which the entry SCORE cannot. Causal.
                thr = float(tok[5:])
                _cs = pd.Series(c); _d = _cs.diff()
                _up = _d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
                _dn = (-_d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
                _rsi = 100.0 - 100.0 / (1.0 + _up / (_dn + 1e-9))
                _slope = (_rsi - _rsi.shift(5)).to_numpy()
                _cond = _slope >= -thr
                _cond[np.isnan(_slope)] = True
                m &= _cond
            if tok.startswith("barpos") and tok[6:].isdigit() and "high" in df.columns and "low" in df.columns:
                # entry bar closes in the upper part of its range (close near the low = weak
                # close = forward drawdown; bar_pos IC +0.40 vs fdd10). Causal (signal bar).
                pct = float(tok[6:]) / 100.0
                hi = df["high"].to_numpy()[idx]; lo = df["low"].to_numpy()[idx]
                rng = np.maximum(hi - lo, 1e-9)
                m &= (c - lo) / rng >= pct
            if tok.startswith("nowick") and tok[6:].isdigit() and "high" in df.columns:
                # upper wick <= pct of the bar range — reject long-upper-wick rejection bars.
                pct = float(tok[6:]) / 100.0
                hi = df["high"].to_numpy()[idx]; lo = df["low"].to_numpy()[idx]
                rng = np.maximum(hi - lo, 1e-9)
                m &= (hi - c) / rng <= pct
            if tok.startswith("rnup") and tok[4:].isdigit() and "high" in df.columns and "low" in df.columns:
                # PRE-ENTRY RUN-UP gate (2026-06-20, pre_entry_runup forensic): require the prior-50-bar
                # run-up (rolling-50 high / rolling-50 low - 1) >= pct. Skips the low-runup "unproven"
                # cohort (Q1 ~14% runup = pnl +0.018 near-worthless); pre-runup ranks realized pnl
                # MONOTONICALLY (Q1->Q4 +0.018->+0.130) where the entry score is flat. Causal (50 bars).
                pct = float(tok[4:]) / 100.0
                hi = df["high"].to_numpy()[idx]; lo = df["low"].to_numpy()[idx]
                hh = pd.Series(hi).rolling(50, min_periods=20).max().to_numpy()
                ll = pd.Series(lo).rolling(50, min_periods=20).min().to_numpy()
                ru = hh / np.where(ll <= 0, np.nan, ll) - 1.0
                cond = ru >= pct
                cond[np.isnan(ru)] = True  # warmup -> allow
                m &= cond
            if tok.startswith("abovema") and tok[7:].isdigit():
                # ENTRY-HEALTH gate: require close >= SMA(N) at the signal bar. young_loser_hunt
                # — below-MA entries are the falling-knife young losers (above_ma10 IC +0.21 vs
                # pnl, +0.26 in non-uptrends); adds beyond the trend/upleg filter. Causal.
                win = int(tok[7:])
                ma = pd.Series(c).rolling(win, min_periods=win).mean().to_numpy()
                cond = c >= ma
                cond[np.isnan(ma)] = True  # warmup -> allow
                m &= cond
            if tok.startswith("abovelow") and tok[8:].isdigit():
                # ENTRY-LOCATION gate: require close >= (1+pct) * trailing 20-bar LOW — skip the
                # weak near-20d-low cohort (entry_selectivity_forensic: dist-above-20low monotone,
                # near-low WR 43% / +1.1% vs far-above 76% / +6.3%). Precise version of the
                # abovema20 proxy (which won +0.7). Causal (past+current low only).
                pct = float(tok[8:]) / 100.0
                base = df["low"].to_numpy()[idx] if "low" in df.columns else c
                rmin = pd.Series(base).rolling(20, min_periods=20).min().to_numpy()
                cond = c >= rmin * (1.0 + pct)
                cond[np.isnan(rmin)] = True  # warmup -> allow
                m &= cond
            if tok.startswith("dd") and tok[2:].isdigit():
                pct = float(tok[2:]) / 100.0
                # trailing 120-bar high (causal: current + past only). Require price to sit
                # >=pct below it -> only buy genuine deep pullbacks, not bars near the high.
                rmax = pd.Series(c).rolling(120, min_periods=20).max().to_numpy()
                cond = c <= rmax * (1.0 - pct)
                cond[np.isnan(rmax)] = False  # warmup: drawdown unknown -> no buy
                m &= cond
            if tok.startswith("ext") and tok[3:].isdigit():
                # EXTENSION cap: block buying when price is >= pct% above its trailing 126-bar
                # low (over-extended from the base). The single dominant crash/grind separator
                # (crash dist_lo126 1.57 vs grind 1.45). Causal; warmup -> allow.
                pct = float(tok[3:]) / 100.0
                rmin = pd.Series(c).rolling(126, min_periods=60).min().to_numpy()
                cond = c < rmin * (1.0 + pct)
                cond[np.isnan(rmin)] = True  # warmup: extension unknown -> allow
                m &= cond
            if tok.startswith("fragrank") and tok[8:].isdigit():
                # CAUSAL combined-fragility rank veto: block buys whose trailing-120 percentile
                # of (extension-from-126d-low + 20d-runup + 20d-vol) >= pct. Targets exactly the
                # top-fragility quintile (entry_extension_confirm: that bucket is BOTH high-crash
                # ~19.6% AND low-return +2.3%), unlike the fixed-threshold 'frag'/'ext' which also
                # cut high-return extended names. Per-symbol, strictly past+current -> no leak.
                pct = float(tok[8:]) / 100.0
                cs = pd.Series(c)
                ext = cs / cs.rolling(126, min_periods=60).min()
                ret20 = cs / cs.shift(20) - 1.0
                rv = cs.pct_change().rolling(20, min_periods=20).std()
                ep = ext.rolling(120, min_periods=60).rank(pct=True)
                rp = ret20.rolling(120, min_periods=60).rank(pct=True)
                vp = rv.rolling(120, min_periods=60).rank(pct=True)
                frag = (ep + rp + vp) / 3.0
                cond = (frag < pct).to_numpy()
                cond[frag.isna().to_numpy()] = True  # warmup -> allow
                m &= cond
            if tok.startswith("er") and len(tok) > 2 and tok[2:].isdigit():
                # REGIME gate (2026-06-19, diag_regime_split): require Kaufman efficiency ratio
                # ER20 >= pct (TRENDING regime). The entry/exit heads are more predictive in trend
                # (score IC 0.26 trend vs 0.16 range; breakout head FLIPS sign neg in range). Test
                # whether confining entries to the trend regime improves composite. pct=token/100.
                thr = float(tok[2:]) / 100.0
                _cs = pd.Series(c)
                _er = ((_cs - _cs.shift(20)).abs()
                       / (_cs.diff().abs().rolling(20, min_periods=20).sum() + 1e-9)).to_numpy()
                _cond = _er >= thr
                _cond[np.isnan(_er)] = True  # warmup -> allow
                m &= _cond
        allow[idx] = m
    # CROSS-SECTIONAL MARKET-BREADTH regime gate (entry_regime_forensic: entries in the
    # weakest-breadth quintile earn ppb 0.28% / WR 54% vs 0.93% / 70% in the strongest;
    # the entry head IGNORES regime — corr(score,pnl)~0.02. The weak-breadth trades are
    # per-bar DILUTIVE, so removing them raises per-bar+PF — composite-aligned). Token
    # 'breadthq<P>': allow a buy only when today's universe breadth (% of names above
    # their own 50-bar SMA) >= its trailing-252 P-th percentile. Applied AFTER the per-symbol
    # loop (which assigns allow[idx]=m). Causal: each day uses only that day's cross-section
    # + the past breadth series for the percentile.
    breadth_tok = next((t for t in tokens if t.startswith("breadthq")), None)
    if breadth_tok is not None:
        if "date" not in df.columns:
            raise ValueError("entry_gate 'breadthq' needs a 'date' column in the signals frame")
        p = float(breadth_tok[len("breadthq"):]) / 100.0
        ma50 = df.groupby("symbol", sort=False)["close"].transform(
            lambda x: x.rolling(50, min_periods=50).mean()).to_numpy()
        cl = df["close"].to_numpy()
        above = np.where(np.isnan(ma50), np.nan, (cl > ma50).astype(float))
        bd = pd.Series(above, index=df["date"].to_numpy()).groupby(level=0).mean().sort_index()
        thr = bd.rolling(252, min_periods=60).quantile(p)
        ok = (bd >= thr)
        ok[thr.isna()] = True  # warmup: regime unknown -> allow
        allow &= df["date"].map(ok.to_dict()).fillna(True).to_numpy().astype(bool)
    # CROSS-SECTIONAL crash-risk veto (entry_lever_diagnostic): big-loser entries cluster in the
    # high cross-sectional realized-vol names WITH weak momentum (csrvol 0.75 vs 0.62, ret20 lower).
    # 'crashx<pct>' blocks buys whose per-DATE crash rank (rank(rvol) − rank(ret20), high = volatile
    # AND weak) is in the top (1-pct) — distinct from 'frag' (volatile AND HOT). Causal per bar.
    crashx_tok = next((t for t in tokens if t.startswith("crashx") and t[6:].isdigit()), None)
    if crashx_tok is not None:
        pct = float(crashx_tok[6:]) / 100.0
        w = df[["symbol", "date", "close"]].copy().sort_values(["symbol", "date"])
        w["rvol"] = w.groupby("symbol")["close"].transform(
            lambda s: s.pct_change().rolling(20, min_periods=20).std())
        w["ret20"] = w.groupby("symbol")["close"].transform(lambda s: s / s.shift(20) - 1.0)
        w["vr"] = w.groupby("date")["rvol"].rank(pct=True)
        w["rr"] = w.groupby("date")["ret20"].rank(pct=True)
        w["crash"] = w["vr"] - w["rr"]
        w["cr"] = w.groupby("date")["crash"].rank(pct=True)
        cr = w["cr"].reindex(df.index)
        allow &= ((cr < pct) | cr.isna()).to_numpy()
    return allow


@functools.lru_cache(maxsize=8)
def _load_regime_index(symbol: str, duck: str = "market_data/market.duckdb") -> pd.Series:
    """Daily close of a real market index/futures (e.g. VN30F1M) from the OHLCV store, for use as
    a regime detector — genuinely-new information vs the traded-universe EW proxy. Cached."""
    import duckdb
    con = duckdb.connect(duck, read_only=True)
    d = con.execute(
        "SELECT date, close FROM ohlcv WHERE symbol=? AND timeframe='1D' ORDER BY date", [symbol]
    ).fetchdf()
    con.close()
    s = d.set_index(pd.to_datetime(d["date"]))["close"].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index()


_BREADTH_CACHE: dict = {}


def _load_market_breadth(metric: str = "pct_above_ma50", ma_win: int = 50,
                         duck: str = "market_data/market.duckdb") -> pd.Series:
    """ORTHOGONAL market-breadth from the FULL OHLCV universe (488 names, NOT just the 61 traded)
    — genuinely-new info the per-symbol/61-EW features never see. Causal, cached.
    pct_above_ma50/200: fraction of all symbols above their N-bar MA each date (regime health).
    adv_pct: fraction of symbols up that day. Breadth at entry separates the champ's signal-LOSERS
    (~0.26-0.36) from WINNERS (~0.33-0.47), corr_w_pnl +0.155..+0.179, and collapses in the 2022
    mdd cluster (pct_above_ma50 0.263 vs 2021 0.548)."""
    key = (metric, ma_win)
    if key in _BREADTH_CACHE:
        return _BREADTH_CACHE[key]
    import duckdb
    con = duckdb.connect(duck, read_only=True)
    oh = con.execute("SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' "
                     "ORDER BY symbol, date").fetchdf()
    con.close()
    oh["date"] = pd.to_datetime(oh["date"])
    piv = oh.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    # A symbol may only cast a breadth vote on a date where it actually TRADES. A NaN price
    # (not-yet-listed / delisted / halted) compares as ``NaN > ma == False`` in pandas — a
    # *finite* False that silently counts as a bearish vote AND inflates the denominator,
    # biasing breadth DOWNWARD (worst in early history when many names are unlisted: up to
    # -0.195 on 2018-01-11 with 124/488 phantom votes). Mask on a valid price (and valid MA)
    # so only genuinely-trading symbols vote. Feeds entry_ensemble3 breadth_features and the
    # exit_force_gate_lowbreadth threshold, so the down-bias falsely fires the low-breadth veto.
    valid = piv.notna()
    if metric == "adv_pct":
        ind = (piv.pct_change() > 0) & valid
    else:
        ma = piv.rolling(ma_win, min_periods=ma_win).mean()
        valid = valid & ma.notna()
        ind = (piv > ma) & valid
    out = ind.sum(axis=1) / valid.sum(axis=1).clip(lower=1)
    _BREADTH_CACHE[key] = out
    return out


_XSEC_CACHE: dict = {}


def _load_xsec_features(metrics: list[str], duck: str = "market_data/market.duckdb") -> pd.DataFrame:
    """Per-symbol CROSS-SECTIONAL features from the FULL universe (this stock's momentum RANK vs ALL
    symbols + the TREND of that rank = leadership rotation / accumulation-vs-distribution at the
    market tier). The strongest per-symbol signal found (cs_rank_trend IC_winner +0.17 vs the champ's
    win/loss; absolute price feats |IC|<=0.05). Long-format [symbol, date, xsec_*]. Causal, cached.
    Metrics: cs_rank20/cs_rank60 (N-day-return percentile vs all), cs_rank_trend (10d change of
    cs_rank20), cs_rank_trend_long (20d change of cs_rank60)."""
    key = tuple(sorted(metrics))
    if key in _XSEC_CACHE:
        return _XSEC_CACHE[key]
    import duckdb
    con = duckdb.connect(duck, read_only=True)
    oh = con.execute("SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' "
                     "ORDER BY symbol, date").fetchdf()
    con.close()
    oh["date"] = pd.to_datetime(oh["date"])
    piv = oh.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    r20 = piv.pct_change(20).rank(axis=1, pct=True)
    r60 = piv.pct_change(60).rank(axis=1, pct=True)
    series = {"cs_rank20": r20, "cs_rank60": r60,
              "cs_rank_trend": r20 - r20.shift(10), "cs_rank_trend_long": r60 - r60.shift(20)}
    # RS-vs-MARKET (vs VNINDEX, 2026-06-21): the strongest per-trade separator found (corr +0.16; strong-
    # RS dips win 72%% vs weak 52%%). RS line = close/VNINDEX; slope/vs-trend/dist-to-high. Causal.
    # Unified with engine._load_vnindex (ENGINE_UPGRADE §1.1): one reader, one fail-loud semantics.
    # Under MARKET_CONTEXT_REQUIRED an absent source RAISES there; else returns None → rs_ skipped
    # (a downstream KeyError for a requested rs_ metric preserves the prior fail-if-requested behavior).
    if any(m.startswith("rs_") for m in metrics):
        from stock_ml.src.backtest.engine import _load_vnindex
        _vni = _load_vnindex()
        if _vni is not None:
            _vni = _vni.copy()
            _vni.index = _vni.index.normalize()
            _vni_a = pd.Series(piv.index.normalize().map(_vni).values, index=piv.index)
            _rs = piv.div(_vni_a.values, axis=0)
            series["rs_sl5"] = _rs / _rs.shift(5) - 1.0
            series["rs_sl20"] = _rs / _rs.shift(20) - 1.0
            series["rs_vsma"] = _rs / _rs.rolling(50, min_periods=20).mean() - 1.0
            series["rs_nh"] = _rs / _rs.rolling(60, min_periods=20).max() - 1.0
    long = None
    for m in metrics:
        s = series[m].stack().rename(f"xsec_{m}")
        s.index.names = ["date", "symbol"]
        long = s.to_frame() if long is None else long.join(s, how="outer")
    long = long.reset_index()
    _XSEC_CACHE[key] = long
    return long


def _rs_market_mask(df: pd.DataFrame, feature: str = "rs_vsma", threshold: float = 0.0) -> np.ndarray:
    """Per-row True where the stock's RS-vs-VNINDEX (`feature`) is WEAK (< threshold) — a LAGGARD whose
    dip is more likely a knife. The RS-vs-market SELECTION strategy suppresses buys here (trade only
    LEADER dips: strong-RS dips win 72%% vs weak 52%%). Causal (RS up to the bar)."""
    long = _load_xsec_features([feature])
    col = f"xsec_{feature}"
    m = df[["symbol", "date"]].merge(long[["symbol", "date", col]], on=["symbol", "date"], how="left")
    return (m[col].to_numpy() < threshold)


def _rs_drop_mask(df: pd.DataFrame, drop: float = 0.12, lookback: int = 10,
                  metric: str = "cs_rank20") -> np.ndarray:
    """Per-row force-SELL where the symbol's CROSS-SECTIONAL RS rank fell more than `drop` over
    `lookback` bars — RS ROLLOVER = the stock is losing relative strength = round-tripping. Exit
    forensic (t1844): giveback trades' RS goes peak 0.73 -> exit 0.54 (rolls over) while WINNERS'
    RS holds peak 0.85 -> exit 0.80; IC(RS-change, giveback) = -0.19. The REALIZABILITY signal
    that RS (rejected on ENTRY as momentum-redundant) provides on the EXIT side. Causal."""
    long = _load_xsec_features([metric]).copy()
    col = f"xsec_{metric}"
    long = long.sort_values(["symbol", "date"])
    long["__fire"] = (long.groupby("symbol")[col].diff(lookback) < -float(drop))
    merged = df[["symbol", "date"]].merge(long[["symbol", "date", "__fire"]], on=["symbol", "date"], how="left")
    return merged["__fire"].fillna(False).to_numpy().astype(bool)


def _market_breadth_mask(df: pd.DataFrame, metric: str, threshold: float, ma_win: int = 50,
                         mode: str = "level", z_lookback: int = 60) -> np.ndarray:
    """Per-row bool: True where broad-market breadth is WEAK (suppress BUYS here). 'level' =
    breadth < threshold; 'zscore' = causal z of breadth < threshold (adapts across regimes)."""
    br = _load_market_breadth(metric, ma_win)
    if mode == "zscore":
        mu = br.rolling(252, min_periods=z_lookback).mean()
        sd = br.rolling(252, min_periods=z_lookback).std()
        sig = ((br - mu) / sd.replace(0, np.nan)) < threshold
    else:
        sig = br < threshold
    sig = sig.fillna(False)
    key = {d.strftime("%Y-%m-%d"): bool(v) for d, v in sig.items()}
    return df["date"].map(
        lambda x: key.get(pd.Timestamp(x).strftime("%Y-%m-%d"), False)
    ).to_numpy().astype(bool)


def _vn30_regime_mask(df: pd.DataFrame, index_symbol: str = "VN30F1M",
                      mode: str = "downtrend", ma_short: int = 20, ma_long: int = 50,
                      dd_thresh: float = 0.10) -> np.ndarray:
    """Per-row bool: True where the REAL market index (VN30F1M futures = large-cap index, genuinely
    new info vs the equal-weight 488-univ breadth) is in a risk-OFF regime. The validated winning
    pattern is regime-cluster EXIT tightening; this is the real-index analog of the EW breadth-
    collapse gate (forensic vn30_regime_1875: the 2022 catastrophe cluster coincides with VN30 in
    confirmed downtrend ~45% of hold-time vs 9-22% other years). 'downtrend' = close < ma_short <
    ma_long (a confirmed large-cap downtrend, not noise); 'drawdown' = index drawdown-from-peak
    <= -dd_thresh. Causal (MA/peak up to the bar). Maps onto each row's date."""
    lvl = _load_regime_index(index_symbol)
    if mode == "drawdown":
        sig = (lvl / lvl.cummax() - 1.0) <= -float(dd_thresh)
    else:
        ms = lvl.rolling(ma_short, min_periods=ma_short).mean()
        ml = lvl.rolling(ma_long, min_periods=ma_long).mean()
        sig = (lvl < ms) & (ms < ml)
    sig = sig.fillna(False)
    key = {d.strftime("%Y-%m-%d"): bool(v) for d, v in sig.items()}
    return df["date"].map(
        lambda x: key.get(pd.Timestamp(x).strftime("%Y-%m-%d"), False)
    ).to_numpy().astype(bool)


def _market_nonbull_mask(df: pd.DataFrame, ma_win: int = 50, persist: int = 3,
                         index_symbol: str | None = None) -> np.ndarray:
    """Per-row bool: True where the market is NOT bull. STICKY (the jumpy MA50+slope version cut
    bull winners during normal pullbacks): nonbull only after the index sits below its `ma_win`-bar
    MA for `persist` CONSECUTIVE days. The regime index is the traded-universe EW proxy by default,
    or a REAL market index (`index_symbol`, e.g. VN30F1M) — new info, ~19% different. Causal."""
    if index_symbol:
        lvl = _load_regime_index(index_symbol)
    else:
        piv = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        mret = piv.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
        lvl = (1.0 + mret.fillna(0.0)).cumprod()
    ma = lvl.rolling(ma_win, min_periods=ma_win).mean()
    below = (lvl < ma)
    nonbull = (below.rolling(persist, min_periods=persist).sum() >= persist)
    nonbull = nonbull.where(ma.notna(), True)  # warmup -> nonbull (apply the tight cut)
    if index_symbol:
        # index has its own calendar -> ffill across all days, then map df dates by string key
        nb = nonbull.astype(float)
        nb.index = pd.to_datetime(nb.index).normalize()
        nb = nb[~nb.index.duplicated(keep="last")].sort_index()
        nb = nb.reindex(pd.date_range(nb.index.min(), nb.index.max(), freq="D")).ffill()
        key = {d.strftime("%Y-%m-%d"): bool(v) for d, v in nb.items() if v == v}
        return df["date"].map(
            lambda x: key.get(pd.Timestamp(x).strftime("%Y-%m-%d"), True)
        ).to_numpy().astype(bool)
    nonbull_by_date = nonbull.astype(bool).to_dict()
    return df["date"].map(nonbull_by_date).fillna(True).to_numpy().astype(bool)


def _market_bull_mask(df: pd.DataFrame, ma_win: int = 50, persist: int = 3,
                      index_symbol: str | None = None,
                      crash_dd: float | None = None) -> np.ndarray:
    """Per-row bool: True where the market is in a SUSTAINED bull — the regime index sits
    ABOVE its `ma_win`-bar MA for `persist` CONSECUTIVE days (the bullish mirror of
    _market_nonbull_mask, same sticky/causal idiom). Used by downleg_skip_bull to release
    the downleg tail-cut only in confirmed uptrends. Warmup -> NOT bull (keep the cut).
    CRASH-BRAKE (`crash_dd`, default None = off, old behavior byte-identical): the sticky
    MA test lags FAST crashes (2026-02-27 peak -> -10.6% in 6 sessions, mask stayed bull
    until 03-09). When the index's drawdown from its rolling 20-bar peak (past-only,
    causal) exceeds `crash_dd`, bull is FORCED False regardless of the MA condition, so
    the downleg tail-cut re-arms immediately."""
    if index_symbol:
        lvl = _load_regime_index(index_symbol)
    else:
        piv = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        mret = piv.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
        lvl = (1.0 + mret.fillna(0.0)).cumprod()
    ma = lvl.rolling(ma_win, min_periods=ma_win).mean()
    above = (lvl > ma)
    bull = (above.rolling(persist, min_periods=persist).sum() >= persist)
    bull = bull.where(ma.notna(), False)  # warmup -> not bull (keep the tight cut)
    if crash_dd is not None:
        # drawdown vs rolling 20-bar peak (window includes only the last 20 bars -> causal)
        peak = lvl.rolling(20, min_periods=1).max()
        crash = (lvl / peak - 1.0) <= -float(crash_dd)
        bull = bull & ~crash
    if index_symbol:
        bb = bull.astype(float)
        bb.index = pd.to_datetime(bb.index).normalize()
        bb = bb[~bb.index.duplicated(keep="last")].sort_index()
        bb = bb.reindex(pd.date_range(bb.index.min(), bb.index.max(), freq="D")).ffill()
        key = {d.strftime("%Y-%m-%d"): bool(v) for d, v in bb.items() if v == v}
        return df["date"].map(
            lambda x: key.get(pd.Timestamp(x).strftime("%Y-%m-%d"), False)
        ).to_numpy().astype(bool)
    bull_by_date = bull.astype(bool).to_dict()
    return df["date"].map(bull_by_date).fillna(False).to_numpy().astype(bool)


def _wilder_rsi_np(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Causal Wilder RSI on a 1-symbol close array (NaN until bar n)."""
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = np.full(len(close), np.nan); rd = np.full(len(close), np.nan)
    if len(close) <= n:
        return ru
    ru[n] = up[1:n + 1].mean(); rd[n] = dn[1:n + 1].mean()
    for i in range(n + 1, len(close)):
        ru[i] = (ru[i - 1] * (n - 1) + up[i]) / n
        rd[i] = (rd[i - 1] * (n - 1) + dn[i]) / n
    rs = ru / np.where(rd == 0, np.nan, rd)
    return 100 - 100 / (1 + rs)


def _force_healthy_mask(c: np.ndarray, suppress: str) -> np.ndarray:
    """Causal 'healthy uptrend context' mask: where True, the SOFT force-sell tokens
    (belowma/bear3/machist/redbar) are suppressed because a dip here is a bounceable
    pullback inside a wave, not a trend break (forcegate_forensic: belowma20p3 sells
    while ABOVE MA50 bounce +5.9%/10d 62% of the time; hold-instead +5.8u). Tokens
    AND-combined: 'abovema<N>' close>SMA(N); 'rsi<K>' RSI14>K. The downleg deep-reversal
    backstop is NEVER suppressed (it stays the tail protection). Strictly past-only."""
    cs = pd.Series(c)
    healthy = np.ones(len(c), dtype=bool)
    for tok in suppress.split("_"):
        if tok.startswith("abovema"):
            n = int(tok[len("abovema"):])
            ma = cs.rolling(n, min_periods=n).mean().to_numpy()
            cond = c > ma
            cond[np.isnan(ma)] = False  # warmup: trend unknown -> not healthy -> allow sell
            healthy &= cond
        elif tok.startswith("rsi"):
            k = float(tok[len("rsi"):])
            r = _wilder_rsi_np(c, 14)
            cond = r > k
            cond[np.isnan(r)] = False
            healthy &= cond
    return healthy


def _exit_force_mask(df: pd.DataFrame, gate: str, suppress: str | None = None) -> np.ndarray:
    """Boolean force-SELL mask from causal OHLCV context. 'downleg' = a confirmed
    zigzag down-leg (price already reversed >=pct from the high): keep selling through
    the decline, fixing the exit head's silence on the right slope of a wave.

    `suppress` (optional) gates only the SOFT tokens (belowma/bear3/machist/redbar) —
    NOT the downleg backstop — off in a healthy uptrend context (see
    _force_healthy_mask), so the gate stops dumping bounceable wave-pullbacks while
    still cutting real breakdowns and keeping the deep-reversal tail protection."""
    if "close" not in df.columns:
        raise ValueError("exit_force_gate needs a 'close' column carried into the signals frame")
    toks = gate.split("_")
    has_open = "open" in df.columns
    force = np.zeros(len(df), dtype=bool)
    for sym, idx in df.groupby("symbol", sort=False).indices.items():
        idx = np.asarray(idx)
        c = df["close"].to_numpy()[idx]
        cs = pd.Series(c)
        backstop = np.zeros(len(idx), dtype=bool)  # downleg deep reversal — ALWAYS fires
        soft = np.zeros(len(idx), dtype=bool)        # belowma/bear3/etc — suppressible
        for tok in toks:
            if tok.startswith("dlvol"):
                # VOL-CONDITIONAL downleg: 'dlvol12' = base 12% reversal, threshold scaled by
                # realized-vol rank (high vol -> sell earlier ~8%, low vol -> wait ~16%).
                suf = tok[len("dlvol"):]
                base = (float(suf) / 100.0) if suf else 0.12
                backstop |= _causal_leg_dir_vol(c, base) < 0
            elif tok.startswith("downleg"):
                # 'downleg' = 6% reversal (default); 'downleg12' = 12% — deeper = only force
                # a sell on a real reversal, not on every shallow pullback inside an uptrend.
                suf = tok[len("downleg"):]
                pct = (float(suf) / 100.0) if suf else 0.06
                backstop |= _causal_leg_dir(c, pct) < 0
            elif tok in ("machist", "macdhist"):
                # MACD histogram < 0 (12/26/9, causal EWM) — momentum has turned down.
                macd = cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean()
                hist = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()
                soft |= hist < 0.0
            elif tok.startswith(("belowma", "bma")):
                # Close below the trailing N-bar SMA — below trend. 'belowma20'/'bma20'
                # = 20-bar (short-term, fires on shallow dips); 'belowma50' = 50-bar
                # (slower, holds through shakeouts that stay above the 50-SMA, forcing
                # the exit only on a real trend break). Optional 'p<K>' suffix
                # ('belowma20p3') requires K CONSECUTIVE bars below the SMA before
                # firing — vetoes a transient shakeout dip (1..K-1 bars then bounce,
                # the sell-rebuy-higher whipsaw) while still exiting fast on a sustained
                # break. All causal (current+past bars only).
                suf = tok[len("belowma"):] if tok.startswith("belowma") else tok[len("bma"):]
                persist = 1
                if "p" in suf:
                    win_s, p_s = suf.split("p", 1)
                    win = int(win_s) if win_s else 20
                    persist = int(p_s) if p_s else 1
                else:
                    win = int(suf) if suf else 20
                ma = cs.rolling(win, min_periods=win).mean().to_numpy()
                below = c < ma
                below[np.isnan(ma)] = False  # warmup: trend unknown -> don't force
                if persist > 1:
                    cond = (pd.Series(below).rolling(persist, min_periods=persist).sum()
                            .to_numpy() >= persist)
                else:
                    cond = below
                soft |= cond
            elif tok.startswith("mabreak"):
                # TREND-BREAK discriminator: force-sell only when close is below MA<short> for
                # <persist> consecutive bars AND below MA<long> (longer trend also broken). A dip
                # below the short MA that stays ABOVE the long MA (uptrend intact) is HELD to capture
                # the bounce; only a real break below BOTH forces the sell — aims for the high-pnl of
                # patient exits WITHOUT the mdd of holding true breakdowns. Format 'mabreak20p3m50'
                # = below MA20 for 3 consecutive bars AND below MA50. Causal (current+past only).
                body = tok[len("mabreak"):]
                short_part, long_s = body.split("m", 1)
                if "p" in short_part:
                    win_s, p_s = short_part.split("p", 1)
                    win = int(win_s) if win_s else 20
                    persist = int(p_s) if p_s else 1
                else:
                    win = int(short_part) if short_part else 20
                    persist = 1
                longw = int(long_s) if long_s else 50
                ma_s = cs.rolling(win, min_periods=win).mean().to_numpy()
                ma_l = cs.rolling(longw, min_periods=longw).mean().to_numpy()
                below_s = c < ma_s
                if persist > 1:
                    below_s = (pd.Series(below_s).rolling(persist, min_periods=persist).sum()
                               .to_numpy() >= persist)
                cond = below_s & (c < ma_l)
                cond[np.isnan(ma_s) | np.isnan(ma_l)] = False  # warmup -> don't force
                soft |= cond
            elif tok == "redbar":
                # Red candle (close < open) — a down bar. Needs the open column.
                if not has_open:
                    raise ValueError("exit_force_gate 'redbar' needs an 'open' column in signals")
                o = df["open"].to_numpy()[idx]
                soft |= c < o
            elif tok == "bear3" or tok.startswith("bear3p"):
                # Confirmed bearish bar = MACD hist<0 AND close<MA20 AND close<open
                # (all three at once) — far more selective than any single condition.
                # Optional 'p<K>' suffix ('bear3p2') requires K CONSECUTIVE confirmed
                # bearish bars before firing — vetoes a one-bar shakeout, holding through
                # to capture the bounce (raises total PnL). Causal (current+past bars).
                if not has_open:
                    raise ValueError("exit_force_gate 'bear3' needs an 'open' column in signals")
                persist = 1
                if tok.startswith("bear3p"):
                    p_s = tok[len("bear3p"):]
                    persist = int(p_s) if p_s else 1
                macd = cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean()
                hist = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()
                ma20 = cs.rolling(20, min_periods=20).mean().to_numpy()
                o = df["open"].to_numpy()[idx]
                cond = (hist < 0.0) & (c < ma20) & (c < o)
                cond[np.isnan(ma20)] = False
                if persist > 1:
                    cond = (pd.Series(cond).rolling(persist, min_periods=persist).sum()
                            .to_numpy() >= persist)
                soft |= cond
        if suppress is not None:
            soft = soft & ~_force_healthy_mask(c, suppress)
        force[idx] = backstop | soft
    return force


def _exit_gate_mask(df: pd.DataFrame, gate: str) -> np.ndarray:
    """Boolean sell-ALLOW mask from causal OHLCV context — restrict the ML-driven sell to
    where the exit head is actually predictive (conditional_exit_ic.py: forward downside is
    leading-predictable by vol MAGNITUDE only NEAR the high — atr->fwd_dd skip-3 = -0.247 at
    dist-to-20d-high > -3%, vs -0.20 overall; far from the high downside is regime-driven and
    the head is noise). Gating the ML sell here removes the low-precision off-high sells that
    amputate the fat tail, WITHOUT touching the downleg force-gate (which stays the ungated
    backstop for the right-slope decline). Tokens: 'nh<pct>' = within pct% of the trailing
    20-bar high (e.g. 'nh3','nh5','nh7'); 'up' = in a causal zigzag up-leg. Strictly past-only."""
    if "close" not in df.columns:
        raise ValueError("exit_gate needs a 'close' column carried into the signals frame")
    tokens = set(gate.split("_"))
    allow = np.ones(len(df), dtype=bool)
    for _sym, idx in df.groupby("symbol", sort=False).indices.items():
        idx = np.asarray(idx)
        c = df["close"].to_numpy()[idx]
        m = np.ones(len(idx), dtype=bool)
        if "up" in tokens:
            m &= _causal_leg_dir(c) > 0
        for tok in tokens:
            if tok.startswith("nh") and tok[2:].isdigit():
                pct = float(tok[2:]) / 100.0
                rmax = pd.Series(c).rolling(20, min_periods=10).max().to_numpy()
                cond = c >= rmax * (1.0 - pct)
                cond[np.isnan(rmax)] = False  # warmup: proximity unknown -> don't allow ML sell
                m &= cond
            # CONSOLIDATION/DISTRIBUTION gate ('cons<N>'): allow the ML sell only when the recent tape
            # is SIDEWAYS — consolidation_score = count of last-10 bars with daily range < 2% of close
            # >= N. exit_struct_discriminate.py (2026-06-18): at signal-exits, consolidation_score is the
            # STRONGEST top-vs-premature separator (sep +0.82sd, IC -postcont +0.276) — real tops are in a
            # distribution/sideways zone, the SOLD_THEN_RAN premature exits fire in a CLEAN-uptrend pullback
            # (low consolidation). Blocking the ML sell there holds the healthy pullback (the mechanical
            # overext/trailing exits stay the ungated downside backstop). Causal. Needs high/low carried.
            elif tok.startswith("cons") and tok[4:].isdigit() and "high" in df.columns and "low" in df.columns:
                n_thr = float(tok[4:])
                # optional window/range overrides via sibling tokens: 'w<bars>' and 'r<range*1000>'
                # (e.g. 'cons2_w20_r25' = count over 20 bars of daily-range<2.5%, >=2). Default 10/2.0%.
                win, rng = 10, 0.02
                for tt in tokens:
                    if tt.startswith("w") and tt[1:].isdigit():
                        win = int(tt[1:])
                    elif tt.startswith("r") and tt[1:].isdigit():
                        rng = int(tt[1:]) / 1000.0
                h = df["high"].to_numpy()[idx]; lo = df["low"].to_numpy()[idx]
                hlp = (h - lo) / np.where(c > 0, c, np.nan)
                consol = pd.Series((hlp < rng).astype(float)).rolling(win, min_periods=3).sum().to_numpy()
                cond = consol >= n_thr
                cond[np.isnan(consol)] = False  # warmup: regime unknown -> don't allow ML sell
                m &= cond
            # ROUNDING-TOP gate ('roundtop', user's "giá giảm kiểu đỉnh tròn"): allow the ML sell only
            # where a dome is forming — the MA20 ascent is DECELERATING (3-bar accel of the 3-bar slope
            # <= 0) AND price is still near the 20-bar high (>=92%), i.e. momentum fading at the top, not
            # a fresh breakout or a deep down-leg. Causal. Combined with AND vs other tokens.
            # EMA-RIBBON CONTRACTION gate ('ribbon', user's "co giãn EMA nhiều chu kỳ"): allow the ML sell
            # when the EMA5-vs-EMA50 ribbon WIDTH is CONTRACTING over 5 bars (Delta<=0) = the uptrend is
            # stalling (a top tell distinct from sideways consolidation). feat_top_probe.py: sep +0.36 (the
            # best of the new multi-period family, below consolidation +0.82). Causal (ewm).
            elif tok == "ribbon":
                cs = pd.Series(c)
                rib = ((cs.ewm(span=5, adjust=False).mean() - cs.ewm(span=50, adjust=False).mean())
                       / np.where(c > 0, c, np.nan))
                chg = (rib - rib.shift(5)).to_numpy()
                cond = chg <= 0
                cond[np.isnan(chg)] = False
                m &= cond
            elif tok == "roundtop":
                cs = pd.Series(c)
                sma20 = cs.rolling(20, min_periods=10).mean()
                slope = sma20 - sma20.shift(3)
                accel = (slope - slope.shift(3)).to_numpy()
                rmax = cs.rolling(20, min_periods=10).max().to_numpy()
                near = c >= rmax * 0.92
                cond = (accel <= 0) & near
                cond[np.isnan(accel) | np.isnan(rmax)] = False
                m &= cond
        # OR-override: 'orext<N>' ALSO allows the ML sell on a parabolic BLOW-OFF top (close >= MA20*
        # (1+N/100)) — the sharp-extended top the consolidation gate BLOCKS (a blow-off is NOT sideways,
        # so cons holds it and gives it back). Lets the gate fire on BOTH a sideways-distribution top
        # (cons) AND a blow-off (orext). Applied as OR after the AND-tokens. Causal.
        for tok in tokens:
            if tok.startswith("orext") and tok[5:].isdigit():
                ma20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
                ext_cond = c >= ma20 * (1.0 + int(tok[5:]) / 100.0)
                ext_cond[np.isnan(ma20)] = False
                m = m | ext_cond
        allow[idx] = m
    return allow


def _reversal_early_mask(
    df: pd.DataFrame, dd_thresh: float = -0.08, min_confirms: int = 2
) -> np.ndarray:
    """Causal reversal-CONFIRMATION early-entry mask. A real V-bottom shows a reversal
    AFTER a dip (up2 / higher-low / reclaim-MA5 / volume-thrust); falling knives keep
    dropping with none. Fires where price is >= ``dd_thresh`` below its trailing 20-bar
    high (a confirmed cheap dip) AND >= ``min_confirms`` of the 4 causal confirms are on.
    Lets the decoupled entry enter EARLIER/cheaper on a confirmed dip than the slower
    momentum z-score would (validated: +4.49% median better entry on shared winners).
    All strictly past-only. Needs close (+ high/low/volume when carried; missing OHLCV
    columns simply drop those confirms). See project_reversal_confirm_gate."""
    if "close" not in df.columns:
        raise ValueError("reversal early-entry needs a 'close' column in the signals frame")
    n = len(df)
    confirms = np.zeros(n, dtype=np.int8)
    cheap = np.zeros(n, dtype=bool)
    has_lv = "low" in df.columns
    has_vol = "volume" in df.columns
    for _sym, idx in df.groupby("symbol", sort=False).indices.items():
        idx = np.asarray(idx)
        c = pd.Series(df["close"].to_numpy()[idx])
        c1, c2 = c.shift(1), c.shift(2)
        sma5 = c.rolling(5, min_periods=2).mean()
        nc = ((c > c1) & (c1 > c2)).astype(np.int8)                       # up2
        nc = nc + ((c >= sma5) & (c1 < sma5.shift(1))).astype(np.int8)    # reclaim5
        if has_lv:
            low = pd.Series(df["low"].to_numpy()[idx])
            nc = nc + (low > low.shift(1)).astype(np.int8)                # higher low
        if has_vol:
            vol = pd.Series(df["volume"].to_numpy()[idx])
            volavg = vol.rolling(20, min_periods=5).mean()
            nc = nc + ((c > c1) & (vol > 1.3 * volavg)).astype(np.int8)   # volume thrust
        dist20 = c / c.rolling(20, min_periods=5).max() - 1.0
        confirms[idx] = nc.to_numpy()
        cheap[idx] = (dist20 <= dd_thresh).fillna(False).to_numpy()
    return (confirms >= min_confirms) & cheap


def _reversal_top_mask(
    df: pd.DataFrame,
    near_thresh: float = 0.03,
    min_confirms: int = 2,
    run_thresh: float = 0.15,
) -> np.ndarray:
    """Causal reversal-DOWN confirmation exit mask — the mirror of ``_reversal_early_mask``.
    A real top ROLLS OVER at the crest of a rally (down2 / lose-MA5 / lower-high /
    distribution volume); a healthy uptrend keeps rising with none. Fires where price is
    (a) still within ``near_thresh`` of its trailing 20-bar HIGH (at/just off the crest —
    the mirror of the entry gate's 'within dd of the 20-bar low'), (b) >= ``run_thresh``
    above its trailing 20-bar low (a genuine rally preceded it, not chop), AND (c) shows
    >= ``min_confirms`` of the 4 causal down-confirms. Sells AT the roll-over instead of
    waiting for the reactive ``downleg`` force-gate's already-fallen -12%. All strictly
    past-only (no look-ahead, so unlike a zigzag-peak target it cannot leak across the
    train/test gap). Needs close (+ high/low/volume when carried; missing OHLCV columns
    simply drop those confirms). See project_reversal_confirm_gate."""
    if "close" not in df.columns:
        raise ValueError("reversal top-exit needs a 'close' column in the signals frame")
    n = len(df)
    confirms = np.zeros(n, dtype=np.int8)
    at_top = np.zeros(n, dtype=bool)
    has_hi = "high" in df.columns
    has_vol = "volume" in df.columns
    for _sym, idx in df.groupby("symbol", sort=False).indices.items():
        idx = np.asarray(idx)
        c = pd.Series(df["close"].to_numpy()[idx])
        c1, c2 = c.shift(1), c.shift(2)
        sma5 = c.rolling(5, min_periods=2).mean()
        nc = ((c < c1) & (c1 < c2)).astype(np.int8)                       # down2
        nc = nc + ((c < sma5) & (c1 >= sma5.shift(1))).astype(np.int8)    # lose MA5
        if has_hi:
            high = pd.Series(df["high"].to_numpy()[idx])
            nc = nc + (high < high.shift(1)).astype(np.int8)             # lower high
        if has_vol:
            vol = pd.Series(df["volume"].to_numpy()[idx])
            volavg = vol.rolling(20, min_periods=5).mean()
            nc = nc + ((c < c1) & (vol > 1.3 * volavg)).astype(np.int8)  # distribution
        roll_max = c.rolling(20, min_periods=5).max()
        roll_min = c.rolling(20, min_periods=5).min()
        dist20high = c / roll_max - 1.0    # <= 0; 0 == at the 20-bar high
        dist20low = c / roll_min - 1.0     # >= 0; size of the up-leg
        zone = (dist20high >= -near_thresh) & (dist20low >= run_thresh)
        confirms[idx] = nc.to_numpy()
        at_top[idx] = zone.fillna(False).to_numpy()
    return (confirms >= min_confirms) & at_top


def _recombine_dual_ml_signals(
    signals: pd.DataFrame,
    sum_threshold: float,
    exit_threshold: float,
    window: int,
    min_periods: int,
    direction: str,
    lower_threshold: float | None = None,
    use_raw_exit: bool = True,
    exit_z_threshold: float | None = None,
    exit_z_ema_span: int | None = None,
    entry_z_threshold: float | None = None,
    entry_z_low_threshold: float | None = None,
    entry_z_ema_span: int | None = None,
    ema_span: int | None = None,
    entry_gate: str | None = None,
    entry_xs_mom_pct: float | None = None,
    exit_gate: str | None = None,
    entry_raw_threshold: float | None = None,
    exit_force_gate: str | None = None,
    exit_force_gate_nonbull: str | None = None,
    exit_force_gate_lowbreadth: dict | None = None,
    exit_force_gate_vn30: dict | None = None,
    exit_rs_drop: dict | None = None,
    exit_force_suppress: str | None = None,
    nonbull_ma_win: int = 50,
    nonbull_persist: int = 3,
    entry_skip_nonbull_persist: int = 0,
    regime_index_symbol: str | None = None,
    early_entry: dict | None = None,
    top_exit: dict | None = None,
    entry_zX_floor: dict | None = None,
    entry_rollover_exit: dict | None = None,
    entry2_z_threshold: float | None = None,
    exit2_z_threshold: float | None = None,
    downleg_skip_bull: dict | None = None,
    entry3_z_threshold: float | None = None,
    entry4_z_threshold: float | None = None,
    entry5_z_threshold: float | None = None,
    entry6_z_threshold: float | None = None,
    entry2_norm: str | None = None,
    entry3_norm: str | None = None,
    entry4_norm: str | None = None,
    entry5_norm: str | None = None,
    entry6_norm: str | None = None,
    entry_breadth_gate: dict | None = None,
    entry_head_csrank_gate: dict | None = None,
    entry_rs_gate: dict | None = None,
) -> pd.DataFrame:
    """Recombine the two dual-ML heads at the AGGREGATE (post-walk-forward) level.

    entry = causal_z(entry_pred) + causal_z(exit_pred) > sum_threshold
    exit  = (use_raw_exit AND exit_pred > exit_threshold)              # raw downside head
            OR (lower_threshold is not None AND z-sum < lower_threshold)  # symmetric band
            OR (exit_z_threshold is not None AND z(exit_pred) > exit_z_threshold)  # rising-risk band
    Sell takes precedence over buy (same as the prototype).

    ``exit_z_threshold`` decouples the sell from the entry z-sum: it fires when the exit
    head's OWN drawdown prediction is unusually HIGH vs its trailing window (risk rising,
    typically near a top) — the opposite trigger to ``lower_threshold`` (which sells once
    risk has fallen / the move has calmed). Used by ``regression_dual_ml_recombine_xrise``.

    The z-score spans the whole concatenated test-prediction series per symbol, which is
    why it must run here (after all folds) and not inside per-fold ``train_fold`` (one
    fold holds <1 year, too short for the 252-bar window). All look-back, no look-ahead.

    ``entry_z_threshold`` decouples the BUY from the composite z-sum: when set, the entry
    fires on the ENTRY head's OWN z alone (``z(entry) > entry_z_threshold``), independent
    of the exit head. Paired with ``exit_z_threshold`` this makes entry-conviction and
    exit-risk two SEPARATE signals instead of one composite — used by the ``*_decoupled``
    strategies. Defaults to ``None`` = canonical z-sum buy.

    ``lower_threshold`` / ``use_raw_exit`` / ``entry_z_threshold`` default to the original
    behaviour (raw-exit only, no lower band, z-sum buy) so the canonical
    ``regression_dual_ml_recombine`` strategy is byte-for-byte unchanged; the two-sided
    z-exit, rising-risk and decoupled variants pass them explicitly.
    """
    if "score" not in signals.columns or "exit_score" not in signals.columns:
        raise ValueError(
            "dual-ML recombine needs 'score' (entry pred) and 'exit_score' (exit pred) "
            "columns — the dual-ML dispatch branch must run first."
        )
    df = signals.sort_values(["symbol", "date"]).reset_index(drop=True)
    zE = _causal_zscore_by_symbol(df["score"], df["symbol"], window, min_periods)
    zX = _causal_zscore_by_symbol(df["exit_score"], df["symbol"], window, min_periods)
    z_combined = zE + zX
    if ema_span is not None:
        # Causal EMA-smooth the combined z-sum per symbol (denoise; ~halves whipsaws).
        # ewm looks strictly backwards, so no look-ahead.
        z_combined = z_combined.groupby(df["symbol"], sort=False).transform(
            lambda s: s.ewm(span=ema_span, adjust=False).mean()
        )
    combined = z_combined.to_numpy()
    pred_exit = df["exit_score"].to_numpy()
    sell = np.zeros(len(df), dtype=bool)
    if use_raw_exit:
        sell |= pred_exit > exit_threshold
    if lower_threshold is not None:
        sell |= combined < lower_threshold
    if exit_z_threshold is not None:
        zX_eval = zX
        if exit_z_ema_span is not None:
            # Causal EMA-smooth z(exit) per symbol before the rising-risk band — fewer
            # false-alarm sells (each z(exit) blip alone won't trip it). Strictly backward.
            zX_eval = zX.groupby(df["symbol"], sort=False).transform(
                lambda s: s.ewm(span=exit_z_ema_span, adjust=False).mean()
            )
        sell |= zX_eval.to_numpy() > exit_z_threshold
    # Exit-context gate (causal): restrict the ML-driven sells above to the OHLCV context
    # where the exit head is predictive (near-high). Applied BEFORE the force-gates so the
    # downleg backstop stays ungated. Removes low-precision off-high ML sells (tail-saving).
    if exit_gate:
        sell = sell & _exit_gate_mask(df, exit_gate)
    # Buy signal. Default: the entry z-sum (zE+zX) over sum_threshold — the canonical
    # composite trigger. ``entry_z_threshold`` decouples it: buy on the ENTRY head's OWN
    # z alone (zE), independent of the exit head — used by the *_decoupled strategies so
    # entry conviction and exit risk are two separate signals, not one composite.
    if entry_raw_threshold is not None:
        # RAW entry: buy on the entry head's own prediction, no z-normalization. Only
        # meaningful for a bounded/interpretable label (triple-barrier P(profit) in [0,1]):
        # an absolute, fixed, exit-independent cutoff with no trailing-window drift.
        buy = df["score"].to_numpy() > entry_raw_threshold
    elif entry_z_threshold is not None:
        zE_eval = zE
        if entry_z_ema_span is not None:
            # Causal EMA-smooth z(entry) per symbol before the buy band (same denoise
            # rationale as the exit side; strictly backward, no look-ahead).
            zE_eval = zE.groupby(df["symbol"], sort=False).transform(
                lambda s: s.ewm(span=entry_z_ema_span, adjust=False).mean()
            )
        buy = zE_eval.to_numpy() > entry_z_threshold
        # Two-sided (U-shaped) entry: ALSO buy the extreme-LOW z(entry) tail. Diagnostic
        # (forward-return-by-decile) showed fwd return is U-shaped in the entry score —
        # the lowest decile (deep washout, ~-12% from the 20d-high) bounces +2.4%/20d, a
        # mean-reversion edge the monotonic continuation buy (top band only) throws away.
        # OFF by default (None) so every existing decoupled run is byte-identical.
        if entry_z_low_threshold is not None:
            buy = buy | (zE_eval.to_numpy() < entry_z_low_threshold)
    else:
        buy = combined > sum_threshold
    # ENSEMBLE union buy: a 2nd entry head (score2, e.g. reversal-trained) fires an independent
    # buy when its causal z exceeds entry2_z_threshold. The union keeps BOTH heads' winners
    # (momentum continuation + dip reversal) instead of the single-head trade-off. zE2 is
    # z-normalized per symbol like zE. See project_champion958_loss_forensics.
    # Capture the reversal-head buy SEPARATELY (don't union yet): the dip/V-bottom head buys
    # BELOW trend by nature, so it must BYPASS the momentum-head gates (upleg_abovema20 / xsec /
    # nonbull) that follow — unioning it here would let those gates kill it. Unioned at the very
    # end after all momentum gates (still subject to sell-precedence = downleg knife-veto).
    # An ensemble head fires when its score crosses a threshold. norm='zscore' (default, legacy) =
    # per-symbol causal z (high vs the symbol's OWN history); norm='csrank' = CROSS-SECTIONAL rank
    # vs ALL symbols that day (high vs the whole tape). Mask forensic: per-symbol z FLIPS the
    # outcome signal of the momentum/mfe heads (raw IC +0.05 vs winner -> z IC -0.03); csrank
    # preserves it (lo->hi quintile WR 22%->37%, per-bar 0.16->0.25). thr is a z-score for zscore,
    # a [0,1] percentile for csrank. breakout's z (+0.18) beats its raw so keep it zscore.
    def _ens_buy(scol, thr, norm):
        if thr is None or scol not in df.columns:
            return np.zeros(len(df), dtype=bool)
        if norm == "csrank":
            sc = df.groupby("date")[scol].rank(pct=True).to_numpy()
        else:
            sc = _causal_zscore_by_symbol(df[scol], df["symbol"], window, min_periods).to_numpy()
        return sc > float(thr)
    rev_buy = _ens_buy("score2", entry2_z_threshold, entry2_norm)
    brk_buy = _ens_buy("score3", entry3_z_threshold, entry3_norm)
    brk_buy2 = _ens_buy("score4", entry4_z_threshold, entry4_norm)
    brk_buy3 = _ens_buy("score5", entry5_z_threshold, entry5_norm)
    brk_buy4 = _ens_buy("score6", entry6_z_threshold, entry6_norm)
    # Entry-regime gate (causal): restrict buys to the OHLCV context where the entry head
    # is actually predictive (per the conditional-IC analysis). Sells are never gated.
    if entry_gate:
        buy = buy & _entry_gate_mask(df, entry_gate)
    # CROSS-SECTIONAL momentum gate (a DIFFERENT axis than the per-symbol head): xsec_probe —
    # buying the names that are relatively WEAK vs peers TODAY underperforms (WR 0.52 bottom
    # quintile vs 0.74 top, IC +0.13 vs pnl). Require the symbol's same-day 20d-momentum
    # percentile-rank across the universe >= entry_xs_mom_pct. Causal (same-bar cross-section,
    # like a daily relative-strength screen). None = off (byte-identical).
    if entry_xs_mom_pct is not None:
        cs = df[["symbol", "date", "close"]].copy()
        cs["_pos"] = np.arange(len(cs))
        cs = cs.sort_values(["symbol", "date"])
        cs["mom20"] = cs.groupby("symbol")["close"].transform(lambda c: c / c.shift(20) - 1.0)
        cs["xs"] = cs.groupby("date")["mom20"].rank(pct=True)
        cs = cs.sort_values("_pos")
        allow = cs["xs"].to_numpy() >= float(entry_xs_mom_pct)
        allow[np.isnan(cs["mom20"].to_numpy())] = True  # warmup -> allow
        buy = buy & allow
    # Scoped zX floor on the MAIN buy: in the deep-drawdown zone the decoupled zE-only buy
    # spends the single lot on low-reward/risk knives (zX absent from the buy → it buys the
    # +1.6% knife and the +3.5% bottom indiscriminately). Require the exit head's reward_risk
    # z >= floor THERE so the slot stays free for the high-zX bottom instead of a knife.
    # Scoped to deep dd so the decoupling elsewhere is byte-unchanged. See
    # project_champion958_loss_forensics (root cause: buy blind to zX + single-lot occupancy).
    if entry_zX_floor:
        zfloor = float(entry_zX_floor.get("floor", 1.5))
        deep_dd = float(entry_zX_floor.get("deep_dd", -0.12))
        c_arr = df["close"].to_numpy()
        deep = np.zeros(len(df), dtype=bool)
        for _sym, _idx in df.groupby("symbol", sort=False).indices.items():
            cs = pd.Series(c_arr[_idx])
            rmax = cs.rolling(20, min_periods=20).max().to_numpy()
            dmask = (c_arr[_idx] / rmax - 1.0) <= deep_dd
            dmask[np.isnan(rmax)] = False
            deep[_idx] = dmask
        buy = buy & (~deep | (zX.to_numpy() >= zfloor))
    # Early reversal-confirm entry (causal): enter EARLIER/cheaper on a confirmed dip of a
    # name the entry head doesn't dislike (zE above a floor below the normal buy bar). With
    # single-lot + entry_first this REPLACES the later momentum entry on dip-then-recover
    # names (better fill, +4.49% median) and adds confirmed-dip entries on quality names;
    # the zE floor drops persistent decliners (the unique-loser knives). See
    # project_reversal_confirm_gate.
    if early_entry:
        em = _reversal_early_mask(
            df,
            dd_thresh=float(early_entry.get("dd", -0.08)),
            min_confirms=int(early_entry.get("min_confirms", 2)),
        )
        floor = early_entry.get("zE_floor")
        if floor is not None:
            em = em & (zE.to_numpy() > float(floor))
        # Grade the confirmed V-bottom by the EXIT head's reward_risk z (zX). The decoupled
        # buy fires on zE alone, walling the exit head off the buy — but in the post-crash
        # zone zX has ~2x the forward-IC of zE and is near-independent (corr~0.27): it splits
        # the real bottoms (+3.5%/+9.7%) from the knives the buy takes blindly (+1.6%). Keep
        # only confirmed dips the exit head reads as high reward/risk. See
        # project_champion958_loss_forensics (root cause: buy is blind to zX).
        zx_floor = early_entry.get("zX_floor")
        if zx_floor is not None:
            em = em & (zX.to_numpy() >= float(zx_floor))
        # Require the causal down-leg to have RELEASED (leg dir >= 0, price reclaimed up_pct
        # off the trough). Confirmed bottoms still IN an active 12% downleg rebound only +1.4%
        # and collide with the downleg force-sell; those that have reclaimed run +9.7%. Off by
        # default so existing early-entry runs are byte-identical.
        if early_entry.get("require_up"):
            up_pct = float(early_entry.get("up_pct", 0.12))
            leg = np.zeros(len(df), dtype=np.int8)
            for _sym, _idx in df.groupby("symbol", sort=False).indices.items():
                leg[_idx] = _causal_leg_dir(df["close"].to_numpy()[_idx], up_pct)
            em = em & (leg >= 0)
        buy = buy | em
        _em_vbottom = em
    else:
        _em_vbottom = None
    # Exit force-gate (causal): force a SELL in a confirmed down-leg so the position is
    # closed through the whole decline (fixes the exit head going silent on the right slope).
    if exit_force_gate:
        fm = _exit_force_mask(df, exit_force_gate, exit_force_suppress)
        # Regime-conditional TAIL-CUT RELEASE (downleg_skip_bull): in a confirmed SUSTAINED
        # bull (index above its ma_win-MA for persist days) suppress the downleg force-sell so
        # trend-year runners are not cut, while it stays active in chop/bear where the mdd
        # lives. Per-year forensic: removing downleg gains +5.6u in 2021 bull but costs
        # -3.3u/2022 + doubles mdd; conditioning aims to keep the gain, not the risk. OFF by
        # default (None) so the champion is byte-for-byte unchanged.
        if downleg_skip_bull:
            _cd = downleg_skip_bull.get("crash_dd")
            fm = fm & ~_market_bull_mask(
                df, int(downleg_skip_bull.get("ma_win", 50)),
                int(downleg_skip_bull.get("persist", 3)), regime_index_symbol,
                crash_dd=(float(_cd) if _cd is not None else None))
        sell = sell | fm
    # Regime-conditional TIGHT force-gate: apply this (eager) gate ONLY where the market is NOT
    # bull, so chop/bear positions are cut fast while bull positions hold through the wave. The
    # loose exit_force_gate (e.g. downleg10) still applies in all regimes as the deep backstop.
    if exit_force_gate_nonbull:
        _nb = _exit_force_mask(df, exit_force_gate_nonbull)
        # Q1 fix: suppress the SOFT nonbull cut (belowma) in a per-symbol HEALTHY uptrend context
        # (above MA / RSI). Forensic: signal-exits in an UP-leg sell shallow pullbacks too early —
        # MFE 5.6% vs realized 1.6% (4% left on table), 43.9% rally >5% after. exit_force_suppress
        # tokens (abovemaN/rsiK) hold those instead. Off unless exit_force_suppress set.
        if exit_force_suppress:
            for _sym, _idx in df.groupby("symbol", sort=False).indices.items():
                _ix = np.asarray(_idx)
                _h = _force_healthy_mask(df["close"].to_numpy()[_ix], exit_force_suppress)
                _nb[_ix] = _nb[_ix] & ~_h
        sell = sell | (_nb & _market_nonbull_mask(df, nonbull_ma_win, nonbull_persist, regime_index_symbol))
    # Regime-conditional TIGHT force-gate keyed on the ORTHOGONAL 488-universe breadth: in a
    # broad-market breadth COLLAPSE (the 2022 mdd cluster, pct_above_ma50 0.263 vs 2021 0.548),
    # apply a tighter cut (e.g. downleg8) to bleed the bear-cluster losers faster. Breadth flags
    # the 2022 leg the 61-EW proxy under-reads. Off by default.
    if exit_force_gate_lowbreadth:
        _g = exit_force_gate_lowbreadth
        sell = sell | (_exit_force_mask(df, _g.get("gate", "downleg8"))
                       & _market_breadth_mask(df, _g.get("metric", "pct_above_ma50"),
                                              float(_g.get("threshold", 0.35)),
                                              int(_g.get("ma_win", 50)),
                                              _g.get("mode", "level"),
                                              int(_g.get("z_lookback", 60))))
    # REAL-INDEX (VN30F1M) regime-cluster EXIT gate: real-index analog of the EW breadth-collapse
    # cut — apply a tighter force-gate (e.g. downleg6) only while the actual VN30 large-cap index is
    # in a risk-OFF regime (confirmed downtrend / drawdown). New info vs the 488-EW proxy. Off by
    # default. A/B both standalone and STACKED on exit_force_gate_lowbreadth (they may overlap).
    if exit_force_gate_vn30:
        _v = exit_force_gate_vn30
        sell = sell | (_exit_force_mask(df, _v.get("gate", "downleg6"))
                       & _vn30_regime_mask(df, _v.get("index_symbol", "VN30F1M"),
                                           _v.get("mode", "downtrend"),
                                           int(_v.get("ma_short", 20)),
                                           int(_v.get("ma_long", 50)),
                                           float(_v.get("dd_thresh", 0.10))))
    # RS-ROLLOVER protective exit: force-sell a HELD trade when its cross-sectional RS rank rolls
    # over (drops > `drop` over `lookback`) — the REALIZABILITY signal (rejected on entry as
    # momentum-redundant, strong on exit: IC(RS-change, giveback) -0.19; givebacks roll over while
    # winners hold RS). Targets the deadband-giveback leak. Off by default.
    if exit_rs_drop:
        _r = exit_rs_drop
        sell = sell | _rs_drop_mask(df, float(_r.get("drop", 0.12)),
                                    int(_r.get("lookback", 10)), _r.get("metric", "cs_rank20"))
    # Downleg VETO for fresh V-bottom entries. 73% of the confirmed V-bottom buys land in an
    # active 12% downleg, so the same-bar downleg force-SELL (sell wins over buy in `sig`)
    # CANCELS the buy at the source — the verified blocker on the bottom alpha. Where a
    # V-bottom buy fires (and for `downleg_veto_bars` bars after, the grace period through the
    # bounce), suppress the downleg sell so the entry survives. zX-graded so we don't disarm
    # the backstop on a true knife. See project_champion958_loss_forensics.
    if early_entry and _em_vbottom is not None and int(early_entry.get("downleg_veto_bars", 0)) > 0:
        vb_bars = int(early_entry.get("downleg_veto_bars", 0))
        veto = _em_vbottom.copy()
        for _sym, _idx in df.groupby("symbol", sort=False).indices.items():
            v = veto[_idx].copy()
            ext = v.copy()
            for k in range(1, vb_bars + 1):
                ext[k:] |= v[:-k]
            veto[_idx] = ext
        sell = sell & ~veto
    # Reversal-confirm TOP exit (causal): sell EARLIER on a confirmed roll-over at an
    # extended top (>= up above the trailing 20-bar low + >= min_confirms down-confirms),
    # instead of waiting for the reactive downleg gate's already-fallen -12%. Optional
    # zX_floor keeps only tops where the exit head's risk z is not deeply negative (don't
    # force-sell a name the model still reads as safe). See project_reversal_confirm_gate.
    if top_exit:
        tm = _reversal_top_mask(
            df,
            near_thresh=float(top_exit.get("near", 0.03)),
            min_confirms=int(top_exit.get("min_confirms", 2)),
            run_thresh=float(top_exit.get("run", 0.15)),
        )
        zx_floor = top_exit.get("zX_floor")
        if zx_floor is not None:
            tm = tm & (zX.to_numpy() > float(zx_floor))
        sell = sell | tm
    # Entry-momentum ROLLOVER sell (causal, the actionable mirror of the bottom blind spot):
    # the decoupled sell uses zX + downleg only, but at a top the entry-momentum z (zE) PEAKS
    # ~1 bar BEFORE the price (leads 58%) while zX LAGS by +2 bars and downleg waits for -12%.
    # Sell when zE rolls over hard (3-bar zE drop >= `drop`) near a 20-bar high after a real
    # run-up — flagging the top ~3 bars earlier than the lagging exit head. Unlike buy changes,
    # sells are not occupancy-bound so this can actually stick. See project_champion958_loss_forensics.
    if entry_rollover_exit:
        drop = float(entry_rollover_exit.get("drop", 1.0))
        near = float(entry_rollover_exit.get("near", 0.03))
        run = float(entry_rollover_exit.get("run", 0.10))
        c_arr = df["close"].to_numpy()
        dE3 = np.full(len(df), np.nan)
        nearhi = np.zeros(len(df), dtype=bool)
        runup = np.zeros(len(df), dtype=bool)
        zE_arr = zE.to_numpy()
        for _sym, _idx in df.groupby("symbol", sort=False).indices.items():
            ze = zE_arr[_idx]
            d = np.full(len(ze), np.nan)
            d[3:] = ze[3:] - ze[:-3]
            dE3[_idx] = d
            cs = pd.Series(c_arr[_idx])
            hi = cs.rolling(20, min_periods=20).max().to_numpy()
            lo = cs.rolling(20, min_periods=20).min().to_numpy()
            nearhi[_idx] = c_arr[_idx] >= hi * (1.0 - near)
            runup[_idx] = (c_arr[_idx] / lo - 1.0) >= run
        roll_sell = (dE3 <= -drop) & nearhi & runup
        # Optional zX gate: only fire the rollover sell where the exit head ALSO reads elevated
        # risk (zX >= floor). Real tops carry higher zX (+0.36) than false tops/pullbacks (+0.23),
        # so this should cut the tail-amputating false-top sells. Mirror of the bottom zX grade.
        rzx = entry_rollover_exit.get("zX_floor")
        if rzx is not None:
            roll_sell = roll_sell & (zX.to_numpy() >= float(rzx))
        sell = sell | roll_sell
    # ENSEMBLE union SELL: a 2nd exit head (exit_score2, e.g. zigzag pre-peak) fires a sell
    # when its causal z exceeds exit2_z_threshold — selling AT the top that the lagging
    # reward_risk head (+2 bars late) misses. Sells aren't occupancy-bound, so unlike the
    # entry union this can actually stick. See project_champion958_loss_forensics.
    if exit2_z_threshold is not None and "exit_score2" in df.columns:
        zX2 = _causal_zscore_by_symbol(df["exit_score2"], df["symbol"], window, min_periods)
        sell = sell | (zX2.to_numpy() > float(exit2_z_threshold))
    # Regime-conditional ENTRY skip: suppress buys while the EW market is in a SUSTAINED bear
    # (index below MA50 for entry_skip_nonbull_persist consecutive days) — dl10_downleg_regime
    # shows ~98% of profit comes from bull, bear-regime trades net-negative. Sticky persist so
    # bull dips / brief chop still trade. Causal.
    if entry_skip_nonbull_persist > 0:
        buy = buy & ~_market_nonbull_mask(df, 50, entry_skip_nonbull_persist, regime_index_symbol)
    # MASK-MAP probe (entry_head_csrank_gate): express a head's CROSS-SECTIONAL rank as a
    # SELECTOR on the MAIN momentum buy — an AND-gate that keeps the protective gates, instead
    # of the union's OR-bypass. Forensic: per-symbol z FLIPS the head's outcome signal (raw IC
    # +0.05 -> z -0.03) but csrank PRESERVES it (lo->hi quintile WR 22%->37%). The open zE>-1.9
    # main buy never USES the ranking; this routes csrank>=pct as a gate so the predictive WR
    # signal can SELECT (not just add bypass trades). col = score|score4(mfe) etc; applied to
    # `buy` BEFORE the union so the ensemble heads are untouched. Off by default.
    if entry_head_csrank_gate:
        _g = entry_head_csrank_gate
        _col = _g.get("col", "score")
        _pct = float(_g.get("pct", 0.3))
        if _col in df.columns:
            _cr = df.groupby("date")[_col].rank(pct=True).to_numpy()
            buy = buy & (_cr >= _pct)
    # Ensemble reversal buy: union AFTER the momentum gates so the contra-phase dip/V-bottom head
    # is not killed by the upleg/xsec/nonbull gates. Sell still wins (a confirmed downleg vetoes it).
    buy = buy | rev_buy | brk_buy | brk_buy2 | brk_buy3 | brk_buy4
    # ORTHOGONAL breadth gate (488-universe market breadth, NOT the 61-EW proxy): suppress ALL
    # buys (incl. ensemble heads) when broad-market breadth is weak — the entries made into low
    # breadth are the signal-exit losers AND the 2022 mdd cluster (corr_w_pnl +0.18). Off by
    # default. Applied AFTER the unions so it filters every entry style.
    if entry_breadth_gate:
        weak = _market_breadth_mask(
            df, entry_breadth_gate.get("metric", "pct_above_ma50"),
            float(entry_breadth_gate.get("threshold", 0.35)),
            int(entry_breadth_gate.get("ma_win", 50)),
            entry_breadth_gate.get("mode", "level"),
            int(entry_breadth_gate.get("z_lookback", 60)))
        buy = buy & ~weak
    # RS-vs-MARKET SELECTION gate: trade only LEADER dips (strong RS-vs-VNINDEX); suppress laggard dips
    # (more likely knives). The realization of the strongest separator (win 72%% vs 52%%) as its own
    # selection strategy. Applied after the unions so it filters every entry style. Off by default.
    if entry_rs_gate:
        weak_rs = _rs_market_mask(df, entry_rs_gate.get("feature", "rs_vsma"),
                                  float(entry_rs_gate.get("threshold", 0.0)))
        buy = buy & ~weak_rs
    sig = np.where(sell, -1, np.where(buy, 1, 0)).astype(np.int8)
    if direction == "short":
        sig = (sig * -1).astype(np.int8)
    df["signal"] = sig
    # WR-EXPRESSION unmask: per-date cross-sectional rank (pct) of the momentum head's RAW
    # score. csrank PRESERVES the head's outcome ordering (per-symbol z FLIPS it). The engine
    # uses this to apply a tighter (tier-2) trailing stop ONLY to high-csrank entries — the
    # predicted-loser cohort (high momentum/mfe csrank trades that end via signal-exit peak at
    # ~5% MFE then fade, while the winners blow past the trail arm). Causal: same-date cross-
    # section only. Off unless the engine's tier2_csr_threshold is set.
    if "score" in df.columns:
        df["entry_csr"] = df.groupby("date")["score"].rank(pct=True).to_numpy()
    return df


def _xsec_rank_membership_signals(signals: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """P1-E2 ``xsec_rank_topk``: per-bar lambdarank scores -> TOP-K MEMBERSHIP signals.

    Every ``rebalance_bars`` bars (monthly cadence = the 20-bar label horizon; P1-E1 §2.2
    showed this is the only cost-viable cadence) the universe is ranked by score:
      - a NON-member entering the top-K becomes a member -> BUY signal at the rebalance
        bar (the engine fills at close of t+1, ``close_next``; NO pullback-limit —
        P1-E1 §3: the limit's fill selection is adverse for this signal family);
      - a member whose rank drops below ``k_exit`` (top-(K+10)) leaves -> SELL signal;
      - hysteresis mirrors P1-E1 p1_03's 'hyst' scheme exactly: stay = rank<=k_exit,
        add = top-K, so membership floats between K and k_exit.
    A BUY is (re-)emitted for EVERY current member at each rebalance (not only new
    entries) so a position closed mid-cycle by the force gate re-enters at the next
    rebalance while the name still holds a slot; the engine ignores buys while the
    position is open, so continuing members are no-ops.

    ``force_exit_gate`` (downleg12) adds the per-bar FORCE sell as the crash brake —
    reuses the champion's ``_exit_force_mask`` (confirmed zigzag down-leg >= 12%); the
    engine ignores sells while flat. All causal: same-date cross-section + trailing
    prices only. Config lives under ``engine.xsec_rank`` and is parsed ONLY here.
    """
    x = (cfg.engine.get("xsec_rank") or {}) if isinstance(cfg.engine, dict) else {}
    k = int(x.get("k", 20))
    k_exit = int(x.get("k_exit", k + 10))
    reb = int(x.get("rebalance_bars", 20))
    force_gate = x.get("force_exit_gate")
    df = signals.sort_values(["symbol", "date"]).reset_index(drop=True)
    piv = df.pivot_table(index="date", columns="symbol", values="score", aggfunc="last")
    reb_dates = list(piv.index[::reb])
    members: set = set()
    buy_keys: set = set()
    sell_keys: set = set()
    for t in reb_dates:
        s = piv.loc[t].dropna()
        if s.empty:
            continue
        rk = s.rank(ascending=False, method="first")  # deterministic tie-break (col order)
        top = set(rk[rk <= k].index)
        stay = {m for m in members if m in rk.index and rk[m] <= k_exit}
        new_members = stay | top
        for m in members - new_members:
            sell_keys.add((m, t))
        for m in new_members:
            buy_keys.add((m, t))
        members = new_members
    buy = np.fromiter(
        ((sym, d) in buy_keys for sym, d in zip(df["symbol"], df["date"])),
        dtype=bool, count=len(df),
    )
    sell = np.fromiter(
        ((sym, d) in sell_keys for sym, d in zip(df["symbol"], df["date"])),
        dtype=bool, count=len(df),
    )
    if force_gate:
        sell = sell | _exit_force_mask(df, force_gate)
    sig = np.where(sell, -1, np.where(buy, 1, 0)).astype(np.int8)
    if cfg.direction == "short":
        sig = (sig * -1).astype(np.int8)
    df["signal"] = sig
    print(
        f"[{cfg.name}] xsec_rank_topk membership: K={k} exit>{k_exit} reb={reb} "
        f"({len(reb_dates)} rebalances) buys={int((sig > 0).sum())} "
        f"sells={int((sig < 0).sum())}" + (f" force={force_gate}" if force_gate else "")
    )
    return df


_XRISE_STRATEGIES = (
    "regression_dual_ml_recombine_xrise",       # sell z(exit) > k (raw exit off)
    "regression_dual_ml_recombine_xrise_ema",   # sell EMA(z(exit)) > k — denoise
    "regression_dual_ml_recombine_xrise_raw",   # sell z(exit) > k OR raw exit > thr
)
_DECOUPLED_STRATEGIES = (
    # Heads fully decoupled: BUY z(entry) > entry_threshold, SELL z(exit) > signal_threshold.
    # No composite z-sum, no raw-exit floor — entry conviction & exit risk are two
    # independent z-signals. _ema EMA5-smooths BOTH z's before their bands.
    "regression_dual_ml_recombine_decoupled",
    "regression_dual_ml_recombine_decoupled_ema",
)
_RECOMBINE_STRATEGIES = (
    "regression_dual_ml_recombine",
    "regression_dual_ml_recombine_zexit",
    "regression_dual_ml_recombine_zexit_both",
    "regression_dual_ml_recombine_ema",
    *_XRISE_STRATEGIES,
    *_DECOUPLED_STRATEGIES,
)


def recombine_signals(signals_all: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Aggregate (post-walk-forward) dual-ML recombine — shared by backtest and serving.

    Rebuilds entry/exit signals from a CAUSAL z-score of the two heads over the full
    concatenated prediction series (per symbol). Done at the aggregate level (not per
    fold) because the trailing 252-bar z-window needs more history than one fold holds.

    Extracted verbatim from ``run_experiment`` so the production serving path reproduces
    the exact champion signal chain (z-sum / decoupled bands / downleg force-gate). A
    no-op pass-through for strategies outside ``_RECOMBINE_STRATEGIES``.
    """
    # P1-E2 xsec_rank_topk (append-only, EARLY RETURN): membership top-K signals from the
    # per-date lambdarank scores — the ranking analog of the recombine step. Every other
    # strategy path below is byte-unchanged.
    if cfg.strategy == "xsec_rank_topk":
        return _xsec_rank_membership_signals(signals_all, cfg)
    # MARKET-BREADTH entry gate for NON-recombine strategies (e.g. single_ml_action_classifier):
    # the recombine path applies entry_breadth_gate internally, but SMAC bypasses recombine, so the
    # per-symbol model is BLIND to broad-market crashes — forensic shows ~100% of the high-trade
    # model's net loss is in the 2022/2026 market-grind years (488-univ breadth 0.26 vs 0.55 in
    # bull). Suppress buys (signal>0) on weak-breadth dates. 488-universe, NOT the 61-EW proxy.
    _bg = cfg.engine.get("entry_breadth_gate") if isinstance(cfg.engine, dict) else None
    if _bg and cfg.strategy not in _RECOMBINE_STRATEGIES and "date" in signals_all.columns:
        weak = _market_breadth_mask(
            signals_all, _bg.get("metric", "pct_above_ma50"),
            float(_bg.get("threshold", 0.35)), int(_bg.get("ma_win", 50)),
            _bg.get("mode", "level"), int(_bg.get("z_lookback", 60)))
        signals_all = signals_all.copy()
        signals_all.loc[(signals_all["signal"] > 0) & weak, "signal"] = 0

    if cfg.strategy in _RECOMBINE_STRATEGIES:
        n_buys_before = int((signals_all["signal"] > 0).sum())
        # _ema: causal EMA-smooth the z-sum (span=5) before the buy threshold — denoise
        # the entry signal (analysis: IC 0.057->0.061, sign-flips halved). Exit unchanged.
        ema_span = 5 if cfg.strategy == "regression_dual_ml_recombine_ema" else None
        # zexit*: add a symmetric LOWER z-band sell (cfg.signal_threshold). 'zexit'
        # REPLACES the raw downside-head sell; '_both' keeps BOTH triggers; the canonical
        # recombine keeps only the raw exit (lower band off — params default unchanged).
        two_sided = cfg.strategy in (
            "regression_dual_ml_recombine_zexit",
            "regression_dual_ml_recombine_zexit_both",
        )
        # _xrise*: sell when the exit head's OWN drawdown z RISES above cfg.signal_threshold
        # (risk climbing, near a top) — decoupled from the entry z-sum, OPPOSITE trigger to
        # zexit's lower band. _ema denoises z(exit) (span=5, fewer false alarms); _raw also
        # keeps the absolute raw-exit floor so big drops still fire even if z(exit) is calm.
        x_rise = cfg.strategy in _XRISE_STRATEGIES
        # _decoupled*: BUY on z(entry) alone (decoupled from the exit head), SELL on the
        # rising-risk z(exit) band — same sell trigger as xrise, so both share the
        # exit_z_threshold path. _ema EMA5-smooths both z's.
        decoupled = cfg.strategy in _DECOUPLED_STRATEGIES
        sell_on_zexit = x_rise or decoupled
        exit_z_ema = 5 if cfg.strategy in (
            "regression_dual_ml_recombine_xrise_ema",
            "regression_dual_ml_recombine_decoupled_ema",
        ) else None
        entry_z_threshold = (
            (cfg.entry_threshold if cfg.entry_threshold is not None else 1.0)
            if decoupled else None
        )
        entry_z_ema = 5 if cfg.strategy == "regression_dual_ml_recombine_decoupled_ema" else None
        use_raw_exit = cfg.strategy not in (
            "regression_dual_ml_recombine_zexit",
            "regression_dual_ml_recombine_xrise",
            "regression_dual_ml_recombine_xrise_ema",
            *_DECOUPLED_STRATEGIES,
        )
        signals_all = _recombine_dual_ml_signals(
            signals_all,
            sum_threshold=cfg.entry_threshold if cfg.entry_threshold is not None else 1.0,
            exit_threshold=cfg.exit_threshold if cfg.exit_threshold is not None else 0.0,
            window=(
                int(cfg.engine.get("z_norm_window", 252)) if isinstance(cfg.engine, dict) else 252
            ),
            min_periods=(
                int(cfg.engine.get("z_norm_min_periods", 60)) if isinstance(cfg.engine, dict) else 60
            ),
            direction=cfg.direction,
            lower_threshold=cfg.signal_threshold if two_sided else None,
            use_raw_exit=use_raw_exit,
            exit_z_threshold=cfg.signal_threshold if sell_on_zexit else None,
            exit_z_ema_span=exit_z_ema,
            entry_z_threshold=entry_z_threshold,
            entry_z_low_threshold=(
                cfg.engine.get("entry_z_low_threshold") if isinstance(cfg.engine, dict) else None
            ),
            entry_z_ema_span=entry_z_ema,
            ema_span=ema_span,
            entry_gate=cfg.engine.get("entry_gate") if isinstance(cfg.engine, dict) else None,
            entry_xs_mom_pct=(
                cfg.engine.get("entry_xs_mom_pct") if isinstance(cfg.engine, dict) else None
            ),
            exit_gate=cfg.engine.get("exit_gate") if isinstance(cfg.engine, dict) else None,
            entry_raw_threshold=(
                cfg.engine.get("entry_raw_threshold") if isinstance(cfg.engine, dict) else None
            ),
            exit_force_gate=(
                cfg.engine.get("exit_force_gate") if isinstance(cfg.engine, dict) else None
            ),
            exit_force_gate_nonbull=(
                cfg.engine.get("exit_force_gate_nonbull") if isinstance(cfg.engine, dict) else None
            ),
            exit_force_gate_lowbreadth=(
                cfg.engine.get("exit_force_gate_lowbreadth") if isinstance(cfg.engine, dict) else None
            ),
            exit_force_gate_vn30=(
                cfg.engine.get("exit_force_gate_vn30") if isinstance(cfg.engine, dict) else None
            ),
            exit_rs_drop=(
                cfg.engine.get("exit_rs_drop") if isinstance(cfg.engine, dict) else None
            ),
            exit_force_suppress=(
                cfg.engine.get("exit_force_suppress") if isinstance(cfg.engine, dict) else None
            ),
            nonbull_ma_win=(
                cfg.engine.get("nonbull_ma_win", 50) if isinstance(cfg.engine, dict) else 50
            ),
            nonbull_persist=(
                cfg.engine.get("nonbull_persist", 3) if isinstance(cfg.engine, dict) else 3
            ),
            entry_skip_nonbull_persist=(
                cfg.engine.get("entry_skip_nonbull_persist", 0) if isinstance(cfg.engine, dict) else 0
            ),
            regime_index_symbol=(
                cfg.engine.get("regime_index_symbol") if isinstance(cfg.engine, dict) else None
            ),
            early_entry=(
                cfg.engine.get("early_entry_reversal") if isinstance(cfg.engine, dict) else None
            ),
            top_exit=(
                cfg.engine.get("top_reversal_exit") if isinstance(cfg.engine, dict) else None
            ),
            entry_zX_floor=(
                cfg.engine.get("entry_zX_floor") if isinstance(cfg.engine, dict) else None
            ),
            entry_rollover_exit=(
                cfg.engine.get("entry_rollover_exit") if isinstance(cfg.engine, dict) else None
            ),
            entry2_z_threshold=(
                (cfg.engine.get("entry_ensemble") or {}).get("z_threshold")
                if isinstance(cfg.engine, dict) else None
            ),
            entry3_z_threshold=(
                (cfg.engine.get("entry_ensemble2") or {}).get("z_threshold")
                if isinstance(cfg.engine, dict) else None
            ),
            entry4_z_threshold=(
                (cfg.engine.get("entry_ensemble3") or {}).get("z_threshold")
                if isinstance(cfg.engine, dict) else None
            ),
            entry6_z_threshold=(
                (cfg.engine.get("entry_ensemble5") or {}).get("z_threshold")
                if isinstance(cfg.engine, dict) else None
            ),
            entry6_norm=((cfg.engine.get("entry_ensemble5") or {}).get("norm") if isinstance(cfg.engine, dict) else None),
            entry5_z_threshold=(
                (cfg.engine.get("entry_ensemble4") or {}).get("z_threshold")
                if isinstance(cfg.engine, dict) else None
            ),
            entry2_norm=((cfg.engine.get("entry_ensemble") or {}).get("norm") if isinstance(cfg.engine, dict) else None),
            entry3_norm=((cfg.engine.get("entry_ensemble2") or {}).get("norm") if isinstance(cfg.engine, dict) else None),
            entry4_norm=((cfg.engine.get("entry_ensemble3") or {}).get("norm") if isinstance(cfg.engine, dict) else None),
            entry5_norm=((cfg.engine.get("entry_ensemble4") or {}).get("norm") if isinstance(cfg.engine, dict) else None),
            entry_rs_gate=(
                cfg.engine.get("entry_rs_gate") if isinstance(cfg.engine, dict) else None
            ),
            entry_breadth_gate=(
                cfg.engine.get("entry_breadth_gate") if isinstance(cfg.engine, dict) else None
            ),
            entry_head_csrank_gate=(
                cfg.engine.get("entry_head_csrank_gate") if isinstance(cfg.engine, dict) else None
            ),
            downleg_skip_bull=(
                cfg.engine.get("downleg_skip_bull") if isinstance(cfg.engine, dict) else None
            ),
            exit2_z_threshold=(
                (cfg.engine.get("exit_ensemble") or {}).get("z_threshold")
                if isinstance(cfg.engine, dict) else None
            ),
        )
        _buy_desc = (
            f"buy {'EMA' if entry_z_ema else ''}z(entry) > {cfg.entry_threshold}"
            if decoupled else f"buy z-sum > {cfg.entry_threshold}"
        )
        print(
            f"[{cfg.name}] dual-ML recombine ({cfg.strategy}): {_buy_desc}"
            + (f" (EMA{ema_span})" if ema_span else "")
            + (f"; sell z-sum < {cfg.signal_threshold}" if two_sided else "")
            + (f"; sell {'EMA' if exit_z_ema else ''}z(exit) > {cfg.signal_threshold}" if sell_on_zexit else "")
            + (f"; raw-exit > {cfg.exit_threshold}" if use_raw_exit else "")
            + f"; buys {n_buys_before} -> {int((signals_all['signal'] > 0).sum())}, "
            f"sells {int((signals_all['signal'] < 0).sum())} (causal z, 252/60)"
        )
    return signals_all


def build_feature_frame(
    ohlcv: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    requested_symbols: list[str],
    data_root: str = "",
    with_targets: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str], dict, dict]:
    """Resolve the per-slot feature matrix (+ optional targets) for ``cfg``.

    Extracted verbatim from ``run_experiment`` so the serving path reuses the EXACT
    feature pipeline (same DSL catalog, same cross-sectional inputs, same warmup trim).

    Args:
        ohlcv: [symbol, date, open, high, low, close, volume].
        cfg: experiment config (entry/exit feature sets + targets).
        requested_symbols: symbols present in ``ohlcv`` (for the sector grouping).
        data_root: passed to the resolver for its content-addressed cache key.
        with_targets: backtest/export need labels (True); serving only predicts (False).

    Returns:
        (feat, entry_feat_cols, exit_feat_cols, entry_target_cfg, exit_target_cfg).
        When ``with_targets`` is False the target columns are absent from ``feat`` but
        the target configs are still returned (for the leakage-gap audit / metadata).
    """
    # Phase 0.4: Resolve per-slot features/targets
    entry_fs = cfg.entry_features or cfg.feature_set
    exit_fs = cfg.exit_features or cfg.feature_set
    # Ensemble 2nd entry head can use its OWN feature set (e.g. reversal-confirm/zigzag-structure
    # feats the momentum set lacks) via engine_config.entry_ensemble.features. None = share the
    # primary entry features (legacy).
    _ens = cfg.engine.get("entry_ensemble") if isinstance(cfg.engine, dict) else None
    entry2_fs = (_ens or {}).get("features")
    # 3rd entry head (breakout) can likewise use its OWN feature set via entry_ensemble2.features.
    _ens2 = cfg.engine.get("entry_ensemble2") if isinstance(cfg.engine, dict) else None
    entry3_fs = (_ens2 or {}).get("features")
    _ens3 = cfg.engine.get("entry_ensemble3") if isinstance(cfg.engine, dict) else None
    entry4_fs = (_ens3 or {}).get("features")
    _ens4 = cfg.engine.get("entry_ensemble4") if isinstance(cfg.engine, dict) else None
    entry5_fs = (_ens4 or {}).get("features")
    _ens5 = cfg.engine.get("entry_ensemble5") if isinstance(cfg.engine, dict) else None
    entry6_fs = (_ens5 or {}).get("features")
    _sets = [entry_fs, exit_fs]
    for _fs in (entry2_fs, entry3_fs, entry4_fs, entry5_fs, entry6_fs):
        if _fs and _fs not in _sets:
            _sets.append(_fs)

    # DSL feature store: materialize the UNION of entry+exit set members once,
    # caching each feature content-addressed. A feature shared by both sets is
    # computed a single time (resolver reports cache hits across runs).
    print(f"[{cfg.name}] resolving features via DSL store: entry={entry_fs}, exit={exit_fs}")
    resolver = FeatureResolver.from_catalog()

    # Cross-sectional sets need a sector grouping and/or a market index. Build only
    # what the requested sets actually reference (per-symbol sets need neither).
    needed_inputs = resolver.required_raw_inputs(_sets)
    sector_map = build_sector_map(requested_symbols) if "sector" in needed_inputs else None
    market_df = build_equal_weight_index(ohlcv) if "market_close" in needed_inputs else None
    if market_df is not None:
        print(f"  [resolver] built equal-weight market index ({len(market_df)} dates)")

    feat, feat_cols_by_set, cache_hits = resolver.resolve(
        ohlcv,
        _sets,
        timeframe="1D",
        data_root=str(data_root),
        market_df=market_df,
        sector_map=sector_map,
    )
    print(f"  [resolver] feature matrix ready ({cache_hits} cache hits)")

    # Per-slot feature columns
    entry_feat_cols = feat_cols_by_set[entry_fs]
    exit_feat_cols = feat_cols_by_set[exit_fs]
    entry2_feat_cols = feat_cols_by_set[entry2_fs] if entry2_fs else None
    entry3_feat_cols = feat_cols_by_set[entry3_fs] if entry3_fs else None
    entry4_feat_cols = feat_cols_by_set[entry4_fs] if entry4_fs else None
    entry5_feat_cols = feat_cols_by_set[entry5_fs] if entry5_fs else None
    entry6_feat_cols = feat_cols_by_set[entry6_fs] if entry6_fs else None

    # BREADTH features for ensemble heads: any entry_ensembleN.breadth_features (list of metrics,
    # e.g. ["pct_above_ma50","adv_pct"]) appends 488-UNIVERSE market-breadth columns to that head's
    # feature set — the orthogonal regime-TIMING dimension the per-symbol sets lack (corr_w_pnl
    # +0.18). Market-wide (per-date), causal (ffill in the loader), warmup -> 0.5 neutral.
    def _breadth_append(ens_cfg, cols):
        bf = (ens_cfg or {}).get("breadth_features")
        if not bf:
            return cols
        names = []
        for m in bf:
            cname = f"breadth_{m}"
            if cname not in feat.columns:
                br = _load_market_breadth(m, int((ens_cfg or {}).get("breadth_ma_win", 50)))
                feat[cname] = feat["date"].map(br).astype("float32").fillna(0.5)
            names.append(cname)
        return list(cols if cols else entry_feat_cols) + names
    entry3_feat_cols = _breadth_append(_ens2, entry3_feat_cols)
    entry4_feat_cols = _breadth_append(_ens3, entry4_feat_cols)
    entry5_feat_cols = _breadth_append(_ens4, entry5_feat_cols)
    entry6_feat_cols = _breadth_append(_ens5, entry6_feat_cols)
    # P1-E2 xsec_rank_topk (append-only): the MAIN entry head gets the 2 breadth columns
    # (488-univ regime timing — the exact 47-col set of P1-E1) via
    # engine.xsec_rank.breadth_features, reusing the ensemble-head mechanism. Gated on the
    # new strategy so every existing feature path is byte-unchanged.
    if cfg.strategy == "xsec_rank_topk":
        _xr_cfg = cfg.engine.get("xsec_rank") if isinstance(cfg.engine, dict) else None
        entry_feat_cols = _breadth_append(_xr_cfg, entry_feat_cols)

    # XSEC features: per-symbol CROSS-SECTIONAL rank/trend from the FULL universe (leadership
    # rotation — the strongest per-symbol signal, cs_rank_trend IC_winner +0.17). Any
    # entry_ensembleN.xsec_features (list) merges those per-(symbol,date) columns into that head's
    # feature set (the per-symbol use of the broad universe the breadth-EXIT only used market-wide).
    _exit_xsec = (cfg.engine.get("exit_xsec_features") if isinstance(cfg.engine, dict) else None) or []
    _entry_xsec = (cfg.engine.get("entry_xsec_features") if isinstance(cfg.engine, dict) else None) or []
    _xsec_reqs = []
    for _ec in (_ens, _ens2, _ens3, _ens4):
        for _m in ((_ec or {}).get("xsec_features") or []):
            if _m not in _xsec_reqs:
                _xsec_reqs.append(_m)
    for _m in (_exit_xsec + _entry_xsec):
        if _m not in _xsec_reqs:
            _xsec_reqs.append(_m)
    if _xsec_reqs:
        _xlong = _load_xsec_features(_xsec_reqs)
        feat = feat.merge(_xlong, on=["symbol", "date"], how="left")
        for _m in _xsec_reqs:
            feat[f"xsec_{_m}"] = feat[f"xsec_{_m}"].fillna(0.5)

    def _xsec_append(ens_cfg, cols):
        xf = (ens_cfg or {}).get("xsec_features")
        if not xf:
            return cols
        return list(cols if cols else entry_feat_cols) + [f"xsec_{m}" for m in xf]
    entry2_feat_cols = _xsec_append(_ens, entry2_feat_cols)
    entry3_feat_cols = _xsec_append(_ens2, entry3_feat_cols)
    entry4_feat_cols = _xsec_append(_ens3, entry4_feat_cols)
    entry5_feat_cols = _xsec_append(_ens4, entry5_feat_cols)
    entry6_feat_cols = _xsec_append(_ens5, entry6_feat_cols)
    # EXIT head can use the RS features too (the REALIZABILITY signal on the exit side): the
    # velocity/exit head learns "RS-rollover + [vol/price] = terminal round-trip = sell", which a
    # blunt RS-drop rule cannot (it clips winners on temporary RS dips). engine.exit_xsec_features.
    if _exit_xsec:
        exit_feat_cols = list(exit_feat_cols) + [f"xsec_{m}" for m in _exit_xsec]
    # MAIN entry head can use the RS-vs-market features (the strongest per-trade separator +0.16):
    # the entry head learns "strong RS-vs-market dip = bounces (leader), weak = knife" — the selection
    # signal the deterministic gate/conv-fill can't realize. engine.entry_xsec_features (list).
    if _entry_xsec:
        entry_feat_cols = list(entry_feat_cols) + [f"xsec_{m}" for m in _entry_xsec]

    # Abort loudly if features are pathologically NaN before any silent dropna shrinks
    # the universe (e.g. the leading_v2 ADX index-misalignment bug).
    _all_feat_cols = (set(entry_feat_cols) | set(exit_feat_cols) | set(entry2_feat_cols or [])
                      | set(entry3_feat_cols or []) | set(entry4_feat_cols or [])
                      | set(entry5_feat_cols or []) | set(entry6_feat_cols or []))
    assert_feature_integrity(feat, sorted(_all_feat_cols))

    # Per-slot targets
    entry_target_cfg = cfg.entry_target or cfg.target
    exit_target_cfg = cfg.exit_target or cfg.target

    if with_targets:
        print(
            f"[{cfg.name}] applying targets: entry={entry_target_cfg['type']}, exit={exit_target_cfg['type']}"
        )

        entry_target = build_target(entry_target_cfg)
        feat["target_entry"] = entry_target.apply(feat.copy())["target"]

        if exit_target_cfg != entry_target_cfg:
            exit_target = build_target(exit_target_cfg)
            feat["target_exit"] = exit_target.apply(feat.copy())["target"]
        else:
            feat["target_exit"] = feat["target_entry"]

        # Backward compat: feat["target"] = entry target
        feat["target"] = feat["target_entry"]

        # ENSEMBLE second entry head (causal): train a 2nd entry regressor on a DIFFERENT target
        # (e.g. reversal_entry alongside the primary continuation) so the union buy captures BOTH
        # the momentum winners (+13.5%) and the dip/V-bottom winners (+10.4%) that a single head
        # trades off. Off unless engine_config.entry_ensemble.target is set. See
        # project_champion958_loss_forensics (two entry styles, comparable quality, union them).
        _ens_cfg = cfg.engine.get("entry_ensemble") if isinstance(cfg.engine, dict) else None
        if _ens_cfg and _ens_cfg.get("target"):
            _ens_target = build_target(_ens_cfg["target"])
            feat["target_entry2"] = _ens_target.apply(feat.copy())["target"]
        # ENSEMBLE 3rd entry head target (e.g. continuation/breakout), shares the primary entry
        # features. Off unless engine_config.entry_ensemble2.target is set.
        _ens2_cfg = cfg.engine.get("entry_ensemble2") if isinstance(cfg.engine, dict) else None
        if _ens2_cfg and _ens2_cfg.get("target"):
            feat["target_entry3"] = build_target(_ens2_cfg["target"]).apply(feat.copy())["target"]
        _ens3_cfg = cfg.engine.get("entry_ensemble3") if isinstance(cfg.engine, dict) else None
        if _ens3_cfg and _ens3_cfg.get("target"):
            feat["target_entry4"] = build_target(_ens3_cfg["target"]).apply(feat.copy())["target"]
        _ens5_cfg = cfg.engine.get("entry_ensemble5") if isinstance(cfg.engine, dict) else None
        if _ens5_cfg and _ens5_cfg.get("target"):
            feat["target_entry6"] = build_target(_ens5_cfg["target"]).apply(feat.copy())["target"]
        _ens4_cfg = cfg.engine.get("entry_ensemble4") if isinstance(cfg.engine, dict) else None
        if _ens4_cfg and _ens4_cfg.get("target"):
            feat["target_entry5"] = build_target(_ens4_cfg["target"]).apply(feat.copy())["target"]
        # ENSEMBLE second EXIT head (causal, the actionable side): a 2nd exit regressor on a peak/
        # sell-at-top target (e.g. zigzag pre-peak) unioned into the SELL. The reward_risk head lags
        # the price top by +2 bars; a peak-proximity head fires AT the top. Sells aren't occupancy-
        # bound so this can stick. zigzag targets need gap>=85 (in place) to stay leak-free
        # ([[project_peak_minfwdleg]]). Off unless engine_config.exit_ensemble.target is set.
        _xens_cfg = cfg.engine.get("exit_ensemble") if isinstance(cfg.engine, dict) else None
        if _xens_cfg and _xens_cfg.get("target"):
            feat["target_exit2"] = build_target(_xens_cfg["target"]).apply(feat.copy())["target"]

    # Trim each symbol's feature-warmup head (structurally-NaN leading rows) before the
    # split, so train/test folds don't trip the fail-loud require_no_nan on expected
    # warmup NaN. Mid-series NaN survives and still fails loud (genuine-bug guard).
    feat = trim_feature_warmup(feat, sorted(_all_feat_cols), name=cfg.name)
    # Symmetric to the warmup trim: drop the trailing NaN-label block of DELISTED symbols
    # (their forward-window tail lands inside a train fold and would trip require_no_nan).
    feat = trim_target_tail(feat, ["target_entry", "target_exit", "target"], name=cfg.name)
    return (feat, entry_feat_cols, exit_feat_cols, entry_target_cfg, exit_target_cfg,
            entry2_feat_cols, entry3_feat_cols, entry4_feat_cols, entry5_feat_cols,
            entry6_feat_cols)


def run_experiment(
    cfg: ExperimentConfig,
    data_root: str,
    symbols: list[str],
    out_dir: str,
    run_id: str | None = None,
    export_csv: bool = True,
) -> dict:
    """Run full experiment — load, train, backtest, report.

    Phase 1b.11: Resumable runs via fold checkpointing.

    Args:
        cfg: ExperimentConfig
        data_root: path to OHLCV data directory
        symbols: list of symbols to use
        out_dir: output directory for results CSVs + JSON
        run_id: optional run_id for fold checkpointing; if provided, saves folds to
                {out_dir}/{run_id}/folds/{fold_label}.parquet for resumability
        export_csv: write detail CSVs + summary JSON to disk. P3: the DB pipeline
                (run_template) passes False and persists the returned frames straight
                to repos; the legacy file runner (run_experiments) leaves it True.

    Returns:
        summary dict with metrics, plus ``_run_detail_frames`` carrying the
        trades/signals/yearly/symbol DataFrames for in-memory DB persistence
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fold_cache_dir = None
    if run_id:
        fold_cache_dir = out / run_id / "folds"
        fold_cache_dir.mkdir(parents=True, exist_ok=True)

    # Use config data_source_dir if specified, otherwise use parameter
    if cfg.data_source_dir:
        data_root = cfg.data_source_dir

    print(f"[{cfg.name}] loading {len(symbols)} symbols from {data_root}")

    data_path = Path(data_root).resolve()
    print(f"  [debug] Resolved to: {data_path}")
    if data_path.suffix == ".duckdb":
        print("  [debug] Using DuckDB loader")
    else:
        print(f"  [debug] all_symbols exists: {(data_path / 'all_symbols').exists()}")

    # Dynamic point-in-time universe (docs/UPGRADE_DYNAMIC_UNIVERSE.md): resolve every
    # fold's top-N universe ONCE (causal, prior-year ADV), load the UNION of all fold
    # lists so each fold has its bars, and let the splitter mask each fold down to its
    # own year's list. None → legacy fixed symbols (byte-identical).
    universe_by_year: dict[int, list[str]] | None = None
    if cfg.universe_policy:
        _sp = cfg.split
        if _sp.get("type", "walk_forward_year") != "walk_forward_year" or not (
            "first_test_year" in _sp and "last_test_year" in _sp
        ):
            raise ValueError(
                "universe_policy requires a walk_forward_year split with explicit "
                "first_test_year/last_test_year"
            )
        if data_path.suffix != ".duckdb":
            raise ValueError("universe_policy requires a DuckDB data source")
        from stock_ml.src.data.universe_resolver import resolve_universes

        _u_years = list(
            range(_sp["first_test_year"], _sp["last_test_year"] + 1, _sp.get("test_years", 1))
        )
        universe_by_year = resolve_universes(cfg.universe_policy, _u_years, str(data_path))
        symbols = sorted(set().union(*universe_by_year.values()))
        print(
            f"[{cfg.name}] universe_policy {cfg.universe_policy.get('mode')}: "
            + ", ".join(f"{y}={len(universe_by_year[y])}" for y in _u_years)
            + f"; union={len(symbols)} symbols"
        )
        # §13.3.1 fetch-on-miss: the corrected universe (§13.9) surfaces survivorship-correct names
        # the local cache never had (ROS, delisted tickers). Fetch + upsert them so they are NOT
        # silently dropped below — which would re-shrink the universe to survivors, the exact bias the
        # §13.9 fix removes. Only the dynamic path needs it (static sets are pre-registered/cached).
        from stock_ml.src.data.duckdb_loader import ensure_symbols_cached

        ensure_symbols_cached(str(data_path), symbols)

    from stock_ml.src.data.loader import get_loader

    loader = get_loader(str(data_path))
    available = set(loader.list_symbols())
    print(f"  [debug] Found {len(available)} symbols")
    requested = [s for s in symbols if s in available]
    missing = sorted(set(symbols) - available)
    if missing:
        print(
            f"  [warn] missing symbols skipped: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if not requested:
        raise ValueError("no symbols available in the dataset")

    raw = loader.load_many(requested)
    ohlcv = raw[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()

    # Resolve features + targets + warmup-trim (extracted to build_feature_frame so the
    # serving path reuses the exact feature pipeline). entry_target_cfg/exit_target_cfg
    # are returned for the downstream leakage-gap audit.
    (feat, entry_feat_cols, exit_feat_cols, entry_target_cfg, exit_target_cfg,
     entry2_feat_cols, entry3_feat_cols, entry4_feat_cols,
     entry5_feat_cols, entry6_feat_cols) = build_feature_frame(
        ohlcv, cfg, requested_symbols=requested, data_root=data_root, with_targets=True
    )

    print(f"[{cfg.name}] dataset: {len(ohlcv)} bars across {ohlcv['symbol'].nunique()} symbols")

    split_cfg = cfg.split.copy()
    split_type = split_cfg.get("type", "walk_forward_year")

    if split_type == "walk_forward_year":
        train_years = split_cfg.get("train_years", 4)
        test_years = split_cfg.get("test_years", 1)
        gap_days = split_cfg.get("gap_days", 25)

        if "first_test_year" in split_cfg and "last_test_year" in split_cfg:
            splitter = YearSplitter(
                train_years=train_years,
                test_years=test_years,
                gap_days=gap_days,
                first_test_year=split_cfg["first_test_year"],
                last_test_year=split_cfg["last_test_year"],
            )
            print(
                f"  [split] year range: {split_cfg['first_test_year']}-{split_cfg['last_test_year']} (explicit)"
            )
        else:
            splitter = YearSplitter.from_data(
                feat,
                train_years=train_years,
                test_years=test_years,
                gap_days=gap_days,
            )
            print(
                f"  [split] year range: {splitter.first_test_year}-{splitter.last_test_year} (auto-detected)"
            )

        # Absorb the post-window tail (data beyond last_test_year) into the final
        # fold so the last model also scores it. Otherwise tail bars carry no
        # signal and an open position rides to end_of_data months later.
        data_max = pd.to_datetime(feat["date"]).max()
        splitter.last_test_end = data_max + pd.Timedelta(days=1)
        last_win_end = splitter.windows()[-1].test_end
        if last_win_end > pd.Timestamp(year=splitter.last_test_year + splitter.test_years, month=1, day=1):
            print(
                f"  [split] last fold extended to {last_win_end.date()} "
                f"to score post-window tail (data ends {data_max.date()})"
            )
    elif split_type == "purged_kfold":
        n_splits = split_cfg.get("n_splits", 5)
        embargo_days = split_cfg.get("embargo_days", 0)
        label_horizon = split_cfg.get("label_horizon", 5)

        splitter = PurgedKFoldSplitter(
            n_splits=n_splits,
            embargo_days=embargo_days,
            label_horizon=label_horizon,
        )
        print(
            f"  [split] purged_kfold: {n_splits} splits, embargo_days={embargo_days}, label_horizon={label_horizon}"
        )
        windows = None  # Will be collected during split loop
    else:
        raise NotImplementedError(f"split type '{split_type}' not yet implemented")

    if split_type == "walk_forward_year":
        windows = splitter.windows()
    else:
        windows = None  # For purged_kfold, windows are collected during split()

    signal_frames: list[pd.DataFrame] = []
    windows_list = []  # Collect windows during split for purged_kfold
    # universe_policy is guarded to walk_forward_year above; only YearSplitter.split
    # accepts the kwarg, so pass it conditionally to leave purged_kfold untouched.
    _split_kwargs = {"universe_by_year": universe_by_year} if universe_by_year else {}
    for w, train_df, test_df in splitter.split(feat, **_split_kwargs):
        windows_list.append(w)
        if train_df.empty or test_df.empty:
            print(f"  [fold {w.label}] empty — skipped")
            continue

        fold_cache_path = fold_cache_dir / f"{w.label}.parquet" if fold_cache_dir else None

        # A fold checkpoint is keyed only by the CONFIG fingerprint (run_id), NOT by engine-wheel
        # or data fingerprint — so an engine upgrade or a data refresh leaves stale parquets that a
        # naive restore would silently reuse (the §11.1 re-baseline would recompute NOTHING). Callers
        # that recompute the catalogue (rebaseline.py) set STOCKML_FRESH_FOLDS=1 to bypass restore and
        # retrain+overwrite every fold. Iterative research runs (no flag) keep the fast restore.
        _fresh_folds = bool(os.environ.get("STOCKML_FRESH_FOLDS"))
        if fold_cache_path and fold_cache_path.exists() and not _fresh_folds:
            signals = pd.read_parquet(fold_cache_path)
            n_buys = (signals["signal"] > 0).sum()
            n_sells = (signals["signal"] < 0).sum()
            print(
                f"  [fold {w.label}] restored from checkpoint  buys={n_buys:>5}  sells={n_sells:>5}"
            )
            signal_frames.append(signals)
            continue

        # No try/except swallow here: a fold that errors out must abort the run rather
        # than silently drop part of the backtest window and produce partial results.
        _fold_om: dict[str, Any] = {}
        entry_model, exit_model, signals = train_fold(
            train_df,
            test_df,
            entry_feat_cols,
            cfg,
            exit_feat_cols=exit_feat_cols if exit_feat_cols != entry_feat_cols else None,
            entry_target_col="target_entry",
            exit_target_col="target_exit",
            entry2_target_col="target_entry2" if "target_entry2" in train_df.columns else None,
            exit2_target_col="target_exit2" if "target_exit2" in train_df.columns else None,
            entry2_feat_cols=entry2_feat_cols,
            entry3_target_col="target_entry3" if "target_entry3" in train_df.columns else None,
            entry3_feat_cols=entry3_feat_cols,
            entry4_target_col="target_entry4" if "target_entry4" in train_df.columns else None,
            entry4_feat_cols=entry4_feat_cols,
            entry5_target_col="target_entry5" if "target_entry5" in train_df.columns else None,
            entry5_feat_cols=entry5_feat_cols,
            entry6_target_col="target_entry6" if "target_entry6" in train_df.columns else None,
            entry6_feat_cols=entry6_feat_cols,
            out_models=_fold_om,
        )
        if signals.empty:
            raise ValueError(
                f"[fold {w.label}] train_fold produced zero signal rows — refusing to "
                "silently skip a non-empty fold."
            )

        n_buys = (signals["signal"] > 0).sum()
        n_sells = (signals["signal"] < 0).sum()
        # Use target_entry if exists (per-slot), else fallback to target (backward compat)
        target_col_for_stats = "target_entry" if "target_entry" in train_df.columns else "target"
        print(
            f"  [fold {w.label}] train={len(train_df.dropna(subset=[target_col_for_stats, *entry_feat_cols])):>6}  "
            f"test={len(signals):>6}  buys={n_buys:>5}  sells={n_sells:>5}"
        )

        if fold_cache_path:
            signals.to_parquet(fold_cache_path, index=False, compression="snappy")

        # §11.9: persist the EXACT per-fold model set that produced these signals, so a bundle can ship it
        # verbatim (export_bundle --fold-models-from-run) and serving replays the same models — reproducing
        # THIS backtest by construction (attest 100%). Opt-in via env so ordinary runs pay no disk cost;
        # the re-baseline runner sets it. Export re-training a fold-model differs by ~0.01 in score and
        # flips band-edge signals, which is why the model must come from here, not be refitted downstream.
        if fold_cache_path and os.environ.get("STOCKML_PERSIST_FOLD_MODELS") and hasattr(w, "test_year"):
            import joblib

            fold_model_set: dict[str, Any] = {"entry": entry_model}
            if exit_model is not None:
                fold_model_set["exit"] = exit_model
            fold_model_set.update(_fold_om)  # entry2..6 / exit2 (xsec also sets 'entry', identical)
            joblib.dump(
                {"test_year": int(w.test_year), "models": fold_model_set},
                fold_cache_path.with_suffix(".models.joblib"),
            )

        signal_frames.append(signals)

    if not signal_frames:
        print("[!] no signals produced — abort")
        return {"name": cfg.name, "ok": False, "reason": "no_signals"}

    signals_all = pd.concat(signal_frames, ignore_index=True)
    signals_all = signals_all.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Dual-ML recombine: rebuild entry/exit signals from a CAUSAL z-score of the two
    # heads over the full concatenated prediction series (per symbol). Extracted to
    # recombine_signals() so the serving path reuses the exact same chain. No-op for
    # non-recombine strategies.
    signals_all = recombine_signals(signals_all, cfg)

    # Clip OHLCV to the signal coverage so a position still open at the end of the
    # signal range locks at the window's last candle (exit_reason 'open') instead of
    # riding through a signal-less tail to the dataset end (the misleading multi-month
    # 'end_of_data' hold). When later folds add signals (e.g. 2026), the continuous
    # run resumes that position and finds its real exit, scoring it from the carry bar.
    sig_max_date = pd.to_datetime(signals_all["date"]).max()
    _n_before = len(ohlcv)
    ohlcv = ohlcv[pd.to_datetime(ohlcv["date"]) <= sig_max_date].copy()
    _n_clipped = _n_before - len(ohlcv)
    if _n_clipped > 0:
        print(
            f"[{cfg.name}] clipped {_n_clipped} OHLCV bars beyond last signal "
            f"{sig_max_date.date()} — open positions lock at window end, not dataset end"
        )

    # Single shared deserializer (§11 R3): 3-way partition of the `engine:` block into EngineConfig
    # fields / recombine-pipeline keys / cost keys. The recombine keys (entry_gate, ensemble heads,
    # xsec, regime, …) are consumed by the signal layer, not EngineConfig; portfolio_cfg is the removed
    # legacy tier's sub-block, checked for enabled=true just below.
    engine, portfolio_cfg = engine_config_from_dict(cfg.engine)

    equity_curve = None
    if portfolio_cfg.get("enabled"):
        raise ValueError(
            f"[{cfg.name}] engine.portfolio.enabled is no longer supported — the legacy "
            "alpha->portfolio->execution tier (src/portfolio + src/execution) was removed. "
            "Only refuted xsec top-k templates (629-631) ever enabled it."
        )
    print(f"[{cfg.name}] backtesting {len(signals_all)} signals")
    trades = run_backtest(signals_all, ohlcv, engine)
    trades_df = trades_to_dataframe(trades)

    # A position still open at the last evaluated bar is a window-end carry, not a real
    # exit. Relabel 'end_of_data' -> 'open' so the dashboard shows a locked-but-open
    # trade (PnL marked to the window's last close) instead of a misleading hold that
    # appears dragged to the end of the dataset.
    if not trades_df.empty:
        trades_df.loc[trades_df["exit_reason"] == "end_of_data", "exit_reason"] = "open"

    agg = aggregate_stats(trades_df)
    yearly = per_year_stats(trades_df)
    daily = per_day_stats(trades_df)
    by_sym = per_symbol_stats(trades_df)

    # Calculate scoring metrics for leaderboard (required by composite_score)
    from stock_ml.src.evaluation.scoring import (
        calc_max_drawdown,
        calc_mdd_per_symbol,
        calc_sharpe,
        calc_yearly_consistency,
    )

    trades_list = [row.to_dict() for _, row in trades_df.iterrows()] if len(trades_df) > 0 else []
    sharpe = calc_sharpe(trades_list)
    max_drawdown = calc_max_drawdown(trades_list)
    mdd_per_symbol = calc_mdd_per_symbol(trades_list)
    yearly_consistency = calc_yearly_consistency(trades_list)

    # Gap must cover the LONGEST forward span of any target actually trained, including
    # per-slot targets (Phase 0.4): the exit head can look further forward than the global
    # cfg.target (e.g. risk_exit h40, or a zigzag pivot whose confirmation is unbounded, vs
    # global h10). Sizing the gap to cfg.target's horizon alone let those per-slot forward
    # labels leak across the train/test boundary while this audit silently passed.
    # The ENSEMBLE heads (entry_ensemble / entry_ensemble2 / exit_ensemble) train their own
    # forward-looking targets too — fold their spans in as well, or an ensemble head with a
    # horizon longer than the entry/exit slots would under-size the gap and leak undetected
    # (same bug-class as the per-slot leak above, one slot-type deeper).
    _ens_target_cfgs = []
    if isinstance(cfg.engine, dict):
        for _ens_key in ("entry_ensemble", "entry_ensemble2", "entry_ensemble3",
                         "entry_ensemble4", "entry_ensemble5", "exit_ensemble", "exit_force_gate_vn30"):
            _ens = cfg.engine.get(_ens_key)
            if isinstance(_ens, dict) and _ens.get("target"):
                _ens_target_cfgs.append(_ens["target"])
    _max_span = max(
        target_forward_span(entry_target_cfg),
        target_forward_span(exit_target_cfg),
        *(target_forward_span(_t) for _t in _ens_target_cfgs),
    )
    required_gap = max(_max_span * 2 + 5, 7)
    # Use collected windows for purged_kfold, pre-computed for walk_forward_year
    audit_windows = (
        windows if windows is not None else windows_list if "windows_list" in locals() else None
    )
    report = audit_report(
        trades_df, signals_all, windows=audit_windows, min_gap_days=required_gap,
        portfolio_mode=False,
    )
    print_report(report)

    if cfg.strict_audit and report["overall"] == "FAIL":
        raise ValueError(f"[{cfg.name}] Strict audit failed: {report['n_fail']} check(s) failed")

    trades_path = out / f"trades_{cfg.name}.csv"
    signals_path = out / f"signals_{cfg.name}.csv"
    daily_path = out / f"daily_stats_{cfg.name}.csv"
    yearly_path = out / f"yearly_stats_{cfg.name}.csv"
    symbol_path = out / f"symbol_stats_{cfg.name}.csv"
    summary_path = out / f"summary_{cfg.name}.json"

    # P3: the DB pipeline (run_template) consumes these frames in-memory and writes
    # straight to repos; CSVs are an opt-in debug export kept for the legacy file
    # runner (run_experiments.py).
    if export_csv:
        trades_df.to_csv(trades_path, index=False)
        signals_all.to_csv(signals_path, index=False)
        daily.to_csv(daily_path, index=False)
        yearly.to_csv(yearly_path, index=False)
        by_sym.to_csv(symbol_path, index=False)

    # Portfolio path: persist the capital curve and its risk/return metrics.
    equity_summary = None
    if equity_curve is not None and not equity_curve.empty:
        from stock_ml.src.backtest.stats import equity_stats

        if export_csv:
            equity_curve.to_csv(out / f"equity_{cfg.name}.csv", index=False)
        equity_summary = equity_stats(equity_curve)

    summary = {
        "name": cfg.name,
        "strategy": cfg.strategy,
        "market": cfg.market,
        "feature_set": cfg.feature_set,
        "n_symbols": int(ohlcv["symbol"].nunique()),
        "first_test_year": getattr(splitter, "first_test_year", None),
        "last_test_year": getattr(splitter, "last_test_year", None),
        "n_trades": int(len(trades_df)),
        "n_signals_buy": int((signals_all["signal"] > 0).sum()),
        "n_signals_sell": int((signals_all["signal"] < 0).sum()),
        "entry_model": cfg.entry_model["type"],
        "model_mode": cfg.model_mode,
        "signal_mode": cfg.signal_mode,
        "direction": cfg.direction,
        "exit_model_type": cfg.exit_model["type"],
        "exit_model_enabled": cfg.exit_model.get("enabled", False),
        "universe_slug": (cfg.universe or {}).get("slug"),
        "universe_version": (cfg.universe or {}).get("resolved_version"),
        "aggregate": agg,
        "equity": equity_summary,
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "mdd_per_symbol": float(mdd_per_symbol),
        "yearly_consistency": float(yearly_consistency),
        "audit": report,
        "outputs": {
            "trades": str(trades_path.relative_to(out.parent)),
            "signals": str(signals_path.relative_to(out.parent)),
            "daily_stats": str(daily_path.relative_to(out.parent)),
            "yearly_stats": str(yearly_path.relative_to(out.parent)),
            "symbol_stats": str(symbol_path.relative_to(out.parent)),
        },
        "config": {
            "feature_set": cfg.feature_set,
            # Per-slot feature sets actually used by the entry/exit heads. The global
            # `feature_set` above is the legacy template field and is misleading for
            # decoupled dual-ML models (it ignores the slot overrides).
            "entry_features": cfg.entry_features,
            "exit_features": cfg.exit_features,
            "target": cfg.target,
            "entry_target": entry_target_cfg,
            "exit_target": exit_target_cfg,
            # Max forward span across BOTH per-slot targets (entry h6 / exit h40),
            # the honest label horizon — not cfg.target's global horizon, which under-
            # reports the exit head and made the stored metadata misleading.
            "target_forward_window": int(_max_span),
            "entry_model": cfg.entry_model,
            "exit_model": cfg.exit_model,
            "split": cfg.split,
            "engine": {
                "max_hold_bars": engine.max_hold_bars,
                "min_hold_bars": engine.min_hold_bars,
                "hard_stop_pct": engine.hard_stop_pct,
                "commission": engine.cost.commission,
                "tax": engine.cost.tax,
                "slippage": engine.cost.slippage,
            },
        },
    }
    if cfg.metadata:
        summary["metadata"] = cfg.metadata
    if export_csv:
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[{cfg.name}] wrote outputs to {out}")

    # P3: hand the detail frames back in-memory so the DB pipeline persists straight
    # to repos without a CSV round-trip. Attached after the JSON dump above so the
    # frames are never serialized into summary_*.json.
    summary["_run_detail_frames"] = {
        "trades": trades_df,
        "signals": signals_all,
        "yearly": yearly,
        "symbol": by_sym,
    }
    return summary
