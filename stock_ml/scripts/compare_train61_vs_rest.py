from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_unseen_symbols_api import TOP1_CONFIG_PATH, summarize_trades  # noqa: E402
from src.components.exit_models.registry import get_exit_model  # noqa: E402
from src.components.models.registry import get_model  # noqa: E402
from src.config_loader import load_config, resolve_data_dir  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.splitter import WalkForwardSplitter  # noqa: E402
from src.data.target import TargetGenerator  # noqa: E402
from src.features.engine import FeatureEngine  # noqa: E402
from src.pipeline import ExperimentConfig, Pipeline  # noqa: E402
from src.signal_adapter import canonicalize_predictions  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


COMPARE_61_PATH = ROOT / "results" / "compare_pooled_vs_per_symbol_61.json"
OUT_JSON = ROOT / "results" / "compare_train61_vs_rest486.json"
OUT_CSV = ROOT / "results" / "compare_train61_vs_rest486_by_symbol.csv"


def _load_raw(symbols: list[str], cutoff_date: str) -> pd.DataFrame:
    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)
    loader = DataLoader(abs_data_dir)
    raw = loader.load_all(symbols=symbols, show_progress=True)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).copy()
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    raw = raw[raw["timestamp"] <= cutoff].copy()
    return raw.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _get_universe_symbols() -> list[str]:
    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)
    loader = DataLoader(abs_data_dir)
    return sorted(str(s).upper() for s in loader.symbols)


def _prepare_features_and_labels(
    cfg: ExperimentConfig,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], bool]:
    engine = FeatureEngine(feature_set=cfg.feature_set())
    feat = engine.compute_for_all_symbols(raw)
    feature_cols = engine.get_feature_columns(feat)

    split_cfg = cfg.split
    legacy_split = {
        "split": {
            "method": split_cfg.method,
            "train_years": split_cfg.train_years,
            "test_years": split_cfg.test_years,
            "gap_days": split_cfg.gap_days,
            "first_test_year": split_cfg.first_test_year,
            "last_test_year": split_cfg.last_test_year,
        },
        "target": cfg.target_dict(),
    }

    target_gen = TargetGenerator.from_config(legacy_split)
    labeled = target_gen.generate_for_all_symbols(feat.copy())
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        labeled = TargetGenerator.generate_exit_labels(
            labeled,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in labeled.columns
    if has_exit:
        drop_cols.append("target_sell")
    labeled = labeled.dropna(subset=drop_cols).copy()
    labeled["symbol"] = labeled["symbol"].astype(str).str.upper()
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], utc=True, errors="coerce")
    return labeled, feature_cols, has_exit


def _splitter(cfg: ExperimentConfig) -> WalkForwardSplitter:
    split_cfg = cfg.split
    return WalkForwardSplitter(
        method=split_cfg.method,
        train_years=split_cfg.train_years,
        test_years=split_cfg.test_years,
        gap_days=split_cfg.gap_days,
        first_test_year=split_cfg.first_test_year,
        last_test_year=split_cfg.last_test_year,
    )


def _fit_predict_cache(
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    has_exit: bool,
    symbols: list[str],
) -> list[dict[str, Any]]:
    pred_cache: list[dict[str, Any]] = []
    exit_model_cfg = cfg.signals.exit_model
    target_cfg = cfg.target_dict()

    for window in _splitter(cfg).get_windows():
        train_mask = (train_df["timestamp"] >= window.train_start) & (
            train_df["timestamp"] <= window.train_end
        )
        test_mask = (test_df["timestamp"] >= window.test_start) & (
            test_df["timestamp"] <= window.test_end
        )
        tr = train_df[train_mask]
        te = test_df[test_mask]
        if len(tr) < 20 or te.empty:
            continue

        y_train = tr["target"].values.astype(int)
        if len(np.unique(y_train)) < 2:
            continue

        model = get_model(
            cfg.entry_model_type(),
            device="cpu",
            **cfg.signals.entry_model.extras,
        )
        X_train = np.nan_to_num(tr[feature_cols].values)
        model.fit(X_train, y_train)

        sell_model = None
        if has_exit:
            y_exit_train = tr["target_sell"].values.astype(int)
            if len(np.unique(y_exit_train)) >= 2:
                sell_model = get_exit_model(
                    exit_model_cfg.type,
                    device="cpu",
                    **exit_model_cfg.extras,
                )
                sell_model.fit(X_train, y_exit_train)

        for sym, sym_df in te.groupby("symbol"):
            sym = str(sym).upper()
            if sym not in symbols:
                continue
            sym_df = sym_df.sort_values("timestamp").reset_index(drop=True)
            if len(sym_df) < 10:
                continue
            X_sym = np.nan_to_num(sym_df[feature_cols].values)
            y_pred = canonicalize_predictions(model.predict(X_sym), target_cfg)

            y_proba = None
            classes = None
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_sym)
                    final_est = model.steps[-1][1] if hasattr(model, "steps") else model
                    classes = list(final_est.classes_)
            except Exception:
                y_proba = None

            pred_cache.append(
                {
                    "symbol": sym,
                    "y_pred": y_pred,
                    "y_pred_exit": (
                        sell_model.predict(X_sym).astype(int) if sell_model is not None else None
                    ),
                    "y_proba": y_proba,
                    "classes": classes,
                    "returns": sym_df["return_1d"].values,
                    "sym_test_df": sym_df,
                    "feature_cols": feature_cols,
                }
            )

    return pred_cache


def _fit_predict_cache_per_symbol(
    cfg: ExperimentConfig,
    labeled: pd.DataFrame,
    feature_cols: list[str],
    has_exit: bool,
    symbols: list[str],
) -> list[dict[str, Any]]:
    pred_cache: list[dict[str, Any]] = []
    for idx, sym in enumerate(symbols, 1):
        sym_df = labeled[labeled["symbol"] == sym].copy()
        if sym_df.empty:
            continue
        sym_cache = _fit_predict_cache(
            cfg,
            train_df=sym_df,
            test_df=sym_df,
            feature_cols=feature_cols,
            has_exit=has_exit,
            symbols=[sym],
        )
        pred_cache.extend(sym_cache)
        if idx % 25 == 0 or idx == len(symbols):
            print(f"  per-symbol progress: {idx}/{len(symbols)} cache_items={len(pred_cache)}")
    return pred_cache


def _by_symbol_metrics(trades_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades_df is None or trades_df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for sym, group in trades_df.groupby("symbol"):
        out[str(sym).upper()] = summarize_trades(group.to_dict("records"))
    return out


def _comparison_rows(
    symbols: list[str],
    pooled_by_symbol: dict[str, dict[str, Any]],
    per_by_symbol: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        pooled = pooled_by_symbol.get(sym, {})
        per = per_by_symbol.get(sym, {})
        row = {
            "symbol": sym,
            "train61_trades": int(pooled.get("trades", 0) or 0),
            "train61_wr": float(pooled.get("wr", 0) or 0),
            "train61_total_pnl": float(pooled.get("total_pnl", 0) or 0),
            "train61_pf": float(pooled.get("pf", 0) or 0),
            "train61_comp": float(pooled.get("composite_score", 0) or 0),
            "per_symbol_trades": int(per.get("trades", 0) or 0),
            "per_symbol_wr": float(per.get("wr", 0) or 0),
            "per_symbol_total_pnl": float(per.get("total_pnl", 0) or 0),
            "per_symbol_pf": float(per.get("pf", 0) or 0),
            "per_symbol_comp": float(per.get("composite_score", 0) or 0),
        }
        row["delta_pnl_per_minus_train61"] = round(
            row["per_symbol_total_pnl"] - row["train61_total_pnl"], 4
        )
        row["delta_wr_per_minus_train61"] = round(row["per_symbol_wr"] - row["train61_wr"], 4)
        row["delta_comp_per_minus_train61"] = round(row["per_symbol_comp"] - row["train61_comp"], 4)
        row["winner_by_comp"] = (
            "per_symbol"
            if row["delta_comp_per_minus_train61"] > 0
            else ("train61" if row["delta_comp_per_minus_train61"] < 0 else "tie")
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["delta_comp_per_minus_train61", "delta_pnl_per_minus_train61"],
        ascending=[False, False],
    )


def main() -> int:
    started = time.time()
    cfg = ExperimentConfig.from_yaml(TOP1_CONFIG_PATH)

    compare_61 = json.loads(COMPARE_61_PATH.read_text(encoding="utf-8"))
    train_symbols = sorted(str(s).upper() for s in compare_61["symbols"])
    cutoff_date = compare_61.get("window", {}).get("cutoff_date", "2025-12-31")

    universe_symbols = _get_universe_symbols()
    train_set = set(train_symbols)
    rest_symbols = [s for s in universe_symbols if s not in train_set]

    print(f"Universe symbols: {len(universe_symbols)}")
    print(f"Train-61 symbols: {len(train_symbols)}")
    print(f"Rest symbols: {len(rest_symbols)}")
    print(f"Cutoff date: {cutoff_date}")

    raw = _load_raw(universe_symbols, cutoff_date)
    usable_symbols = sorted(raw["symbol"].unique().tolist())
    rest_symbols = [s for s in rest_symbols if s in usable_symbols]

    print("Computing features and labels once for the full universe...")
    labeled, feature_cols, has_exit = _prepare_features_and_labels(cfg, raw)

    train_df = labeled[labeled["symbol"].isin(train_symbols)].copy()
    rest_df = labeled[labeled["symbol"].isin(rest_symbols)].copy()
    rest_symbols = [s for s in rest_symbols if s in set(rest_df["symbol"].unique())]
    print(f"Usable rest symbols after features/labels: {len(rest_symbols)}")

    print("Running pooled model: train 61 symbols, backtest rest symbols...")
    pooled_cache = _fit_predict_cache(
        cfg,
        train_df=train_df,
        test_df=rest_df,
        feature_cols=feature_cols,
        has_exit=has_exit,
        symbols=rest_symbols,
    )
    print(f"  train61 prediction cache items: {len(pooled_cache)}")
    pooled_result = Pipeline(
        cfg, symbols=rest_symbols, device="cpu", prediction_cache=pooled_cache
    ).run()
    pooled_trades_df = pooled_result.trades_df.copy()
    if not pooled_trades_df.empty:
        pooled_trades_df["symbol"] = pooled_trades_df["symbol"].astype(str).str.upper()

    print("Running per-symbol models on rest symbols...")
    per_cache = _fit_predict_cache_per_symbol(
        cfg,
        labeled=rest_df,
        feature_cols=feature_cols,
        has_exit=has_exit,
        symbols=rest_symbols,
    )
    print(f"  per-symbol prediction cache items: {len(per_cache)}")
    per_result = Pipeline(cfg, symbols=rest_symbols, device="cpu", prediction_cache=per_cache).run()
    per_trades_df = per_result.trades_df.copy()
    if not per_trades_df.empty:
        per_trades_df["symbol"] = per_trades_df["symbol"].astype(str).str.upper()

    pooled_trades = pooled_trades_df.to_dict("records") if not pooled_trades_df.empty else []
    per_trades = per_trades_df.to_dict("records") if not per_trades_df.empty else []
    pooled_by_symbol = _by_symbol_metrics(pooled_trades_df)
    per_by_symbol = _by_symbol_metrics(per_trades_df)
    by_symbol = _comparison_rows(rest_symbols, pooled_by_symbol, per_by_symbol)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(TOP1_CONFIG_PATH),
        "cutoff_date": cutoff_date,
        "train61_symbol_count": len(train_symbols),
        "rest_symbol_count": len(rest_symbols),
        "train61_symbols": train_symbols,
        "overall": {
            "train61_eval_rest": summarize_trades(pooled_trades),
            "per_symbol_eval_rest": summarize_trades(per_trades),
        },
        "summary": {
            "per_symbol_better_by_comp": int((by_symbol["winner_by_comp"] == "per_symbol").sum()),
            "train61_better_by_comp": int((by_symbol["winner_by_comp"] == "train61").sum()),
            "ties": int((by_symbol["winner_by_comp"] == "tie").sum()),
            "avg_delta_pnl_per_minus_train61": round(
                float(by_symbol["delta_pnl_per_minus_train61"].mean()), 4
            ),
            "avg_delta_comp_per_minus_train61": round(
                float(by_symbol["delta_comp_per_minus_train61"].mean()), 4
            ),
        },
        "top20_per_symbol_better": by_symbol.head(20).to_dict("records"),
        "top20_train61_better": by_symbol.sort_values(
            "delta_comp_per_minus_train61", ascending=True
        )
        .head(20)
        .to_dict("records"),
        "runtime_sec": round(time.time() - started, 1),
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    by_symbol.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(OUT_JSON)
    print(OUT_CSV)
    print("\nOverall:")
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))
    print("\nSummary:")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
