"""
Tìm tín hiệu MUA chưa BÁN - Predict trên TOÀN BỘ data (không giới hạn năm)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datetime import datetime

import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("TIM TIN HIEU MUA CHUA BAN - FULL DATA 2026")
    print("=" * 80)

    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.config_loader import get_pipeline_config, resolve_data_dir
    from src.data.loader import DataLoader
    from src.data.target import TargetGenerator
    from src.features.engine import FeatureEngine
    from src.pipeline import ExperimentConfig

    # Load config
    config_path = (
        ROOT
        / "results/experiments/v22_exit_ablation_round42/v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only/config.resolved.yaml"
    )
    cfg = ExperimentConfig.from_yaml(config_path)

    # Load symbols
    manifest_path = ROOT / "visualization/manifest.json"
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols = [str(s).upper() for s in manifest.get("base_symbols", [])][:100]

    print(f"Symbols: {len(symbols)}")

    # Load data
    pipeline_cfg = get_pipeline_config()
    data_dir = resolve_data_dir(pipeline_cfg.get("data_dir"))
    loader = DataLoader(data_dir)

    print("Loading all data...")
    all_df = loader.load_all(symbols=symbols)
    print(
        f"Loaded: {len(all_df)} rows, date range: {all_df['timestamp'].min()} to {all_df['timestamp'].max()}"
    )

    # Features
    engine = FeatureEngine(feature_set=cfg.feature_set())
    all_df = engine.compute_for_all_symbols(all_df)

    # Target
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
    all_df = target_gen.generate_for_all_symbols(all_df)

    # Exit labels
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        all_df = TargetGenerator.generate_exit_labels(
            all_df,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    feature_cols = engine.get_feature_columns(all_df)
    print(f"Features: {len(feature_cols)}")
    print("=" * 80)

    open_signals = []

    for idx, symbol in enumerate(symbols, 1):
        try:
            print(f"[{idx}/{len(symbols)}] {symbol}...", end=" ", flush=True)

            sym_df = all_df[all_df["symbol"].astype(str).str.upper() == symbol.upper()].copy()
            if sym_df.empty:
                print("No data")
                continue

            drop_cols = feature_cols + ["target"]
            if "target_sell" in sym_df.columns:
                drop_cols.append("target_sell")

            sym_df = sym_df.dropna(subset=drop_cols).reset_index(drop=True)
            if len(sym_df) < 20:
                print("Too few")
                continue

            # Train trên TOÀN BỘ data
            X = np.nan_to_num(sym_df[feature_cols].values)
            y = sym_df["target"].values.astype(int)

            model = get_model(
                cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras
            )
            model.fit(X, y)
            y_pred = model.predict(X)

            y_pred_exit = None
            if "target_sell" in sym_df.columns:
                exit_model = get_exit_model(
                    cfg.signals.exit_model.type, device="cpu", **cfg.signals.exit_model.extras
                )
                exit_model.fit(X, sym_df["target_sell"].values.astype(int))
                y_pred_exit = exit_model.predict(X)

            sym_df["signal"] = y_pred
            sym_df["exit_signal"] = y_pred_exit if y_pred_exit is not None else 0

            # Tìm entry gần nhất
            entry_indices = sym_df[sym_df["signal"] == 1].index.tolist()
            if not entry_indices:
                print("No entry")
                continue

            last_entry_idx = entry_indices[-1]

            # Có exit sau entry không?
            has_exit = False
            if y_pred_exit is not None:
                for i in range(last_entry_idx + 1, len(sym_df)):
                    if sym_df.iloc[i]["exit_signal"] == 1:
                        has_exit = True
                        break

            if has_exit:
                print("Closed")
                continue

            # OPEN!
            entry_row = sym_df.iloc[last_entry_idx]
            latest_row = sym_df.iloc[-1]

            entry_date = entry_row.get("timestamp") or entry_row.get("date")
            entry_price = float(entry_row["close"])
            current_price = float(latest_row["close"])
            current_date = latest_row.get("timestamp") or latest_row.get("date")

            pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

            try:
                if hasattr(entry_date, "date"):
                    entry_dt = entry_date
                else:
                    entry_dt = datetime.strptime(str(entry_date)[:10], "%Y-%m-%d")
                if hasattr(current_date, "date"):
                    current_dt = current_date
                else:
                    current_dt = datetime.strptime(str(current_date)[:10], "%Y-%m-%d")
                days = (current_dt - entry_dt).days
            except:
                days = 0

            open_signals.append(
                {
                    "symbol": symbol,
                    "entry_date": str(entry_date)[:10],
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "current_date": str(current_date)[:10],
                    "pnl_pct": pnl,
                    "days_held": days,
                }
            )

            print(f"OPEN | {str(entry_date)[:10]} | {pnl:+.2f}% | {days}d")

        except Exception as e:
            print(f"Error: {str(e)[:40]}")

    print("\n" + "=" * 80)
    print(f"KET QUA: {len(open_signals)} open signals")
    print("=" * 80)

    if open_signals:
        df = pd.DataFrame(open_signals).sort_values("pnl_pct", ascending=False)
        print(
            f"\n{'Symbol':<8} {'Entry':<12} {'Current':<12} {'Entry$':<10} {'Current$':<10} {'PnL%':<10} {'Days':<6}"
        )
        print("-" * 90)
        for _, r in df.iterrows():
            print(
                f"{r['symbol']:<8} {r['entry_date']:<12} {r['current_date']:<12} {r['entry_price']:<10.2f} {r['current_price']:<10.2f} {r['pnl_pct']:+.2f}%    {r['days_held']:<6}"
            )

        out = ROOT / "open_signals_full_data.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {out}")

        avg = df["pnl_pct"].mean()
        pos = (df["pnl_pct"] > 0).sum()
        print(f"\nSTATS: Avg={avg:+.2f}% | Profit={pos}/{len(df)} ({pos / len(df) * 100:.1f}%)")

    print("=" * 80)


if __name__ == "__main__":
    main()
