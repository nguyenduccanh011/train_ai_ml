"""
Tìm các mã cổ phiếu có tín hiệu MUA gần nhất chưa BÁN (open signals)
Sử dụng model top 1: v22_exit_ablation_round42
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Config cho model top 1
TOP1_CONFIG_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "v22_exit_ablation_round42"
    / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
    / "config.resolved.yaml"
)


@dataclass
class OpenSignal:
    symbol: str
    entry_date: str
    entry_price: float
    days_held: int
    current_price: float
    unrealized_pnl_pct: float
    signal_strength: float = 0.0


def load_symbols() -> list[str]:
    """Load danh sách symbols từ manifest"""
    manifest_path = ROOT / "visualization" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(sym).upper() for sym in manifest.get("base_symbols", [])]

    # Fallback: load từ data directory
    from src.config_loader import get_pipeline_config, resolve_data_dir

    pipeline = get_pipeline_config()
    data_dir = resolve_data_dir(
        pipeline.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    )
    data_path = Path(data_dir)
    if data_path.exists():
        symbols = sorted({f.stem.upper() for f in data_path.glob("*.parquet")})
        return symbols

    return []


def generate_signals_for_symbol(symbol: str) -> tuple[list[dict], pd.DataFrame]:
    """Generate signals cho 1 symbol và trả về trades + raw data"""
    import numpy as np
    import src.data.target as target_module
    import src.features.engine as feature_engine_module
    from src.cache.feature_cache import FeatureCacheManager
    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.config_loader import load_config, resolve_data_dir
    from src.data.loader import DataLoader
    from src.data.target import TargetGenerator
    from src.env import get_results_dir
    from src.features.engine import FeatureEngine
    from src.pipeline import ExperimentConfig, Pipeline
    from src.signal_adapter import canonicalize_predictions

    # Load config
    cfg = ExperimentConfig.from_yaml(TOP1_CONFIG_PATH)

    # Load data
    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)

    loader = DataLoader(abs_data_dir)
    feature_set = cfg.feature_set()
    engine = FeatureEngine(feature_set=feature_set)

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

    # Load features with cache
    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__]

    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=[symbol],
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=legacy_split.get("target", {}),
        code_paths=code_paths,
    )

    if df is None:
        raw_df = loader.load_all(symbols=[symbol])
        df = engine.compute_for_all_symbols(raw_df)
        cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=[symbol],
            timeframe=loader.timeframe,
            feature_set=feature_set,
            target_config=legacy_split.get("target", {}),
            code_paths=code_paths,
        )

    df = target_gen.generate_for_all_symbols(df)

    # Generate exit labels
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        df = TargetGenerator.generate_exit_labels(
            df,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    feature_cols = engine.get_feature_columns(df)
    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in df.columns
    if has_exit:
        drop_cols.append("target_sell")

    df = df.dropna(subset=drop_cols)
    sym_df = df[df["symbol"].astype(str).str.upper() == symbol.upper()].reset_index(drop=True)

    if len(sym_df) < 20:
        return [], sym_df

    # Train models
    X = np.nan_to_num(sym_df[feature_cols].values)
    y = sym_df["target"].values.astype(int)

    model = get_model(cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras)
    model.fit(X, y)
    y_pred = canonicalize_predictions(model.predict(X), legacy_split.get("target", {}))

    sell_model = None
    if has_exit:
        exit_model_cfg = cfg.signals.exit_model
        sell_model = get_exit_model(exit_model_cfg.type, device="cpu", **exit_model_cfg.extras)
        sell_model.fit(X, sym_df["target_sell"].values.astype(int))

    # Build prediction cache
    prediction_cache = [
        {
            "symbol": symbol,
            "y_pred": y_pred,
            "y_pred_exit": sell_model.predict(X).astype(int) if sell_model is not None else None,
            "y_proba": None,
            "classes": None,
            "returns": sym_df["return_1d"].values,
            "sym_test_df": sym_df,
            "feature_cols": feature_cols,
        }
    ]

    # Run pipeline
    result = Pipeline(cfg, symbols=[symbol], device="cpu", prediction_cache=prediction_cache).run()

    if result.trades_df is None or result.trades_df.empty:
        return [], sym_df

    trades = result.trades_df[result.trades_df["symbol"].str.upper() == symbol.upper()].to_dict(
        "records"
    )
    return trades, sym_df


def find_open_signals(symbols: list[str], max_symbols: int = None) -> list[OpenSignal]:
    """Tìm các signals đang mở (mua chưa bán)"""
    open_signals = []
    total = len(symbols) if max_symbols is None else min(len(symbols), max_symbols)

    print(f"\nDang quet {total} ma co phieu...")
    print("=" * 80)

    for idx, symbol in enumerate(symbols[:total], 1):
        try:
            print(f"[{idx}/{total}] {symbol}...", end=" ", flush=True)

            trades, sym_df = generate_signals_for_symbol(symbol)

            if not trades:
                print("Khong co trade")
                continue

            # Tìm trade gần nhất
            trades_sorted = sorted(trades, key=lambda t: t.get("entry_date", ""), reverse=True)
            latest_trade = trades_sorted[0]

            # Kiểm tra xem có exit_date không
            if latest_trade.get("exit_date") and latest_trade["exit_date"] != "":
                print(f"Da dong ({latest_trade['exit_date']})")
                continue

            # Trade đang mở!
            entry_date = latest_trade.get("entry_date", "")
            entry_price = latest_trade.get("entry_price", 0)

            # Lấy giá hiện tại (giá close gần nhất)
            if not sym_df.empty:
                latest_data = sym_df.iloc[-1]
                current_price = float(latest_data.get("close", entry_price))
                current_date = latest_data.get("date") or latest_data.get("timestamp")
                if hasattr(current_date, "date"):
                    current_date = current_date.date()
            else:
                current_price = entry_price
                current_date = None

            # Tính unrealized PnL
            if entry_price > 0:
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                unrealized_pnl_pct = 0

            # Tính số ngày hold
            try:
                if isinstance(entry_date, str):
                    entry_dt = datetime.strptime(entry_date[:10], "%Y-%m-%d")
                else:
                    entry_dt = entry_date

                if current_date:
                    if isinstance(current_date, str):
                        current_dt = datetime.strptime(str(current_date)[:10], "%Y-%m-%d")
                    else:
                        current_dt = current_date
                else:
                    current_dt = datetime.now()

                days_held = (current_dt - entry_dt).days
            except Exception:
                days_held = 0

            signal = OpenSignal(
                symbol=symbol,
                entry_date=str(entry_date)[:10],
                entry_price=entry_price,
                days_held=days_held,
                current_price=current_price,
                unrealized_pnl_pct=unrealized_pnl_pct,
            )

            open_signals.append(signal)
            pnl_str = f"{unrealized_pnl_pct:+.2f}%"
            pnl_color = "+" if unrealized_pnl_pct >= 0 else "-"
            print(f"MO | Entry: {entry_date} | PnL: {pnl_color} {pnl_str} | Hold: {days_held}d")

        except Exception as e:
            print(f"Loi: {str(e)[:50]}")
            continue

    return open_signals


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tìm tín hiệu mua chưa bán (open signals)")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Số lượng mã tối đa cần quét (mặc định: tất cả)",
    )
    parser.add_argument(
        "--output", type=str, default="open_signals_top1.csv", help="File output CSV"
    )
    parser.add_argument(
        "--min-pnl",
        type=float,
        default=None,
        help="Lọc chỉ hiển thị signals có PnL >= giá trị này (%)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("TIM TIN HIEU MUA CHUA BAN - MODEL TOP 1")
    print("=" * 80)
    print("Model: v22_exit_ablation_round42")
    print("Leaderboard: #1 | PnL: 15,768.62 | WR: 74.64%")
    print("=" * 80)

    # Load symbols
    symbols = load_symbols()
    print(f"\nTong so ma: {len(symbols)}")

    # Find open signals
    open_signals = find_open_signals(symbols, max_symbols=args.max_symbols)

    # Filter by min PnL if specified
    if args.min_pnl is not None:
        open_signals = [s for s in open_signals if s.unrealized_pnl_pct >= args.min_pnl]

    # Sort by unrealized PnL
    open_signals.sort(key=lambda s: s.unrealized_pnl_pct, reverse=True)

    # Display results
    print("\n" + "=" * 80)
    print(f"KET QUA: Tim thay {len(open_signals)} tin hieu dang mo")
    print("=" * 80)

    if open_signals:
        print(
            f"\n{'Symbol':<8} {'Entry Date':<12} {'Entry Price':<12} {'Current':<12} {'PnL %':<10} {'Days':<6}"
        )
        print("-" * 80)

        for signal in open_signals:
            pnl_str = f"{signal.unrealized_pnl_pct:+.2f}%"
            print(
                f"{signal.symbol:<8} {signal.entry_date:<12} {signal.entry_price:<12.2f} "
                f"{signal.current_price:<12.2f} {pnl_str:<10} {signal.days_held:<6}"
            )

        # Save to CSV
        df = pd.DataFrame([vars(s) for s in open_signals])
        output_path = ROOT / args.output
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nDa luu ket qua vao: {output_path}")

        # Summary stats
        avg_pnl = sum(s.unrealized_pnl_pct for s in open_signals) / len(open_signals)
        positive = sum(1 for s in open_signals if s.unrealized_pnl_pct > 0)
        negative = len(open_signals) - positive

        print("\nTHONG KE:")
        print(f"   - Trung binh PnL: {avg_pnl:+.2f}%")
        print(f"   - Dang lai: {positive} ma ({positive / len(open_signals) * 100:.1f}%)")
        print(f"   - Dang lo: {negative} ma ({negative / len(open_signals) * 100:.1f}%)")
        print(
            f"   - Tot nhat: {open_signals[0].symbol} ({open_signals[0].unrealized_pnl_pct:+.2f}%)"
        )
        print(
            f"   - Te nhat: {open_signals[-1].symbol} ({open_signals[-1].unrealized_pnl_pct:+.2f}%)"
        )

    else:
        print("\nKhong tim thay tin hieu nao dang mo")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
