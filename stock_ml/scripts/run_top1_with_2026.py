"""
Chạy model top 1 với data đến 2026
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import ExperimentConfig, Pipeline  # noqa: E402

# Load config
config_path = (
    ROOT
    / "results/experiments/v22_exit_ablation_round42/v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only/config.resolved.yaml"
)
cfg = ExperimentConfig.from_yaml(config_path)
cfg.split.last_test_year = 2026

# Load symbols
manifest_path = ROOT / "visualization/manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
symbols = [str(s).upper() for s in manifest.get("base_symbols", [])]

print("=" * 80)
print("CHAY MODEL TOP 1 VOI DATA 2026")
print("=" * 80)
print(f"Config: {cfg.name}")
print(f"Test years: {cfg.split.first_test_year} - {cfg.split.last_test_year}")
print(f"Symbols: {len(symbols)}")
print("=" * 80)
print("Chay pipeline (co the mat 5-15 phut)...")

# Chạy pipeline
result = Pipeline(cfg, symbols=symbols, device="cpu").run()

print("\n" + "=" * 80)
print("KET QUA:")
print("=" * 80)
print(f"Total trades: {len(result.trades_df)}")

# Lưu kết quả
output_dir = ROOT / "results/experiments/v22_exit_ablation_round42_2026"
output_dir.mkdir(parents=True, exist_ok=True)
trades_file = output_dir / "trades.csv"
result.trades_df.to_csv(trades_file, index=False)
print(f"Saved trades to: {trades_file}")

# Kiểm tra open signals
import pandas as pd

df = result.trades_df
df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
open_trades = df[df["exit_date"].isna()]

print(f"\nOpen signals: {len(open_trades)}")
if len(open_trades) > 0:
    print("\nCAC MA DANG MO:")
    for _, row in open_trades.head(20).iterrows():
        print(f"  {row['symbol']}: Entry {row['entry_date']} @ {row.get('entry_price', 0):.2f}")

print("=" * 80)
