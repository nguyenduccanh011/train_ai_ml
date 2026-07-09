"""
Retrain model top 1 de kiem tra:
1. Pivot fix co anh huong khong (so sanh voi leaderboard)
2. Boundary leakage (gap_days=0 + forward_window=15-20)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline import ExperimentConfig, Pipeline

# Load config top 1
config_path = (
    ROOT
    / "results/experiments/v22_exit_ablation_round25/v22_exit_ablation_round25_signals_target-earlyv2_fw15_g04_l02-exit_model-exit_fw20_l035-fusion-peak_dist_only/config.resolved.yaml"
)
cfg = ExperimentConfig.from_yaml(config_path)

# Override gap_days = 25 de fix boundary leakage
# Target forward_window=15, exit forward_window=20 -> gap_days >= 20 la an toan
cfg.split.gap_days = 25
print(f"[FIX] Override gap_days = {cfg.split.gap_days} (boundary leakage fix)")

# Load symbols
manifest_path = ROOT / "visualization/manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
symbols = [str(s).upper() for s in manifest.get("base_symbols", [])]

print("=" * 80)
print("RETRAIN MODEL TOP 1 - LEAKAGE CHECK")
print("=" * 80)
print(f"Config: {cfg.name}")
print(f"Test years: {cfg.split.first_test_year} - {cfg.split.last_test_year}")
print(f"Gap days: {cfg.split.gap_days}")
print("Target forward_window: 15")
print("Exit forward_window: 20")
print(f"Symbols: {len(symbols)}")
print("=" * 80)
print()
print("CHECK 1: Pivot fix")
print("  - Feature cache leading_v2 moi se dung pivot fix")
print("  - So sanh metrics voi leaderboard (WR=78.2%, PF=15.16)")
print()
print("CHECK 2: Boundary leakage")
print("  - gap_days=0 -> train end = 31/12/YYYY")
print("  - Target tai 31/12 nhin forward 15-20 ngay -> 15-20/01/YYYY+1 (test set)")
print("  - Day la LEAKAGE nghiem trong")
print("=" * 80)
print()
print("Chay pipeline (5-15 phut)...")
print()
# Chạy pipeline
result = Pipeline(cfg, symbols=symbols, device="cpu").run()

print("\n" + "=" * 80)
print("KET QUA RETRAIN:")
print("=" * 80)

# Tính metrics từ trades_df
df = result.trades_df
total_trades = len(df)
wins = len(df[df['pnl_pct'] > 0])
wr = (wins / total_trades * 100) if total_trades > 0 else 0
total_pnl = df['pnl_pct'].sum()
total_win = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
total_loss = abs(df[df['pnl_pct'] < 0]['pnl_pct'].sum())
pf = (total_win / total_loss) if total_loss > 0 else 0
avg_hold = df['holding_days'].mean()

# Max drawdown (simple)
df_sorted = df.sort_values('exit_date')
cumsum = df_sorted['pnl_pct'].cumsum()
running_max = cumsum.cummax()
drawdown = running_max - cumsum
max_dd = drawdown.max()

metrics = {
    'trades': total_trades,
    'wr': wr,
    'pf': pf,
    'total_pnl': total_pnl,
    'max_drawdown': max_dd,
    'avg_hold': avg_hold,
}

print(f"Total trades: {metrics['trades']}")
print(f"Win rate: {metrics['wr']:.2f}%")
print(f"Profit factor: {metrics['pf']:.2f}")
print(f"Total PnL: {metrics['total_pnl']:.2f}")
print(f"Max drawdown: {metrics['max_drawdown']:.2f}")
print(f"Avg hold: {metrics['avg_hold']:.1f} days")

print("\n" + "=" * 80)
print("SO SANH VOI LEADERBOARD:")
print("=" * 80)
print("Leaderboard (co the leaky):")
print("  WR: 78.2%")
print("  PF: 15.16")
print("  Trades: 1225")
print("  Total PnL: 15977.05")
print()
print("Retrain (pivot fix applied):")
print(f"  WR: {metrics['wr']:.2f}%")
print(f"  PF: {metrics['pf']:.2f}")
print(f"  Trades: {metrics['trades']}")
print(f"  Total PnL: {metrics['total_pnl']:.2f}")
print()

wr_delta = metrics['wr'] - 78.2
pf_delta = metrics['pf'] - 15.16

print(f"Delta WR: {wr_delta:+.2f}%")
print(f"Delta PF: {pf_delta:+.2f}")

if abs(wr_delta) < 0.5 and abs(pf_delta) < 0.5:
    print("\n[!] Metrics gan nhu khong doi -> Pivot fix khong anh huong nhieu")
    print("    HOAC feature cache chua invalidate dung")
elif wr_delta < -2 or pf_delta < -1:
    print("\n[OK] Metrics giam dang ke -> Pivot fix da loai bo leakage")
else:
    print("\n[?] Metrics thay doi nhe -> Can kiem tra them")

print("\n" + "=" * 80)
print("BOUNDARY LEAKAGE:")
print("=" * 80)
print("[!] VAN DE NGHIEM TRONG:")
print("  - gap_days=0 -> khong co khoang cach giua train/test")
print("  - Target tai row cuoi train nhin forward 15-20 ngay vao test set")
print("  - Model hoc tu future data -> metrics bi inflate")
print()
print("FIX DE XUAT:")
print("  - Tang gap_days len 20-25 (de target khong leak vao test)")
print("  - HOAC: Loai bo N rows cuoi train (N = forward_window)")
print("=" * 80)

# Luu ket qua
output_dir = ROOT / "results/leakage_check"
output_dir.mkdir(parents=True, exist_ok=True)
suffix = f"_gap{cfg.split.gap_days}" if cfg.split.gap_days > 0 else ""
trades_file = output_dir / f"top1_retrain{suffix}_trades.csv"
result.trades_df.to_csv(trades_file, index=False)

metrics_file = output_dir / f"top1_retrain{suffix}_metrics.json"
with open(metrics_file, "w") as f:
    json.dump({
        "retrain_metrics": metrics,
        "leaderboard_metrics": {
            "wr": 78.2,
            "pf": 15.16,
            "trades": 1225,
            "total_pnl": 15977.05,
        },
        "delta": {
            "wr": wr_delta,
            "pf": pf_delta,
        },
    }, f, indent=2)

print(f"\nSaved to: {output_dir}")
print(f"  - {trades_file.name}")
print(f"  - {metrics_file.name}")
