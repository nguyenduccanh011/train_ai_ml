# Stock ML System — Hướng Dẫn Workflow Triển Khai Hoàn Chỉnh

**Mục lục**
1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Sơ Đồ Kiến Trúc](#2-sơ-đồ-kiến-trúc)
3. [Những Khái Niệm Chính](#3-những-khái-niệm-chính)
4. [Reproducibility & Experiment Tracking](#4-reproducibility--experiment-tracking)
5. [Bố Trí Dữ Liệu](#5-bố-trí-dữ-liệu)
6. [Tham Khảo Config](#6-tham-khảo-config)
7. [Tham Khảo Feature](#7-tham-khảo-feature)
8. [Workflow A: Chạy Backtest Mới](#8-workflow-a-chạy-backtest-mới)
9. [Workflow B: Đăng Ký Vào Leaderboard](#9-workflow-b-đăng-ký-model-vào-leaderboard)
10. [Workflow C: Ghim Lên Dashboard](#10-workflow-c-ghim-model-lên-dashboard)
11. [Workflow D: Retire Model](#11-workflow-d-retire-model)
12. [Workflow E: Chạy Live Simulation](#12-workflow-e-chạy-live-simulation)
13. [Tham Khảo Output Files](#13-tham-khảo-output-files)
14. [Model Lifecycle States](#14-model-lifecycle-states)
15. [Thêm Model Type Mới](#15-thêm-model-type-mới)

---

## 1. Tổng Quan Hệ Thống

**Stock ML** là hệ thống backtesting ML cho cổ phiếu Việt Nam, VN derivatives (futures), và cryptocurrency. Cung cấp hai chế độ thực thi:

- **Batch Backtest** (`run_v2`): Mô phỏng toàn bộ giai đoạn test như một phân tích lịch sử sử dụng walk-forward theo năm. Tốt để so sánh các tham số khác nhau và tối ưu hóa khoảng thời gian dài.
- **Live Simulation** (`run_live_sim`): Mô phỏng execution từng ngày, sinh signal tại T-1 và thực thi tại T mở cửa. Tốt để hiểu hành vi sẵn sàng deployment và tracking PnL hàng ngày.

**Nguyên Tắc Thiết Kế Cốt Lõi:**
- **Không Lookahead**: Mỗi feature tại bar `t` chỉ dùng dữ liệu ≤ `t`. Bắt buộc khoảng cách giữa training/testing để ngăn label leakage.
- **Signal Freezing**: Signal được sinh tại T-1 (đóng cửa hôm qua) và đóng băng bất biến (SHA1 hash) trước khi thực thi T. Ngăn chặn sửa đổi sau này, mô phỏng ràng buộc thực tế: quyết định phải made trước khi mở cửa.
- **Cost Modeling**: Bao gồm commission (0.15%), tax (0.10%), slippage (0.10%). Chi phí tính vào giá fill và PnL.
- **Tri-class Prediction**: Model dự đoán +1 (mua), 0 (trung lập), -1 (bán) dựa trên ngưỡng forward return.

---

## 2. Sơ Đồ Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                      Các File CSV OHLCV                          │
│        <data_root>/all_symbols/symbol=<SYM>/...                  │
└─────────────────────────────────────────────────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │   DataLoader         │
                    │ load_many() → OHLCV  │
                    └──────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  add_features()      │
                    │ 8 chỉ số kỹ thuật    │
                    └──────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │ ForwardReturnTarget  │
                    │ .apply() → labels    │
                    │  +1, 0, -1           │
                    └──────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  YearSplitter        │
                    │ walk-forward folds   │
                    │with gap enforcement  │
                    └──────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  BaselineModel       │
                    │LightGBM classifier   │
                    └──────────────────────┘
                               ↓
        ┌──────────────────┬──────────────────┐
        ↓                  ↓                  ↓
   run_backtest()    run_live_sim()   (cả 2 chế độ)
   fill bar sau      từng ngày
   hard_stop/signal  freezing signal
   max_hold exits
        ↓                  ↓
    Stats & Audit
   (aggregate, yearly, daily, symbol)
        ↓
    ┌──────────────────────────────────────┐
    │  Output: trades, signals, stats CSV  │
    │           summary JSON               │
    └──────────────────────────────────────┘
        ↓ (tùy chọn)
    ┌──────────────────────────────────────┐
    │  aggregator.append_or_update()       │
    │  → leaderboard.json + dashboard      │
    └──────────────────────────────────────┘
```

---

## 3. Những Khái Niệm Chính

### 3.1 Walk-Forward Splits với Gap

Hệ thống ngăn chặn lookahead bằng cách bắt buộc khoảng cách giữa các giai đoạn training và test.

**Cấu Trúc:** Cho năm test `Y`:
- `train_start` = Năm `Y - train_years` (ví dụ: 2020 nếu Y=2024, train_years=4)
- `train_end` = Năm `Y - 1 - gap_days` (ví dụ: khoảng Dec 1, 2023 nếu Y=2024, gap_days=25)
- `test_start` = Năm `Y` (Jan 1)
- `test_end` = Năm `Y + test_years` (Jan 1 năm sau)

**Tại sao cần gap?** Nếu target là forward return 5 ngày, model chỉ có thể thấy clean labels cho đến 5 ngày trước khi giai đoạn test bắt đầu. Gap (`gap_days`, mặc định 25) bắt buộc điều này. Model được train không có gap sẽ có quyền truy cập vào thông tin một phần về bars trong giai đoạn test, gây lookahead.

**Ví Dụ:** Nếu train 4 năm với gap 25 ngày cho năm test 2024:
- Train: 2020-01-01 → 2023-12-06 (~4 năm, gap trước 2024)
- Test: 2024-01-01 → 2024-12-31

### 3.2 Signal Freezing (T-1 / T)

Trong chế độ live hoặc live-sim, signal được sinh tại T-1 (hôm qua) và khóa trước khi thực thi T.

**Tại sao?** Ngăn chặn lookahead, mô phỏng trading thực tế: bạn quyết định "mua AAPL ngày mai tại open" hôm nay, trước khi bất kỳ giá T nào được biết.

**FrozenSignalSet:**
- Sinh tại T-1 close với dữ liệu ≤ T-1
- Chứa dict `{symbol: signal}` trong đó signal ∈ {-1, 0, 1}
- Integrity hash (SHA1) của dict signal đóng băng để ngăn chặn tampering
- Immutable dataclass — một khi tạo, không thể sửa đổi

**Luồng thực thi:**
- T-1: Sinh signal, đóng băng, ghi vào signals_log.csv
- T open: Đọc signal đóng băng, thực thi entries tại `open_price`
- T close: Evaluate exits cho open positions

### 3.3 Cost Model (Mô Hình Chi Phí)

Mỗi fill được điều chỉnh cho chi phí:
- **Commission**: 0.15% mỗi lệnh (bạn trả để mua, trả để bán)
- **Tax**: 0.10% trên proceeds bán
- **Slippage**: 0.10% (bid-ask và market impact)

**Fills:**
```
fill_buy(raw_price)  = raw_price * (1 + slippage)
fill_sell(raw_price) = raw_price * (1 - slippage)
round_trip_cost()    = 2 * commission + tax  ≈ 0.40% tổng
```

**Tính PnL:**
```
gross_pnl = (exit_price / entry_price) - 1
net_pnl   = gross_pnl - round_trip_cost()
```

Mặc định: commission=0.0015, tax=0.0010, slippage=0.0010 (có thể override via CLI).

### 3.4 Tri-Class Labels (3 Lớp)

Model dự đoán ba lớp dựa trên forward return:

```
fwd_return = close[t + horizon] / close[t] - 1

nếu fwd_return ≥ gain_threshold:      label = +1  (mua)
hay fwd_return ≤ -loss_threshold:     label = -1  (bán)
nếu không:                            label = 0   (trung lập)
```

Ngưỡng mặc định: `gain_threshold=0.04` (4%), `loss_threshold=0.04` (4%).
Horizon mặc định: 5 bars.

Tức là model dự đoán cổ phiếu sẽ tăng/giảm 4% trong 5 ngày tới.

### 3.5 Composite Score Formula

Mỗi model được xếp hạng bằng composite score kết hợp nhiều metrics:

```
quality_score = (
    0.18 * tanh(sharpe / 0.55)
  + 0.28 * tanh(avg_pnl / 12.0)
  + 0.26 * (1 - exp(-max(profit_factor - 1, 0) / 9))
  - 0.12 * clip(1 - exp(-mdd_per_symbol / 35), 0, 1)
  - 0.05 * clip(max(yearly_cv - 0.35, 0) / 2, 0, 1)
  - 0.01 * clip(max(avg_hold - 50, 0) / 25, 0, 1)
) * 1000

confidence = min(1.0, sqrt(n_trades / 1000))
pnl_scale = tanh(max(total_pnl, 0) / 18000)
composite = round(quality_score * confidence + 0.18 * pnl_scale * 1000, 1)
```

**Trọng số (Cái Nào Quan Trọng Nhất):**
- **avg_pnl (28%)**: Trung bình PnL/lệnh — nếu lệnh có lãi, cái này cao
- **sharpe (18%)**: Return điều chỉnh rủi ro — thưởng PnL nhất quán, volatility thấp
- **profit_factor (26%)**: Gross wins / gross losses — models với PF > 1.5 mạnh
- **mdd_per_symbol (-12%)**: Max drawdown/symbol — phạt single-symbol blowups
- **yearly_consistency (-5%)**: Coefficient variation yearly PnL — phạt năm không nhất quán
- **avg_hold (-1%)**: Trung bình hold days — phạt nhẹ long holds (thích quick trades)
- **total_pnl (scale)**: Thêm bonus nếu total PnL rất cao (18% weight)

Model có 100 lệnh, 50% win rate, avg PnL $150/lệnh, Sharpe 0.8 thường score 80–120.

---

## 4. Reproducibility & Experiment Tracking

### 4.1 Deterministic Runs (Same Seed → Same Results)

Mỗi run được kiểm soát bởi seed duy nhất:

```bash
python -m stock_ml.scripts.run_v2 \
    --symbols AAA,SSI,VND \
    --seed 42 \
    --out results/my_experiment
```

**Khi nào two runs giống hệt nhau:**
- Cùng `--seed 42`
- Cùng data (`--data-root` pointing to chính xác symbol/dates)
- Cùng config (target, costs, engine parameters)

**Seed propagation:**
- `src/seed.py::set_global_seed(seed)` thiết lập seed cho: numpy, random, LightGBM, XGBoost, sklearn, TensorFlow
- Hàm được gọi tại đầu `run()` → đảm bảo determinism từ lúc load dữ liệu

### 4.2 MLflow Experiment Tracking

Mỗi run tự động logged tới MLflow:

```
results/{experiment_name}/mlruns/
├── 0/                          # Experiment ID
│   ├── {run_id}/               # Run ID từ MLflow
│   │   ├── artifacts/
│   │   │   ├── trades.csv
│   │   │   ├── signals.csv
│   │   │   └── summary.json
│   │   ├── metrics.json
│   │   └── params.json
```

**Ghi log những gì:**
- **Config**: full RunConfig dict (seed, target, engine, features)
- **Params**: git_commit, data_fingerprint, seed
- **Metrics**: total_pnl, avg_pnl, win_rate, sharpe, max_drawdown, etc.
- **Artifacts**: trades.csv, signals.csv, summary.json

**Cách xem MLflow runs:**
```bash
mlflow ui --backend-store-uri file:///{project_root}/stock_ml/results/{experiment_name}/mlruns
# Mở http://localhost:5000
```

### 4.3 Data Fingerprint

Mỗi run tính toán **data fingerprint** — SHA256 hash của {symbols, start_date, end_date}:

```
data_fingerprint = SHA256("{AAA,SSI,VND}|2020-01-01|2025-12-31")[:16]
# Example: d3cd1c12b57d9cb2
```

**Ghi lại ở:**
- `results/{exp}/{run_id}/data_fingerprint.txt` (file)
- `summary.json` field `data_fingerprint` (JSON)
- MLflow param `data_fingerprint` (tracking)

**Tại sao:** Detect nếu data thay đổi mà config không → giải thích metric changes.

### 4.4 Git Commit Tracking

Mỗi run logs git commit hash (40 chars):

```json
{
  "git_commit": "3406abba48271...",
  "run_id": "20260529-081545-3406abba"
}
```

Giúp trace lại code version từ run result.

### 4.5 Run ID Format

```
{run_id} = {YYYYMMDD}-{HHMMSS}-{git_hash[:8]}
# Example: 20260529-081545-3406abba
```

**Benefit:**
- Timestamp cho ordering, debugging
- Git hash cho code version
- Unique per run (không overwrites)

### 4.6 Summary JSON

Master JSON file chứa tất cả metadata + results:

```json
{
  "run_id": "20260529-081545-3406abba",
  "name": "test_baseline",
  "data_fingerprint": "d3cd1c12b57d9cb2",
  "git_commit": "3406abba...",
  "mlflow_run_id": "05f727a8...",
  "aggregate": {
    "total_pnl": 25000.0,
    "avg_pnl": 172.41,
    "win_rate": 0.45,
    "profit_factor": 1.32,
    "sharpe": 0.85,
    "max_drawdown": 0.15
  },
  "audit": {
    "overall": "PASS",
    "checks": [...]
  },
  "config": {...},
  "generated_at": "2026-05-29T08:15:45Z"
}
```

---

## 5. Bố Trí Dữ Liệu

**Cấu Trúc Thư Mục:**
```
<data_root>/all_symbols/
  symbol=AAA/
    timeframe=1D/
      data.csv
  symbol=SSI/
    timeframe=1D/
      data.csv
  ...
```

**CSV Columns (ví dụ cho cổ phiếu VN):**
```
timestamp          | 2024-01-02T00:00:00Z
symbol             | AAA
exchange           | HOSE
asset_type         | equity
data_provider      | vn_stock_ai_dataset_cleaned
timeframe          | 1D
open               | 24500.0
high               | 24700.0
low                | 24400.0
close              | 24600.0
volume             | 1250000
traded_value       | 30750000000  (in VND)
```

**DataLoader Normalization:**
- Parse `timestamp` as UTC, convert to tz-naive date
- Dedup rows theo date (keep last nếu duplicates)
- Return DataFrame với columns: `[date, open, high, low, close, volume, symbol]`

**Ví Dụ:** `2024-01-02T00:00:00Z` → `2024-01-02`

---

## 6. Tham Khảo Config

### 6.1 RunConfig — Tham Số Batch Backtest

```python
class RunConfig:
    data_root: str              # Đường dẫn dữ liệu OHLCV
    symbols: list[str]          # Danh sách symbols để backtest
    out_dir: str                # Output directory cho results
    name: str                   # Tên run (dùng trong output filenames)
    
    # Cài đặt walk-forward split
    train_years: int = 4        # Năm dữ liệu training/fold
    test_years: int = 1         # Năm dữ liệu test/fold
    gap_days: int = 25          # Gap giữa train_end và test_start
    first_test_year: int = 2020 # Năm đầu tiên của giai đoạn test
    last_test_year: int = 2024  # Năm cuối cùng của giai đoạn test
    
    # Model và trading parameters
    target: ForwardReturnTarget  # Label definition
    engine: EngineConfig         # Exit rules và costs
    seed: int = 42               # Random seed
```

**Ví Dụ Sử Dụng:**
```bash
python -m stock_ml.scripts.run_v2 \
    --data-root /path/to/data \
    --symbols AAA,SSI,VND,HPG,FPT \
    --out results/my_experiment \
    --train-years 4 \
    --test-years 1 \
    --gap-days 25 \
    --first-test-year 2020 \
    --last-test-year 2024 \
    --horizon 5 \
    --gain-threshold 0.04 \
    --loss-threshold 0.04 \
    --max-hold-bars 20 \
    --hard-stop-pct -0.08 \
    --commission 0.0015 \
    --tax 0.0010 \
    --slippage 0.0010
```

### 6.2 LiveSimConfig — Tham Số Day-by-Day Simulation

```python
class LiveSimConfig:
    data_root: str              # Đường dẫn dữ liệu OHLCV
    symbols: list[str]          # Symbols để simulate
    out_dir: str                # Output directory
    
    # Thời kỳ mô phỏng (ngày cụ thể, không phải năm)
    sim_start: str              # Ngày trading đầu tiên (YYYY-MM-DD)
    sim_end: str                # Ngày trading cuối cùng (YYYY-MM-DD)
    
    # Training window (khái niệm tương tự RunConfig)
    train_years: int = 4        # Năm trước sim_start để train
    gap_days: int = 25          # Gap trước sim_start
    
    seed: int = 42              # Random seed
    target: ForwardReturnTarget  # Label definition
    engine: EngineConfig         # Exit rules và costs
    min_volume_filter: float = 0.0  # Tùy chọn: neutralize low-liquidity symbols
```

**Ví Dụ Sử Dụng:**
```bash
python -m stock_ml.scripts.run_live_sim \
    --symbols AAA,SSI,VND \
    --sim-start 2025-01-02 \
    --sim-end 2025-12-31 \
    --out results/live_sim_test \
    --train-years 4 \
    --horizon 5 \
    --gain-threshold 0.04 \
    --loss-threshold 0.04 \
    --max-hold-bars 20 \
    --min-hold-bars 1 \
    --hard-stop-pct -0.08 \
    --commission 0.0015 \
    --tax 0.0010 \
    --slippage 0.0010
```

### 6.3 Market Profiles

**Location:** `config/markets/`

| File | Timeframe | Instrument | Symbols | Ghi chú |
|---|---|---|---|---|
| `vn_stock.yaml` | 1D | Cổ phiếu Spot | ~60 cổ phiếu VN | Mặc định. 100M VND capital. |
| `vn_derivatives.yaml` | 1H | Futures | VN30F1M, VN30F2M | 100M VND capital. Multiplier 100,000 VND. |
| `vn_derivatives_1d.yaml` | 1D | Futures | Như trên | Cùng symbols ở daily. |
| `vn_derivatives_30m.yaml` | 30m | Futures | Như trên | Cùng symbols ở 30-min. |
| `vn_derivatives_15m.yaml` | 15m | Futures | Như trên | Cùng symbols ở 15-min. |
| `crypto_spot.yaml` | 1D | Spot | BTC, ETH, BNB, SOL, ADA | 10,000 USDT capital. Không tax. |
| `crypto_perp.yaml` | 1H | Perpetual | Như crypto_spot | 10,000 USDT capital. Funding rates. Short enabled. |

### 6.4 Feature Sets

**Location:** `config/feature_sets/`

| File | Features | Ghi chú |
|---|---|---|
| `leading.yaml` | OHLCV basics, moving averages, momentum, trend, volatility, volume, leading signals | V22 baseline. 8 features trong FEATURE_COLS. |
| `leading_v2.yaml` | leading + market structure + exhaustion + volatility regime + multi-timeframe | Extended feature set. |
| `leading_v3.yaml` | leading_v2 + accumulation + relative strength | V29 retrain. |
| `leading_v4.yaml` | leading_v3 + heikin_ashi | V34 retrain. Phiên bản mới nhất. |
| `leading_deriv.yaml` | leading_v2 + heikin_ashi (không cross-sectional RS) | Optimized cho derivatives. |

CLI args `--horizon`, `--gain-threshold`, `--loss-threshold` configure `ForwardReturnTarget`.

### 6.5 Engine Configuration

```python
class CostModel:
    commission: float = 0.0015  # 0.15% mỗi lệnh
    tax: float = 0.0010         # 0.10% trên bán
    slippage: float = 0.0010    # 0.10% market impact

class EngineConfig:
    max_hold_bars: int = 20     # Force exit sau N bars
    min_hold_bars: int = 1      # Không exit trước N bars (trừ hard_stop)
    hard_stop_pct: float = -0.08  # Exit nếu MtM ≤ -8%
    cost: CostModel
```

**Exit Priority (Ưu tiên Exit):**
1. **hard_stop** (cao nhất): Nếu unrealized loss ≤ hard_stop_pct và hold ≥ min_hold_bars, exit ngay.
2. **signal**: Nếu model dự đoán -1 (bán) và hold ≥ min_hold_bars, exit.
3. **max_hold** (thấp nhất): Nếu hold ≥ max_hold_bars, force exit.

---

## 7. Tham Khảo Feature

**8 FEATURE_COLS** — đây là features model baseline sử dụng:

| Feature | Công Thức | Warmup Bars |
|---|---|---|
| `ret_1d` | `close.pct_change(1)` | 1 |
| `ret_5d` | `close.pct_change(5)` | 5 |
| `sma_5_ratio` | `close / SMA(5) - 1` | 5 |
| `sma_20_ratio` | `close / SMA(20) - 1` | 20 |
| `rsi_14` | Wilder's RSI(14), NaN → 50 | 14 |
| `volume_ratio_20` | `volume / SMA_vol(20)` | 20 |
| `high_low_pct` | `(high - low) / close` | 0 |
| `atr_14_ratio` | `ATR(14) / close` | 14 |

**add_features()** nhóm OHLCV theo symbol, sort theo date, tính toán 8 features/symbol. 20 rows đầu/symbol sẽ là NaN (max warmup). Pipeline tự động drop những cái này trước training.

---

## 8. Workflow A: Chạy Backtest Mới

### Bước 1: Chọn Symbols

Quyết định symbols nào để backtest:
- **Option A (danh sách cụ thể):** `--symbols AAA,SSI,VND,HPG,FPT`
- **Option B (tất cả available):** `--symbols ALL`
- **Option C (với cap):** `--symbols ALL --max-symbols 30` (test chỉ 30 đầu)

### Bước 2: Cấu Hình Target và Engine

Quyết định entry/exit rules:
- **Entry**: Driven bởi forward return thresholds (horizon, gain/loss targets)
- **Exit**: Driven bởi hard_stop, signal, hoặc max_hold

### Bước 3: Chạy Command

```bash
python -m stock_ml.scripts.run_v2 \
    --data-root portable_data/vn_stock_ai_dataset_cleaned \
    --symbols AAA,SSI,VND \
    --out stock_ml/results/my_first_test \
    --name test_v1 \
    --train-years 4 \
    --test-years 1 \
    --gap-days 25 \
    --first-test-year 2020 \
    --last-test-year 2024 \
    --horizon 5 \
    --gain-threshold 0.04 \
    --loss-threshold 0.04 \
    --max-hold-bars 20 \
    --hard-stop-pct -0.08 \
    --commission 0.0015 \
    --tax 0.0010 \
    --slippage 0.0010
```

**Cái gì xảy ra:**
1. Load OHLCV cho AAA, SSI, VND từ data_root
2. Tính toán 8 technical features
3. Apply tri-class labels với forward 5-day 4% thresholds
4. Split thành 5 walk-forward folds (2020, 2021, 2022, 2023, 2024)
5. Mỗi fold: train LightGBM trên train period, predict trên test period
6. Chạy backtest với next-bar fills và exit logic
7. Tính toán stats và chạy audit checks
8. Ghi output files

### Bước 4: Kiểm Tra Audit Report

Sau khi run hoàn tất, kiểm tra field `audit` trong `summary_test_v1.json`:

```json
"audit": {
    "overall": "PASS",
    "n_fail": 0,
    "n_warn": 0,
    "checks": [
        {
            "name": "split_no_overlap",
            "status": "PASS",
            "detail": "5 windows clean"
        },
        ...
    ]
}
```

| Status | Nghĩa |
|---|---|
| `PASS` | An toàn sử dụng. Không phát hiện leakage. |
| `WARN` | Vấn đề nhỏ (ví dụ: signal coverage thấp). Backtest dùng được nhưng có edge cases. |
| `FAIL` | Critical leakage hoặc data issue. KHÔNG dùng. Fix vấn đề và retry. |

### Bước 5: Đọc Output Files

Điều hướng tới `stock_ml/results/my_first_test/`:

- `trades_test_v1.csv` — Danh sách tất cả closed trades (entry/exit dates, prices, PnL)
- `signals_test_v1.csv` — Tất cả generated signals (symbol, date, signal)
- `daily_stats_test_v1.csv` — Per-day PnL và metrics
- `yearly_stats_test_v1.csv` — Per-year summary
- `symbol_stats_test_v1.csv` — Per-symbol summary
- `summary_test_v1.json` — Master summary với metrics và audit

**Kiểm tra key metrics trong summary:**
```json
{
    "name": "test_v1",
    "n_trades": 145,
    "aggregate": {
        "total_pnl": 25000.0,
        "avg_pnl": 172.41,
        "win_rate": 0.45,
        "profit_factor": 1.32
    },
    "audit": { "overall": "PASS" },
    ...
}
```

---

## 9. Workflow B: Đăng Ký Model Vào Leaderboard

Sau khi backtest run, bạn có thể đăng ký nó vào **leaderboard** — file JSON xếp hạng tất cả experiments theo composite score.

### Required Artifacts

Run directory phải chứa những files này:

```
<run_dir>/
  predictions_meta.json         # Model metadata
  ranking_row.json              # Legacy metrics cache
  metrics.json                  # Computed metrics
  config.resolved.yaml          # Full resolved YAML config
  trades.csv                    # Trade log (preferred for recomputation)
  trades_*.csv                  # Alternative naming
  lifecycle.json                # (optional) {"state": "trained"|"pinned"|"retired"}
```

**Ghi chú:** Hệ thống tự động ghi hầu hết những cái này sau `run_v2` backtest. Hàm `run_dir_to_row()` đọc những cái này và build `LeaderboardRow`.

### Run ID Format

Mỗi model được identify duy nhất bằng:
```
run_id = {bundle}/{run_name}#{config_hash[:8]}
```

Ví dụ: `v22_exit_ablation/test_v1#a3f9c4e2`

### Đăng Ký via CLI

```bash
python -m stock_ml.scripts.build_leaderboard append \
    stock_ml/results/my_first_test \
    --output-dir stock_ml/results/leaderboard \
    --bundle my_first_bundle
```

**Cái gì xảy ra:**
1. Đọc existing `leaderboard.json` từ `--output-dir` (hoặc tạo nếu missing)
2. Call `run_dir_to_row(stock_ml/results/my_first_test)` để đọc artifacts và tính metrics
3. Dedup theo `run_id`: nếu row với bundle/run_name same đã tồn tại, mark old one là `superseded=True`
4. Sort tất cả rows theo `composite_score` (descending)
5. Annotate với fairness flags (ví dụ: `is_baseline`, `same_symbols_as_baseline`)
6. Ghi 5 output files tới `--output-dir`:
   - `leaderboard.json` — Array LeaderboardRow objects
   - `leaderboard.csv` — Flattened 48-column CSV
   - `summary.json` — Metadata (row count, top 3/fairness group, etc.)
   - `index.json` — Fast lookup `{run_id: position_index}`
   - `schema.json` — JSON Schema definition

### Output Structure

**leaderboard.json** (simplified example):
```json
[
  {
    "run_id": "my_first_bundle/test_v1#a3f9c4e2",
    "bundle": "my_first_bundle",
    "run_name": "test_v1",
    "state": "trained",
    "config_hash": "a3f9c4e2",
    "generated_at": "2025-05-28T10:30:00Z",
    "n_trades": 145,
    "wr": 0.45,
    "avg_pnl": 172.41,
    "total_pnl": 25000.0,
    "pf": 1.32,
    "composite_score": 87.3,
    "is_baseline": false,
    "warnings": []
  },
  ...
]
```

---

## 10. Workflow C: Ghim Model Lên Dashboard

Một khi model đã trong leaderboard, bạn có thể **pin** nó lên dashboard để visualize.

### Bước 1: Khởi Động API Server

```bash
python scripts/api_server.py
```

Khởi động local FastAPI server tại `http://localhost:5176`.

### Bước 2: Mở Dashboard

Điều hướng tới:
```
http://localhost:5176/visualization/leaderboard.html
```

Bạn sẽ thấy bảng leaderboard với tất cả registered models, sort theo composite_score.

### Bước 3: Pin Model

Xác định `run_id` của model bạn muốn pin (từ bảng hoặc leaderboard.json).

Gửi PATCH request tới API:
```bash
curl -X PATCH http://localhost:5176/api/runs \
    -H "Content-Type: application/json" \
    -d '{"run_id": "my_first_bundle/test_v1#a3f9c4e2", "state": "pinned"}'
```

Hoặc dùng Python requests:
```python
import requests
requests.patch("http://localhost:5176/api/runs", json={
    "run_id": "my_first_bundle/test_v1#a3f9c4e2",
    "state": "pinned"
})
```

### Bước 4: Cái Gì Xảy Ra Tự Động

Khi model được pin:
1. API gọi `export_version()` từ `src/export/unified_export.py`
2. Cho mỗi symbol trong model, nó đọc `trades.csv` và sinh ra:
   - `visualization/data_a3f9c4e2/{SYMBOL}.json` — Trade markers, trade list, stats cho symbol
3. Nó update `visualization/manifest.json` với model metadata mới
4. Dashboard refresh và hiển thị trade model overlaid trên price charts

**Kết quả:** Dashboard bây giờ hiển thị entry/exit signals và PnL cho mỗi symbol dưới model này.

---

## 11. Workflow D: Retire Model

Để ẩn model từ dashboard nhưng giữ nó trong leaderboard history:

```bash
curl -X PATCH http://localhost:5176/api/runs \
    -H "Content-Type: application/json" \
    -d '{"run_id": "my_first_bundle/test_v1#a3f9c4e2", "state": "retired"}'
```

**Khác biệt với delete:**
- **Retire**: Model ở trong `leaderboard.json` với `state: "retired"`, nhưng không hiển thị trên dashboard.
- **Delete**: Run directory bị xóa entirely, model bị xóa khỏi leaderboard.

Dùng retire nếu bạn muốn giữ record cho historical reference nhưng không còn promote model.

---

## 12. Workflow E: Chạy Live Simulation

**Khi nào dùng:** Khi bạn muốn day-by-day execution simulation (ví dụ: test deployment readiness hoặc track daily PnL).

**Khác biệt với batch backtest:**
- Batch: Chạy toàn bộ test period như một fold. Focus annual performance.
- Live sim: Step qua mỗi business day riêng lẻ, freezing signals tại T-1 trước T execution.

### Bước 1: Chạy Command

```bash
python -m stock_ml.scripts.run_live_sim \
    --data-root portable_data/vn_stock_ai_dataset_cleaned \
    --symbols AAA,SSI,VND \
    --sim-start 2025-01-02 \
    --sim-end 2025-12-31 \
    --out stock_ml/results/my_live_sim \
    --train-years 4 \
    --horizon 5 \
    --gain-threshold 0.04 \
    --loss-threshold 0.04 \
    --max-hold-bars 20 \
    --min-hold-bars 1 \
    --hard-stop-pct -0.08 \
    --min-volume-filter 0.0
```

**Cái gì xảy ra:**
1. Load OHLCV cho AAA, SSI, VND
2. Tính features và labels
3. Train một BaselineModel trên 4 năm history trước `sim_start`
4. Mỗi business day từ `sim_start` đến `sim_end`:
   - Sinh signal tại T-1 (đóng cửa hôm qua) với dữ liệu T-1 only
   - Thực thi buy orders tại T open
   - Evaluate exits (hard_stop, signal, max_hold)
   - Record daily activity
5. Force-close tất cả remaining positions ở cuối
6. Ghi output files

### Bước 2: Outputs

Điều hướng tới `stock_ml/results/my_live_sim/`:

- `trades.csv` — Tất cả closed trades
- `live_sim_daily_log.csv` — Per-day activity (entries, exits, open positions, PnL)
- `signals_log.csv` — Tất cả generated signals với integrity hash
- `yearly_stats.csv` — Per-year aggregated stats
- `summary_live_sim.json` — Master summary

**Key metric trong summary:**
```json
{
    "n_trades": 42,
    "metrics": {
        "total_pnl": 15000.0,
        "avg_pnl": 357.14,
        "win_rate": 0.52,
        "profit_factor": 1.45
    },
    ...
}
```

---

## 13. Tham Khảo Output Files

### 12.1 trades_{name}.csv (Batch) hoặc trades.csv (Live Sim)

Closed trades từ backtest. Columns:

| Column | Type | Ghi chú |
|---|---|---|
| `symbol` | str | Stock symbol |
| `entry_date` | date | Ngày position mở |
| `entry_price` | float | Giá trả sau costs |
| `exit_date` | date | Ngày position đóng |
| `exit_price` | float | Giá nhận sau costs |
| `holding_days` | int | Số ngày hold (có thể 0 cho same-day) |
| `pnl_pct` | float | Net PnL as % entry price (include costs) |
| `exit_reason` | str | `signal`, `max_hold`, `hard_stop`, hoặc `end_of_data` (batch) / `end_of_sim` (live) |
| `entry_signal_date` | date | Ngày model sinh buy signal (T-1 cho live, một bar trước T cho batch) |

### 12.2 signals_{name}.csv (Batch) hoặc signals_log.csv (Live Sim)

Tất cả generated signals. Columns:

| Column | Type |
|---|---|
| `symbol` | str |
| `date` | date |
| `signal` | int | -1 (bán), 0 (trung lập), 1 (mua) |
| `integrity_hash` | str | (live sim only) SHA1 của frozen signal set |

### 12.3 daily_stats_{name}.csv (Batch) hoặc live_sim_daily_log.csv (Live Sim)

Per-day aggregated metrics. Columns:

| Column | Type | Ghi chú |
|---|---|---|
| `date` | date | |
| `n_trades` | int | Closed trades ngày này |
| `wins` | int | Profitable trades |
| `losses` | int | Losing trades |
| `win_rate` | float | wins / (wins + losses) |
| `total_pnl` | float | Sum trade PnLs ngày này |
| `avg_pnl` | float | total_pnl / n_trades |
| `max_win` | float | Best trade PnL |
| `max_loss` | float | Worst trade PnL |

### 12.4 yearly_stats_{name}.csv

Per-year aggregated metrics. Columns:

| Column | Type |
|---|---|
| `year` | int |
| `n_trades` | int |
| `wins`, `losses` | int |
| `win_rate` | float |
| `total_pnl` | float |
| `avg_pnl`, `med_pnl`, `std_pnl` | float |
| `max_win`, `max_loss` | float |
| `profit_factor` | float | (wins * avg_win) / (losses * abs(avg_loss)) |
| `avg_hold_days` | float |

### 12.5 symbol_stats_{name}.csv

Per-symbol aggregated metrics. Same columns như yearly_stats (year → symbol).

### 12.6 summary_{name}.json (Batch) hoặc summary_live_sim.json (Live Sim)

Master JSON summary. Key fields:

```json
{
  "name": "...",
  "n_symbols": 3,
  "n_trades": 145,
  "n_signals_buy": 320,
  "n_signals_sell": 105,
  "n_signals_neutral": 2000,
  
  "aggregate": {
    "total_pnl": 25000.0,
    "avg_pnl": 172.41,
    "win_rate": 0.45,
    "profit_factor": 1.32,
    "avg_hold_days": 12.5,
    "sharpe": 0.85,
    "max_drawdown": 0.15,
    "mdd_per_symbol": 0.10,
    "yearly_consistency": 0.25
  },
  
  "audit": {
    "overall": "PASS",
    "n_fail": 0,
    "n_warn": 0,
    "checks": [
      {
        "name": "split_no_overlap",
        "status": "PASS",
        "detail": "5 windows clean",
        "examples": []
      },
      ...
    ]
  },
  
  "outputs": {
    "trades_csv": "stock_ml/results/my_first_test/trades_test_v1.csv",
    "signals_csv": "stock_ml/results/my_first_test/signals_test_v1.csv",
    ...
  },
  
  "config": {
    "train_years": 4,
    "test_years": 1,
    "gap_days": 25,
    "target": { "horizon": 5, "gain_threshold": 0.04, "loss_threshold": 0.04 },
    "cost": { "commission": 0.0015, "tax": 0.0010, "slippage": 0.0010 },
    "engine": { "max_hold_bars": 20, "hard_stop_pct": -0.08 }
  },
  
  "generated_at": "2025-05-28T10:30:00.123456+00:00"
}
```

### 12.7 Cấu Trúc Audit Report

Field `audit` trong summary JSON. Example:

```json
{
  "overall": "PASS|WARN|FAIL",
  "n_fail": 0,
  "n_warn": 0,
  "checks": [
    {
      "name": "split_no_overlap",
      "status": "PASS",
      "detail": "Train và test windows không overlap",
      "examples": []
    },
    {
      "name": "split_gap",
      "status": "PASS",
      "detail": "Tất cả gaps >= 25 ngày",
      "examples": []
    },
    {
      "name": "entry_integrity",
      "status": "PASS",
      "detail": "Tất cả 145 trades trace tới buy signal",
      "examples": []
    },
    {
      "name": "fill_offset",
      "status": "PASS",
      "detail": "Tất cả entries fill ở next bar sau signal",
      "examples": []
    },
    {
      "name": "signal_coverage",
      "status": "PASS",
      "detail": "145 trades / 320 buy signals (45%)",
      "examples": []
    }
  ]
}
```

| Check Name | Validates |
|---|---|
| `split_no_overlap` | Train và test periods không overlap |
| `split_gap` | Gap giữa train_end và test_start đủ |
| `entry_integrity` | Mỗi entry traces tới buy signal |
| `fill_offset` | Entries fill ở next bar (không same-bar) |
| `signal_coverage` | Trade conversion rate từ buy signals (warn nếu thấp) |

### 12.8 leaderboard.json Structure

Sau khi aggregator chạy, leaderboard.json chứa array LeaderboardRow objects:

```json
[
  {
    "run_id": "v22_exit_ablation/test_v1#a3f9c4e2",
    "bundle": "v22_exit_ablation",
    "run_name": "test_v1",
    "config_hash": "a3f9c4e2",
    "generated_at": "2025-05-28T10:30:00Z",
    "superseded": false,
    "state": "trained",
    
    "market": "vn_stock",
    "market_family": "vn_stock",
    "currency": "VND",
    "timeframe": "1D",
    "schema": "run_backtest_v2",
    "strategy": "baseline_lightgbm",
    "feature_set": "leading",
    "entry_model": "lightgbm",
    "exit_model_type": "none",
    "exit_model_enabled": false,
    
    "target": {
      "type": "forward_return",
      "forward_window": 5,
      "gain_threshold": 0.04,
      "loss_threshold": 0.04
    },
    
    "trades": 145,
    "wr": 0.45,
    "avg_pnl": 172.41,
    "total_pnl": 25000.0,
    "pf": 1.32,
    "avg_hold": 12.5,
    "sharpe": 0.85,
    "max_drawdown": 0.15,
    "mdd_per_symbol": 0.10,
    "yearly_consistency": 0.25,
    
    "composite_score": 87.3,
    "score_mode": "live",
    
    "n_symbols": 3,
    "first_test_year": 2020,
    "last_test_year": 2024,
    "backtest_window_key": "2020-2024",
    "cost_profile": {
      "commission": 0.0015,
      "tax": 0.0010,
      "slippage": 0.0010
    },
    "fairness_group_key": "7f9a3c2e1b5d6a9c",
    
    "is_baseline": false,
    "same_symbols_as_baseline": true,
    "same_window_as_baseline": true,
    "same_target_as_baseline": false,
    "same_cost_as_baseline": true,
    "same_market_family_as_baseline": true,
    
    "warnings": []
  },
  ...
]
```

**Sorted theo `composite_score` descending.**

---

## 14. Model Lifecycle States

Models transition qua ba states:

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  TRAINED    │────────▶│   PINNED    │────────▶│  RETIRED    │
│             │         │             │         │             │
│ Trong       │         │ Trong       │         │ Trong       │
│ leaderboard │         │ leaderboard │         │ leaderboard │
│ KHÔNG trên  │         │ VÀ trên     │         │ KHÔNG trên  │
│ dashboard   │         │ dashboard   │         │ dashboard   │
└─────────────┘         └─────────────┘         └─────────────┘
```

| State | Visibility | Sử Dụng |
|---|---|---|
| **trained** | Chỉ leaderboard (sort by score). Không trên dashboard. | Sau khi backtest hoàn tất. Model sẵn sàng nhưng không active trong production. |
| **pinned** | Cả leaderboard và dashboard. Per-symbol trade data exported và visualized. | Khi bạn quyết định model tốt và muốn show nó. Thường max 1-3 pinned models. |
| **retired** | Chỉ leaderboard history. Không trên dashboard. | Khi bạn muốn ẩn model nhưng giữ record. |

**Transitions:**
- `trained` → `pinned`: Gọi `PATCH /api/runs` với `state: "pinned"` (triggers export)
- `pinned` → `trained`: Gọi `PATCH /api/runs` với `state: "trained"` (ẩn từ dashboard)
- `pinned` → `retired`: Gọi `PATCH /api/runs` với `state: "retired"`
- `trained` → `retired`: Gọi `PATCH /api/runs` với `state: "retired"`
- Delete (any state): Gọi `DELETE /api/runs?run_id=...` (xóa run_dir và leaderboard record)

---

## 15. Thêm Model Type Mới

Nếu bạn muốn thêm model class khác (ví dụ: XGBoost, Random Forest) hoặc đáng kể thay đổi signal generation logic:

### Bước 1: Implement Model Class

Tạo file mới trong `src/models/` (ví dụ: `src/models/xgboost.py`):

```python
import numpy as np
from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, seed=42):
        self.clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed,
            verbosity=0
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        # Map -1/0/1 labels to 0/1/2
        y_mapped = y + 1  # -1→0, 0→1, 1→2
        self.clf.fit(X, y_mapped)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.clf.predict(X)  # 0/1/2
        return preds - 1  # Back to -1/0/1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)
```

### Bước 2: Configure RunConfig

Trong backtest script hoặc CLI, override model:

```python
from src.pipeline.run import run, build_default_config
from src.models.xgboost import XGBoostModel

cfg = build_default_config(
    data_root="...",
    symbols=["AAA", "SSI"],
    out_dir="results/xgb_test",
    name="xgb_v1"
)

# Monkey-patch hoặc parameterize model creation
summary = run(cfg)
```

### Bước 3: Chạy Backtest và Kiểm Tra Audit

```bash
python -m stock_ml.scripts.run_v2 \
    --symbols AAA,SSI \
    --out results/xgb_test \
    --name xgb_v1
```

Verify `summary_xgb_v1.json` có `audit.overall == "PASS"`.

### Bước 4: Đăng Ký Vào Leaderboard

```bash
python -m stock_ml.scripts.build_leaderboard append \
    results/xgb_test \
    --output-dir results/leaderboard \
    --bundle xgb_experiments
```

### Bước 5 (Tùy Chọn): Pin Lên Dashboard

```bash
curl -X PATCH http://localhost:5176/api/runs \
    -H "Content-Type: application/json" \
    -d '{"run_id": "xgb_experiments/xgb_v1#...", "state": "pinned"}'
```

---

## Appendix: Quick Reference

### Most Used Commands

```bash
# Chạy backtest
python -m stock_ml.scripts.run_v2 --symbols AAA,SSI --out results/test1

# Chạy live simulation
python -m stock_ml.scripts.run_live_sim --sim-start 2025-01-02 --sim-end 2025-12-31

# Đăng ký vào leaderboard
python -m stock_ml.scripts.build_leaderboard append \
    results/test1 \
    --output-dir results/leaderboard \
    --bundle my_bundle

# Khởi động dashboard server
python scripts/api_server.py

# Pin model (via API)
curl -X PATCH http://localhost:5176/api/runs \
    -H "Content-Type: application/json" \
    -d '{"run_id": "my_bundle/test1#...", "state": "pinned"}'
```

### Key Files to Know

- **Data**: `portable_data/vn_stock_ai_dataset_cleaned/`
- **Results**: `stock_ml/results/`
- **Configs**: `stock_ml/config/` (base.yaml, markets/, feature_sets/)
- **Source code**: `stock_ml/src/` (pipeline, live_sim, backtest, models, leaderboard, export)
- **Scripts**: `stock_ml/scripts/` (run_v2.py, run_live_sim.py, api_server.py, build_leaderboard.py)

### Default Values

- Timeframe: `1D`
- Market: `vn_stock`
- Commission: `0.15%`, Tax: `0.10%`, Slippage: `0.10%`
- Hard stop: `-8%`
- Max hold: `20` bars
- Horizon: `5` bars (forward return window)
- Thresholds: `±4%` (gain/loss)
- Train years: `4`, Test years: `1`, Gap: `25` ngày

---

**Cập Nhật Lần Cuối:** 2025-05-28  
**Phiên Bản:** 1.0  
**Ngôn Ngữ:** Tiếng Việt
