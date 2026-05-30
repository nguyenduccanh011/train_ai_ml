# Stock ML Trading System — Research-Grade Architecture

Hệ thống ML giao dịch chuyên nghiệp (Two Sigma / AHL / de Prado methodology), hiện đang refactor thành modular research platform với proper validation, reproducibility, và alpha gate.

## Status (2026-05-29)

**Phase 0 + Phase 1b.1-3**: ✅ Complete
- ✅ Foundation infrastructure (MLflow tracking, reproducibility, logging, output layout)
- ✅ Model/feature/target registries
- ✅ **Regression approach** for entry model (single professional model predicts forward return)
- ✅ Vectorized signal generation (threshold-based, O(n))
- ✅ Score column for confidence-weighted sizing (Phase 3)

**In Progress**: Phase 1b.4-11 (unify paths, YAML schema, strict audit, others)

See `IMPLEMENTATION_ROADMAP.md` for detailed progress and next phases.

## Cài đặt

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # ruff, mypy, pytest (dev only)
```

**Yêu cầu:** Python 3.11+

## REST API Server (Model Management & Lifecycle)

Stock ML Platform includes a production REST API for model management, leaderboard, and cache operations.

### Start API Server
```bash
# Development mode (auto-reload)
python -m uvicorn stock_ml.api.main:app --reload --port 8000

# Or production mode
uvicorn stock_ml.api.main:app --host 0.0.0.0 --port 8000
```

### API Documentation
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **API Reference**: `docs/API.md` (all endpoints documented)

### Key Endpoints

**Leaderboard & Models:**
- `GET /api/v1/leaderboard` — Get all model runs ranked by composite score
- `GET /api/v1/models` — List available models

**Model Lifecycle (Pin/Unpin/Retire):**
- `GET /api/v1/runs` — List all runs
- `PATCH /api/v1/runs/{run_id}/state` — Change state (trained → pinned → retired)
- `DELETE /api/v1/runs/{run_id}` — Delete model entirely

**Cache Management:**
- `GET /api/v1/cache/stats` — Cache disk usage and orphan stats
- `POST /api/v1/gc/sweep` — Run garbage collection (quarantine orphans)
- `POST /api/v1/cache/purge-trash` — Purge old trash batches

**Bulk Operations:**
- `POST /api/v1/runs/bulk-state` — Batch pin/retire/train models
- `DELETE /api/v1/runs/bulk` — Delete all runs in a state

See `docs/API.md` for complete endpoint documentation with request/response examples.

### Dashboard & Leaderboard UI
The API serves a web-based dashboard with:
- **Candlestick chart** with signal overlays from pinned models
- **Leaderboard table** with ranking, filtering, pin/unpin buttons
- **Cache management panel** for cleanup and GC operations

**Access Dashboard:**
- Direct: http://localhost:9001 (development)
- Via Nginx: http://localhost (production/containerized)

---

## Cấu trúc dự án

```
stock_ml/
├── config/
│   ├── base.yaml                    # Cấu hình pipeline mặc định (market, device, split)
│   ├── markets/                     # MarketProfile: data/execution/symbols/features/models/target
│   │   ├── vn_stock.yaml
│   │   ├── crypto_spot.yaml
│   │   ├── crypto_perp.yaml
│   │   └── vn_derivatives.yaml
│   ├── feature_sets/                # YAML định nghĩa feature block composition
│   │   ├── leading.yaml
│   │   ├── leading_v2.yaml
│   │   ├── leading_v3.yaml
│   │   └── leading_v4.yaml
│   └── experiments/
│       ├── champions/               # YAML cho champion versions
│       │   ├── v22.yaml
│       │   ├── v35b.yaml
│       │   └── ...
│       └── matrix/                  # Grid search matrix configs
│
├── src/
│   ├── components/                  # ← KIẾN TRÚC MỚI (v2)
│   │   ├── base.py                  # BarContext, FusionResult, Position, Trade, Action
│   │   ├── registry.py              # ComponentRegistry (thread-safe)
│   │   ├── features/
│   │   │   ├── blocks/              # 14 feature blocks (momentum, heikin_ashi, ...)
│   │   │   ├── engine.py            # ComposableFeatureEngine
│   │   │   └── registry.py
│   │   ├── targets/                 # TrendRegime, EarlyWave, EarlyWaveV2, EarlyWaveDual
│   │   ├── models/                  # LightGBM, XGBoost, CatBoost, RandomForest, GRU
│   │   ├── fusion/
│   │   │   ├── stack.py             # FusionStack — chain strategies theo layer
│   │   │   ├── registry.py          # register_strategy / list_strategies
│   │   │   └── strategies/          # fusion strategy classes (pre_entry/entry/hold/exit_rules)
│   │   ├── backtest/
│   │   │   └── engine.py            # SimpleLongBacktester
│   │   └── runners/                 # Generic runners cho champion groups + rule special-case
│   │       ├── generic_fusion.py
│   │       ├── v34_runner.py
│   │       ├── rule_runner.py
│   │       └── runner_registry.py
│   │
│   ├── pipeline/                    # ← ORCHESTRATOR MỚI
│   │   ├── config.py                # ExperimentConfig (Pydantic)
│   │   ├── orchestrator.py          # Pipeline class + PipelineResult
│   │   ├── trainer.py               # build_prediction_cache()
│   │   ├── build_predictions.py     # Shared prediction builder (không import run_pipeline.py)
│   │   ├── walker.py                # walk_forward() splits
│   │   ├── cache.py                 # PredictionCacheManager (pickle + sha256 key)
│   │   ├── matrix_expander.py       # expand_matrix(yaml) → list[ExperimentConfig]
│   │   └── validate.py              # validate_config() → list[ValidationError]
│   │
│   ├── data/                        # DataLoader theo schema/timestamp/timezone từ MarketProfile
│   ├── backtest/                    # Unified backtest + PnL calculators theo pnl_mode
│   ├── leaderboard/                 # Leaderboard scoped theo market/schema/timeframe
│   ├── market_profile.py            # MarketProfile loader + ResolvedRunContext + run_identity
│   ├── features/engine.py           # FeatureEngine legacy (vẫn dùng bởi runners)
│   ├── models/registry.py           # ModelRegistry legacy
│   ├── export/unified_export.py     # CSV → JSON cho dashboard
│   └── env.py / config_loader.py    # Path resolution, YAML loading legacy-compatible
│
├── scripts/
│   ├── cli.py                       # Entry point CLI
│   └── benchmark.py                 # Pipeline timing
│
├── tests/
│   ├── signals/                     # Feature, target, signal/rule helpers
│   ├── strategy/                    # Fusion stack, entry/hold/exit rules
│   ├── execution/                   # Backtest engine behavior
│   └── regression/
│       ├── golden/                  # CSV + checksums.txt (CPU, deterministic)
│       ├── test_champions.py        # Hash check champions vs golden
│       └── test_*_parity.py         # Per-champion exact parity tests
│
├── archive/
│   └── scripts/                     # Historical scripts kept as reference
│
├── visualization/                   # Dashboard (HTML + JS + per-symbol JSON)
├── docs/refactor/                   # ARCHITECTURE.md, roadmap, HOW_TO guides
├── pyproject.toml                   # ruff + mypy + pytest config
├── requirements.txt
├── requirements-dev.txt
└── run_pipeline.py                  # DEPRECATED — dùng `python -m stock_ml run`
```

## CLI mới (`python -m stock_ml`)

```bash
# Chạy một champion experiment
python -m stock_ml run champions/v22
python -m stock_ml run champions/v22 --device gpu

# Chạy + lưu CSV + export dashboard
python -m stock_ml run champions/v22 --save-results --export

# Chạy tất cả experiments trong matrix YAML
python -m stock_ml run-matrix matrix/test_2x2

# Xếp hạng artifact đã lưu của một matrix
python -m stock_ml compare-matrix results/experiments/finalists_entry_exit --top 10

# Chạy & xếp hạng toàn bộ champion với 1 split chung (aggregator)
python -m stock_ml compare-champions --first-test-year 2023 --last-test-year 2024 --device gpu --resume
python -m stock_ml compare-champions --first-test-year 2019 --last-test-year 2025 --champions v22,v34,v37a,v39d --device gpu --resume

# Validate config
python -m stock_ml validate champions/v22
python -m stock_ml validate matrix/q3_2026   # matrix có thể dùng axis market

# Export trades CSV → dashboard JSON (tất cả active models)
python -m stock_ml export
python -m stock_ml export --versions v22,v34,v37a

# So sánh nhiều experiments
python -m stock_ml compare champions/v22 champions/v35b champions/v34

# List components đã đăng ký
python -m stock_ml list-components
python -m stock_ml list-components --type fusion

# List experiments
python -m stock_ml list-experiments

# Benchmark pipeline
python -m stock_ml benchmark --versions v22,v34,v37a
```

### Flow bắt đầu mô hình phái sinh

1. Chuẩn hóa dataset theo `config/markets/vn_derivatives.yaml`: OHLCV, `timestamp`, `volume`, `expiry_date` nếu dùng rollover, và các cột spread như `next_volume`/`next_oi` nếu dùng smart roll.
   Hỗ trợ cả 2 layout dữ liệu: `all_symbols/symbol=.../timeframe=.../data.csv` và `symbol=.../timeframe=.../data.csv`.
2. Tạo experiment YAML riêng với `market: vn_derivatives`, chọn feature set/model/target giống matrix hiện có hoặc override theo market.
3. Validate trước khi chạy:

```bash
python -m stock_ml validate my_vn_derivatives_exp
```

4. Chạy smoke test 1 symbol/hợp đồng chỉ để kiểm tra pipeline/config không lỗi:

```bash
python -m stock_ml run my_vn_derivatives_exp --symbols-limit 1 --save-results
```

5. Khi smoke test ổn, chạy score chính thức theo đầy đủ danh sách symbol trong `config/markets/vn_derivatives.yaml` (không dùng `--symbols-limit`):

```bash
python -m stock_ml run-matrix matrix/my_vn_derivatives_matrix --resume
python -m stock_ml compare-matrix results/experiments/my_vn_derivatives_matrix --top 10
```

## Visualization / Leaderboard

Serve từ thư mục `stock_ml/` (không phải `visualization/`):

```bash
cd stock_ml
python -m http.server 8181
```

Mở browser:
- **Dashboard:** http://localhost:8181/visualization/dashboard.html
- **Leaderboard:** http://localhost:8181/visualization/leaderboard.html

> **Lưu ý:** Phải serve từ `stock_ml/` để browser có thể truy cập `results/leaderboard/` đúng path. Nếu serve từ `visualization/` thì path `../results/` sẽ không hoạt động.

Nếu port 8181 bị chiếm, dùng port khác:

```bash
python -m http.server 8080
```

Sau khi rebuild leaderboard, nhớ hard refresh browser (`Ctrl+F5`) để xóa cache JS cũ.

## Kiến trúc v2 — Component Pipeline + MarketProfile

### Luồng xử lý

```
ExperimentConfig (YAML)
        │
        ▼
  resolve_run_context()
        │
        ├── MarketProfile (data/execution/symbols/features/models/target)
        ├── ResolvedRunContext (market, schema, timeframe, run_identity)
        │
        ▼
   Pipeline.run()
        │
        ├── build_prediction_cache()   ← WalkForwardSplitter + Model train/predict
        │        │
        │        └── PredictionCacheManager (cache key scoped theo run_identity)
        │
        └── Champion Runner (per strategy)
                 │
                 ├── DataLoader → FeatureEngine/blocks → FusionStack
                 │
                 └── Unified Backtest Engine → PnlCalculator → list[Trade]
                          │
                          ▼
                    trades_df + market/schema/timeframe metadata
```

### MarketProfile

Mỗi run có `market` explicit. `ExperimentConfig` được resolve cùng `config/markets/<market>.yaml` thành `ResolvedRunContext`, sau đó context này đi xuyên trainer, cache, runner, backtest, artifact và leaderboard.

MarketProfile chỉ chứa defaults theo market:

- `data`: schema, timeframe, timestamp column, timezone, required columns, benchmark
- `execution`: `pnl_mode`, commission/tax/slippage, capital, currency, multiplier, funding, leverage/liquidation, rollover
- `symbols`: universe mặc định và symbol groups
- `features`, `models`, `target`: defaults theo market

Market hiện có:

| Market | Profile | PnL mode | Ghi chú |
|--------|---------|----------|---------|
| VN stock | `config/markets/vn_stock.yaml` | `equity_spot` | Stock daily, VND, có tax |
| Crypto spot | `config/markets/crypto_spot.yaml` | `equity_spot` | Spot OHLCV, USDT, không tax |
| Crypto perp | `config/markets/crypto_perp.yaml` | `linear_usdt_perp` | Funding/leverage/liquidation, cross-margin, short risk controls |
| VN derivatives | `config/markets/vn_derivatives.yaml` | `futures_contract` | Contract multiplier + rollover, cross-margin/short-ready |

### Backtest PnL modes

Backtest engine dispatch qua `src/backtest/pnl.py` theo `execution.pnl_mode`:

| Mode | Dùng cho |
|------|----------|
| `equity_spot` | VN stock, crypto spot |
| `linear_usdt_perp` | USDT-margined perpetual |
| `inverse_perp` | Inverse perpetual |
| `futures_contract` | Futures có contract multiplier/expiry rollover |

Derivatives path hiện đã có production-risk support trong scope hiện tại:

- short enable/disable rõ ràng qua MarketProfile/config
- funding/borrow fee, borrow availability/recall
- hard stop, hard cap, fast profit, zombie exit, squeeze exit cho short
- cross-margin portfolio engine với margin exhaustion, liquidation và aggregate short cap
- forced rollover và smart rollover theo `volume_crossover`, `oi_crossover`, `n_days_before_expiry`

Gap còn lại không block production path: ATR sizing hiện chỉ có trong portfolio engine, và tooling tạo spread columns (`next_volume`, `next_oi`) từ raw contract chain chưa có.

### Fusion Stack (4 layers)

```
Bar i:
  1. pre_entry   → skip_choppy, sma200_filter, anti_fomo, ...
  2. entry       → ML signal, hybrid_entry, rule_ensemble, ...
  3. hold        → trend_persistence, confirm_bars, min_hold, ...
  4. exit_rules   → hard_stop, ATR_stop, trailing, peak_protect, zombie, ...
```

### Walk-Forward Validation

```
2015 ──────────── 2019 | 2020 (test)
2016 ──────────── 2020 | 2021 (test)
2017 ──────────── 2021 | 2022 (test)
2018 ──────────── 2022 | 2023 (test)
2019 ──────────── 2023 | 2024 (test)
2020 ──────────── 2024 | 2025 (test)
     ← 4yr train →   ← 1yr test →
```

## Champion Versions (11)

| Version | Feature Set | Target | Entry Model | Exit Model | Ghi chú |
|---------|------------|--------|-------------|------------|---------|
| **v22** | leading_v2 | trend_regime | LightGBM | rule-only | Baseline `generic_fusion`, không bật exit model. |
| **v22_with_exit_model** | leading_v2 | early_wave | LightGBM | LightGBM | v22 + exit model (champion `entry_exit` track). |
| **v32** | leading_v3 | early_wave | LightGBM | LightGBM | Lineage `v32_runner` với patch_smart_hardcap, v26–v32 params. |
| **v34** | leading_v4 | early_wave | LightGBM | LightGBM | Lineage `v34_runner`. |
| **v35b** | leading_v4 | early_wave | LightGBM | LightGBM | `run_v35b` với strategy_v3 rule set. |
| **v37a** | leading_v4 | early_wave | LightGBM | LightGBM | Lineage `v37a_runner`. |
| **v37a_exit** | leading_v4 | early_wave_dual | LightGBM | LightGBM (fw=8) | v37a + exit model forward_window ngắn. |
| **v37d** | leading_v4 | early_wave | GRU | LightGBM | GRU entry, lineage `v37d_runner`. |
| **v39d** | leading_v4 | early_wave | LightGBM | LightGBM | Lineage `v39d_runner`. |
| **v42_a** | leading_v4 | early_wave_dual (fw=15) | LightGBM | LightGBM | Lineage `v42_a_runner`. |
| **rule** | leading_v2 | early_wave | Rule | — | Rule baseline (MACD+MA20), benchmark cho ML champion. |

> **Primary champion hiện tại (Pha 6, 2026-05-03):** `leading_v2 + random_forest + lightgbm_exit` (strategy `v22`), được chốt trong [docs/refactor/MODEL_SELECTION_RUN_PLAN.md](docs/refactor/MODEL_SELECTION_RUN_PLAN.md). Cấu hình này sống dưới dạng row trong các matrix `finalists_*`, không có file `champions/*.yaml` riêng.

Tất cả champion có golden baseline (CPU, deterministic) tại `tests/regression/golden/` và pass exact parity test.

## Thêm component mới

### Feature Block mới (~1 giờ)

```bash
# 1. Implement
src/components/features/blocks/my_block.py   # class MyBlock(FeatureBlock)

# 2. Register
src/components/features/registry.py          # _BLOCK_REGISTRY["my_block"] = MyBlock

# 3. Dùng trong YAML
config/feature_sets/leading_v5.yaml          # blocks: [..., my_block]
```

→ Xem [HOW_TO_ADD_FEATURE_BLOCK.md](docs/refactor/HOW_TO_ADD_FEATURE_BLOCK.md)

### Entry Model mới (~30 phút)

```bash
# 1. Implement + register
src/components/models/transformer.py
src/components/models/registry.py

# 2. Dùng trong YAML
config/experiments/exp_transformer_v1.yaml   # signals.entry_model.type: transformer
```

→ Xem [HOW_TO_ADD_ENTRY_MODEL.md](docs/refactor/HOW_TO_ADD_ENTRY_MODEL.md)

### Exit terminology

| Thuật ngữ | Ý nghĩa | Ví dụ |
|-----------|---------|-------|
| `exit_model` | Model supervised dự đoán điểm thoát riêng; khi bật sẽ train `y_pred_exit` và strategy consume tín hiệu này. | `signals.exit_model.type: lightgbm`, `v22_with_exit_model` |
| `strategy.exit_rules` | Rule thoát trong strategy layer; không cần train exit model riêng. | `hard_stop`, `v22_fast_exit`, `ma_cross_hybrid_exit` |
| `v22` | Champion baseline không dùng exit model; vẫn dùng rule exits. | `champions/v22` |
| `v22_with_exit_model` | Champion có bật exit model, cộng thêm rule exits an toàn. | `champions/v22_with_exit_model` |

Tránh dùng tên `hybrid` đơn lẻ cho artifact mới vì dễ nhầm với `ma_cross_hybrid_exit`; nếu cần mô tả combo exit model + rule, dùng tên cụ thể như `exit_model_plus_rules`.

### Tạo tổ hợp mới bằng Matrix YAML

Matrix dùng để backtest nhiều tổ hợp `feature_set × entry_model × exit_model × strategy` mà không cần viết thêm Python file.

#### 1. Tạo file matrix

Tạo YAML mới dưới `config/experiments/matrix/`, ví dụ `config/experiments/matrix/q3_2026.yaml`:

```yaml
name_prefix: q3_2026
axes:
  features: [leading_v2, leading_v4]
  model_type: [lightgbm, random_forest]
  exit_model:
    - {label: no_exit_model, enabled: false, type: "null"}
    - {label: lightgbm_exit, enabled: true, type: lightgbm, forward_window: 15, loss_threshold: 0.05}
  strategy:
    - v22
    - v34
base:
  signals:
    target:
      type: early_wave
  split:
    first_test_year: 2020
    last_test_year: 2025
```

Tham khảo matrix thật tại `config/experiments/matrix/model_selection.yaml`. Các giá trị hợp lệ có thể xem bằng:

```bash
python -m stock_ml list-components --type models
python -m stock_ml list-components --type exit_models
python -m stock_ml list-components --type runners
python -m stock_ml list-experiments
```

#### 2. Validate và dry-run trước khi backtest

```bash
python -m stock_ml validate matrix/q3_2026
python -m stock_ml run-matrix matrix/q3_2026 --dry-run
python -m stock_ml run-matrix matrix/q3_2026 --dry-run --limit 3
```

#### 3. Chạy backtest matrix

```bash
# Chạy full, mặc định lưu artifact vào results/experiments/q3_2026/
python -m stock_ml run-matrix matrix/q3_2026 --device cpu

# Chạy nhanh trên ít mã để kiểm tra flow
python -m stock_ml run-matrix matrix/q3_2026 --symbols-limit 10

# Resume matrix lớn, skip tổ hợp đã có ranking_row.json
python -m stock_ml run-matrix matrix/q3_2026 --resume

# Preview tất cả tổ hợp trên ít mã, sau đó chạy full top 3
python -m stock_ml run-matrix matrix/q3_2026 --symbols-limit 10 --top-k-preview 3 --resume

# Chạy song song CPU
python -m stock_ml run-matrix matrix/q3_2026 --jobs 4 --device cpu
```

Mỗi tổ hợp sẽ có artifact riêng gồm `trades.csv`, `metrics.json`, `ranking_row.json`, `predictions_meta.json`, `config.resolved.yaml`. Sau khi chạy xong, bundle matrix có thêm `ranking.csv` và `ranking.json`.

#### 4. Xếp hạng kết quả matrix

```bash
python -m stock_ml compare-matrix results/experiments/q3_2026 --top 10
python -m stock_ml compare-matrix results/experiments/q3_2026 --top 20 --min-trades 20 --max-mdd 20
python -m stock_ml compare-matrix results/experiments/q3_2026 --sort total_pnl --top 10
```

#### 5. Thêm kết quả vào leaderboard

Khi `run-matrix` lưu artifact, leaderboard tự cập nhật vào `results/leaderboard/` trừ khi dùng `--skip-leaderboard`.

```bash
# Cập nhật leaderboard tự động
python -m stock_ml run-matrix matrix/q3_2026 --resume

# Không cập nhật leaderboard nếu chỉ chạy thử
python -m stock_ml run-matrix matrix/q3_2026 --symbols-limit 10 --skip-leaderboard
```

#### 6. Benchmark tổ hợp/champion

`benchmark` nhận danh sách version champion trong `config/experiments/champions/*.yaml`:

```bash
python -m stock_ml benchmark --versions v22,v34,v37a --symbols-limit 10
python -m stock_ml benchmark --versions v22,v34,v37a --device gpu --output results/benchmark_q3_2026.json
```

Nếu winner đang nằm trong matrix, promote thành champion YAML trước rồi mới benchmark bằng tên version đó.

#### 7. Xuất top matrix vào leaderboard HTML/dashboard

Dashboard đọc dữ liệu trong `visualization/manifest.json` và các JSON được export. Với matrix, dùng `export-matrix` để lấy top-K từ bundle đã backtest:

```bash
# Export top 5 theo composite_score vào visualization/
python -m stock_ml export-matrix q3_2026 --top-k 5

# Hoặc truyền đường dẫn đầy đủ
python -m stock_ml export-matrix results/experiments/q3_2026 --top-k 10 --sort total_pnl

# Nếu cần tính/lấp composite_score trong manifest
python -m stock_ml score-models --force
```

Mở HTML bằng HTTP server:

```bash
cd visualization && python -m http.server 8080
# Mở http://localhost:8080/dashboard.html
```

Không cần sửa `visualization/dashboard.html`; file này đọc `manifest.json` và tự render model được export.

→ Xem thêm [HOW_TO_RUN_MATRIX.md](docs/refactor/HOW_TO_RUN_MATRIX.md)

## Regression & Tooling

Các lệnh dưới đây chạy từ thư mục `stock_ml/`. Nếu chạy từ repo root, thêm prefix `PYTHONPATH=stock_ml python -m pytest stock_ml/...`.

```bash
# Hash check champion CSVs vs golden (nhanh, ~1s)
pytest tests/regression/test_champions.py -q

# Full parity test (tất cả champion CSVs, ~336s CPU)
PYTHONHASHSEED=42 pytest tests/regression/ -q

# Unit tests components
pytest tests/components/ -q -k "not integration"

# Lint + format
ruff check src/ scripts/ tests/
ruff format src/ scripts/ tests/

# Type check
mypy src/components/ src/pipeline/ --ignore-missing-imports

# Pre-commit hooks (ruff + regression hash check)
pre-commit run --all-files

# Benchmark pipeline mới vs legacy
python -m stock_ml benchmark --versions v22,v34,v37a --symbols-limit 10
```

**Lưu ý regression:** Golden baseline được tạo bằng `--device cpu` (LightGBM GPU non-deterministic). Regression test bắt buộc CPU.

## Dashboard

```bash
# Export trades CSV → JSON cho dashboard
python -m stock_ml export                        # tất cả active models
python -m stock_ml export --versions v22,v34     # chọn lọc

# Mở dashboard (cần HTTP server do CORS)
cd visualization && python -m http.server 8080
# Mở http://localhost:8080/dashboard.html
```

Dashboard (`visualization/dashboard.html`) đọc `manifest.json` và tự render — không cần sửa HTML khi thêm/xóa model.

## Google Colab

```python
# Cell 1 — Clone + setup
!git clone https://github.com/nguyenduccanh011/train_ai_ml.git /content/repo 2>/dev/null \
  || git -C /content/repo pull --ff-only
%cd /content/repo/stock_ml
%run colab_setup.py

# Cell 2 — Run champion
!PYTHONHASHSEED=42 python -m stock_ml run champions/v22 --save-results
```

Data tự động được mount từ Google Drive qua `colab_setup.py`. Xem [docs/refactor/HOW_TO_RUN_MATRIX.md](docs/refactor/HOW_TO_RUN_MATRIX.md) để chạy grid search trên Colab.

## Tài liệu

| File | Nội dung |
|------|----------|
| [docs/refactor_multi_market.md](docs/refactor_multi_market.md) | Kế hoạch và trạng thái refactor multi-market, MarketProfile, PnL modes |
| [docs/refactor/ARCHITECTURE.md](docs/refactor/ARCHITECTURE.md) | Kiến trúc đích v2, component interfaces, YAML schema |
| [docs/refactor/CHAMPION_VERSIONS.md](docs/refactor/CHAMPION_VERSIONS.md) | 11 champion: lý do chọn, coverage matrix |
| [docs/refactor/REFACTOR_ROADMAP.md](docs/refactor/REFACTOR_ROADMAP.md) | Foundation v2 và diary refactor ban đầu |
| [docs/refactor/ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md](docs/refactor/ENTRY_EXIT_RESEARCH_REFACTOR_PLAN.md) | Entry/exit research matrix, champion `v22_with_exit_model`, phase còn partial |
| [docs/refactor/HOW_TO_ADD_FEATURE_BLOCK.md](docs/refactor/HOW_TO_ADD_FEATURE_BLOCK.md) | Thêm feature block mới |
| [docs/refactor/HOW_TO_ADD_FUSION_STRATEGY.md](docs/refactor/HOW_TO_ADD_FUSION_STRATEGY.md) | Thêm fusion strategy mới |
| [docs/refactor/HOW_TO_ADD_ENTRY_MODEL.md](docs/refactor/HOW_TO_ADD_ENTRY_MODEL.md) | Thêm entry model mới |
| [docs/refactor/HOW_TO_PORT_LEGACY_VERSION.md](docs/refactor/HOW_TO_PORT_LEGACY_VERSION.md) | Promote legacy version lên component runner |
| [docs/refactor/HOW_TO_RUN_MATRIX.md](docs/refactor/HOW_TO_RUN_MATRIX.md) | Grid search qua YAML |
| [docs/refactor/FUSION_STRATEGY_INVENTORY.md](docs/refactor/FUSION_STRATEGY_INVENTORY.md) | Inventory fusion strategies, mapping flag cũ → class mới |
| [docs/refactor/EXIT_MODEL_BUG.md](docs/refactor/EXIT_MODEL_BUG.md) | Bug exit model trained-but-dropped, status, fix plan |

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) chạy tự động khi push:
- **lint**: `ruff check` + `ruff format --check`
- **typecheck**: `mypy src/components/ src/pipeline/`
- **test-unit**: `pytest tests/components/ -k "not integration"`
- **test-regression**: hash check champion CSVs vs golden (push-only, không chạy mỗi PR)

---

*Cập nhật: 2026-05-06 — Multi-market Phase 0–7 đã hoàn tất: `MarketProfile`, `ResolvedRunContext`, cache/artifact/leaderboard scoping theo `run_identity`, profile `vn_stock`, `crypto_spot`, `crypto_perp`, `vn_derivatives`, backtest PnL dispatch cho spot/perp/futures, cross-margin portfolio engine, short risk controls và smart futures rollover. `run_pipeline.py` vẫn deprecated, dùng `python -m stock_ml run` thay thế.*
