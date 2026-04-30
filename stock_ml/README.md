# Stock ML Trading System v2

Hệ thống ML dự đoán xu hướng giá cổ phiếu Việt Nam (dữ liệu 2015-2025), sử dụng walk-forward validation với kiến trúc component-based composable (v2.0).

## Cài đặt

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # ruff, mypy, pytest (dev only)
```

**Yêu cầu:** Python 3.11+

## Cấu trúc dự án

```
stock_ml/
├── config/
│   ├── models.yaml                  # Registry legacy — 60+ versions (vẫn dùng cho adapter)
│   ├── base.yaml                    # Cấu hình pipeline (data_dir, device, symbols)
│   ├── feature_sets/                # YAML định nghĩa feature block composition
│   │   ├── leading.yaml
│   │   ├── leading_v2.yaml
│   │   ├── leading_v3.yaml
│   │   └── leading_v4.yaml
│   └── experiments/
│       ├── champions/               # YAML cho 11 champion versions
│       │   ├── v22.yaml
│       │   ├── v35b.yaml
│       │   └── ...
│       ├── matrix/                  # Grid search matrix configs
│       └── legacy/                  # Auto-generated từ migrate-legacy
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
│   │   │   └── strategies/          # ~92 strategy classes (pre_entry/entry/hold/exit_override)
│   │   ├── backtest/
│   │   │   └── engine.py            # SimpleLongBacktester
│   │   └── runners/                 # Dedicated runner cho 11 champions
│   │       ├── rule_runner.py
│   │       ├── v22_runner.py
│   │       ├── v32_runner.py
│   │       └── ... (11 runners)
│   │
│   ├── pipeline/                    # ← ORCHESTRATOR MỚI
│   │   ├── config.py                # ExperimentConfig (Pydantic)
│   │   ├── orchestrator.py          # Pipeline class + PipelineResult
│   │   ├── trainer.py               # build_prediction_cache()
│   │   ├── build_predictions.py     # Shared legacy prediction builder (không import run_pipeline.py)
│   │   ├── walker.py                # walk_forward() splits
│   │   ├── cache.py                 # PredictionCacheManager (pickle + sha256 key)
│   │   ├── matrix_expander.py       # expand_matrix(yaml) → list[ExperimentConfig]
│   │   ├── validate.py              # validate_config() → list[ValidationError]
│   │   └── legacy_adapter.py        # LegacyVersionAdapter — wrap 60+ backtest_vXX
│   │
│   ├── data/                        # DataLoader, WalkForwardSplitter, TargetGenerator (legacy)
│   ├── features/engine.py           # FeatureEngine legacy (vẫn dùng bởi runners)
│   ├── models/registry.py           # ModelRegistry legacy
│   ├── export/unified_export.py     # CSV → JSON cho dashboard
│   └── env.py / config_loader.py    # Path resolution, YAML loading
│
├── scripts/
│   ├── cli.py                       # Entry point CLI
│   ├── migrate_legacy.py            # models.yaml entry → ExperimentConfig YAML
│   └── benchmark.py                 # Pipeline mới vs legacy timing
│
├── tests/
│   ├── components/                  # Unit + equivalence tests (200+ tests)
│   └── regression/
│       ├── golden/                  # 11 CSV + checksums.txt (CPU, deterministic)
│       ├── test_champions.py        # Hash check 11 champions vs golden
│       └── test_*_parity.py         # Per-champion exact parity tests
│
├── experiments/                     # Legacy/champion backtest functions (adapter + parity reference)
├── archive/
│   ├── results_legacy/              # Historical backtest results (giữ nguyên)
│   └── scripts/                     # run_v10_compare.py → run_v22_compare.py (model config reference)
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

# Validate config
python -m stock_ml validate champions/v22

# Export trades CSV → dashboard JSON (tất cả active models)
python -m stock_ml export
python -m stock_ml export --versions v22,v34,v37a

# So sánh nhiều experiments
python -m stock_ml compare champions/v22 champions/v35b champions/v34

# List components đã đăng ký
python -m stock_ml list-components
python -m stock_ml list-components --type fusion

# List experiments / legacy versions
python -m stock_ml list-experiments
python -m stock_ml list-legacy

# Chạy version legacy (qua adapter, không cần component runner)
python -m stock_ml run legacy/v25

# Migrate legacy config → YAML
python -m stock_ml migrate-legacy v25
python -m stock_ml migrate-legacy --all

# Benchmark pipeline mới vs legacy
python -m stock_ml benchmark --versions v22,v34,v37a
```

## Kiến trúc v2 — Component Pipeline

### Luồng xử lý

```
ExperimentConfig (YAML)
        │
        ▼
   Pipeline.run()
        │
        ├── build_prediction_cache()   ← WalkForwardSplitter + Model train/predict
        │        │
        │        └── PredictionCacheManager (sha256 cache key)
        │
        └── Champion Runner (per strategy)
                 │
                 ├── DataLoader → FeatureEngine → FusionStack
                 │
                 └── SimpleLongBacktester → list[Trade]
                          │
                          ▼
                    trades_df (CSV format)
```

### Fusion Stack (4 layers)

```
Bar i:
  1. pre_entry   → skip_choppy, sma200_filter, anti_fomo, ...
  2. entry       → ML signal, hybrid_entry, rule_ensemble, ...
  3. hold        → trend_persistence, confirm_bars, min_hold, ...
  4. exit_override → hard_stop, ATR_stop, trailing, peak_protect, zombie, ...
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

| Version | Feature Set | Target | Model | Score |
|---------|------------|--------|-------|------:|
| **v22** | leading_v2 | trend_regime | LightGBM | 686.3 |
| **v32** | leading_v3 | early_wave | LightGBM | 353.4 |
| **v34** | leading_v4 | early_wave | LightGBM | 598.4 |
| **v35b** | leading_v4 | early_wave | LightGBM | 603.7 |
| **v37a** | leading_v4 | early_wave_dual | LightGBM | 603.4 |
| **v37a_exit** | leading_v4 | early_wave_dual | LightGBM | — |
| **v37d** | leading_v4 | early_wave | GRU | — |
| **v39d** | leading_v4 | early_wave_dual | LightGBM | **611.7** |
| **v42_a** | leading_v4 | early_wave_dual fw=15 | LightGBM | 550.1 |
| **v19_3** | leading_v2 | early_wave | LightGBM | — |
| **rule** | — | — | Rule (MACD+MA20) | 295.2 |

Tất cả 11 champion có golden baseline (CPU, deterministic) tại `tests/regression/golden/` và pass exact parity test.

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
config/experiments/exp_transformer_v1.yaml   # components.entry_model.type: transformer
```

→ Xem [HOW_TO_ADD_ENTRY_MODEL.md](docs/refactor/HOW_TO_ADD_ENTRY_MODEL.md)

### Grid Search (~5 phút setup)

```bash
# 1. Định nghĩa matrix
config/experiments/matrix/q3_2026.yaml

# 2. Chạy
python -m stock_ml run-matrix matrix/q3_2026

# 3. So sánh
python -m stock_ml compare matrix/q3_2026/* --top 10
```

→ Xem [HOW_TO_RUN_MATRIX.md](docs/refactor/HOW_TO_RUN_MATRIX.md)

## Regression & Tooling

Các lệnh dưới đây chạy từ thư mục `stock_ml/`. Nếu chạy từ repo root, thêm prefix `PYTHONPATH=stock_ml python -m pytest stock_ml/...`.

```bash
# Hash check 11 champions vs golden (nhanh, ~1s)
pytest tests/regression/test_champions.py -q

# Full parity test (tất cả 11 champions, ~336s CPU)
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
| [docs/refactor/ARCHITECTURE.md](docs/refactor/ARCHITECTURE.md) | Kiến trúc đích v2, component interfaces, YAML schema |
| [docs/refactor/CHAMPION_VERSIONS.md](docs/refactor/CHAMPION_VERSIONS.md) | 11 champion: lý do chọn, coverage matrix |
| [docs/refactor/REFACTOR_ROADMAP.md](docs/refactor/REFACTOR_ROADMAP.md) | 6 phase đã hoàn thành, diary từng ngày |
| [docs/refactor/HOW_TO_ADD_FEATURE_BLOCK.md](docs/refactor/HOW_TO_ADD_FEATURE_BLOCK.md) | Thêm feature block mới |
| [docs/refactor/HOW_TO_ADD_FUSION_STRATEGY.md](docs/refactor/HOW_TO_ADD_FUSION_STRATEGY.md) | Thêm fusion strategy mới |
| [docs/refactor/HOW_TO_ADD_ENTRY_MODEL.md](docs/refactor/HOW_TO_ADD_ENTRY_MODEL.md) | Thêm entry model mới |
| [docs/refactor/HOW_TO_PORT_LEGACY_VERSION.md](docs/refactor/HOW_TO_PORT_LEGACY_VERSION.md) | Promote legacy version lên component runner |
| [docs/refactor/HOW_TO_RUN_MATRIX.md](docs/refactor/HOW_TO_RUN_MATRIX.md) | Grid search qua YAML |
| [docs/refactor/FUSION_STRATEGY_INVENTORY.md](docs/refactor/FUSION_STRATEGY_INVENTORY.md) | 92 strategy classes, mapping flag cũ → class mới |
| [docs/refactor/EXIT_MODEL_BUG.md](docs/refactor/EXIT_MODEL_BUG.md) | Bug exit model trained-but-dropped, status, fix plan |

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) chạy tự động khi push:
- **lint**: `ruff check` + `ruff format --check`
- **typecheck**: `mypy src/components/ src/pipeline/`
- **test-unit**: `pytest tests/components/ -k "not integration"`
- **test-regression**: hash check 11 champions vs golden (push-only, không chạy mỗi PR)

---

*Cập nhật: 2026-04-29 — Phase 7 Step 0 hoàn tất: `_build_predictions` đã tách khỏi `run_pipeline.py`; `run_pipeline.py` vẫn deprecated, dùng `python -m stock_ml run` thay thế.*
