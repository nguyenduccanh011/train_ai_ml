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
│   ├── base.yaml                    # Cấu hình pipeline (data_dir, device, symbols)
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
│   ├── data/                        # DataLoader, WalkForwardSplitter, TargetGenerator (legacy)
│   ├── features/engine.py           # FeatureEngine legacy (vẫn dùng bởi runners)
│   ├── models/registry.py           # ModelRegistry legacy
│   ├── export/unified_export.py     # CSV → JSON cho dashboard
│   └── env.py / config_loader.py    # Path resolution, YAML loading
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

### Grid Search (~5 phút setup)

```bash
# 1. Định nghĩa matrix
config/experiments/matrix/q3_2026.yaml

# 2. Chạy hoặc dry-run nhanh
python -m stock_ml run-matrix matrix/q3_2026
python -m stock_ml run-matrix matrix/q3_2026 --dry-run --limit 3

# 3. Resume / preview top-k nếu matrix lớn
python -m stock_ml run-matrix matrix/q3_2026 --resume
python -m stock_ml run-matrix matrix/q3_2026 --symbols-limit 10 --top-k-preview 3

# 4. So sánh artifact đã lưu
python -m stock_ml compare-matrix results/experiments/q3_2026
```

→ Xem [HOW_TO_RUN_MATRIX.md](docs/refactor/HOW_TO_RUN_MATRIX.md)

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

*Cập nhật: 2026-05-03 — Pha 6 model selection đã chốt primary champion `leading_v2 + random_forest + lightgbm_exit` (strategy `v22`); thêm CLI `compare-champions` để chạy & xếp hạng toàn bộ champion với 1 split chung. Phase 5 terminology giữ nguyên: `exit_model`, `strategy.exit_rules`, `signals.entry_model`, `v22_with_exit_model`, `LegacyAdapter`; `run_pipeline.py` vẫn deprecated, dùng `python -m stock_ml run` thay thế.*
