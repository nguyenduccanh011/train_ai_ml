# Stock ML Trading System

## Tong quan

He thong ML du doan xu huong gia co phieu Viet Nam (370+ ma, 2015-2025) su dung LightGBM voi walk-forward validation. He thong ho tro quan ly nhieu phien ban model, backtest tu dong, va dashboard so sanh truc quan.

## Cai dat

```bash
pip install -r requirements.txt
```

**Yeu cau:** Python 3.8+, numpy, pandas, lightgbm, scikit-learn, pyyaml

## GPU Acceleration

He thong ho tro GPU acceleration cho LightGBM, XGBoost va CatBoost. Mac dinh se **auto-detect** GPU.

### Cau hinh

**Cach 1: Config file** (`config/base.yaml`):
```yaml
training:
  device: "auto"   # auto | gpu | cuda | cpu
```

**Cach 2: Command line** (override config):
```bash
python run_pipeline.py --version v25 --device gpu     # Force GPU
python run_pipeline.py --version v25 --device cpu     # Force CPU
python run_pipeline.py --version v25 --device auto    # Auto-detect
```

### Device mapping theo thu vien

| Setting | LightGBM | XGBoost | CatBoost |
|---------|----------|---------|----------|
| `gpu` | `device="gpu"` (OpenCL) | `device="cuda"` | `task_type="GPU"` |
| `cpu` | `device="cpu"` | `device="cpu"` | `task_type="CPU"` |
| `auto` | Tu detect NVIDIA GPU -> gpu/cpu | Tu detect -> cuda/cpu | Tu detect -> GPU/CPU |

### Yeu cau GPU

- **NVIDIA GPU** voi driver cap nhat (kiem tra: `nvidia-smi`)
- **LightGBM GPU:** Can OpenCL runtime (thuong co san voi NVIDIA driver)
- **XGBoost GPU:** Can CUDA toolkit (thuong co san neu da cai PyTorch/TensorFlow)
- **CatBoost GPU:** Can CUDA toolkit

### Kiem tra GPU hoat dong

```bash
# Kiem tra NVIDIA GPU
nvidia-smi

# Test LightGBM GPU
python -c "import lightgbm as lgb; import numpy as np; m=lgb.LGBMClassifier(device='gpu',n_estimators=10,verbose=-1); X=np.random.rand(100,5); y=np.random.randint(0,3,100); m.fit(X,y); print('LightGBM GPU OK')"

# Test XGBoost GPU
python -c "import xgboost as xgb; import numpy as np; m=xgb.XGBClassifier(device='cuda',n_estimators=10,tree_method='hist',verbosity=0); X=np.random.rand(100,5); y=np.random.randint(0,3,100); m.fit(X,y); print('XGBoost GPU OK')"
```

### Luu y hieu nang

- GPU hieu qua nhat khi dataset **lon** (>100K rows, >50 features)
- Voi dataset nho, CPU co the nhanh hon do overhead GPU data transfer
- Walk-forward validation train 6 models tuan tu -> GPU giup moi fold nhanh hon
- Tren RTX 3060 12GB, expect ~2-5x speedup cho LightGBM training phase

## Cau truc du an

```
stock_ml/
+-- config/
|   +-- models.yaml              # Registry trung tam -- dinh nghia TAT CA model versions
|   +-- base.yaml                # Cau hinh training device (gpu/cpu/auto)
|
+-- src/                          # Core ML pipeline (shared code)
|   +-- config_loader.py          # Doc models.yaml, cung cap helper functions
|   +-- safe_io.py                # Fix UnicodeEncodeError tren Windows console
|   +-- data/
|   |   +-- loader.py             # DataLoader -- load OHLCV tu CSV (Hive-partitioned)
|   |   +-- splitter.py           # WalkForwardSplitter -- rolling 4yr train / 1yr test
|   |   +-- target.py             # TargetGenerator -- 3-class trend regime (dual MA)
|   +-- features/
|   |   +-- engine.py             # FeatureEngine -- leading/leading_v2 feature sets
|   +-- models/
|   |   +-- registry.py           # ModelRegistry -- LightGBM, XGBoost, RandomForest
|   +-- evaluation/
|   |   +-- scoring.py            # Unified scoring -- composite_score() + calc_metrics()
|   |   +-- metrics.py            # Classification metrics (F1, accuracy)
|   |   +-- backtest.py           # Backtest engine v1 (legacy)
|   |   +-- backtest_v2.py        # Backtest engine v2 (legacy)
|   +-- export/
|       +-- unified_export.py     # Export thong nhat -- CSV -> JSON cho dashboard
|
+-- visualization/                # Web dashboard
|   +-- dashboard.html            # Dashboard dong -- tu render tu manifest.json
|   +-- manifest.json             # Auto-generated: danh sach models + config
|   +-- js/
|   |   +-- state.js              # Global state
|   |   +-- chart.js              # TradingView Lightweight Charts
|   |   +-- ui.js                 # Dynamic UI rendering
|   |   +-- app.js                # Main app logic + data loading
|   +-- data/                     # Base OHLCV data (per-symbol JSON)
|   +-- data_v27/                 # V27 trades + markers (per-symbol JSON)
|   +-- data_v26/                 # V26 trades + markers
|   +-- data_v25/                 # V25 trades + markers
|   +-- data_v24/                 # V24 trades + markers
|   +-- data_v23/                 # V23 trades + markers
|   +-- data_v22/                 # V22 trades + markers
|   +-- data_v19_1/               # V19.1 trades + markers
|   +-- data_rule/                # Rule trades + markers
|
+-- results/                      # Backtest output
|   +-- trades_v27.csv            # V27 trades (entry/exit/pnl per trade)
|   +-- trades_v27.meta.json      # Metadata: symbols, conditions, timestamp
|   +-- trades_v26.csv            # V26 trades
|   +-- trades_v25.csv            # V25 trades
|   +-- trades_v24.csv            # V24 trades
|   +-- trades_v23.csv            # V23 trades
|   +-- trades_v22.csv            # V22 trades
|   +-- trades_rule.csv           # Rule-based trades
|   +-- trades_rule.meta.json     # Metadata cho rule baseline
|
+-- model_manager.py              # CLI quan ly model (list/compare/retire/add)
+-- run_pipeline.py               # Pipeline runner thong nhat (shared ML cache)
|
+-- run_v27.py                    # V27 backtest function + standalone runner
+-- run_v26.py                    # V26 backtest function + runner
+-- run_v25.py                    # V25 ablation study + backtest function
+-- run_v24.py                    # V24 backtest function + runner
+-- run_v23_optimal.py            # V23 backtest function
+-- run_v22_final.py              # V22 backtest function
+-- run_v19_1_compare.py          # V19.1 backtest + run_test() + calc_metrics()
+-- compare_rule_vs_model.py      # Rule-based backtest
+-- run_feature_ablation.py       # Feature group ablation experiment
|
+-- docs/                         # Tai lieu bo sung
+-- requirements.txt              # Python dependencies
```

## Pipeline ML

### Luong xu ly chinh

```
+-------------+    +--------------+    +--------------+    +--------------+
|  Data Load   |--->|  Feature Eng |--->|  Train Model |--->|   Backtest   |
|  (loader.py) |    |  (engine.py) |    |  (LightGBM)  |    |  (run_v*.py) |
+-------------+    +--------------+    +--------------+    +------+-------+
                                                                  |
                                                                  v
+-------------+    +--------------+    +--------------+    +--------------+
|  Dashboard   |<---|  manifest.json|<---|  Unified Export|<---| trades_v*.csv|
|  (HTML+JS)   |    |  (auto-gen)  |    |  (export.py)  |    |  (results/)  |
+-------------+    +--------------+    +--------------+    +--------------+
```

### Walk-Forward Validation

```
2015 ------------ 2019 | 2020 (test)
2016 ------------ 2020 | 2021 (test)
2017 ------------ 2021 | 2022 (test)
2018 ------------ 2022 | 2023 (test)
2019 ------------ 2023 | 2024 (test)
2020 ------------ 2024 | 2025 (test)
     <-- 4yr train -->   <-- 1yr test -->
```

- **Model:** LightGBM classifier (3 classes: UPTREND / SIDEWAYS / DOWNTREND)
- **Target:** Dual MA crossover (SMA5/SMA20), shifted -1 de du doan ngay mai
- **Features:** leading_v2 feature set (Groups A+B+C+D: Market Structure, Exhaustion, Volatility Regime, Multi-timeframe)
- **Symbols:** 370+ ma co phieu VN (HOSE + HNX), auto-detect voi min 2000 rows

### Backtest Engine

Moi version co mot ham `backtest_vXX()` rieng voi logic entry/exit khac nhau:

| Component | Mo ta |
|-----------|-------|
| **Entry Logic** | ML signal + breakout + V-shape + rule ensemble |
| **Exit Logic** | Hard stop, ATR stop, trailing, profit lock, peak protect, zombie, signal confirm |
| **Position Sizing** | Trend-based (strong/moderate/weak) + ATR-adjusted |
| **Mod Flags** | 10 toggles (a-j) bat/tat tung module |
| **Regime Adapter** | Symbol profiles (bank, high_beta, momentum, defensive) |

### Chay pipeline

```bash
# Chay 1 version
python run_pipeline.py --version v24

# Chay + so sanh (SMART: chi chay v27, tu dung CSV cu cho v26/v25/rule)
python run_pipeline.py --version v27 --compare v26,v25,rule

# Chay tat ca active models
python run_pipeline.py --all

# Chay tat ca nhung skip version da co CSV
python run_pipeline.py --all --skip-existing

# Bat buoc chay lai tat ca (khong skip)
python run_pipeline.py --version v27 --compare v26,v25 --force

# Chi export (khong train lai)
python run_pipeline.py --export-all

# Chi export 1 version
python run_pipeline.py --version v24 --export-only

# Custom symbols
python run_pipeline.py --version v24 --symbols ACB,FPT,HPG
```

### Pipeline Architecture (Unified)

Pipeline runner (`run_pipeline.py`) dam bao **cong bang** khi so sanh:

1. **Shared Symbol Resolution:** Tat ca models duoc test tren CUNG symbol list (auto-detect hoac tu config). Khong con tinh trang V24 chay 14 symbols nhung V27 chay 61 symbols.

2. **Shared ML Predictions:** Models cung `feature_set` chia se 1 lan ML training. Pipeline group models theo feature_set, train LightGBM 1 lan per group, roi chay nhieu backtest functions tren cached predictions.

   ```
   feature_set="leading_v2" (6 models: v22-v27):
     Data -> Features -> Train (1 lan) -> Cache y_pred
       -> backtest_v22(y_pred) -> trades_v22.csv
       -> backtest_v23(y_pred) -> trades_v23.csv
       -> ...
       -> backtest_v27(y_pred) -> trades_v27.csv
   
   strategy="rule" (1 model):
     Data -> Walk-forward split -> backtest_rule(per fold) -> trades_rule.csv
   ```

3. **Fair Rule Baseline:** Rule-based model chay qua CUNG walk-forward split windows nhu ML models (truoc day rule baseline chi filter date >= "2020-01-01" lien tuc, khong chia fold).

4. **Metadata Tracking:** Moi trades CSV co companion `.meta.json` ghi lai dieu kien sinh ra (symbols, min_rows, feature_set, timestamp). Smart cache kiem tra metadata truoc khi tai su dung CSV cu.

5. **Unified Scoring:** Tat ca comparisons su dung cung 1 `composite_score()` formula (cau hinh weights trong `config/models.yaml`).

### Smart Cache (Tai su dung ket qua backtest)

Ket qua backtest duoc luu duoi dang CSV trong `results/trades_v{XX}.csv` kem theo metadata `results/trades_v{XX}.meta.json`. He thong **tu dong tai su dung** ket qua cu khi so sanh:

```
Vi du: Ban da co CSV cho v26, v25, v24. Tao v27 moi va muon so sanh:

# Chi chay backtest cho v27, tu dung CSV cu cho v26/v25/v24
python run_pipeline.py --version v27 --compare v26,v25,v24

Output:
  Symbols: 367 (shared across all models)

  SMART CACHE: Reusing existing CSV for 3 version(s):
    v26: 9928 trades (cached 2026-04-22 02:27)
    v25: 10116 trades (cached 2026-04-22 02:30)
    v24: 10041 trades (cached 2026-04-22 02:32)

  WILL RUN backtest for: v27
```

| Flag | Hanh vi |
|------|---------|
| *(mac dinh)* | `--version` luon chay, `--compare` tu skip neu co CSV |
| `--skip-existing` | Skip TAT CA versions da co CSV (ke ca `--version`) |
| `--force` | Chay lai TAT CA, bo qua cache |

**Metadata validation:** Khi smart cache tai su dung CSV cu, he thong kiem tra `.meta.json` de dam bao CSV duoc tao voi cung dieu kien (symbols, min_rows). Neu khac biet, hien canh bao:

```
  WARNING: 2 cached CSV(s) have mismatched conditions:
    v24: symbol list differs (14 changes: cached=14, current=367)
    Use --force to re-generate these for a fair comparison.
```

> **Luu y:** Export va Compare luon bao gom TAT CA versions (ca cached lan moi chay).

## Quan ly Model

### Registry (`config/models.yaml`)

Day la **single source of truth** cho tat ca model versions. Moi model co:

```yaml
v27:
  name: "V27"                     # Ten hien thi
  description: "V26 + patches"    # Mo ta ngan
  color: "#9C27B0"                # Mau tren dashboard
  active: true                    # true = dang dung, false = da retire
  strategy: "v27"                 # Key de map toi backtest function
  feature_set: "leading_v2"       # Feature set cho ML training
  mods: {a: true, b: true, ...}   # Mod flags cho backtest
  params: {hard_cap_weak: -0.10}  # Params rieng cho version nay
  marker_shape: "arrowUp"         # Hinh dang marker tren chart
  order: 0                        # Thu tu hien thi
```

### Unified Scoring

He thong su dung 1 canonical `composite_score()` (dinh nghia trong `src/evaluation/scoring.py`).  Weights duoc cau hinh trong `config/models.yaml`:

```yaml
scoring:
  weights:
    total_pnl: 0.30
    profit_factor: 0.25
    win_rate: 0.20
    avg_pnl: 0.15
    max_loss_penalty: 0.10
```

`model_manager.py compare` va `run_pipeline.py` deu su dung cung formula nay de xep hang.

### Symbol Resolution

Symbols duoc cau hinh trong `config/models.yaml`:

```yaml
pipeline:
  symbols:
    mode: auto          # auto | explicit
    min_rows: 2000      # Chi lay symbols co >= 2000 dong du lieu
    explicit_list: ""   # Danh sach cu the (chi dung khi mode=explicit)
```

Khi `mode: auto`, he thong tu dong scan thu muc du lieu va chon tat ca symbols du dieu kien. Tat ca models trong cung 1 pipeline run se dung CUNG 1 symbol list.

### CLI Commands

```bash
# Xem tat ca models + trang thai CSV
python model_manager.py list

# Chi xem active models
python model_manager.py list --active

# So sanh metrics tat ca active models (voi composite score)
python model_manager.py compare

# So sanh cac versions cu the (ke ca retired)
python model_manager.py compare --versions v27,v26,v25,rule

# Retire model (danh dau loi thoi)
python model_manager.py retire v18 --reason "Superseded by V19.1"

# Kich hoat lai model da retire
python model_manager.py activate v18

# Them model moi vao registry
python model_manager.py add v28 --name "V28 New" --color "#FF5722" --strategy v28
```

### Trang thai Model

| Trang thai | Y nghia | Dashboard |
|------------|---------|-----------|
| `active: true` | Dang su dung, hien thi tren dashboard | Hien thi |
| `active: false` | Da loi thoi, khong hien thi | An |

Khi retire model, chi can chay `python model_manager.py retire vXX` -> dashboard tu an model do.

## Workflow: Them Model Moi

### Buoc 1: Dang ky model

```bash
python model_manager.py add v28 --name "V28 Improved" --color "#FF5722" --strategy v28
```

### Buoc 2: Viet backtest function

Tao file `run_v28.py` voi ham `backtest_v28()`. Co the copy tu version gan nhat va chi sua phan logic khac:

```python
def backtest_v28(y_pred, returns, df_test, feature_cols,
                 initial_capital=100_000_000, commission=0.0015, tax=0.001,
                 record_trades=True,
                 mod_a=True, mod_b=True, ...):
    # Logic entry/exit moi o day
    ...
```

### Buoc 3: Dang ky backtest function

Them vao `get_backtest_function()` trong `run_pipeline.py`:

```python
strategy_map = {
    "v28": ("run_v28", "backtest_v28"),
    ...
}
```

### Buoc 4: Chay backtest

```bash
# Chay V28 + so sanh voi V27 va Rule
python run_pipeline.py --version v28 --compare v27,rule
```

Output: `results/trades_v28.csv` + `results/trades_v28.meta.json` + `visualization/data_v28/` + `manifest.json` updated

### Buoc 5: Xem ket qua

Mo `visualization/dashboard.html` trong browser -> V28 tu xuat hien tren dashboard!

### Buoc 6: Quyet dinh

```bash
# So sanh tat ca models (composite score)
python model_manager.py compare

# Neu V28 tot hon V26 -> retire V26
python model_manager.py retire v26 --reason "Superseded by V28"

# Re-export de dashboard cap nhat
python run_pipeline.py --export-all
```

## Visualization (Dashboard)

### Cach hoat dong

Dashboard (`visualization/dashboard.html`) hoat dong **hoan toan dong**:

1. Doc `manifest.json` -> biet co bao nhieu models, mau gi, data o dau
2. Tu tao toggle buttons cho moi model
3. Tu tao legend, stat cards, trade panels
4. Load data per-symbol tu `data_{version}/{SYM}.json`

**Khong can sua HTML khi them/bo model!**

### Mo dashboard

```bash
# Can HTTP server (do CORS policy)
cd stock_ml/visualization
python -m http.server 8080
# Mo http://localhost:8080/dashboard.html
```

Hoac dung VS Code Live Server extension.

### Tinh nang

- **Toggle models:** Click button de an/hien tung model tren chart
- **Stat cards:** So sanh WR, PnL, PF, MaxDD cho moi model (auto BEST badge)
- **Trade tables:** Xem chi tiet tung trade (entry/exit/reason/trend)
- **Symbol selector:** Dropdown chon ma co phieu, hien thi PnL tong hop

### File structure

```
visualization/
+-- dashboard.html          # HTML + CSS (static)
+-- manifest.json           # Auto-generated boi unified_export
+-- js/
|   +-- state.js            # Global variables
|   +-- chart.js            # Chart creation + marker management
|   +-- ui.js               # Dynamic rendering (buttons, stats, trades)
|   +-- app.js              # Init, data loading, event handlers
+-- data/                   # Base OHLCV (generated by export)
|   +-- index.json          # Symbol list + file paths
|   +-- ACB.json            # Per-symbol: ohlcv data
|   +-- ...
+-- data_v27/               # V27 overlay (generated by unified_export)
|   +-- index.json
|   +-- ACB.json            # {v27_markers, v27_trades, v27_stats}
|   +-- ...
+-- data_rule/              # Rule overlay
    +-- ...
```

## Du lieu

### Nguon du lieu

```
portable_data/vn_stock_ai_dataset_cleaned/
+-- all_symbols/
|   +-- symbol=ACB/
|   |   +-- timeframe=1D/
|   |       +-- data.csv    # OHLCV daily: date, open, high, low, close, volume
|   +-- symbol=FPT/
|   +-- ...
+-- clean_symbols.txt       # Danh sach ma hop le
```

### Tien xu ly

```bash
python preprocess_stocks.py  # Tu raw -> cleaned dataset
```

### Trades CSV Format

Moi file `results/trades_{version}.csv` co cac cot:

| Cot | Mo ta |
|-----|-------|
| `entry_date` | Ngay vao lenh (YYYY-MM-DD) |
| `exit_date` | Ngay thoat lenh |
| `pnl_pct` | Loi nhuan % cua trade |
| `holding_days` | So ngay giu |
| `exit_reason` | Ly do thoat (stop_loss, trailing_stop, signal, ...) |
| `entry_trend` | Xu huong luc vao (strong/moderate/weak) |
| `symbol` / `entry_symbol` | Ma co phieu |
| `position_size` | Kich thuoc vi the (0.25 - 1.0) |
| `entry_profile` | Profile ma (bank/high_beta/momentum/defensive/balanced) |
| `vshape_entry` | True neu vao bang V-shape bypass |
| `breakout_entry` | True neu vao bang breakout |
| `quick_reentry` | True neu vao lai nhanh sau trailing stop |

### Metadata JSON Format

Moi file `results/trades_{version}.meta.json`:

```json
{
  "version": "v27",
  "generated_at": "2026-04-22T15:30:00",
  "generator": "run_pipeline.py",
  "symbols": ["AAA", "AAS", "AAV", "..."],
  "n_symbols": 367,
  "min_rows": 2000,
  "feature_set": "leading_v2",
  "n_trades": 9915
}
```

## Kien truc Model Versions

### Lich su phat trien

```
V11 (baseline) -> V17 (modules A-J) -> V18 (adaptive) -> V19 (regime)
    -> V19.1 (risk-tuned) -> V19.3 (binary sizing) -> V22 (SMA200 + hybrid)
    -> V23 (graduated exits) -> V24 (5 patches) -> V25 (ablation-validated)
    -> V26 (skip choppy + rule ensemble) -> V27 (hardcap two-step + trend persistence)
```

### Active Models (hien tai)

| Version | Mo ta | Feature Set |
|---------|-------|-------------|
| **V27** | V26 + hardcap two-step + rule priority + dynamic score5 penalty + trend persistence | leading_v2 |
| **V26** | V25 + skip choppy + relaxed strong-trend entry + stronger rule ensemble | leading_v2 |
| **V25** | V24 all patches + pp_threshold=0.12 (ablation-validated) | leading_v2 |
| **V24** | V23 + smart hard_cap + peak_protect + long-horizon carry + symbol tuning + rule ensemble | leading_v2 |
| **V23** | Graduated exits + restored peak_protect + trend-specific caps | leading_v2 |
| **V22** | V19.1 + SMA200 filter + hybrid entry + fast_exit_loss + signal_hard_cap + time_decay | leading_v2 |
| **Rule** | MACD_hist > 0 AND Close > MA20 AND Close > Open (baseline) | N/A |

### Retired Models

| Version | Ly do |
|---------|-------|
| V19.3 | Superseded by V22/V23 |
| V19.1 | Superseded by V22/V23/V24/V25 |
| V18 | Superseded by V19.1 |

### Mod Flags (a-j)

| Flag | Module | Mo ta |
|------|--------|-------|
| `a` | V-shape bypass | Cho phep vao lenh khi phat hien V-shape reversal |
| `b` | Peak protection | Bao ve loi nhuan khi gia giam tu dinh |
| `c` | Fast loss cut | Cat lo nhanh (disabled mac dinh) |
| `d` | Adaptive exit confirm | Exit confirm thich ung (disabled mac dinh) |
| `e` | Secondary breakout | Scanner breakout thu cap (tieu chi long hon) |
| `f` | BO quality filter | Loc chat luong breakout (MACD + volume) |
| `g` | Bear regime defense | Chan entry trong bear market |
| `h` | Confirmed signal exit | Yeu cau bearish score du cao moi exit |
| `i` | Trend-carry override | Giu vi the trong uptrend manh |
| `j` | Anti-chop filter | Chan entry trong sideways/choppy market |

## Feature Sets

### leading_v2 (default)

Feature set mac dinh cho tat ca active models. Bao gom:

- **Base leading features:** Price action, volume, moving averages, momentum, trend indicators, volatility, breakout signals
- **Group A:** Market Structure (pivot points, swing highs/lows, break of structure)
- **Group B:** Exhaustion & Failure Signals (upthrust, spring, climax volume, gaps)
- **Group C:** Volatility Regime (ATR percentile, compression, overnight gaps)
- **Group D:** Multi-timeframe (weekly indicators, price vs weekly MA)

### Feature Ablation

Su dung `run_feature_ablation.py` de test tung group:

```bash
python run_feature_ablation.py              # Test tat ca groups
python run_feature_ablation.py --group A    # Chi test Group A
python run_feature_ablation.py --group A,B  # Test Group A va B
```

## Quy uoc & Best Practices

### Dat ten file

- `run_v{N}.py` -- Backtest function + standalone runner cho version N
- `run_v{N}_experiments.py` -- Script experiment cho version N
- `trades_v{N}.csv` -- Output trades CSV
- `trades_v{N}.meta.json` -- Metadata cho trades CSV

### Quy trinh phat trien

1. **KHONG copy toan bo file** khi tao version moi
2. Chi viet ham `backtest_vXX()` voi logic KHAC BIET
3. Dang ky trong `config/models.yaml`
4. Dang ky trong `run_pipeline.py` `get_backtest_function()`
5. Dung `run_pipeline.py` de chay thong nhat
6. Dung `model_manager.py compare` de quyet dinh retire

### Metrics quan trong

| Metric | Y nghia | Muc tieu |
|--------|---------|----------|
| **Composite Score** | Diem tong hop (normalized, weighted) | Cang cao cang tot |
| **Total PnL** | Tong loi nhuan % | Cang cao cang tot |
| **Win Rate** | % trades thang | > 40% |
| **Profit Factor** | Gross profit / Gross loss | > 2.0 |
| **Max Loss** | Trade thua lon nhat | > -20% |
| **Avg Hold** | So ngay giu trung binh | 15-25 ngay |
| **Max Drawdown** | Drawdown lon nhat | Cang nho cang tot |

### Khi nao nen retire model?

- Composite Score thap hon dang ke so voi model moi
- Total PnL am hoac thap hon dang ke
- Win Rate < 35%
- Profit Factor < 1.5
- Max Loss > -30% (qua rui ro)
- Da co version moi supersede hoan toan

---

*Cap nhat lan cuoi: 2026-04-22*
