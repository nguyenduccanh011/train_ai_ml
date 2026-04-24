# Stock ML Trading System

## Tong quan

He thong ML du doan xu huong gia co phieu Viet Nam (du lieu 2015-2025) su dung LightGBM voi walk-forward validation. He thong ho tro quan ly nhieu phien ban model, backtest tu dong, va dashboard so sanh truc quan.

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
+-- src/                          # Core ML pipeline (shared code â€” khong co runner scripts o day)
|   +-- env.py                    # Auto-detect local/Colab, resolve paths
|   +-- config_loader.py          # Doc models.yaml, cung cap helper functions
|   +-- safe_io.py                # Fix UnicodeEncodeError tren Windows console
|   +-- signal_adapter.py         # Canonicalize model predictions -> {-1,0,1}
|   +-- experiment_runner.py      # run_test() + run_rule_test() -- dung chung cho moi run_vXX
|   +-- cache/
|   |   +-- feature_cache.py      # FeatureCacheManager -- cache feature DataFrame theo feature_set/signature
|   +-- data/
|   |   +-- loader.py             # DataLoader -- load OHLCV tu CSV (Hive-partitioned)
|   |   +-- splitter.py           # WalkForwardSplitter -- rolling 4yr train / 1yr test
|   |   +-- target.py             # TargetGenerator -- 3-class trend regime (dual MA)
|   +-- features/
|   |   +-- engine.py             # FeatureEngine -- leading/leading_v2 feature sets
|   +-- models/
|   |   +-- registry.py           # ModelRegistry -- LightGBM, XGBoost, RandomForest
|   +-- backtest/
|   |   +-- engine.py             # backtest_unified() -- engine chinh dung chung cho v22-v27
|   |   +-- defaults.py           # DEFAULT_PARAMS, FEATURE_DEFAULTS, symbol configs (doc tu models.yaml)
|   |   +-- indicators.py         # compute_indicators(), detect_trend_strength(), get_regime_adapter()
|   +-- evaluation/
|   |   +-- scoring.py            # composite_score() + calc_metrics() -- single source of truth
|   |   +-- metrics.py            # Classification metrics (F1, accuracy)
|   +-- strategies/
|   |   +-- legacy.py             # backtest_v19_1/v19_3 -- dung boi src/ luc train, khong dung truc tiep
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
+-- run_pipeline.py               # Pipeline runner thong nhat (shared ML cache + matrix mode)
+-- colab_setup.py                # 1-click setup cho Google Colab
+-- compare_rule_vs_model.py      # Rule-based backtest (backtest_rule)
|
+-- experiments/
|   +-- run_v27.py                # V27: backtest_v27() + standalone runner
|   +-- run_v26.py                # V26: backtest_v26() + runner
|   +-- run_v26_experiments.py    # V26: ablation cac patch A-H (dung src/experiment_runner)
|   +-- run_v25.py                # V25: backtest_v25() + ablation study
|   +-- run_v24.py                # V24: backtest_v24() + runner
|   +-- run_v23_optimal.py        # V23: backtest_v23()
|   +-- run_v22_final.py          # V22: backtest_v22()
|   +-- run_feature_ablation.py   # Feature group ablation experiment (Groups A/B/C/D/E/F)
|
+-- archive/                      # Code khong con active -- giu de tham khao lich su
|   +-- scripts/                  # run_v2 -> run_v22_compare, run_v26_feature_compare, run_v27_experiments
|   +-- src/                      # pipeline.py + evaluation/ (da duoc thay boi src/ chinh)
|   +-- analysis/                 # Deep analysis scripts (v10-v23)
|   +-- exports/                  # Export scripts (v10-v23)
|   +-- misc/                     # Visualization + cleanup scripts
|
+-- docs/                         # Tai lieu bo sung
+-- requirements.txt              # Python dependencies
```

## Pipeline ML

### Luong xu ly chinh

```
+-------------+    +--------------+    +------------------+    +--------------+
|  Data Load   |--->|  Feature Eng |--->|  Target Generate |--->|  Train Model |
|  (loader.py) |    |  (engine.py) |    |  entry_wave      |    |  Model A     |
+-------------+    +--------------+    |  exit_signal (*) |    |  (entry)     |
                                        +------------------+    |  Model B (*) |
                                                                |  (exit)      |
                                                                +------+-------+
                                                                       |
                                                                       v
+-------------+    +--------------+    +--------------+    +------------------+
|  Dashboard   |<---|  manifest.json|<---|  Unified Export|<---| Backtest Engine  |
|  (HTML+JS)   |    |  (auto-gen)  |    |  (export.py)  |    | y_pred_entry     |
+-------------+    +--------------+    +--------------+    | y_pred_exit (*)  |
                                                            +------------------+

(*) Exit model — chi active neu model co exit_model.enabled=true trong models.yaml
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

- **Model A (entry):** LightGBM, 3 classes — early_wave BUY / NEUTRAL / AVOID
- **Model B (exit):** LightGBM, binary — EXIT_NOW / HOLD (chi active khi exit_model.enabled=true)
- **Target entry:** early_wave (sideways N ngay + forward gain >= threshold)
- **Target exit:** forward drawdown >= loss_threshold trong N ngay (doc lap voi entry target)
- **Target config:** moi version tu khai bao trong `config/models.yaml` -> `models.<version>.target`
- **Exit config:** `config/models.yaml` -> `models.<version>.exit_model` (forward_window, loss_threshold)
- **Features:** leading_v4 (V34+) — leading_v3 (V29-V33) — leading_v2 (V22-V28)
- **Symbols:** Lay tu `config/models.yaml` (`pipeline.symbols`)

### Dual Model Architecture (Entry + Exit)

```
Train time (per fold):
  X_train = features(train_period)
  Model A.fit(X_train, y_entry)   <- early_wave label: BUY/NEUTRAL/AVOID
  Model B.fit(X_train, y_exit)    <- exit label: EXIT(1)/HOLD(0)  [neu enabled]

Backtest time (per bar):
  y_pred_entry[i] = Model A.predict(features[i])
  y_pred_exit[i]  = Model B.predict(features[i])   [neu enabled, else None]

  Engine logic:
    if not in_position and y_pred_entry[i] == 1: -> entry
    if in_position:
      if hard_stop:           -> exit (highest priority)
      elif y_pred_exit[i]==1  -> exit "model_b_exit"  [override engine rules]
      elif trailing_stop:     -> exit
      elif ...other rules...  -> exit
```

### Backtest Engine

Moi version co mot ham `backtest_vXX()` rieng voi logic entry/exit khac nhau:

| Component | Mo ta |
|-----------|-------|
| **Entry Logic** | ML signal (Model A) + breakout + V-shape + rule ensemble |
| **Exit Logic** | Model B exit (override) > Hard stop > ATR stop > trailing > peak protect > zombie > signal |
| **Model B Exit** | `model_b_exit` — active sau min 3 ngay giu, chi bi block boi hard_stop |
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

# Matrix mode: test nhieu feature set x nhieu ML model
python run_pipeline.py --versions v26,v27 --feature-sets leading,leading_v2 --ml-models lightgbm,xgboost

# Matrix mode + compare list (auto append vao versions)
python run_pipeline.py --version v27 --compare v26 --feature-sets leading_v2 --ml-models lightgbm
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

**Metadata validation:** Khi smart cache tai su dung CSV cu, he thong kiem tra `.meta.json` de dam bao CSV duoc tao voi cung dieu kien (symbols, min_rows, feature_set, target_fingerprint). Neu khac biet, hien canh bao:

```
  WARNING: 2 cached CSV(s) have mismatched conditions:
    v24: symbol list differs (14 changes: cached=14, current=367)
    Mismatched versions se tu dong bi cache invalidation va duoc re-run de dam bao fairness.
```

> **Luu y:** Export va Compare luon bao gom TAT CA versions (ca cached lan moi chay).

### Feature Cache (run_vXX / run_test)

`src/experiment_runner.py` co cache rieng cho buoc tao feature (qua `FeatureCacheManager`) de tang toc cac script `experiments/run_v24.py`, `run_v25.py`, `run_v26.py`, `run_v26_experiments.py`, `run_v27.py`.

- Thu muc cache: `results/cache/features/<feature_set>/`
- Cache key duoc tao tu: `symbols + target_config + source_fingerprint(data) + code_fingerprint + schema_version`
- Ho tro nhieu feature set (`leading`, `leading_v2`, va cac set moi sau nay) ma khong xung dot cache
- Khi data hoac code thay doi, key thay doi -> cache miss tu nhien (khong can xoa tay)

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
    mode: explicit      # auto | explicit
    min_rows: 2000      # Chi lay symbols co >= 2000 dong du lieu
    explicit_list: "ACB,FPT,..."  # Danh sach cu the (chi dung khi mode=explicit)
```

Khi `mode: explicit`, he thong dung danh sach co san trong config. Neu chuyen sang `mode: auto`, he thong se scan thu muc du lieu va chon tat ca symbols du dieu kien `min_rows`. Tat ca models trong cung 1 pipeline run se dung CUNG 1 symbol list.

### Symbol Profiles

Phan loai symbol (bank/high_beta/momentum/defensive) duoc cau hinh trong `config/models.yaml`:

```yaml
symbol_profiles:
  bank: [ACB, BID, MBB, TCB]
  high_beta: [AAV, AAS, SSI, VND]
  momentum: [DGC, HPG, VIC]
  defensive: [FPT, REE, VNM]

rule_priority_symbols: [AAA, SSN, TEG, GAS, PLX, IJC, DQC]
score5_risky_symbols: [AAA, IJC, ITC, VHM, TEG, QBS, KMR, SSN, PLX]
```

Backtest engine (`src/backtest/engine.py`) tu dong doc cac config nay qua `src/backtest/defaults.py`. Them/sua symbol chi can chinh `config/models.yaml`, khong can sua code.

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

### Buoc 1: Dang ky model trong models.yaml

```yaml
# config/models.yaml
models:
  v43:
    name: V43
    description: 'Mo ta ngan gon'
    color: '#FF5722'
    active: true
    strategy: v43
    feature_set: leading_v4
    target:
      type: early_wave
      forward_window: 8
      gain_threshold: 0.06
      loss_threshold: 0.03
      classes: 3
    exit_model:           # Bo qua neu khong muon exit model
      enabled: true
      forward_window: 15
      loss_threshold: 0.05
    mods:
      a: true
      b: true
      # ... c-j
    marker_shape: arrowUp
    order: -52
```

Hoac dung CLI:
```bash
python model_manager.py add v43 --name "V43" --color "#FF5722" --strategy v43
```

### Buoc 2: Viet backtest function

Tao `experiments/run_v43.py`. Backtest function chi can nhan `**kwargs` de tu dong nhan `y_pred_exit`:

```python
# experiments/run_v43.py
from experiments.run_v37a import backtest_v37a

def backtest_v43(y_pred, returns, df_test, feature_cols, **kwargs):
    # Them logic rieng neu can, roi delegate xuong engine
    # y_pred_exit duoc tu dong truyen qua **kwargs -> backtest_unified
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **kwargs)
```

### Buoc 3: Dang ky trong run_pipeline.py

```python
strategy_map = {
    "v43": ("experiments.run_v43", "backtest_v43"),
    ...
}
```

### Buoc 4: Chay backtest

```bash
# Chay V43 (tu dong dung exit model neu exit_model.enabled=true trong yaml)
python run_pipeline.py --version v43 --compare v37a,rule
```

Output: `results/trades_v43.csv` + `.meta.json` + `visualization/data_v43/` + `manifest.json`

### Buoc 5: Xem ket qua

Mo `visualization/dashboard.html` trong browser -> V43 tu xuat hien tren dashboard!

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

## Google Colab â€” Hybrid Workflow

He thong ho tro chay tren ca **local** (RTX 3060) va **Google Colab** (T4/A100). Code tu dong detect moi truong va chuyen path phu hop.

### Kien truc

```
+----------------------------------------------------------+
|               GitHub (source of truth)                    |
|   stock_ml/src/*, config/*, run_*.py, requirements.txt   |
|   CHI CODE â€” khong data, khong model, khong results      |
+---------------------------+------------------------------+
                            |
                +-----------+-----------+
                v                       v
+---------------------+     +---------------------------+
|   Local (RTX 3060)  |     |   Colab (T4/A100)         |
|                     |     |                           |
|   git pull          |     |   git clone/pull          |
|   data: local disk  |     |   data: Mount Drive       |
|   dev & debug       |     |   train nang              |
|   train nho         |     |   batch backtest all      |
+--------+------------+     +----------+----------------+
         |                              |
         v                              v
+----------------------------------------------------------+
|            Google Drive (shared storage)                   |
|   stock_ml_hub/portable_data/ + results/ + models/       |
+----------------------------------------------------------+
```

### Khi nao dung Local, khi nao dung Colab?

| Task | Local (3060) | Colab |
|------|-------------|-------|
| Dev/debug strategy moi | Chay 1-2 symbol nhanh | |
| Train 1 model (v27) | OK neu LightGBM/XGBoost | Neu can deep learning |
| **Chay ALL models so sanh** | | **Colab** â€” `--all` |
| **Feature ablation toan bo** | | **Colab** |
| **Batch backtest 200+ symbols x 7 models** | | **Colab** |
| Export JSON cho dashboard | Local (nhe) | |

### Buoc 1: Upload data len Google Drive (1 lan duy nhat)

Tren may local, nen data:
```bash
cd portable_data
tar -czf vn_stock_ai_dataset_cleaned.tar.gz vn_stock_ai_dataset_cleaned/
```

Upload file `.tar.gz` len Google Drive:
- Tao thu muc: `MyDrive/stock_ml_hub/portable_data/`
- Upload `vn_stock_ai_dataset_cleaned.tar.gz` vao do

### Buoc 2: Lan dau tren Colab â€” Setup

Mo Colab, tao notebook moi, paste 3 cell:

**Cell 1 â€” Clone repo:**
```python
!git clone https://github.com/nguyenduccanh011/train_ai_ml.git /content/repo
%cd /content/repo/stock_ml
```

**Cell 2 â€” Setup (mount Drive, install deps, check data):**
```python
%run colab_setup.py
```

**Cell 3 â€” Giai nen data (chi lan dau):**
```python
!cd /content/drive/MyDrive/stock_ml_hub/portable_data && tar -xzf vn_stock_ai_dataset_cleaned.tar.gz
```

### Buoc 3: Chay pipeline tren Colab

```python
# Chay tat ca active models
!python run_pipeline.py --all

# Chay model cu the
!python run_pipeline.py --version v27

# So sanh models
!python run_pipeline.py --version v27 --compare v26,v25,v24,rule

# Feature ablation
!python experiments/run_feature_ablation.py
```

Results tu dong luu vao `MyDrive/stock_ml_hub/results/` tren Drive.

### Buoc 4: Lay results ve local

Vao Drive > `stock_ml_hub/results/` > tai cac file CSV can thiet ve `stock_ml/results/` tren local.
Hoac cai Google Drive for Desktop de tu sync.

### Lan sau dung Colab (khong can upload data lai)

Chi can 2 cell:

```python
# Cell 1 â€” Pull latest code + setup
!git clone https://github.com/nguyenduccanh011/train_ai_ml.git /content/repo 2>/dev/null || git -C /content/repo pull --ff-only
%cd /content/repo/stock_ml
%run colab_setup.py

# Cell 2 â€” Run
!python run_pipeline.py --all
```

Data da nam san tren Drive, mount la dung ngay.

### Auto-detect moi truong

Code tu dong detect local/Colab thong qua `src/env.py`:

```python
from src.env import is_colab, resolve_data_dir, get_results_dir

is_colab()         # True tren Colab, False tren local
resolve_data_dir() # Tra ve Drive path tren Colab, local path tren may
get_results_dir()  # Tra ve Drive results/ tren Colab, local results/ tren may
```

Override bang environment variable neu can:
```bash
export STOCK_DATA_DIR="/custom/path/to/data"
export STOCK_RESULTS_DIR="/custom/path/to/results"
```

### Cau truc Google Drive

```
MyDrive/stock_ml_hub/
+-- portable_data/
|   +-- vn_stock_ai_dataset_cleaned/     # Data OHLCV (upload 1 lan)
|   +-- vn_stock_ai_dataset_cleaned.tar.gz  # File nen goc
+-- results/                              # Backtest results (tu Colab)
|   +-- trades_v27.csv
|   +-- trades_v27.meta.json
|   +-- ...
+-- models/                               # Model weights (neu can luu)
    +-- lightgbm_fold1.pkl
    +-- ...
```

### MCP Colab (tuong tac tu local)

Du an co cau hinh MCP Colab (`colab-mcp`) cho phep Claude Code tuong tac truc tiep voi Colab notebook tu local. Cau hinh trong `.claude/settings.json`:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"]
    }
  }
}
```

**Luu y:** Khi dung MCP Colab, can mo notebook tren trinh duyet truoc, sau do goi `open_colab_browser_connection` de ket noi. Sau khi ket noi, cac tool execute se available (co the can restart Claude Code session de refresh tool list).

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
  "target_config": {"type": "trend_regime", "trend_method": "dual_ma", "short_window": 5, "long_window": 20, "classes": 3},
  "target_fingerprint": "{\"classes\":3,\"long_window\":20,\"short_window\":5,\"trend_method\":\"dual_ma\",\"type\":\"trend_regime\"}",
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
    -> V28 (early wave target) -> V29 (leading_v3 retrain, +26)
    -> V30 (signal_exit_defer) -> V31 (SHEF + HAP)
    -> V32 (HAP preempt fix, +11) -> V33 (recovery_peak_filter, +4)
    -> V34 (leading_v4 HA features, +32) -> V35/V36 (V35 flags)
    -> V37a (per-profile dispatch) -> V38/V39 (HA exit, HAP reform, rule_confirm)
    -> V37a+ExitModel ★ CURRENT BEST (separate exit model B, score +187 vs V37a)
    -> V42 (fw=15 entry + exit model)
```

### Ket qua so sanh hien tai (61 symbols, 2020-2025)

| Version | Trades | WR | AvgPnL | TotPnL | PF | MaxLoss | AvgHold | Score |
|---------|--------|----|--------|--------|-----|---------|---------|-------|
| **V37a+ExitModel** ★ | 3694 | 57.4% | +3.35% | +12386% | 3.18 | -23.9% | 8.5d | **597** |
| V42a+ExitModel | 4087 | 56.0% | +2.73% | +11142% | 2.76 | -23.9% | 7.2d | 550 |
| V37a | 1355 | 49.0% | +6.19% | +8392% | 2.84 | -57.4% | 35.5d | ~410 |
| V42_base | 1383 | 48.6% | +5.95% | +8230% | 2.75 | -57.4% | 34.4d | 401 |
| V34 | 1299 | 50.2% | +6.35% | +8243% | 2.89 | — | — | 465 |
| Rule | 2185 | 39.8% | +2.37% | +5178% | 1.84 | -27.2% | 18.1d | — |

**Key insight:** Exit model B (+ExitModel) tang WR +8pp, giam MaxLoss 58%, tang TotalPnL 47%
bang cach thoat som o dinh song thay vi giu den trailing stop.

### Active Models (hien tai)

| Version | Mo ta | Feature Set | Exit Model |
|---------|-------|-------------|------------|
| **V37a+ExitModel** ★ | V37a engine + Model B exit (fw=15, loss=5%) | leading_v4 | enabled |
| **V42a** | V37a engine, fw=15 entry + Model B exit | leading_v4 | enabled |
| **V42_base** | V37a engine, fw=15 (khong exit model, baseline) | leading_v4 | — |
| **V39g** | V37a + HAP reform (8%/15d) + selective rule_confirm | leading_v4 | enabled |
| **V39f** | V37a + HAP (8%/15d) + rule_confirm | leading_v4 | enabled |
| **V39d** | V39e + per-symbol rule-exit hybrid | leading_v4 | enabled |
| **V39e** | V37a + signal_exit_min_hold=35 + HAP reform | leading_v4 | enabled |
| **V39b** | V37a + HAP reform (trigger 8%, min_hold=15) | leading_v4 | enabled |
| **V39a2** | V37a + rule_confirm_exit (selective) | leading_v4 | enabled |
| **V39a** | V37a + signal_exit_min_hold=35 | leading_v4 | enabled |
| **V37a** | V34 engine + per-profile V35 flag dispatch | leading_v4 | enabled |
| **V36a/b/c** | V35 flag variants | leading_v4 | enabled |
| **V35b** | V34 + V35 flags (rule_override + skip_proximity) | leading_v4 | enabled |
| **V34** | V32 engine + leading_v4 (18 HA features) + HAP(t4,f7) | leading_v4 | enabled |
| **V32/V31/V30/V29** | HAP preempt / SHEF / signal_defer / early_wave | leading_v3 | — |
| **Rule** | MACD_hist > 0 AND Close > MA20 AND Close > Open (baseline) | N/A | — |

### Exit Model Config (models.yaml)

```yaml
v39g:
  target:
    type: early_wave        # entry label (buy/neutral/avoid)
    forward_window: 8
    gain_threshold: 0.06
  exit_model:               # exit label (exit/hold) — doc lap
    enabled: true
    forward_window: 15      # nhin truoc 15 ngay (dai hon entry fw=8)
    loss_threshold: 0.05    # exit neu co luc giam >= 5%
```

`exit_model` duoc doc tu `config/models.yaml`. Model khong co block nay chay nhu cu (backward compatible).

### Retired Models

| Version | Ly do |
|---------|-------|
| V19.3 | Superseded by V22/V23 |
| V19.1 | Superseded by V22/V23/V24/V25 |
| V18 | Superseded by V19.1 |
| V25/V23 | Superseded by V26+ |

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
- **Group G (accumulation):** Volatility contraction, range compression, dist-to-highs, volume dry-then-spike, sideway score, BB squeeze

### leading_v4 (V34 — current best)

leading_v3 + **18 Heikin-Ashi wave-position features** (`src/features/engine.py::_heikin_ashi_features`):

| Feature | Y nghia |
|---------|---------|
| `ha_green`, `ha_green_streak`, `ha_red_streak` | Mau va chuoi lien tiep |
| `ha_color_switch` | Doi mau ngay hom nay (tin hieu dao chieu som) |
| `ha_upper_shadow_ratio`, `ha_lower_shadow_ratio` | Ti le rau tren/duoi |
| `ha_no_lower_shadow`, `ha_no_upper_shadow` | Uptrend/downtrend manh (khong rau) |
| `ha_upper_shadow_growing`, `ha_lower_shadow_growing` | Rau tang dan (phan phoi/tich luy) |
| `ha_body_ratio`, `ha_body_shrinking` | Suc manh than nen, da giam dan |
| `ha_streak_position` | Vi tri trong song (0=dau, 1=dinh) |
| `ha_doji` | Doji HA (canh bao dao chieu) |
| `ha_bearish_reversal_signal` | green_streak>=4 + upper_shadow growing + body shrinking |
| `ha_bullish_reversal_signal` | Truoc do do >=3 ngay + doi sang xanh + lower shadow growing |
| `ha_early_wave` | streak<=2 + no_lower_shadow + body>0.5 (mua dau song) |
| `ha_late_wave` | streak>=5 + upper_shadow>0.3 + body_shrinking (tranh fomo) |

**Ly do HA features hieu qua (tu phan tich trades_v32.csv):**
- Losers avg entry_ret_5d=+3.55% → model mua khi da tang (cuoi song)
- `ha_streak_position` va `ha_late_wave` giup model nhan dien "cuoi song/fomo"
- `ha_upper_shadow_growing` + `ha_body_shrinking` → phan phoi som hon MACD/RSI
- Ket hop target gain=6% (vs 5%) → label chat luong cao hon, it nhieu hon

### Feature Ablation

Su dung `experiments/run_feature_ablation.py` de test tung group:

```bash
python experiments/run_feature_ablation.py              # Test tat ca groups
python experiments/run_feature_ablation.py --group A    # Chi test Group A
python experiments/run_feature_ablation.py --group A,B  # Test Group A va B
```

## Quy uoc & Best Practices

### Dat ten file

- `experiments/run_v{N}.py` -- Backtest function `backtest_vN()` + standalone runner cho version N
- `experiments/run_v{N}_experiments.py` -- Script ablation/experiment cho version N
- `trades_v{N}.csv` -- Output trades CSV
- `trades_v{N}.meta.json` -- Metadata cho trades CSV

### Quy tac import (QUAN TRONG)

**Dependency graph hop le:**

```
experiments/run_vXX.py  ->  src/backtest/engine.py       (backtest_unified)
                         ->  src/evaluation/scoring.py   (composite_score, calc_metrics)
                         ->  experiments/run_v{N-1}.py   (chi khi can compare voi version truoc, trong if __name__ == "__main__)

experiments/run_vXX_experiments.py  ->  src/experiment_runner.py   (run_test, run_rule_test)
                                     ->  src/evaluation/scoring.py  (composite_score, calc_metrics)
                                     ->  experiments/run_vXX.py      (backtest_vXX)

run_pipeline.py  ->  src/*  (tat ca src modules)
                 ->  experiments/run_vXX.py (qua dynamic import trong get_backtest_function)

experiments/run_feature_ablation.py  ->  experiments/run_v27.py
                                      ->  src/features/engine.py
```

**KHONG duoc phep:**
- Import bat ky thu gi tu `archive/` trong code dang active
- Import `experiments/run_vXX.py` mot cach tuy tien trong utility scripts khong lien quan pipeline/ablation
- Import `comp_score` tu `run_v25` -- dung truc tiep `from src.evaluation.scoring import composite_score`

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

*Cap nhat lan cuoi: 2026-04-24 - Them Separate Exit Model (Model B): train doc lap tren exit signal, override engine khi predict EXIT. V37a+ExitModel la BEST hien tai (score 597, WR 57.4%, MaxLoss -23.9%). Architecture: `exit_model:` config trong models.yaml, `TargetGenerator.generate_exit_labels()`, `_build_predictions(exit_model_cfg)`, `backtest_unified(y_pred_exit)`. Fix critical bug y_pred_sell/y_pred_exit naming. 14 active models da duoc cap nhat exit_model config.*

