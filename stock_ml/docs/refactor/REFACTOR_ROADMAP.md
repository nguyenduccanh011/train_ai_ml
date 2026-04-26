# Refactor Roadmap — chi tiết từng bước

## Overview

8 giai đoạn, ~8 tuần làm full-time. Mỗi giai đoạn:
- **Goal**: kết quả cụ thể đo được
- **Cần xem xét trước**: câu hỏi phải trả lời TRƯỚC khi code
- **Cần biết trước**: kiến thức/file cần đọc TRƯỚC khi bắt đầu
- **Steps**: các bước thực hiện theo thứ tự
- **Verification**: cách verify "DONE"
- **Risk**: vấn đề có thể gặp + cách phòng

---

## Phase 0 — Chuẩn bị (Tuần 1)

### Goal
- Có golden baseline cho 11 champion versions
- Có ARCHITECTURE.md được lock-in (không sửa nữa trong phase 1+)
- Setup tooling: ruff, mypy, pytest, pre-commit
- Branch strategy đã rõ

### Phase 0.1 — Lock random seeds (1 ngày) ✅ DONE 2026-04-26

**Cần biết trước**:
- File `src/models/registry.py` có hàm `build_model()` cho mỗi loại
- LightGBM dùng `random_state` param
- XGBoost dùng `random_state` param
- GRU (PyTorch) cần `torch.manual_seed()` + `torch.backends.cudnn.deterministic = True`

**Cần xem xét**:
- Có deprecated warning về reproducibility với GPU không?
- LightGBM GPU mode có deterministic không? → check docs, có thể phải force CPU cho GRU

**Steps**:
1. Audit `src/models/registry.py`: tất cả model phải có `random_state=42`
2. Add deterministic flag cho GRU: `torch.use_deterministic_algorithms(True)` 
3. Set `PYTHONHASHSEED=42` ở env
4. Set numpy global seed: `np.random.seed(42)` ở pipeline start

**Verification**:
```bash
# Run twice, kết quả giống nhau
python run_pipeline.py --version v22 --device gpu --force > run1.log
python run_pipeline.py --version v22 --device gpu --force > run2.log
diff results/trades_v22.csv results/trades_v22.csv.bak
# Phải empty
```

**Risk**:
- GPU non-determinism với cuDNN → fallback CPU cho GRU regression test
- Floating point khác giữa runs → tolerance 1e-9 cho metrics, exact match cho integer counts

**Result (2026-04-26)**:
- Edits: [src/models/sequence.py:84-92](../../src/models/sequence.py#L84-L92) (cudnn.deterministic + use_deterministic_algorithms warn_only), [run_pipeline.py:670-694](../../run_pipeline.py#L670-L694) (`_lock_global_seeds(42)` ở đầu main).
- Verify: v22 GPU run 2 lần, hash sha256 trùng `cb7283ef...`, 1784 trades / WR 46.4% / PnL +6843.6%. Diff CSV empty.
- Diary: [diary/2026-04-26.md](diary/2026-04-26.md).

### Phase 0.2 — Tạo golden baseline (1 ngày) ✅ DONE 2026-04-27

**Cần biết trước**:
- 11 champion list đã chốt (xem `CHAMPION_VERSIONS.md`)
- Disk space đủ: ~50MB cho 11 trades CSV
- **CPU mode bắt buộc** (xem Result bên dưới)

**Steps**:
1. Run 11 champions với `--force --device cpu`:
```bash
PYTHONHASHSEED=42 python run_pipeline.py \
  --version v22 \
  --compare v32,v34,v35b,v37a,v37a_exit,v37d,v39d,v42_a,v19_3,rule \
  --device cpu --force --no-export
```

2. Backup vào `tests/regression/golden/`:
```bash
mkdir -p tests/regression/golden
cp results/trades_{v22,v32,v34,v35b,v37a,v37a_exit,v37d,v39d,v42_a,v19_3,rule}.csv tests/regression/golden/
cp results/trades_{v22,...}.meta.json tests/regression/golden/
```

3. Hash:
```bash
cd tests/regression/golden
sha256sum *.csv > checksums.txt
git add -f *.csv *.json checksums.txt README.md
git commit -m "Add golden baseline for 11 champion versions"
```

**Verification**:
- 11 file CSV + 11 file JSON trong `tests/regression/golden/`
- `checksums.txt` có 11 hash
- Re-run pipeline → hash giống

**Risk**:
- File quá lớn (>10MB) → dùng git-lfs
- Dữ liệu thay đổi (ai modify portable_data) → invalidate. Lock data_dir hash trong meta.

**Result (2026-04-27)**:
- GPU Run 1 vs GPU Run 2: 9/11 match, **2 lệch** (v22: 1784→1785 trades, hash `cb7283ef`→`1cb508a4`; v42_a: 1442→1441 trades, hash `c45b2cc5`→`1f1b37b3`).
- Investigation: PYTHONHASHSEED=42 set trước Python KHÔNG fix. Cache + logs identical. CPU Run A vs Run B → **11/11 hash exact match**.
- Root cause: **LightGBM GPU (OpenCL) non-deterministic** giữa invocations Python ([microsoft/LightGBM#2479](https://github.com/microsoft/LightGBM/issues/2479)). Boundary samples ở v22, v42_a bị classify khác giữa 2 runs.
- GRU (v37d) GPU lại reproducible nhờ `cudnn.deterministic=True` Phase 0.1.
- **Decision**: Golden baseline dùng CPU mode. Production runtime GPU vẫn OK cho research, nhưng regression test bắt buộc CPU. Trade-off: training chậm hơn ~20-40%, chấp nhận được.
- Files: [tests/regression/golden/](../../tests/regression/golden/) (11 csv + 11 meta.json + checksums.txt + README.md).
- Diary: [diary/2026-04-27.md](diary/2026-04-27.md).

### Phase 0.3 — Tooling setup (1 ngày) ✅ DONE 2026-04-28

**Cần biết trước**:
- Project hiện không có `pyproject.toml`, có `requirements.txt`
- Không có pre-commit hook

**Steps**:
1. Tạo `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]
ignore = ["E501"]  # tạm bỏ qua line length

[tool.mypy]
python_version = "3.11"
strict = false  # bắt đầu lỏng, sau strict dần
files = ["src/components/", "src/pipeline/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

2. Install dev dependencies:
```bash
pip install ruff mypy pytest pytest-xdist pre-commit pydantic
pip freeze > requirements-dev.txt
```

3. Tạo `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: local
    hooks:
      - id: regression-quick
        name: regression-quick
        entry: pytest tests/regression/test_quick.py
        language: system
        pass_filenames: false
```

4. Install hook:
```bash
pre-commit install
```

5. Tạo regression-quick test (dùng cached predictions, ~1 phút):
```python
# tests/regression/test_quick.py
def test_v22_score_unchanged():
    """Quick smoke test - chỉ run v22 (cached)."""
    ...
```

**Verification**:
- `pre-commit run --all-files` pass
- `pytest tests/regression/test_quick.py` pass
- `mypy src/components/` không error (chưa có file nên 0 errors)

**Result (2026-04-28)**:
- Files: [pyproject.toml](../../../pyproject.toml), [stock_ml/requirements-dev.txt](../../requirements-dev.txt), [.pre-commit-config.yaml](../../../.pre-commit-config.yaml), [tests/regression/test_champions.py](../../tests/regression/test_champions.py).
- Per-file-ignores cho legacy code (legacy.py, engine.py, run_pipeline.py, experiments/, analysis/) — siết sau khi port qua `src/components/` ở Phase 1+.
- Pre-commit 3 hooks: ruff (--fix), ruff-format, regression-champions (hash check 11 trades CSV vs golden, ~1s, KHÔNG full pipeline).
- Ruff auto-fix 479 issues + format 90 files. Regen 11 champions CPU 336.3s → pytest 12/12 pass, hash match 100% — code logic không đổi.
- Pre-commit install: phải `git config --unset-all core.hooksPath` (đang trỏ default path) trước.
- Diary: [diary/2026-04-28.md](diary/2026-04-28.md).

### Phase 0.4 — Branch strategy (30 phút)

**Steps**:
1. Tạo long-running branch: `git checkout -b refactor/v2-clean-arch`
2. Mỗi sub-phase = sub-branch:
```bash
# Working
git checkout -b refactor/phase-0-baseline
# After done
git checkout refactor/v2-clean-arch
git merge --no-ff refactor/phase-0-baseline
```
3. Main branch không touch cho đến khi refactor xong (tag `v2.0`)

**Rules**:
- KHÔNG merge sub-branch vào main
- KHÔNG để sub-branch sống quá 1 tuần
- Mỗi sub-branch phải pass regression trước merge

### Phase 0.5 — Lock ARCHITECTURE.md (30 phút) ✅ DONE 2026-04-28

**Steps**:
1. Đọc lại ARCHITECTURE.md
2. Trả lời 5 open questions ở section 15
3. Update doc với decisions
4. Commit + tag `arch-locked`

**Verification**: file `ARCHITECTURE.md` được commit, không có TBD/TODO ở phần kiến trúc chính.

**Result (2026-04-28)**:
- Section 15 đổi từ open questions sang quyết định lock-in: profile dispatch ở outer orchestrator, per-symbol model defer but supported by design, có ensemble skeleton, giữ dashboard manifest cũ trong transition, tách training/inference cho single-bar mode.
- `CHAMPION_VERSIONS.md` được đồng bộ với `EXIT_MODEL_BUG.md` (v37a_exit/v42_a hiện là `trained-but-dropped`, chưa active Model B).

---

## Phase 1 — Foundation: Components framework (Tuần 2)

### Goal
- Có skeleton `src/components/` với base interfaces
- Có 1-2 implementations đơn giản cho mỗi component type
- Tests pass

### Phase 1.1 — Base interfaces (1-2 ngày)

**Cần biết trước**:
- Python Protocol vs ABC: dùng Protocol cho type hints, ABC khi cần shared logic
- Pydantic v2 syntax đã thay đổi từ v1

**Steps**:
1. Tạo `src/components/base.py` với common types:
```python
@dataclass
class BarContext: ...

@dataclass
class FusionResult: ...

@dataclass
class Position: ...

@dataclass
class Trade: ...

@dataclass
class Action: ...
```

2. Tạo base cho mỗi component type:
```
src/components/features/base.py
src/components/targets/base.py
src/components/models/base.py
src/components/exit_models/base.py
src/components/fusion/base.py
src/components/backtest/base.py
src/components/evaluation/base.py
```

3. Tạo registry pattern shared:
```python
# src/components/registry.py
class ComponentRegistry:
    _instances: dict[str, dict[str, type]] = {}
    
    @classmethod
    def register(cls, category: str, name: str, component_cls: type): ...
    
    @classmethod
    def get(cls, category: str, name: str) -> type: ...
    
    @classmethod
    def list_components(cls, category: str | None = None) -> list[str]: ...
```

**Verification**:
- `mypy src/components/` không error
- Import được không circular

**Risk**:
- Circular imports giữa base classes → giải pháp: forward references + TYPE_CHECKING

### Phase 1.2 — Feature blocks (3-4 ngày)

**Cần biết trước**:
- File `src/features/engine.py` hiện tại implement `leading`, `leading_v2`, `leading_v3`, `leading_v4`
- Audit: feature nào ở mỗi set, depend cái gì

**Cần xem xét**:
- Có feature nào tính ngược nhau (set A có nhưng set B không, ngược lại)?
- HA features ở leading_v4 có depend leading_v3 features không?

**Steps**:

1. Audit feature sets cũ:
```bash
# Liệt kê columns mỗi feature_set produce
python -c "
from src.features.engine import FeatureEngine
import pandas as pd
df = pd.DataFrame({'open':[1,2,3], 'high':[1,2,3], 'low':[1,2,3], 'close':[1,2,3], 'volume':[1,2,3]})
for fs in ['leading','leading_v2','leading_v3','leading_v4']:
    eng = FeatureEngine(feature_set=fs)
    out = eng.compute_for_all_symbols(df.assign(symbol='X', timestamp=pd.date_range('2020',periods=3)))
    print(fs, list(out.columns))
"
```

2. Phân tách features thành blocks:
   - `OhlcvBasicBlock`: returns, SMA20, SMA50, ATR
   - `MomentumBlock`: RSI, MACD, ROC, ret_5, ret_10, ret_20
   - `VolumeBlock`: vol_z, OBV
   - `RegimeBlock`: trend_strength, choppy_index, dist_sma200
   - `HeikinAshiBlock`: ha_*, ha_streak

3. Implement từng block:
```python
# src/components/features/blocks/momentum.py
class MomentumBlock(FeatureBlock):
    name = "momentum"
    requires = ["close", "high", "low"]
    
    def compute(self, df):
        df = df.copy()
        df["ret_5"] = df["close"].pct_change(5)
        df["ret_10"] = df["close"].pct_change(10)
        df["rsi"] = compute_rsi(df["close"])
        # ...
        return df
    
    def get_feature_names(self):
        return ["ret_5", "ret_10", "rsi", "macd", "macd_signal", ...]
```

4. Tạo composer:
```python
# src/components/features/engine.py
class ComposableFeatureEngine:
    def __init__(self, blocks): ...
    def compute(self, df): ...
    def signature(self): ...  # for cache key
```

5. Map old feature_sets → blocks:
```yaml
# config/feature_sets/leading.yaml
blocks: [ohlcv_basic, momentum, regime]

# config/feature_sets/leading_v2.yaml  
blocks: [ohlcv_basic, momentum, volume, regime]

# config/feature_sets/leading_v3.yaml
blocks: [ohlcv_basic, momentum, volume, regime, sma200]

# config/feature_sets/leading_v4.yaml
blocks: [ohlcv_basic, momentum, volume, regime, sma200, heikin_ashi]
```

6. Equivalence test:
```python
# tests/components/test_features_equivalence.py
def test_leading_v4_matches_old():
    df = load_test_data()
    
    old = FeatureEngine(feature_set="leading_v4").compute_for_all_symbols(df)
    
    new = ComposableFeatureEngine.from_yaml("config/feature_sets/leading_v4.yaml").compute(df)
    
    pd.testing.assert_frame_equal(
        old[sorted(old.columns)],
        new[sorted(new.columns)],
        check_exact=True,
    )
```

**Verification**: 4 feature sets test pass

**Risk**:
- Old `engine.py` có hidden state hoặc side effects → khó reproduce
- Feature ordering matter cho LightGBM (column order in X) → giữ thứ tự ổn định

### Phase 1.3 — Targets (2 ngày)

**Cần biết trước**:
- File `src/data/target.py` có 4 target types
- Có hàm `generate_exit_labels` đã exist

**Steps**:
1. Tạo `TargetGenerator` base class
2. Port 4 target types: `TrendRegime`, `EarlyWave`, `EarlyWaveV2`, `EarlyWaveDual`
3. Map flag `supports_exit_labels`
4. Equivalence test với target gen cũ

**Verification**: 4 target types test pass

### Phase 1.4 — Models (1-2 ngày)

**Cần biết trước**:
- File `src/models/registry.py` có `build_model()` cho mỗi loại
- Có pipeline wrapping (StandardScaler + classifier?)

**Steps**:
1. Implement `EntryModel` base
2. Port LightGBM, XGBoost, CatBoost, RandomForest, GRU
3. Hyperparam dataclass
4. Smoke test: fit + predict

**Risk**:
- GRU non-deterministic với GPU → document, có thể skip ở regression

---

## Phase 2 — Fusion stack (Tuần 3-4) — KHÓ NHẤT

### Goal
- Implement đủ 30+ fusion strategies cần cho 11 champion
- Fusion stack composable qua YAML
- Validation rules ở config load

### Phase 2.1 — Audit existing fusion logic (1 ngày)

**Cần biết trước**:
- 60 backtest function đang scattered trong `experiments/`, `src/strategies/`
- Mỗi function có 30-40 flag (v22_*, v26_*, v27_*, ...)

**Steps**:
1. Đọc và categorize toàn bộ flag từ `models.yaml`:
```bash
python -c "
import yaml
cfg = yaml.safe_load(open('config/models.yaml'))
all_flags = set()
for v, c in cfg['models'].items():
    all_flags |= set((c.get('params') or {}).keys())
for f in sorted(all_flags):
    print(f)
" > all_flags.txt
```

2. Phân loại từng flag vào layer:
   - `pre_entry`: skip_choppy, sma200_filter, anti_fomo, ...
   - `entry`: hybrid_entry, rule_ensemble, ...
   - `hold`: trend_persistence_hold, time_decay, ...
   - `exit_override`: hap_preempt, early_loss_cut, ...

3. Output: file `FUSION_STRATEGY_INVENTORY.md` map mỗi flag → strategy class dự kiến.

**Verification**: 100% flags được categorize, không có flag bí ẩn.

### Phase 2.2 — Base fusion interface (1 ngày)

**Cần biết trước**: Section 4.6 của ARCHITECTURE.md.

**Steps**:
1. Tạo `BarContext`, `FusionResult`, `Position` dataclasses
2. Tạo `FusionStrategy` Protocol
3. Tạo `FusionStack`: chain strategies theo layer + priority
4. Unit test với dummy strategy

**Verification**: Test với fake strategy chain qua được.

### Phase 2.3 — Implement strategies cho v22 (2-3 ngày)

**Mục tiêu**: Đủ strategies để v22 chạy.

**v22 cần**:
- `SMA200Filter` (pre_entry)
- `HybridEntry` (entry)
- `MlOnly` (entry default)
- `FastExitLoss` (exit_override)
- `SignalHardCap` (exit_override)
- `TimeDecay` (exit_override/hold)

**Steps**:
1. Implement 6 strategies
2. Mỗi strategy có unit test
3. Tạo experiment YAML cho v22:
```yaml
# config/experiments/champions/v22.yaml
name: V22
components:
  features: leading_v2
  target: trend_regime
  entry_model:
    type: lightgbm
fusion:
  pre_entry:
    - {name: sma200_filter}
    - {name: hybrid_entry}
  entry:
    - {name: ml_only}
  exit_override:
    - {name: fast_exit_loss, params: {...}, priority: 1}
    - {name: signal_hard_cap, params: {...}, priority: 2}
    - {name: time_decay, params: {...}, priority: 3}
```

4. Run v22 qua pipeline mới
5. Compare với golden baseline

**Verification**: `tests/regression/test_v22.py` pass với golden CSV.

**Risk**:
- Logic exit phức tạp, dễ off-by-one error
- Fusion order matter, sai order = sai kết quả

### Phase 2.4 — Implement strategies cho remaining champions (4-5 ngày)

Theo thứ tự tăng dần độ khó:

1. **v19_3** — đơn giản, ít flag (1 ngày)
2. **v34** — leading_v4 + early_wave + nhiều flag v26/v27/v28 (1 ngày)
3. **v35b** — engine reform, single_bar_signal, rule_override (1 ngày)
4. **v32** — full v26-v32 stack (1 ngày)
5. **v37a** — per-profile dispatch (1 ngày)
6. **v39d** — per-symbol rule routing (1/2 ngày)
7. **v37a_exit, v42_a** — exit_model thật sự active (1 ngày)
8. **v37d** — GRU model (1/2 ngày)
9. **rule** — non-ML baseline (1/2 ngày)

**Sau mỗi version**:
- Run regression
- Commit
- Update FUSION_STRATEGY_INVENTORY.md

**Risk control**: Nếu 1 version stuck > 2 ngày → skip, đánh dấu, move on. Quay lại sau.

---

## Phase 3 — Pipeline orchestrator (Tuần 5)

### Goal
- Pipeline mới chạy được end-to-end
- Cache hệ thống mới
- CLI mới

### Phase 3.1 — Orchestrator core (2 ngày)

**Cần biết trước**:
- `run_pipeline.py` hiện 1025 dòng — đọc kỹ flow
- Hiện grouping logic ở [run_pipeline.py:907-927](stock_ml/run_pipeline.py#L907-L927)

**Steps**:
1. Tạo `src/pipeline/orchestrator.py`:
```python
class Pipeline:
    def __init__(self, experiment: ExperimentConfig): ...
    
    def run(self) -> PipelineResult:
        symbols = self._resolve_symbols()
        df = self._load_data(symbols)
        
        for fold_idx, (train_df, test_df) in self._walk_forward(df):
            entry_model = self._train_entry(train_df, fold_idx)
            exit_model = self._train_exit(train_df, fold_idx) if has_exit else None
            
            for symbol in test_df["symbol"].unique():
                trades = self._run_symbol(symbol, entry_model, exit_model, test_df)
                ...
        
        return PipelineResult(...)
```

2. Tạo `src/pipeline/walker.py` (walk-forward split)
3. Tạo `src/pipeline/trainer.py` (train Model A + B)

**Verification**: Run v22 qua orchestrator mới, kết quả match golden.

### Phase 3.2 — Caching layer (2 ngày)

**Cần biết trước**:
- Cache hiện ở `src/cache/feature_cache.py`
- Format: parquet + JSON metadata

**Steps**:
1. Tạo `src/pipeline/cache.py` với 4 cấp cache:
   - Data cache (rare invalidation)
   - Feature cache (per feature_blocks signature)
   - Target cache (per target signature)
   - Prediction cache (per model + fold)

2. Cache key = hash của:
```python
def cache_key(component) -> str:
    return hashlib.sha256(
        f"{component.name}:{json.dumps(component.config, sort_keys=True)}".encode()
    ).hexdigest()[:16]
```

3. Atomic write: tmp → rename
4. Cache invalidation rules

**Verification**: Run v22 lần 2 → cache hit (nhanh hơn 5x)

### Phase 3.3 — Matrix expander (1 ngày)

**Steps**:
1. Tạo `src/pipeline/matrix_expander.py`:
```python
def expand_matrix(matrix_yaml: Path) -> list[ExperimentConfig]:
    """Expand axes into individual experiments."""
```

2. Test với matrix nhỏ (2 axes × 2 values = 4 experiments)
3. Verify share cache đúng (3 experiments cùng feature_set → 1 lần compute)

**Verification**: Matrix 2×2×2 = 8 experiments share 1 feature compute.

### Phase 3.4 — CLI (1 ngày)

**Steps**:
1. Tạo `scripts/cli.py` với subcommands (dùng `argparse` hoặc `click`):
   - `run`
   - `run-matrix`
   - `validate`
   - `list-components`
   - `list-experiments`
   - `export`
   - `compare`
   - `migrate-legacy`

2. Tạo `__main__.py`:
```python
# stock_ml/__main__.py
from scripts.cli import main
main()
```

3. Test mỗi subcommand

**Verification**: 
```bash
python -m stock_ml run champions/v22  # works
python -m stock_ml validate champions/v22  # works
python -m stock_ml list-components --type fusion  # lists strategies
```

### Phase 3.5 — Validation rules (1 ngày)

**Steps**:
1. Pydantic schema cho experiment config
2. Validation rules:
   - Component exists in registry
   - Exit model + fusion compatibility
   - Target supports exit labels nếu cần
   - Hyperparams hợp lệ
3. Validate at load time, not run time

**Verification**: Sai config → error rõ ràng, không crash sau 5 phút.

---

## Phase 4 — Migration & legacy adapter (Tuần 6)

### Goal
- 11 champion ported, all pass regression
- 49 legacy versions chạy qua adapter
- Migration tool functional

### Phase 4.1 — Legacy adapter (2-3 ngày)

**Cần biết trước**:
- 49 versions có functions ở `experiments/run_vXX.py` và `src/strategies/legacy.py`
- Strategy map ở `run_pipeline.py:50-112`

**Steps**:
1. Tạo `legacy/` folder:
```
legacy/
├── README.md
├── adapter.py
├── strategies/         # copy từ experiments/, src/strategies/legacy.py
└── configs/            # extract entries từ models.yaml
```

2. Tạo `LegacyVersionAdapter`:
```python
class LegacyVersionAdapter(BacktestStrategy):
    """Wraps old backtest_vXX function to fit new pipeline."""
    
    def __init__(self, version_key: str):
        self.version_key = version_key
        self.old_fn = self._import_old_function(version_key)
        self.old_config = self._load_legacy_config(version_key)
    
    def run(self, prediction_cache, ...):
        # Translate new state → old args
        # Call old function
        # Translate result back
```

3. Test: 5 random legacy versions chạy qua adapter, score gần với historical

**Verification**:
```bash
python -m stock_ml run legacy/v25  # works
python -m stock_ml run legacy/v25 --compare champions/v22  # compare
```

**Risk**:
- Old config có flags không map được → adapter raise warning
- Old function expects old DataFrame format → adapter converts

### Phase 4.2 — Migration tool (1-2 ngày)

**Steps**:
1. Tạo `scripts/migrate_legacy.py`:
```python
def migrate(version_key: str) -> NewExperimentConfig:
    """Convert legacy YAML entry → new schema."""
    old = load_legacy_config(version_key)
    return NewExperimentConfig(
        name=old["name"],
        components=resolve_components(old),
        fusion=translate_flags_to_fusion(old["params"]),
        ...
    )
```

2. Translation table: flag → fusion strategy
3. Test với 5 versions: migrate → run → compare với legacy adapter

**Verification**:
```bash
python -m stock_ml migrate-legacy v25
# Tạo config/experiments/champions/v25.yaml
python -m stock_ml run champions/v25
# Result gần match với legacy v25
```

### Phase 4.3 — Cleanup old pipeline (1 ngày)

**Steps**:
1. Đảm bảo không có code nào còn reference old pipeline
2. Mark `run_pipeline.py` deprecated:
```python
# run_pipeline.py
import warnings
warnings.warn("Deprecated, use 'python -m stock_ml run'", DeprecationWarning)
# ... vẫn forward sang pipeline mới qua adapter
```
3. Update README

**Verification**: `git grep -i "from run_pipeline import"` returns 0 (no internal usages).

---

## Phase 5 — Testing & verification (Tuần 7)

### Goal
- Full regression test pass
- Performance benchmark
- Coverage report

### Phase 5.1 — Full regression (2 ngày)

**Steps**:
1. Tests cho 11 champions:
```python
# tests/regression/test_champions.py
@pytest.mark.parametrize("champion", CHAMPIONS)
def test_champion_matches_golden(champion):
    result = run_pipeline(f"champions/{champion}")
    golden = load_golden(champion)
    assert_csv_equal(result.trades_csv, golden, exact=True)
```

2. Smoke tests cho 10 random legacy versions
3. Property-based tests cho fusion stack

**Verification**: 100% pass.

### Phase 5.2 — Performance benchmark (1 ngày)

**Steps**:
1. Benchmark từng champion: time + memory
2. Compare với baseline (pre-refactor):
```
Version       Old (s)    New (s)    Diff
v22           45.2       42.1       -7%
v34           48.5       46.0       -5%
v37a          52.0       50.5       -3%
...
```

3. Mục tiêu: <20% slower. Nếu slower hơn → profile.

**Verification**: Benchmark report committed.

### Phase 5.3 — Documentation (2 ngày)

**Steps**:
1. Update `README.md` với kiến trúc mới
2. Viết user guides:
   - `HOW_TO_ADD_FEATURE_BLOCK.md`
   - `HOW_TO_ADD_FUSION_STRATEGY.md`
   - `HOW_TO_ADD_ENTRY_MODEL.md`
   - `HOW_TO_PORT_LEGACY_VERSION.md`
   - `HOW_TO_RUN_MATRIX.md`
3. Update CLAUDE.md / .clinerules với conventions mới

**Verification**: Doc reviewable, có ví dụ runnable.

---

## Phase 6 — Tooling enhancements (Tuần 8)

### Goal
- Dashboard tương thích với pipeline mới
- CI/CD setup
- Pre-commit hooks complete

### Phase 6.1 — Dashboard compatibility (1-2 ngày)

**Cần biết trước**:
- `visualization/manifest.json` format hiện tại
- `visualization/data_*/` structure

**Steps**:
1. Pipeline mới sinh manifest format giống cũ (transition period)
2. Sau khi stable → update dashboard schema nếu cần

**Verification**: Dashboard mở được, hiển thị 11 champions + legacy.

### Phase 6.2 — CI/CD (1 ngày)

**Steps**:
1. GitHub Actions workflow (nếu có):
   - Lint (ruff)
   - Type check (mypy)
   - Test (pytest, quick subset)
2. Pre-commit hook đầy đủ
3. Release tagging

### Phase 6.3 — Final cleanup (1 ngày)

**Steps**:
1. Xóa code chết
2. Dọn `archive/` (xem CLEANUP_PLAN.md)
3. Tag `v2.0`
4. Merge `refactor/v2-clean-arch` vào main

---

## Phase 7+ — Tận dụng kiến trúc mới (vô thời hạn)

Sau refactor xong, mỗi research iteration:

### Thêm 1 entry model mới (~30 phút)
```bash
# 1. Implement
src/components/models/transformer.py

# 2. Register
src/components/models/registry.py

# 3. Use in YAML
config/experiments/exp_transformer_v1.yaml
```

### Thêm 1 feature block mới (~1 giờ)
```bash
# 1. Implement
src/components/features/blocks/sentiment.py

# 2. Compose into feature_set
config/feature_sets/leading_v5_sentiment.yaml

# 3. Use in experiment
config/experiments/champions/v50_sentiment.yaml
```

### Run grid search (5 phút setup, time-limited only by GPU)
```bash
# Define matrix
config/experiments/matrix/q3_2026.yaml

# Run
python -m stock_ml run-matrix matrix/q3_2026 --parallel 4

# Compare
python -m stock_ml compare matrix/q3_2026/* --top 10
```

---

## Tóm tắt timeline

| Tuần | Phase | Output chính |
|------|-------|--------------|
| 1 | Phase 0 | Golden baseline + tooling locked |
| 2 | Phase 1 | Component framework + features + targets + models |
| 3 | Phase 2.1-2.3 | Fusion audit + v22 ported |
| 4 | Phase 2.4 | Còn 10 champions ported |
| 5 | Phase 3 | Pipeline orchestrator + CLI |
| 6 | Phase 4 | Legacy adapter + migration tool |
| 7 | Phase 5 | Tests + benchmark + docs |
| 8 | Phase 6 | CI/CD + cleanup + tag v2.0 |

**Tổng: 8 tuần** (full-time, solo).

## Solo dev discipline rules

1. **Không skip golden baseline** — không có nó thì refactor mù
2. **Mỗi sub-branch ≤ 1 tuần** — nếu lâu hơn → chia nhỏ
3. **Pre-commit hook bắt buộc** — không bypass
4. **Diary daily** — 5 dòng mỗi ngày: done/stuck/next
5. **Regression sau mỗi sub-phase** — không tích lũy debt
6. **Stop khi confused > 2 ngày** — quay lại đọc ARCHITECTURE.md, không guess

## Khi nào pause refactor

Refactor nên pause nếu:
- Stuck 1 phase > 1 tuần
- Phát hiện kiến trúc có flaw fundamental → revise ARCHITECTURE.md
- Có incident production cần fix
- Burnout (solo dev critical)

Không nên pause nếu:
- "Cảm giác" sai → trust the plan
- Một bug nhỏ → fix in-place, move on

## Success criteria toàn bộ refactor

✅ 11 champion versions chạy qua pipeline mới, kết quả match golden 100%  
✅ 49 legacy versions chạy được qua adapter (không cần exact match)  
✅ Thêm 1 component mới (model/feature/fusion) ≤ 1 giờ  
✅ Grid search 50 experiments setup ≤ 5 phút  
✅ Codebase pass `ruff check` + `mypy --strict src/components/`  
✅ Documentation đầy đủ + runnable examples  
✅ Performance regression < 20%
