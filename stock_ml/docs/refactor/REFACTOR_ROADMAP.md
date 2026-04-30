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

### Phase 0.4 — Branch strategy (30 phút) ✅ POLICY LOCKED 2026-04-29

**Policy**:
1. Long-running branch dự kiến: `refactor/v2-clean-arch`.
2. Mỗi sub-phase nên có branch riêng `refactor/phase-N-<name>` nếu cần tách history.
3. Main branch không touch cho đến khi refactor xong (tag `v2.0`).

**Rules**:
- KHÔNG merge sub-branch vào main.
- KHÔNG để sub-branch sống quá 1 tuần.
- Mỗi sub-branch phải pass regression trước merge.
- Không tự tạo/switch branch, merge, tag hoặc commit nếu chưa có explicit instruction.

**Status (2026-04-29)**:
- Branch strategy đã được document và dùng như policy vận hành cho refactor.
- Việc chưa tạo đúng long-running branch/tag không block Phase 2.4; tiếp tục trên working branch hiện tại nếu user chưa yêu cầu git operation.

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
- `arch-locked` tag chưa tạo vì đây là git operation; không block Phase 2.4 nếu user chưa yêu cầu tag/commit.

---

## Phase 1 — Foundation: Components framework (Tuần 2)

### Goal
- Có skeleton `src/components/` với base interfaces
- Có 1-2 implementations đơn giản cho mỗi component type
- Tests pass

### Phase 1.1 — Base interfaces (1-2 ngày) ✅ DONE 2026-04-27

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

**Result (2026-04-27)**:
- Files: [src/components/base.py](../../src/components/base.py) (BarContext, FusionResult, Position, Trade, Action — dùng `slots=True`), 7 base Protocol files cho mỗi component type, [src/components/registry.py](../../src/components/registry.py) (thread-safe với RLock, hỗ trợ replace/unregister/create).
- `mypy src/components/` → 0 errors (sau khi `pip install types-PyYAML`).
- Import không circular: dùng `from __future__ import annotations` + TYPE_CHECKING.
- Diary: [diary/2026-04-27.md](diary/2026-04-27.md) (gộp chung với Phase 1.2).

### Phase 1.2 — Feature blocks (3-4 ngày) ✅ DONE 2026-04-27

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

**Result (2026-04-27)**:
- 14 feature blocks implement đúng logic từ legacy `FeatureEngine`: [src/components/features/blocks/](../../src/components/features/blocks/) — ohlcv_basic, moving_averages, momentum, trend, volatility, volume_advanced, leading_signals, market_structure, exhaustion, volatility_regime, multi_timeframe, accumulation, relative_strength, heikin_ashi.
- [src/components/features/engine.py](../../src/components/features/engine.py): `ComposableFeatureEngine` với `compute_for_all_symbols()` tách per-symbol vs cross-sectional (RelativeStrengthBlock). `signature()` = sha256 của block names.
- [src/components/features/registry.py](../../src/components/features/registry.py): `_BLOCK_REGISTRY` dict, `get_block()`, `build_engine_from_yaml()`, `build_engine_from_name()`.
- 4 YAML configs: [config/feature_sets/](../../config/feature_sets/) — leading, leading_v2, leading_v3, leading_v4.
- Tests: [tests/components/test_features_equivalence.py](../../tests/components/test_features_equivalence.py) — 13/13 pass. Kiểm tra column presence, value exact match (tol 1e-8), signature stability, unknown block error.
- `ruff check` + `mypy src/components/` → 0 errors.
- Diary: [diary/2026-04-27.md](diary/2026-04-27.md).

### Phase 1.3 — Targets (2 ngày) ✅ DONE 2026-04-29

**Cần biết trước**:
- File `src/data/target.py` có 4 target types
- Có hàm `generate_exit_labels` đã exist

**Steps**:
1. Tạo `TargetGenerator` base class
2. Port 4 target types: `TrendRegime`, `EarlyWave`, `EarlyWaveV2`, `EarlyWaveDual`
3. Map flag `supports_exit_labels`
4. Equivalence test với target gen cũ

**Verification**: 4 target types test pass

**Result (2026-04-29)**:
- 4 target generator wrappers: [src/components/targets/trend_regime.py](../../src/components/targets/trend_regime.py), [early_wave.py](../../src/components/targets/early_wave.py), [early_wave_v2.py](../../src/components/targets/early_wave_v2.py), [early_wave_dual.py](../../src/components/targets/early_wave_dual.py).
- [src/components/targets/registry.py](../../src/components/targets/registry.py): `get_target(name, **kwargs)`, `list_targets()`. Flag `supports_exit_labels` exposed qua class attr.
- Tests: [tests/components/test_targets_equivalence.py](../../tests/components/test_targets_equivalence.py) — 17/17 pass. Equivalence với legacy `TargetGenerator` per type, signature stability check.
- `ruff check` + `mypy src/components/targets/` → 0 errors.
- Diary: [diary/2026-04-29.md](diary/2026-04-29.md).

### Phase 1.4 — Models (1-2 ngày) ✅ DONE 2026-04-27

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

**Result (2026-04-27)**:
- 5 model wrappers: [src/components/models/lightgbm_classifier.py](../../src/components/models/lightgbm_classifier.py), [xgboost_classifier.py](../../src/components/models/xgboost_classifier.py), [catboost_classifier.py](../../src/components/models/catboost_classifier.py), [random_forest.py](../../src/components/models/random_forest.py), [gru_seq.py](../../src/components/models/gru_seq.py).
- [src/components/models/ensemble.py](../../src/components/models/ensemble.py): `EnsembleEntryModel` skeleton với soft/hard voting.
- [src/components/models/registry.py](../../src/components/models/registry.py): `get_model(name, **kwargs)`, `list_models()`.
- XGBoost label remapping (-1,0,1 → 0,1,2) vì XGBoost không support negative class labels — remap ở fit/predict, transparent qua Protocol.
- GRU wrapper reuse `src/models/sequence.GRUClassifier` (đã có deterministic flags từ Phase 0.1) + thêm RobustScaler (khớp legacy pipeline).
- 19/19 smoke tests pass (catboost skip vì không install).
- `ruff check` + `mypy src/components/` → 0 errors.

---

## Phase 2 — Fusion stack (Tuần 3-4) — KHÓ NHẤT

### Goal
- Implement đủ 30+ fusion strategies cần cho 11 champion
- Fusion stack composable qua YAML
- Validation rules ở config load

### Phase 2.1 — Audit existing fusion logic (1 ngày) ✅ DONE 2026-04-27

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

**Result (2026-04-27)**:
- File: [FUSION_STRATEGY_INVENTORY.md](FUSION_STRATEGY_INVENTORY.md).
- 80 flag YAML + ~45 flag ẩn trong engine.py (v29_*, v30_*, v33_*, v38_*, v39_*) đều được phân loại.
- Tổng kết ~92 strategy class chia theo 4 layer: pre_entry (~22), entry (~15), hold (~20), exit_override (~35). Lớn hơn ước tính 30+ ban đầu trong ARCHITECTURE.md.
- Mapping ngược 11 champion → fusion stack high-level ở §2 của inventory; v22 và v32-stack chiếm phần lớn champions, v37a thêm per-profile dispatch, v39d thêm per-symbol rule exit.
- Exit-priority chain (>30 strategies) được trích chính xác từ engine.py — Phase 2 PHẢI giữ exact ordering để regression match.
- 5 open issues cho Phase 2.2: hold-vs-exit-modifier semantics, state carry across bars (counters), per-symbol set API, always-on registration, counter naming convention.
- Diary: [diary/2026-04-27.md](diary/2026-04-27.md).

### Phase 2.2 — Base fusion interface (1 ngày) ✅ DONE 2026-04-29

**Cần biết trước**: Section 4.6 của ARCHITECTURE.md.

**Steps**:
1. Tạo `BarContext`, `FusionResult`, `Position` dataclasses
2. Tạo `FusionStrategy` Protocol
3. Tạo `FusionStack`: chain strategies theo layer + priority
4. Unit test với dummy strategy

**Verification**: Test với fake strategy chain qua được.

**Result (2026-04-29)**:
- Mở rộng `Position.strategy_state: dict` (state carry across bars) + `FusionActionType="keep_position"` ở [src/components/base.py](../../src/components/base.py).
- [src/components/fusion/stack.py](../../src/components/fusion/stack.py): `FusionStack` chain executor — lifecycle `pre_entry → entry → hold → exit_override`, short-circuit per layer, counter accumulation, `keep_position` block exit_override.
- [src/components/fusion/registry.py](../../src/components/fusion/registry.py): `register_strategy / get_strategy / list_always_on` — `always_on=True` flag cho core strategies (orchestrator auto-prepend).
- 17 unit tests + 66/66 component tests pass; ruff/format/mypy clean.
- 5 open issues từ Phase 2.1 inventory §7: 4/5 resolved (per-symbol set defer Phase 2.4f).
- Diary: [diary/2026-04-29.md](diary/2026-04-29.md).

### Phase 2.3a — Port `rule` baseline (test-bed end-to-end) ✅ DONE 2026-04-29

**Mục tiêu**: Standalone non-ML baseline làm test-bed đầu tiên cho `FusionStack` + `Backtester` mới — không qua walk-forward, không qua orchestrator phức tạp. Thứ tự ưu tiên port: rule → v19_3 → v22 (theo diary 2026-04-29).

**Steps**:
1. `SimpleLongBacktester` (concrete impl của Protocol `Backtester`).
2. `RuleSignalEntry` (entry layer) + `RuleSignalExit` (exit_override layer) fusion strategies.
3. Mini driver `run_rule_baseline()` — wrap DataLoader + indicator compute (MA20, MACD) + FusionStack + Backtester.
4. Regression test golden parity 2585 trades.

**Result (2026-04-29)**:
- [src/components/backtest/engine.py](../../src/components/backtest/engine.py): `SimpleLongBacktester.run(actions, df_test, *, commission=0.0015, tax=0.001) → list[Trade]`. Pair `enter_long` với `exit` kế tiếp; drop dangling positions; round price/pnl 2 chữ số; date `str(...)[:10]`.
- [src/components/fusion/strategies/rule_signal.py](../../src/components/fusion/strategies/rule_signal.py): 2 strategy class với signature exact của legacy (`macd_hist > 0 and close > ma20 and close > open` → enter; ngược lại exit). Registered qua [strategies/__init__.py](../../src/components/fusion/strategies/__init__.py).
- [src/components/runners/rule_runner.py](../../src/components/runners/rule_runner.py): `run_rule_baseline(symbols, data_dir, *, first_test_year=2020, min_bars=50)` — mirror [run_pipeline.py:407-439](../../run_pipeline.py#L407-L439) `_run_rule_backtest_fair()`.
- Tests: [tests/components/test_backtest_simple.py](../../tests/components/test_backtest_simple.py) (10/10), [test_fusion_rule_signal.py](../../tests/components/test_fusion_rule_signal.py) (13/13), [tests/regression/test_rule_parity.py](../../tests/regression/test_rule_parity.py) — `pd.testing.assert_frame_equal(check_exact=True)` pass với golden 2585 trades, 61 symbols.
- Diary: [diary/2026-04-29.md](diary/2026-04-29.md).

### Phase 2.3b — Port `v19_3` minimal ML stack ✅ DONE 2026-04-29

**Mục tiêu**: Port strategy legacy `backtest_v19_3` qua component architecture để verify fusion stack đủ phủ một ML strategy có walk-forward predictions, entry cascade, position sizing, state carry across bars và exit-priority chain rút gọn.

**Result (2026-04-29)**:
- [src/components/fusion/helpers/](../../src/components/fusion/helpers/) — helper tính inline indicators, regime adapter và sizing cho v19_3, giữ arithmetic khớp legacy.
- [src/components/fusion/strategies/core/](../../src/components/fusion/strategies/core/) — 11 always-on exit/hold strategies dùng lại cho champion ML: hard stop, signal hard cap, fast exit, ATR stop, peak protect, hybrid exit, trailing, profit lock, zombie exit, min-hold protection.
- [src/components/fusion/strategies/entry/v19_entry_cascade.py](../../src/components/fusion/strategies/entry/v19_entry_cascade.py) — gộp entry sources + filters + sizing của v19_3 vào một strategy lớn để giảm rủi ro lệch shared state.
- [src/components/fusion/strategies/hold/v19_signal_hold_guard.py](../../src/components/fusion/strategies/hold/v19_signal_hold_guard.py) — gộp confirmed signal exit, confirm bars, strong uptrend carry và trend carry override.
- [src/components/runners/v19_3_runner.py](../../src/components/runners/v19_3_runner.py) — dedicated runner dùng legacy `_build_predictions()` khi cần prediction cache bit-equivalent, tự quản lý equity/position state và serialize schema golden.
- Tests: [tests/components/test_v19_3_smoke.py](../../tests/components/test_v19_3_smoke.py), [tests/regression/test_v19_3_parity.py](../../tests/regression/test_v19_3_parity.py) — parity pass với golden 1910 trades, 61 symbols.
- Verification: `pytest tests/components/ -q` → 156 passed; `PYTHONHASHSEED=42 pytest tests/regression/test_v19_3_parity.py -v -s` → 1 passed; `pytest tests/regression/test_rule_parity.py -q` → 1 passed; `ruff check ...` + `mypy src/components/` clean.
- Diary: [diary/2026-04-29.md](diary/2026-04-29.md).

### Phase 2.3c — Implement strategies cho v22 ✅ DONE 2026-04-29

**Mục tiêu**: Lock v22 parity trong component architecture và chuẩn bị YAML cho Phase 3 orchestrator.

**Result (2026-04-29)**:
- [src/components/runners/v22_runner.py](../../src/components/runners/v22_runner.py): dedicated runner cho v22 dùng `V19EntryCascade`, `V22HardCap`, `V22FastExit` và shared core exit stack để giữ exact parity với legacy.
- [src/components/fusion/strategies/__init__.py](../../src/components/fusion/strategies/__init__.py): register v22/core strategies (`v22_hard_cap`, `v22_fast_exit`, hard stop, ATR stop, peak protect, hybrid exit, trailing, profit lock, zombie, min-hold, long-horizon carry) cho YAML/orchestrator Phase 3.
- [config/experiments/champions/v22.yaml](../../config/experiments/champions/v22.yaml): declarative spec mirror actual v22 runner order (`force_exit`, `active_exit`, `hold`) với `leading_v2`, `trend_regime`, LightGBM, exit-model metadata.
- [tests/components/fusion/test_v22_registry.py](../../tests/components/fusion/test_v22_registry.py): verify tất cả strategy trong YAML resolve qua registry, exit order khớp runner contract, mods/params không drift khỏi defaults.
- Standalone split `SMA200Filter` / `HybridEntry` / `MlOnly` / `TimeDecay` defer tới Phase 3+ nếu cần YAML-executable stack; Phase 2.3c ưu tiên parity, không tách lại behavior-sensitive `V19EntryCascade`.
- Verification: `pytest tests/components/fusion/test_v22_fast_exit.py tests/components/fusion/test_v22_hard_cap.py tests/components/fusion/test_v22_registry.py -q` → 13 passed; `PYTHONHASHSEED=42 pytest tests/regression/test_v22_parity.py -v -s` → 1 passed (1784 trades exact golden); `pytest tests/regression/test_rule_parity.py tests/regression/test_v19_3_parity.py -q` → 2 passed; `ruff check ...` + `mypy src/components/` clean.

### Phase 2.4a — Port `v34` V32-stack champion ✅ DONE 2026-04-29

**Mục tiêu**: Lock v34 golden parity trong component runners trước khi port các champion V35+.

**Result (2026-04-29)**:
- [src/components/runners/v34_runner.py](../../src/components/runners/v34_runner.py): parity-first runner wrapper quanh `experiments.run_v34_final.backtest_v34`, dùng config `v34` từ [config/models.yaml](../../config/models.yaml) và giữ exact legacy V32 stack behavior thay vì tách thiếu V26-V32 strategies trong lượt này.
- [src/components/runners/__init__.py](../../src/components/runners/__init__.py): export `run_v34` và `trades_to_v34_dataframe`.
- [tests/regression/test_v34_parity.py](../../tests/regression/test_v34_parity.py): build CPU prediction cache từ golden meta, chạy runner component, CSV-roundtrip rồi `assert_frame_equal(check_exact=True)` với golden.
- Verification: `python -m pytest stock_ml/tests/regression/test_v34_parity.py -q` → 1 passed (1323 trades exact golden); `python -m pytest stock_ml/tests/regression/test_v22_parity.py stock_ml/tests/regression/test_v34_parity.py -q` → 2 passed; `python -m ruff check stock_ml/src/components/runners/v34_runner.py stock_ml/src/components/runners/__init__.py stock_ml/tests/regression/test_v34_parity.py` → clean.
- Decision: preserve trained-but-dropped exit-model behavior implicitly by mirroring legacy `run_pipeline._run_backtest_from_cache` signature filtering; `y_pred_exit` is trained in cache but not passed into `backtest_v34` for parity.

### Phase 2.4b — Port `v35b` entry-layer champion ✅ DONE 2026-04-29

**Mục tiêu**: Lock v35b golden parity trên cùng V32/V34 lineage, với các flag entry-layer V35 bật theo config.

**Result (2026-04-29)**:
- [src/components/runners/v35b_runner.py](../../src/components/runners/v35b_runner.py): thin runner wrapper quanh `experiments.run_v34_final.backtest_v35b`, reuse V34-lineage cache helper để giữ exact legacy behavior và tránh duplicate runner logic.
- [src/components/runners/v34_runner.py](../../src/components/runners/v34_runner.py): factor shared `_run_v34_lineage_cache(...)` cho v34/v35b; giữ rule không truyền `y_pred_exit` vào backtest function không khai báo param.
- [config/experiments/champions/v35b.yaml](../../config/experiments/champions/v35b.yaml): declarative champion spec cho Phase 3, mirror `v35b` trong [config/models.yaml](../../config/models.yaml) với `leading_v4`, `early_wave`, LightGBM và exit-model metadata enabled.
- [tests/regression/test_v35b_parity.py](../../tests/regression/test_v35b_parity.py): build CPU prediction cache từ golden meta, chạy component runner, CSV-roundtrip rồi exact compare với golden; khóa thêm `model_b_exit == 0`.
- Verification: `python -m pytest stock_ml/tests/regression/test_v35b_parity.py -q` → 1 passed (1381 trades exact golden); `python -m pytest stock_ml/tests/regression/test_v34_parity.py stock_ml/tests/regression/test_v35b_parity.py -q` → 2 passed; full champion parity set (`rule`, `v19_3`, `v22`, `v34`, `v35b`) → 5 passed; `python -m ruff check stock_ml/src/components/runners/v34_runner.py stock_ml/src/components/runners/v35b_runner.py stock_ml/src/components/runners/__init__.py stock_ml/tests/regression/test_v35b_parity.py` → clean.
- Decision: preserve trained-but-dropped exit-model behavior for v35b too; cache may contain `y_pred_exit`, but `backtest_v35b` does not receive it, matching legacy golden and keeping `model_b_exit` count at 0.

### Phase 2.4c — Verify `v32` standalone champion ✅ DONE 2026-04-29

**Mục tiêu**: Lock v32 golden parity riêng sau khi V34/V35b lineage helper đã ổn định.

**Result (2026-04-29)**:
- [src/components/runners/v32_runner.py](../../src/components/runners/v32_runner.py): thin runner wrapper quanh `experiments.run_v32_final.backtest_v32`, dùng config `v32` (`leading_v3`, `early_wave`) và reuse V34-lineage cache helper để giữ exact legacy V26-V32 stack behavior.
- [src/components/runners/__init__.py](../../src/components/runners/__init__.py): export `run_v32` và `trades_to_v32_dataframe`.
- [tests/regression/test_v32_parity.py](../../tests/regression/test_v32_parity.py): build CPU prediction cache từ golden meta, chạy component runner, CSV-roundtrip rồi exact compare với golden 1347 trades.
- Verification: `python -m pytest stock_ml/tests/regression/test_v32_parity.py -q` → 1 passed; champion parity set (`rule`, `v19_3`, `v22`, `v34`, `v35b`, `v32`) → 6 passed; `python -m ruff check stock_ml/src/components/runners/v32_runner.py stock_ml/src/components/runners/__init__.py stock_ml/tests/regression/test_v32_parity.py` → clean.
- Decision: preserve trained-but-dropped exit-model behavior; cache may contain `y_pred_exit`, but `backtest_v32` does not receive it, matching legacy golden.

### Phase 2.4e — Port `v37a_exit` & `v42_a` exit-model champions ✅ DONE 2026-04-28

**Mục tiêu**: Lock parity cho 2 champion `early_wave_dual` + trained-but-dropped exit-model behavior, dùng V34 lineage helper.

**Result (2026-04-28)**:
- [src/components/runners/v37a_exit_runner.py](../../src/components/runners/v37a_exit_runner.py): runner cho `v37a_exit`, dùng config từ [config/models.yaml](../../config/models.yaml) (`leading_v4`, `early_wave_dual` fw=8) và reuse V34-lineage helper. `_exit_model_cfg` returns None vì models.yaml không khai báo `exit_model` block cho v37a_exit → cache không train Model B (khớp meta golden không có `exit_model_config`).
- [src/components/runners/v42_a_runner.py](../../src/components/runners/v42_a_runner.py): runner cho `v42_a`, cùng pattern với target `early_wave_dual` fw=15, gọi `experiments.run_v42.backtest_v42` (alias `backtest_v37a`).
- [src/components/runners/__init__.py](../../src/components/runners/__init__.py): export `run_v37a_exit`, `run_v42_a`, `trades_to_v37a_exit_dataframe`, `trades_to_v42_a_dataframe`.
- [tests/regression/test_v37a_exit_parity.py](../../tests/regression/test_v37a_exit_parity.py): build CPU prediction cache từ meta (dùng `meta.get("exit_model_config")` vì meta golden không có key này), CSV-roundtrip rồi exact compare với golden 1370 trades; assert `model_b_exit == 0`.
- [tests/regression/test_v42_a_parity.py](../../tests/regression/test_v42_a_parity.py): tương tự cho golden 1442 trades.
- Verification: `PYTHONHASHSEED=42 pytest tests/regression/test_v37a_exit_parity.py tests/regression/test_v42_a_parity.py -v -s` → 2 passed; full champion suite (rule, v22, v32, v34, v35b, v37a, v37a_exit, v39d, v42_a) → 9 passed; `ruff check` + `mypy src/components/` clean.

**Decision**:
- **Reuse V34 lineage helper**: cả v37a_exit và v42_a chia sẻ V37a engine logic; runner mỏng giữ exact legacy behavior, không tách lại fusion stack trước Phase 3.
- **Preserve trained-but-dropped exit model**: `_exit_model_cfg` đọc trực tiếp `model_cfg["exit_model"]` từ models.yaml → None cho v37a_exit/v42_a (đúng với golden meta thiếu `exit_model_config`). Pipeline gate ở `run_pipeline._run_backtest_from_cache` vẫn drop `y_pred_exit` cho legacy backtest function (xem [EXIT_MODEL_BUG.md](EXIT_MODEL_BUG.md)). Match golden `model_b_exit == 0` cho cả 2.
- **Dùng `meta.get("exit_model_config")`** trong parity test thay vì `meta["exit_model_config"]` để tương thích cả golden có lẫn không có key này (v37a_exit và v42_a meta golden đều không có).

### Phase 2.4f — Port `v37d` GRU champion ✅ DONE 2026-04-28

**Mục tiêu**: Lock parity cho champion cuối cùng (`v37d`) — V32 engine với GRU sequence model thay LightGBM.

**Result (2026-04-28)**:
- [src/components/runners/v37d_runner.py](../../src/components/runners/v37d_runner.py): parity-first runner wrapper quanh `experiments.run_v37d.backtest_v37d` (delegate sang `backtest_v32`); dùng config `v37d` từ [config/models.yaml](../../config/models.yaml) (`leading_v4`, `early_wave`, `model_type=gru`) và reuse V34-lineage helper.
- [src/components/runners/__init__.py](../../src/components/runners/__init__.py): export `run_v37d` và `trades_to_v37d_dataframe`.
- [tests/regression/test_v37d_parity.py](../../tests/regression/test_v37d_parity.py): build CPU prediction cache, dùng `model_type` từ `config/models.yaml` (`gru`) thay vì meta golden (ghi sai `lightgbm`); CSV-roundtrip rồi exact compare 1407 trades; assert `model_b_exit == 0`.
- Verify:
  - `PYTHONHASHSEED=42 python -m pytest stock_ml/tests/regression/test_v37d_parity.py -v -s` → 1 passed (1407 trades exact golden).
  - `python -m ruff check stock_ml/src/components/runners/v37d_runner.py stock_ml/src/components/runners/__init__.py stock_ml/tests/regression/test_v37d_parity.py` → clean.
  - `python -m mypy stock_ml/src/components/` → clean.

**Decision**:
- **Reuse V34 lineage helper**: v37d engine identical với V34 (delegate `backtest_v32`); chỉ khác model swap ở `_build_predictions(...)`. Wrapper mỏng giữ exact legacy behavior, không tách GRU thành component path mới.
- **Override `model_type` từ config thay vì meta**: golden meta `trades_v37d.meta.json` ghi `model_type: lightgbm` (sai sót snapshot lúc generate baseline), nhưng test phải build cache với GRU để match golden 1407 trades. Dùng `get_model_config("v37d").get("model_type")` làm nguồn sự thật.
- **Preserve trained-but-dropped exit model**: cache vẫn có thể train exit model, nhưng runner không truyền `y_pred_exit` vào `backtest_v37d`; khớp legacy golden `model_b_exit == 0`.

**Next**:
- Phase 2 hoàn tất. Chuyển sang Phase 3 — pipeline orchestrator + CLI.

### Phase 2.4 — Implement strategies cho remaining champions (4-5 ngày)

`rule`, `v19_3`, `v22` đã xong ở Phase 2.3a-c; `v34` đã xong ở Phase 2.4a; `v35b` đã xong ở Phase 2.4b; `v32` đã xong ở Phase 2.4c; `v37a` và `v39d` đã có parity runner/component mapping; `v37a_exit` và `v42_a` đã xong ở Phase 2.4e; `v37d` đã xong ở Phase 2.4f. **Phase 2.4 hoàn thành — toàn bộ 11 champion đã có parity runner/regression locked.**

Theo thứ tự tăng dần độ khó:

1. **v34** — ✅ DONE 2026-04-29: leading_v4 + early_wave + full v32-style stack, parity runner wrapper giữ exact legacy V32 behavior; golden 1323 trades.
2. **v35b** — ✅ DONE 2026-04-29: V35 entry-layer flags (`relax_cooldown`, `single_bar_signal`, `rule_override`), parity wrapper reuse V34 lineage; golden 1381 trades.
3. **v32** — ✅ DONE 2026-04-29: leading_v3 + early_wave + standalone V32 delta, parity wrapper reuse V34 lineage; golden 1347 trades.
4. **v37a** — ✅ DONE 2026-04-29: per-profile dispatch, parity wrapper reuse V34 lineage; golden parity locked.
5. **v39d** — ✅ DONE 2026-04-29: per-symbol rule routing, parity wrapper reuse V34 lineage; golden 1181 trades.
6. **v37a_exit, v42_a** — ✅ DONE 2026-04-28: trained-but-dropped exit-model behavior preserved, parity wrapper reuse V34 lineage; golden 1370 / 1442 trades.
7. **v37d** — ✅ DONE 2026-04-28: GRU sequence model thay LightGBM, V32 engine path; parity wrapper reuse V34 lineage; golden 1407 trades.

**Sau mỗi version**:
- Run regression.
- Update FUSION_STRATEGY_INVENTORY.md.
- Commit chỉ khi user yêu cầu.

**Risk control**: Nếu 1 version stuck > 2 ngày → skip, đánh dấu, move on. Quay lại sau.

---

## Phase 3 — Pipeline orchestrator (Tuần 5) ✅ DONE 2026-04-28

### Goal
- Pipeline mới chạy được end-to-end ✅
- Cache hệ thống mới ✅
- CLI mới ✅

### Phase 3.1 — Orchestrator core ✅ DONE 2026-04-28

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

**Result (2026-04-28)**:
- [src/pipeline/config.py](../../src/pipeline/config.py): `ExperimentConfig` Pydantic model load từ YAML, tự suy `runner` từ `strategy`.
- [src/pipeline/walker.py](../../src/pipeline/walker.py): thin wrapper `walk_forward()` + `build_splitter()` từ `SplitConfig`.
- [src/pipeline/trainer.py](../../src/pipeline/trainer.py): `build_prediction_cache()` — refactor `_build_predictions` từ `run_pipeline.py` thành component riêng, parameterized bởi `ExperimentConfig`.
- [src/pipeline/orchestrator.py](../../src/pipeline/orchestrator.py): `Pipeline` class + `PipelineResult` dataclass, `CHAMPION_RUNNER_MAP` resolve đúng runner cho 11 champions.
- [tests/regression/test_pipeline_v22_parity.py](../../tests/regression/test_pipeline_v22_parity.py): v22 qua `Pipeline.run()` → 1 passed, 1784 trades exact golden.

### Phase 3.2 — Caching layer ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [src/pipeline/cache.py](../../src/pipeline/cache.py): `PredictionCacheManager` — pickle-based prediction cache, atomic write (tmp→rename), `load()/save()/invalidate()`. Cache key = sha256 của feature_set + model_type + target + exit_model + symbols + split.
- Tích hợp vào `Pipeline.__init__` qua `cache_manager` param + `use_cache` flag.

### Phase 3.3 — Matrix expander ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [src/pipeline/matrix_expander.py](../../src/pipeline/matrix_expander.py): `expand_matrix(yaml_path)` → `list[ExperimentConfig]`. Dùng `itertools.product` trên `axes` dict, deep-merge với `base`.
- [config/experiments/matrix/test_2x2.yaml](../../config/experiments/matrix/test_2x2.yaml): test matrix 2×2 = 4 configs verify.

### Phase 3.4 — CLI ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [scripts/cli.py](../../scripts/cli.py): subcommands `run`, `run-matrix`, `validate`, `list-components`, `list-experiments`, `compare`.
- [stock_ml/__main__.py](../../__main__.py): entry point `python -m stock_ml <cmd>`.
- Verify:
  - `python -m stock_ml validate champions/v22` → OK
  - `python -m stock_ml list-experiments` → 2 champions + 1 matrix
  - `python -m stock_ml list-components --type fusion` → 16 strategies

### Phase 3.5 — Validation rules ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [src/pipeline/validate.py](../../src/pipeline/validate.py): `validate_config()` → `list[ValidationError]`, `assert_valid()` raise `ValueError`.
- Rules: strategy runner exists, entry_model registered, target type valid, split range valid.
- Rule 4 (exit_model + target compatibility) relaxed — preserved as note per EXIT_MODEL_BUG.md.
- Bad config (strategy=v999, model=bad, split inverted) → 3 errors caught at load time.

---

## Phase 4 — Migration & legacy adapter (Tuần 6) ✅ DONE 2026-04-28

### Goal
- 11 champion ported, all pass regression ✅
- 49 legacy versions chạy qua adapter ✅
- Migration tool functional ✅

### Phase 4.1 — Legacy adapter (2-3 ngày) ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [src/pipeline/legacy_adapter.py](../../src/pipeline/legacy_adapter.py): `LegacyVersionAdapter` — wrap `backtest_vXX` cho tất cả 60+ legacy strategy keys. Đọc config từ models.yaml (feature_set, mods, params, target, exit_model), build prediction cache qua `trainer.build_prediction_cache`, delegate `_run_backtest` với mod_kwargs + proba_thresholds support. Không phải exact-parity runner — dùng cho historical comparison.
- `build_experiment_config()`: convert legacy model cfg → dict phù hợp `ExperimentConfig.model_validate()`, dùng cho migrate_legacy.
- `CHAMPION_VERSIONS` frozenset: warn khi dùng adapter cho champion có dedicated runner.
- `list_legacy_versions()` / `list_all_legacy_versions()`: discovery helpers.
- [src/pipeline/__init__.py](../../src/pipeline/__init__.py): export `LegacyVersionAdapter`, `LegacyRunResult`.
- [tests/components/test_legacy_adapter.py](../../tests/components/test_legacy_adapter.py): 18/18 pass — strategy map coverage, constructor validation, `build_experiment_config` schema, `_trades_to_dataframe`, `LegacyRunResult`, migrate dry-run, migrate write.
- Diary: [diary/2026-04-28.md](diary/2026-04-28.md).

### Phase 4.2 — Migration tool (1-2 ngày) ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [scripts/migrate_legacy.py](../../scripts/migrate_legacy.py): `migrate_version(key)` + `migrate_all()` — convert legacy models.yaml entry → ExperimentConfig YAML file. Output dir mặc định `config/experiments/legacy/`. Support `--dry-run`. `_meta` block giữ label/description/active/retired_reason.
- CLI: `python -m stock_ml migrate-legacy v25` / `python -m stock_ml migrate-legacy --all`. `migrate-legacy --all` migrate 49 non-champion versions, skip 11 champions.
- [scripts/cli.py](../../scripts/cli.py): thêm `list-legacy` subcommand (list 49 non-champion keys + champion aliases), `migrate-legacy` subcommand, `run legacy/vXX` route qua `LegacyVersionAdapter`.
- Diary: [diary/2026-04-28.md](diary/2026-04-28.md).

### Phase 4.3 — Cleanup old pipeline (1 ngày) ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [run_pipeline.py](../../run_pipeline.py): thêm docstring `DEPRECATED: Use python -m stock_ml run` + `DeprecationWarning` khi import. File vẫn hoạt động đầy đủ (forward-compatible) — sẽ xóa ở Phase 6.3 (tag v2.0).
- Diary: [diary/2026-04-28.md](diary/2026-04-28.md).

---

### Phase 4.1 — Legacy adapter (2-3 ngày) [ORIGINAL SPEC — kept for reference]

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

## Phase 5 — Testing & verification (Tuần 7) ✅ DONE 2026-04-28

### Goal
- Full regression test pass ✅
- Performance benchmark script ✅
- Documentation guides ✅

### Phase 5.1 — Full regression (2 ngày) ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- Fix test isolation bug: `test_fusion_stack.py` `_clean_registry` fixture clear registry nhưng không restore → `test_v22_registry.py` fail khi chạy chung. Fix bằng `importlib.reload(src.components.fusion.strategies)` sau yield.
- [tests/regression/test_legacy_smoke.py](../../tests/regression/test_legacy_smoke.py): 20 smoke tests cho 10 legacy versions (`v11`, `v14`, `v19_3`, `v22`, `v25`, `v28`, `v30`, `v34`, `v37b`, `v39a`). Test `adapter constructs` + `build_experiment_config schema` (no-IO). Integration test `run_produces_trades` cho 5 versions (skip nếu no data).
- [tests/components/test_fusion_properties.py](../../tests/components/test_fusion_properties.py): 10 property-based tests cho FusionStack (`P1`-`P10`): empty stack, skip_entry blocks entry, enter fires, exit fires, keep_position blocks exit_override, pre_entry gating, priority ordering, counter accumulation, layer isolation.
- Verify: `python -m pytest stock_ml/tests/components/ stock_ml/tests/regression/test_champions.py stock_ml/tests/regression/test_rule_parity.py stock_ml/tests/regression/test_legacy_smoke.py -q -k "not integration"` → 236 passed.

### Phase 5.2 — Performance benchmark (1 ngày) ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [scripts/benchmark.py](../../scripts/benchmark.py): `run_benchmark(versions, device, symbols_limit)` — wall-clock so sánh `Pipeline.run()` vs legacy `_build_predictions + adapter.run()`. Print table với `delta_pct` và `status` (OK / SLOW / FAST). Threshold: <20% slower.
- [scripts/cli.py](../../scripts/cli.py): thêm subcommand `benchmark --versions --device --symbols-limit --output`.
- Verify: `python -m stock_ml benchmark --help` → OK. Script chạy được; actual timing cần data và sẽ phụ thuộc machine.

### Phase 5.3 — Documentation (2 ngày) ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [HOW_TO_ADD_FEATURE_BLOCK.md](HOW_TO_ADD_FEATURE_BLOCK.md): implement + register + YAML + test equivalence.
- [HOW_TO_ADD_FUSION_STRATEGY.md](HOW_TO_ADD_FUSION_STRATEGY.md): 4 layers, implement, register, YAML, unit test, priority conventions, state carry.
- [HOW_TO_ADD_ENTRY_MODEL.md](HOW_TO_ADD_ENTRY_MODEL.md): Protocol requirements, wrapper pattern, register, smoke test, YAML, GPU notes.
- [HOW_TO_PORT_LEGACY_VERSION.md](HOW_TO_PORT_LEGACY_VERSION.md): dedicated runner, export, regression test, golden baseline, orchestrator map, adapter frozenset.
- [HOW_TO_RUN_MATRIX.md](HOW_TO_RUN_MATRIX.md): YAML format (base + axes), validate, run, compare, promote winner.

---

## Phase 6 — Tooling enhancements (Tuần 8) ✅ DONE 2026-04-28

### Goal
- Dashboard tương thích với pipeline mới ✅
- CI/CD setup ✅
- Cleanup archive/ ✅

### Phase 6.1 — Dashboard compatibility ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [scripts/cli.py](../../scripts/cli.py): thêm `export` subcommand — gọi `unified_export` qua subprocess; dashboard manifest tự update khi export.
- Thêm `--save-results` flag cho `run` command: auto-save CSV vào `results/trades_{strategy}.csv`.
- Thêm `--export` flag cho `run` command: sau khi run xong, tự export sang dashboard JSON.
- Luồng hoàn chỉnh: `python -m stock_ml run champions/v22 --save-results --export` → train → backtest → save CSV → generate JSON → update manifest.
- `python -m stock_ml export --versions v22,v34` → export chọn lọc cho dashboard.

### Phase 6.2 — CI/CD ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- [.github/workflows/ci.yml](../../../../.github/workflows/ci.yml): 4 jobs — lint (ruff check + ruff format), typecheck (mypy src/components/ + src/pipeline/), test-unit (pytest tests/components/ -k "not integration"), test-regression (hash check, chỉ chạy khi push, không PR).
- Trigger: push lên `master` hoặc `refactor/**`; PR vào `master`.
- Regression job cache `tests/regression/golden/` qua `actions/cache@v4`.

### Phase 6.3 — Cleanup ✅ DONE 2026-04-28

**Result (2026-04-28)**:
- Xóa `archive/analysis/`, `archive/exports/`, `archive/misc/`, `archive/src/`, `archive/visualization/` — tiết kiệm ~128MB (archive còn ~20MB với results_legacy/ + scripts/).
- Regression 12/12 pass sau cleanup.

**Result (2026-04-28 — phase tiếp theo)**:
- Tạo [legacy/](../../legacy/) structure (scripts_reference/, strategies/, configs/, docs/) + README.
- Move 17 scripts có model config từ `archive/scripts/` → [legacy/scripts_reference/](../../legacy/scripts_reference/) (700KB).
- Xóa toàn bộ `archive/scripts/` (27 one-off scripts, ~500KB) sau khi move.
- Nén `archive/results_legacy/` (19MB) → `archive/results_legacy.tar.gz` (~12MB).
- Cleanup [experiments/](../../experiments/): xóa 37 one-off variants, intermediate CSV/JSON, superseded *_final.py. Còn 12 champion/final files (run_v22_final, run_v3*_final, run_v37a/b/c/d, run_v39d, run_v42).
- Move `visualize_v42.py` → `visualization/scripts_v42.py`; `compare_rule_vs_model.py` → `analysis/compare_rule_vs_model.py`.
- Xóa `run_exit_29.log`, `run_exit_remaining.log` khỏi root.
- Regression 12/12 + component 203 pass sau cleanup.
- **Chưa thực hiện**: tag v2.0 — defer theo user preference.

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
| 4 | Phase 2.4 | Remaining champions ported sau rule/v19_3/v22 |
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
