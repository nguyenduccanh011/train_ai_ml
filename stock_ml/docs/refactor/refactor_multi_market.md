# Refactor: Multi-Market Support (Revised)

**Mục tiêu**: Mở rộng dự án từ VN stock sang crypto spot/perp, forex, phái sinh VN mà **không fork project** và **không trộn kết quả giữa các market**.

**Phạm vi phiên bản revised**: Giữ triết lý ban đầu (tách market config khỏi hardcode), nhưng bổ sung các hạng mục bắt buộc để kiến trúc chạy đúng end-to-end trong pipeline hiện tại.

---

## 0. Vì sao cần chỉnh kế hoạch

Kế hoạch cũ đúng hướng nhưng còn 4 lỗ hổng kiến trúc lớn:

1. `market` chưa đi xuyên schema + pipeline.
- Nếu chỉ thêm `market: crypto_spot` vào YAML mà model schema không chứa field này, override sẽ bị rơi.

2. Config global + cache dễ rò market.
- Nhiều module đang gọi `load_config()` theo kiểu global, không gắn context của run cụ thể.

3. Scope hardcode còn thiếu.
- Ngoài các file đã liệt kê, vẫn còn fallback VN trong `env.py`, `pipeline/*`, `run_pipeline.py`, và profile logic trong `backtest/indicators.py` + `components/fusion/helpers/regime.py`.

4. Champion scoping mới xử lý một nhánh.
- Hiện có cả nhánh legacy (`config_loader`) và nhánh CLI/Pipeline (`ExperimentConfig.from_yaml`). Cần scope champion theo market cho cả 2 nhánh.

---

## 1. Nguyên tắc thiết kế (khóa cứng)

1. **Market context là explicit**.
- Mọi run đều có market xác định rõ ràng từ đầu, không suy diễn ngầm.

2. **Không dùng runtime config global mutable cho core path**.
- Core pipeline phải nhận context đã resolve theo run.

3. **Ưu tiên config theo thứ tự**:
- `experiment YAML > MarketProfile > base.yaml defaults`.

4. **MarketProfile chỉ chứa defaults theo market**.
- Không đưa logic model/backtest vào profile.

5. **Không trộn leaderboard/caching cross-market**.
- Mỗi market phải có namespace rõ trong champion, cache key, artifact metadata.

---

## 2. Kiến trúc mục tiêu sau refactor

### 2.1 MarketProfile

```
config/
  markets/
    vn_stock.yaml
    crypto_spot.yaml
    crypto_perp.yaml
    vn_derivatives.yaml
```

`MarketProfile` chỉ mô tả đặc tính mặc định của thị trường, không chứa logic model/backtest. Profile gồm các namespace:
- `data`: `data_dir`, `schema`, `default_timeframe`, `benchmark_symbol`, `timestamp_column`, `timezone`, `required_columns`, `optional_columns`, `volume_unit`
- `execution`: `instrument_type`, `pnl_mode`, `commission`, `tax`, `slippage`, `initial_capital`, `currency`, `contract_multiplier`, `funding_rate_column`, `margin_mode`, `leverage`, `maintenance_margin_rate`, `liquidation_fee`
- `symbols`: `default_list`, `groups`, `default_group`
- `strategy_overrides`: override theo tên strategy đã đăng ký
- `features`: default feature blocks theo market
- `models`: default model stack theo market
- `target`: default target/horizon/unit theo market

Không tạo registry riêng theo market. `MarketProfile` chỉ chọn config mặc định; implementation vẫn đi qua registry hiện có của strategy/feature/model/backtest.

### 2.2 Experiment schema có `market`

`ExperimentConfig` phải có:

```yaml
market: vn_stock   # optional, default từ base.yaml
```

Đồng thời matrix expander phải cho phép axis/field `market`.

### 2.3 Run context bất biến

Tạo object runtime kiểu `ResolvedRunContext` (tên file tùy chọn) để gom:
- `experiment_cfg`
- `market_profile`
- `market`
- `resolved_data_dir`
- `resolved_symbols`
- `resolved_symbol_groups`
- `execution_costs`
- `timeframe/schema`
- `feature_set`
- `model_stack`
- `target_config`
- `run_identity`

`ResolvedRunContext` là nơi duy nhất merge theo thứ tự `experiment YAML > MarketProfile > base.yaml defaults`. Mọi module chính (trainer/orchestrator/runner/backtest/leaderboard/cache) đọc từ context này thay vì tự gọi global config fallback hoặc tự merge defaults.

### 2.4 RunIdentity/fingerprint chung

Tạo một định danh run dùng chung cho cache, artifact, champion, leaderboard:
- `market`
- `data.schema`
- `timeframe`
- `symbol_universe_hash`
- `feature_set_hash`
- `target_config_hash`
- `model_stack_hash`
- `strategy_config_hash`

Không để từng subsystem tự tạo key riêng, vì sẽ dễ lệch namespace và rò kết quả cross-market.

---

## 3. Chuẩn hóa schema MarketProfile

### 3.1 Enum và semantic thống nhất

`execution.pnl_mode` chỉ dùng 1 bộ enum thống nhất:
- `equity_spot`
- `linear_usdt_perp`
- `inverse_perp`
- `futures_contract`

Lưu ý: Không dùng `spot` nếu engine dispatch đang dùng `equity_spot`.

### 3.2 Ví dụ profile VN

```yaml
name: vn_stock
market_type: equity_spot

data:
  data_dir: "../portable_data/vn_stock_ai_dataset_cleaned"
  schema: ohlcv_daily
  default_timeframe: "1D"
  benchmark_symbol: "HNXINDEX"
  timestamp_column: date
  timezone: Asia/Ho_Chi_Minh
  required_columns: [open, high, low, close, volume]
  optional_columns: []
  volume_unit: shares

execution:
  instrument_type: spot
  pnl_mode: equity_spot
  commission: 0.0015
  tax: 0.001
  slippage: 0.001
  initial_capital: 100000000
  currency: VND
  contract_multiplier: 1
  funding_rate_column: null
  margin_mode: null
  leverage: 1
  maintenance_margin_rate: 0.0
  liquidation_fee: 0.0

symbols:
  default_list: [ACB, BID, MBB, SSI, VND]
  groups:
    ACB: bank
    BID: bank
    MBB: bank
    SSI: high_beta
    VND: high_beta

strategy_overrides:
  min_hold_protection:
    rule_priority: [AAA, SSN, TEG, GAS, PLX, IJC, DQC]
  v19_entry_cascade:
    score5_risky: [AAA, IJC, ITC, VHM, TEG, QBS, KMR, SSN, PLX]

features:
  enabled_blocks: null

models:
  default_stack: null

target:
  type: forward_return
  horizon: 5
  unit: bars
```

---

## 4. Kế hoạch thực hiện (Revised)

## Phase 0 - Schema + wiring nền [DONE]

**Mục tiêu**: `market` đi xuyên toàn bộ đường chạy config.

**Việc cần làm**:
1. ✅ Thêm field `market` vào `ExperimentConfig`.
2. ✅ Cho phép matrix axis `market` trong `matrix_expander`.
3. ✅ Validate `market` hợp lệ (file profile tồn tại).
4. ✅ `base.yaml` thêm `market: vn_stock`.

**Files**:
- `src/pipeline/config.py`
- `src/pipeline/matrix_expander.py`
- `src/pipeline/validate.py`
- `config/base.yaml`

---

## Phase 1 - MarketProfile loader + runtime resolution [DONE]

**Mục tiêu**: Có loader profile chuẩn + fail-fast validation.

**Việc cần làm**:
1. ✅ Tạo `src/market_profile.py` với dataclass/pydantic + validation, nhưng tái dùng cơ chế parse/validate config hiện có thay vì tạo config layer song song.
2. ✅ Tạo `config/markets/vn_stock.yaml`.
3. ✅ Resolve market theo ưu tiên:
- `experiment.market`
- `base.market`
- fallback cuối: `vn_stock`
4. ✅ Validate `strategy_overrides` bằng registry:
- key strategy phải tồn tại
- payload phải đúng schema
5. ✅ Tạo resolver duy nhất để build `ResolvedRunContext` (market + profile + merged defaults).
6. ✅ Core Pipeline/Trainer dùng `ResolvedRunContext` thay vì tự gọi `load_config()` hoặc tự merge fallback VN.
7. ✅ Các nhánh legacy còn lại đã được dọn trong Phase 2 theo từng call path. Fallback `vn_stock` cho champion YAML legacy không có `market` được giữ có chủ đích để backward-compatible.

**Files**:
- `src/market_profile.py` (new)
- `src/config_loader.py`
- `config/markets/vn_stock.yaml` (new)

---

## Phase 2 - Propagate market defaults vào toàn pipeline [DONE]

**Mục tiêu**: Không còn fallback VN trong call path chính.

**Việc đã làm**:
1. ✅ Bỏ hardcode path VN trong:
- `src/config_loader.py`
- `src/experiment_runner.py`
- `src/pipeline/build_predictions.py`
- `src/pipeline/trainer.py`
- `src/pipeline/orchestrator.py`
- `run_pipeline.py` (deprecated nhưng còn dùng)
- `src/env.py` (nhánh Colab)

2. ✅ `DataLoader.load_context()`:
- bỏ default `HNXINDEX`
- nếu `benchmark_symbol = null` thì trả DataFrame rỗng để caller có thể skip context join an toàn.

3. ✅ `get_pipeline_symbols()`:
- nếu user không truyền `--symbols` và không có explicit list trong experiment,
  dùng `market_profile.symbols.default_list` làm default chuẩn market.

4. ✅ Timeframe:
- propagate `data.default_timeframe` xuyên DataLoader/trainer/cache key.

5. ✅ Timestamp/schema:
- DataLoader đọc `timestamp_column`, `timezone`, `required_columns`, `optional_columns` từ context.
- Không giả định dữ liệu luôn là daily VN stock hoặc luôn có cột `date` ở layer loader.

**Files**:
- `src/data/loader.py`
- `src/config_loader.py`
- `src/experiment_runner.py`
- `src/pipeline/build_predictions.py`
- `src/pipeline/trainer.py`
- `src/pipeline/orchestrator.py`
- `src/env.py`
- `run_pipeline.py`

---

## Phase 3 - Dọn VN-specific profile logic còn sót [DONE]

**Mục tiêu**: Profile/grouping không còn hardcode rải rác.

**Việc đã làm**:
1. ✅ Thêm `symbol_groups`, `rule_priority_symbols`, `score5_risky_symbols` vào `DEFAULT_PARAMS` trong `src/backtest/defaults.py` (default `None` = dùng fallback module-level).

2. ✅ `src/backtest/engine.py`:
- Đọc `symbol_groups`, `rule_priority_symbols`, `score5_risky_symbols` từ cfg thay vì global.
- Truyền `symbol_groups` vào `get_regime_adapter`.
- Dùng `_rule_priority` / `_score5_risky` thay vì `RULE_PRIORITY_SYMBOLS` / `SCORE5_RISKY_SYMBOLS` global.

3. ✅ `src/backtest/indicators.py` — `get_regime_adapter` nhận `symbol_groups` param, fallback về `SYMBOL_PROFILES` nếu `None`.

4. ✅ `src/components/fusion/helpers/regime.py` — `get_regime_adapter` nhận `symbol_groups` param, fallback về `SYMBOL_PROFILES` nếu `None`.

5. ✅ `src/components/runners/generic_fusion.py`:
- `_regime()` nhận + truyền `symbol_groups`.
- `_run_cache_item()` nhận `symbol_groups`.
- `run_fusion()` nhận `symbol_groups`.

**Chiến lược backward-compat**: Các module-level fallback (`SYMBOL_PROFILES`, `RULE_PRIORITY_SYMBOLS`, `SCORE5_RISKY_SYMBOLS`) vẫn giữ nguyên. Khi `symbol_groups=None` (không truyền context), engine tự động fallback về hardcode VN. Khi caller truyền context từ `ResolvedRunContext.resolved_symbol_groups`, VN defaults bị bypass.

**Lưu ý**:
- Các symbol tuning đặc thù (ví dụ HPG/VND trong `indicators.py`) vẫn còn trong code — cần đưa vào `strategy_overrides` hoặc `regime_overrides` trong Phase 5 khi thêm market mới.
- Module-level fallback data có thể xóa sau khi toàn bộ callers đều truyền context.

---

## Phase 4 - Champion scoping theo market [DONE]

**Mục tiêu**: Không trộn champion VN/crypto trong cùng run.

**Việc đã làm**:
1. ✅ Thêm `market: vn_stock` vào toàn bộ champion YAML hiện có.
2. ✅ Nhánh legacy `config_loader._load_champion_models()` filter champion theo market hiện tại.
3. ✅ `ExperimentConfig.from_yaml()` backward-compatible: champion không có `market` mặc định về `vn_stock`.
4. ✅ `scripts/cli.py compare-champions`:
- Nếu user truyền `--champions`, chạy đúng danh sách explicit.
- Nếu không truyền, chỉ quét champion cùng market với `base.yaml`.
5. ✅ `scripts/cli.py list-experiments` chỉ liệt kê champion cùng market hiện tại để tránh chọn nhầm cross-market.
6. ✅ `get_pipeline_symbols()` dùng `ResolvedRunContext` + schema/timestamp/timezone từ MarketProfile khi resolve symbol list.

**Khuyến nghị rollout tiếp**:
- Bước hiện tại dùng field `market` trong YAML (ít churn).
- Khi số champion lớn thì migrate sang thư mục con `champions/<market>/`.

---

## Phase 5 - Feature/model/target/data routing theo market [DONE]

**Mục tiêu**: Hỗ trợ dataset, feature, target và model riêng cho từng market mà không sửa core mỗi lần.

**Việc đã làm**:
1. ✅ `src/market_profile.py` — `resolve_run_context` merge experiment overrides vào feature/model/target:
   - `feature_set`: `profile.features.enabled_blocks` (nếu non-null) > `cfg.feature_set()` string từ experiment.
   - `model_stack`: `profile.models.default_stack` (nếu non-null) > `[cfg.entry_model_type()]` từ experiment.
   - `target_config`: merge `{**profile_target, **experiment_target}` — experiment wins per-field.
   - `run_identity` build từ các giá trị **sau merge**.

2. ✅ `src/components/features/registry.py` — thêm `build_engine_from_blocks(names: list[str])`: tạo `ComposableFeatureEngine` từ block name list (dùng khi MarketProfile define `enabled_blocks`).

3. ✅ `src/pipeline/trainer.py` — dùng `run_context` thay vì `cfg` trực tiếp:
   - Feature: nếu `run_context.feature_set` là list → `build_engine_from_blocks`; nếu là string → `FeatureEngine(feature_set=...)`.
   - Model: `run_context.model_stack[0]` nếu non-null, else `cfg.entry_model_type()`.
   - Target: `run_context.target_config` (merged).

4. ✅ `src/pipeline/cache.py` — `_build_prediction_cache_key` nhận `run_context` optional:
   - Nếu có `run_context`: hash từ `{**run_context.run_identity, split, exit_model, execution}`.
   - Else: giữ logic cũ (backward compat nhánh legacy).
   - `PredictionCacheManager.load/save/invalidate/key` đều nhận `run_context` optional.

5. ✅ `src/pipeline/orchestrator.py` — truyền `run_context` vào `_build_cache` và dùng `run_context.feature_set` cho metadata label.

6. ✅ `src/pipeline/build_predictions.py` — bỏ hardcode VN target fallback:
   - Thứ tự ưu tiên: `target_cfg` param > `pipeline_cfg["target"]` > `run_context.target_config` (MarketProfile).

7. ✅ `src/pipeline/validate.py` — mở `VALID_TARGET_TYPES` theo MarketProfile:
   - Load profile của market hiện tại, thêm `profile.target.type` vào valid set trước khi check.
   - Xử lý `resolved_market` trước try block để đảm bảo luôn có giá trị cho rule 3.

8. ✅ `src/data/loader.py` (đã đủ từ trước) — `load_symbol` raise `ValueError` nếu thiếu `required_columns`.

9. ✅ Thêm `config/markets/crypto_spot.yaml` cho crypto spot OHLCV, dùng `pnl_mode: equity_spot`, currency `USDT`, timestamp `timestamp`, timezone `UTC`.

**Backward compat**:
- VN stock (`enabled_blocks: null`, `default_stack: null`): behavior không đổi — trainer vẫn dùng experiment string/model type.
- Khi thêm market mới với `enabled_blocks` defined: trainer tự động dùng block list mà không cần sửa experiment YAML.

**Ví dụ khác biệt market cần hỗ trợ bằng YAML**:
- VN stock daily: target `forward_return`, horizon `5`, unit `bars`.
- Crypto spot intraday: target `forward_return`, horizon `12`, unit `hours` hoặc `bars`.
- Perp: feature block có `funding`, data schema yêu cầu `funding_rate`, execution dùng `linear_usdt_perp`.

---

## Phase 6 - Backtest engine cho phái sinh/perp [DONE]

**Mục tiêu**: Engine dispatch theo `execution.pnl_mode` và dùng metadata execution từ `ResolvedRunContext`.

**Việc đã làm trong bước đầu**:
1. ✅ Tạo `src/backtest/pnl.py` với `PnlCalculator` registry và implementation `equity_spot`.
2. ✅ `src/backtest/engine.py` dispatch qua `PnlCalculator` cho `equity_spot` và áp dụng `slippage` khi config khác 0.
3. ✅ `src/pipeline/orchestrator.py` forward `commission`, `tax`, `slippage`, `initial_capital`, `pnl_mode` từ `ResolvedRunContext.execution_costs` vào runner thay vì để runner dùng hardcoded defaults.
4. ✅ Artifact metadata ưu tiên resolved execution từ MarketProfile và ghi thêm `schema`, `timeframe`.
5. ✅ Leaderboard row/loader thêm `schema`, `timeframe`; fairness key tách thêm theo `schema`.
6. ✅ `src/pipeline/orchestrator.py` forward đầy đủ execution metadata (`contract_multiplier`, funding, margin/liquidation, rollover) từ `ResolvedRunContext` vào runner.
7. ✅ Thêm unit test đối chiếu tay cho `equity_spot` PnL calculator.

**Việc đã làm tiếp**:
1. ✅ Thêm implementation `linear_usdt_perp` trong `src/backtest/pnl.py`.
2. ✅ Thêm unit test đối chiếu tay cho `linear_usdt_perp` PnL calculator.
3. ✅ Thêm market profile `config/markets/crypto_perp.yaml` cho USDT-margined perpetual.
4. ✅ Thêm implementation `inverse_perp` trong `src/backtest/pnl.py`.
5. ✅ Thêm unit test đối chiếu tay cho `inverse_perp` PnL calculator.
6. ✅ `src/backtest/engine.py` đọc `execution.funding_rate_column` và trừ funding fee khi đang giữ long position.

**Việc đã làm tiếp sau đó**:
1. ✅ Thêm implementation `futures_contract` trong `src/backtest/pnl.py`.
2. ✅ `src/backtest/engine.py` đọc `contract_multiplier` và scale bar-level return theo multiplier.
3. ✅ Thêm default `contract_multiplier: 1.0` và `funding_rate_column: null` trong `src/backtest/defaults.py`.
4. ✅ Thêm unit test cho `futures_contract` và multiplier scaling.

**Việc đã làm tiếp theo**:
1. ✅ `MarketExecutionConfig` + market YAML hỗ trợ `leverage`, `maintenance_margin_rate`, `liquidation_fee`.
2. ✅ `src/backtest/defaults.py` thêm default leverage/liquidation tắt mặc định (`leverage: 1`, `maintenance_margin_rate: 0`, `liquidation_fee: 0`).
3. ✅ `src/backtest/engine.py` kiểm tra liquidation cho long-only derivatives bằng intrabar `low` khi `leverage > 1`.
4. ✅ Thêm trade-level backtest mẫu cho liquidation và guard đảm bảo spot path không bị ảnh hưởng.

**Việc đã làm trong bước này**:
1. ✅ `FuturesContractCalculator` có helper `compute_roll_cost()` để model chi phí rollover/basis ở tầng calculator.
2. ✅ Thêm market profile `config/markets/vn_derivatives.yaml` cho VN futures (`pnl_mode: futures_contract`, `contract_multiplier: 100000`).
3. ✅ Thêm integration test cho funding fee deduction trong `backtest_unified()`.
4. ✅ Thêm integration test cho `inverse_perp` engine path.
5. ✅ Thêm smoke test cho `futures_contract` engine path.

**Việc đã làm hoàn tất Phase 6 trong bước này**:
1. ✅ `src/backtest/defaults.py` thêm `roll_cost_rate` và `expiry_date_column` để bật rollover bằng config.
2. ✅ `src/backtest/engine.py` hỗ trợ short signal ở tầng engine (`y_pred = -1`) và kiểm tra liquidation short bằng intrabar `high`.
3. ✅ `src/backtest/engine.py` thêm expiry-date forced rollover cho `futures_contract`: giữ position, reset `entry_close`, trừ exit cost + roll cost + re-entry cost.
4. ✅ Thêm test trade-level cho rollover thật khi có `expiry_date_column` và guard khi thiếu cột expiry.
5. ✅ `MarketExecutionConfig` + market YAML expose `roll_cost_rate`, `expiry_date_column` để bật rollover end-to-end bằng MarketProfile.
6. ✅ Thêm test liquidation short engine-level, không thay đổi strategy để tự sinh short.
7. ✅ Rollover futures áp dụng cho cả long và short khi giữ nguyên vị thế qua expiry-date forced roll.
8. ✅ Thêm test short futures rollover để đảm bảo reset entry + roll cost đúng cho short path.
9. ✅ Rollover chỉ kích hoạt cho `pnl_mode: futures_contract`; nếu perp/spot lỡ có `expiry_date_column` thì engine bỏ qua rollover config.

**Kết luận Phase 6**: Hoàn tất engine-level support cho spot/perp/futures: PnL dispatch, funding, leverage/liquidation, contract multiplier, short engine signal và forced rollover bằng `expiry_date_column`.

## Phase 7 - Production-risk sizing perp/futures [DONE]

**Mục tiêu**: Nâng engine từ trade-level derivatives support lên portfolio/risk sizing production-ready.

**Việc đã làm trong Phase 7.1 — short strategy official support**:
1. ✅ Thêm `short_enabled` vào `DEFAULT_PARAMS`, `MarketExecutionConfig`, market profile `crypto_perp`, và forwarding từ `Pipeline` vào runner.
2. ✅ `src/backtest/engine.py` mặc định chặn signal `y_pred = -1`; chỉ mở short khi config/MarketProfile bật `short_enabled: true`.
3. ✅ Sửa funding fee cho short: funding rate dương credit cho short thay vì luôn deduct như long.
4. ✅ Thêm short exit tối thiểu ở engine-level:
   - hard stop khi short lỗ vượt `HARD_STOP`
   - zombie exit khi short flat/stall quá `ZOMBIE_BARS`
5. ✅ `InversePerpCalculator.bar_return()` không còn inherit linear price-return path; long inverse dùng quan hệ ngược với price return.
6. ✅ Thêm unit/integration tests cho:
   - short disabled mặc định
   - short funding credit
   - short hard stop
   - short zombie exit
   - short liquidation vẫn hoạt động khi explicit bật short
   - inverse perp bar return

**Việc đã làm trong Phase 7.2 — short sizing + risk controls**:
1. ✅ Thêm `short_position_size`, `short_hard_cap`, `borrow_rate_column` vào `DEFAULT_PARAMS`, `MarketExecutionConfig`, `crypto_perp`, và `vn_derivatives`.
2. ✅ `src/backtest/engine.py` hỗ trợ short-specific position sizing khi mở vị thế short.
3. ✅ Short path có profit-taking hard cap (`signal_hard_cap`) và fast profit exit (`fast_exit_profit`) ngoài hard stop/zombie exit đã có.
4. ✅ Engine đọc `borrow_rate_column` và trừ borrow cost theo bar khi đang giữ short.
5. ✅ Thêm tests cho short position sizing, short hard cap, short fast exit và borrow fee deduction.

**Việc đã làm trong Phase 7.3 — short production-risk controls**:
1. ✅ Thêm `short_squeeze_exit`, `short_squeeze_vol_mult`, `short_squeeze_price_pct` vào `DEFAULT_PARAMS`, `MarketExecutionConfig`, và `crypto_perp` profile.
2. ✅ `src/backtest/engine.py` hỗ trợ short squeeze exit khi volume spike so với rolling 20-bar avg và candle tăng vượt ngưỡng.
3. ✅ Thêm `borrow_available_column` để block short entry khi không borrow được và forced exit `borrow_recalled` khi đang giữ short mà availability mất.
4. ✅ Thêm `max_short_notional` để cap deployed notional của short position trong single-symbol engine, làm nền cho exposure limit portfolio-level sau này.
5. ✅ Thêm tests cho short squeeze, borrow availability/recalled và max short notional cap.

**Việc đã làm trong Phase 7.4 — derivatives guardrail cleanup**:
1. ✅ Tạo `SUPPORTED_PNL_MODES` chung từ registry PnL và dùng trong `MarketExecutionConfig` để fail-fast khi profile khai báo `pnl_mode` không hợp lệ.
2. ✅ `src/backtest/engine.py` đọc các field derivatives/short mới từ `DEFAULT_PARAMS` thống nhất bằng `cfg[...]`, tránh fallback hardcode lệch với MarketProfile.
3. ✅ Sửa cooldown sau khi đóng short dùng PnL đã ký dấu theo vị thế, tránh coi short đang lời là long loss.
4. ✅ Giảm lặp normalize symbol list trong `src/market_profile.py` và tránh import registry strategy lặp mỗi lần validate profile.
5. ✅ Thêm tests cho unsupported `pnl_mode` và cooldown signed-PnL của short.

**Việc đã làm trong Phase 7.5 — cross-margin portfolio engine**:
1. ✅ Tạo `src/backtest/portfolio_engine.py` với `PortfolioState`, `PositionState` và `backtest_portfolio()` chạy bar-synchronized cho nhiều symbol.
2. ✅ `margin_mode: cross` route qua portfolio engine trong `src/experiment_runner.py`; `margin_mode: isolated` giữ đường chạy per-symbol hiện tại.
3. ✅ Cross-margin state theo dõi portfolio equity, total notional, initial margin, maintenance margin và available margin.
4. ✅ Entry mới bị block khi không đủ available margin; liquidation cross-margin force close vị thế tệ nhất trước khi portfolio equity thấp hơn maintenance margin.
5. ✅ Short portfolio-level có aggregate cap `max_total_short_notional` và vẫn tôn trọng borrow availability/borrow recall.
6. ✅ Thêm tests cho equity consolidation, margin exhaustion, cross-margin liquidation, aggregate short cap và borrow recall.

**Việc đã làm trong Phase 7.6 — portfolio short exits + vol sizing**:
1. ✅ `src/backtest/portfolio_engine.py` port short exit controls từ single-symbol engine sang cross-margin portfolio path:
   - hard stop
   - short hard cap / profit target
   - fast profit exit
   - zombie exit
   - short squeeze exit theo volume spike + candle tăng mạnh
2. ✅ Portfolio short path tiếp tục ưu tiên borrow recall và aggregate `max_total_short_notional`, đồng thời hỗ trợ thêm per-trade `max_short_notional`.
3. ✅ Thêm ATR-based short position sizing (`atr_position_sizing`, `atr_risk_target`) để size per-symbol theo volatility thay vì dùng một scalar chung cho toàn bộ symbol.
4. ✅ Thêm tests cho portfolio short hard stop, hard cap, zombie exit, squeeze exit và ATR sizing.

**Việc đã làm trong Phase 7.7 — portfolio futures rollover parity**:
1. ✅ `src/backtest/portfolio_engine.py` hỗ trợ forced rollover cho cross-margin futures khi `pnl_mode: futures_contract` và có `expiry_date_column`.
2. ✅ Portfolio rollover áp dụng exit cost + roll cost + re-entry cost, reset `entry_close`, `entry_day`, `entry_date`, `entry_equity`, `max_equity`, `cumulative_pnl` và `hold_days` cho vị thế đang giữ.
3. ✅ Rollover config bị bỏ qua cho spot/perp path, giữ parity với single-symbol engine.
4. ✅ Thêm tests cho portfolio futures rollover và guard non-futures.

**Việc đã làm trong Phase 7.8 — smart roll calendar theo spread columns**:
1. ✅ Thêm config roll calendar vào `DEFAULT_PARAMS`, `MarketExecutionConfig`, `vn_derivatives` profile và forward từ pipeline runner:
   - `roll_rule`: `null`, `volume_crossover`, `oi_crossover`, `n_days_before_expiry`
   - `roll_days_before_expiry`
   - `next_volume_column`
   - `next_oi_column`
2. ✅ `src/backtest/engine.py` hỗ trợ smart rollover cho single-symbol futures:
   - giữ behavior cũ khi `roll_rule: null` (roll khi `date >= expiry_date`)
   - roll trước expiry khi next contract volume/OI vượt front contract
   - roll N ngày trước expiry khi dùng rule `n_days_before_expiry`
   - guard chỉ roll 1 lần cho mỗi expiry period.
3. ✅ `src/backtest/portfolio_engine.py` có parity với single-symbol engine cho smart rollover trong cross-margin futures.
4. ✅ Thêm regression tests cho volume crossover, N-days-before-expiry, behavior cũ khi `roll_rule: null`, guard roll một lần mỗi expiry và portfolio volume crossover.

**Việc đã hoàn tất trong Phase 7**:
1. ✅ Short strategy với đầy đủ risk controls: hard stop, hard cap, zombie exit, squeeze exit, borrow fee/availability/recall.
2. ✅ Cross-margin portfolio engine với equity consolidation, margin exhaustion, liquidation và aggregate short cap.
3. ✅ ATR-based position sizing trong portfolio path.
4. ✅ Smart futures rollover với 3 roll rules: `volume_crossover`, `oi_crossover`, `n_days_before_expiry`.
5. ✅ Comprehensive test coverage cho derivatives features trong single-symbol và portfolio path.

**Gap nhỏ còn lại**:
1. ⚠️ ATR position sizing chưa có trong single-symbol engine; hiện chỉ có trong portfolio engine.
2. ⚠️ Tooling tạo spread columns (`next_volume`, `next_oi`) từ raw contract chain chưa có; data cần chuẩn bị sẵn trước khi chạy smart roll.

**Kết luận Phase 7**: Hoàn tất production-risk support cho derivatives trong scope hiện tại. Các gap còn lại là DX/parity cleanup, không block việc chạy production path.

**Ghi chú triển khai phái sinh**:
- Với USDT-margined perpetual (`linear_usdt_perp`, `contract_multiplier: 1`), có thể bắt đầu chạy model bằng YAML market profile; `margin_mode: cross` đã có portfolio engine cơ bản cho nhiều symbol.
- `contract_multiplier` chỉ là blocker nếu dataset/venue dùng contract size khác 1 hoặc inverse/futures contract tính PnL theo số hợp đồng.
- Liquidation hiện hỗ trợ long bằng intrabar `low` và short bằng intrabar `high` trong single-symbol engine; portfolio engine có thêm cross-margin maintenance check và force liquidation theo vị thế tệ nhất.
- Rollover hiện chỉ bật cho `futures_contract`, hỗ trợ forced roll theo `expiry_date_column` và smart roll theo `roll_rule` (`volume_crossover`, `oi_crossover`, `n_days_before_expiry`) bằng spread columns của next contract; áp dụng cost và reset entry trong cùng bar ở cả single-symbol và cross-margin portfolio path.

---

## Phase 8 - Cleanup & Tooling (Optional)

**Mục tiêu**: Nâng cấp DX và đồng bộ parity giữa single-symbol và portfolio path. Phase này không block production deployment.

**Việc có thể làm tiếp**:
1. Port ATR position sizing từ `src/backtest/portfolio_engine.py` sang `src/backtest/engine.py` để single-symbol engine có parity với portfolio engine.
2. Tạo helper script để generate spread columns (`next_volume`, `next_oi`) từ raw long-format futures contract chain.
3. Thêm validation helper cho roll calendar config để fail-fast khi bật `roll_rule` nhưng thiếu cột dữ liệu tương ứng.
4. Viết user guide cho derivatives strategy configuration khi có nhu cầu vận hành nhiều venue/dataset khác nhau.

**Files dự kiến nếu triển khai Phase 8**:
- `src/backtest/engine.py`
- `scripts/build_futures_spread.py` (new)
- `docs/derivatives_guide.md` (new, chỉ tạo khi cần guide chính thức)

---

## 5. Cập nhật rủi ro và giảm thiểu

| Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|
| `market` không đi qua schema nên override mất tác dụng | Cao | Thêm field `market` vào `ExperimentConfig` + matrix/validator |
| Rò config giữa các run do global cache | Cao | Dùng `ResolvedRunContext`; hạn chế đọc global runtime config |
| Trộn champion cross-market | Cao | Champion phải có `market`, loader filter cứng theo market |
| Symbol universe sai market | Cao | Resolver ưu tiên `symbols.default_list` từ MarketProfile |
| Fallback VN còn sót trong pipeline/env | Cao | Dọn toàn bộ fallback path VN ở phase 2 |
| Mỗi subsystem tự tạo cache/artifact key khác nhau | Cao | Dùng `run_identity` chung từ `ResolvedRunContext` |
| Feature/target/model vẫn global nên khó thêm crypto/phái sinh | Cao | Phase 5 phải resolve cả feature, target, model theo market |
| Enum `pnl_mode` không nhất quán | Trung bình | Chuẩn hóa enum và validate khi load profile |
| Leaderboard/currency so sánh lệch | Trung bình | Rank theo `%` + tách nhóm theo market/currency nếu cần |

---

## 6. Thứ tự ưu tiên khi triển khai thị trường mới

### 6.1 Thêm crypto spot (OHLCV đơn giản)

1. ✅ Hoàn tất Phase 0 -> 5.
2. ✅ Tạo `config/markets/crypto_spot.yaml`.
3. ✅ Khai báo schema/timestamp/timezone/feature/target/model stack riêng trong YAML nếu khác VN stock.
4. ⏳ Tạo champion có `market: crypto_spot` khi có dataset/model chính thức.
5. ✅ Chạy pipeline không cần Phase 6 nếu chỉ là spot không margin/funding.

### 6.2 Thêm perp/phái sinh

1. Hoàn tất Phase 0 -> 5.
2. Tạo profile `crypto_perp` hoặc `vn_derivatives` với schema/columns phù hợp.
3. Triển khai xong Phase 6 rồi mới chạy backtest chính thức.

---

## 7. Definition of Done (DoD)

## DoD A - Generic multi-market cho spot ✅ ĐẠT

- ✅ Có thể chạy 2 market khác nhau (vd `vn_stock`, `crypto_spot`) trong cùng codebase.
- ✅ Không sửa code khi đổi market, chỉ đổi YAML.
- ✅ Data schema, timestamp, timezone, symbol universe, feature set, target và model stack đều resolve từ YAML/context.
- ✅ Champion/filter/cache/artifact/leaderboard không trộn lẫn market nhờ `run_identity` chung.

## DoD B - Perp/futures production-ready ✅ ĐẠT

- ✅ Engine dispatch đúng theo `pnl_mode` (`equity_spot`, `linear_usdt_perp`, `inverse_perp`, `futures_contract`).
- ✅ Funding/margin/liquidation/rollover có test và pass.
- ✅ Kết quả trade-level đối chiếu tay khớp qua unit/integration tests cho PnL calculator và derivatives backtest.

---

## 8. Danh sách file ảnh hưởng (Revised)

### Core bắt buộc (Phase 0-4)
- `config/base.yaml`
- `config/markets/vn_stock.yaml` (new)
- `src/market_profile.py` (new)
- `src/config_loader.py`
- `src/pipeline/config.py`
- `src/pipeline/matrix_expander.py`
- `src/pipeline/validate.py`
- `src/data/loader.py`
- `src/experiment_runner.py`
- `src/pipeline/build_predictions.py`
- `src/pipeline/trainer.py`
- `src/pipeline/orchestrator.py`
- `src/env.py`
- `run_pipeline.py`
- `scripts/cli.py`
- `src/backtest/defaults.py`
- `src/backtest/indicators.py`
- `src/components/fusion/helpers/regime.py`

### Mở rộng (Phase 5-6)
- `src/components/features/registry.py`
- `src/components/models/registry.py` (nếu thêm resolver stack theo market)
- `src/components/backtest/engine.py`
- `src/components/backtest/pnl/*` (new)
- `config/markets/crypto_spot.yaml` (new) ✅
- `config/markets/crypto_perp.yaml` (new)
- `config/markets/vn_derivatives.yaml` (new)

---

## 9. Kết luận triển khai

- ✅ Kiến trúc `MarketProfile` đã hoàn tất và là nền tảng chung cho multi-market.
- ✅ Phase 0 -> 7 đã hoàn tất: có thể chạy spot/perp/futures bằng YAML config mà không fork project.
- ✅ DoD A và DoD B đã đạt: config/context, cache/artifact/champion/leaderboard đều tách theo market; derivatives có PnL dispatch, funding, margin/liquidation, short risk controls và rollover có test.
- ⚠️ Gap còn lại chỉ là DX/parity cleanup: ATR sizing chưa có trong single-symbol path và tooling tạo spread columns từ raw contract chain chưa có.
- 🎯 Trạng thái hiện tại đủ nền tảng production-ready cho VN stock, crypto spot, crypto perp và VN derivatives trong scope đã định nghĩa.
