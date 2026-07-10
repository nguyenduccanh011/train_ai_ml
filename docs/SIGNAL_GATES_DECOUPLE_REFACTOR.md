# Signal-Gates Decoupling Refactor

**Status:** Proposed (Phase 0 landed behind a flag) · **Owner:** TBD · **Created:** 2026-06-15
**Related:** `docs/ARCHITECTURE.md`, `STORAGE_ARCHITECTURE_REFACTOR.md`, the in-progress
alpha → portfolio → execution refactor (`project_alpha_portfolio_execution_refactor`).

---

## TL;DR (Tiếng Việt)

Các "gate" tạo nên edge của chiến lược (upleg entry, downleg/nonbull exit, cross-sectional/
raw entry, …) đang **bị trộn vào pipeline dual-ML** (`recombine_signals`) — chỉ chạy được khi
có ML, và `engine_config` lẫn lộn 2 loại key (recombine-level vs engine-level). Hệ quả: **không
dựng được model rule-thuần mà không đụng ML** (đã phải thêm cờ `rule_only_no_ml` để vá tạm).
Đây **không phải bug** — là **giới hạn linh hoạt do thiết kế** giả định "mọi model đều dùng ML".

Kế hoạch: tách tầng gate ra một module dùng chung `src/signals/gates.py` để **ML, rule, hybrid
đều gọi chung**; gom config gate vào một schema rõ ràng (bỏ list `pop()` thủ công). Điều kiện
bắt buộc: **golden parity** — mọi champion hiện có phải ra composite Y HỆT trước/sau.

---

## 1. Problem / Motivation

The strategies that define the current leaderboard (`regression_dual_ml_recombine_decoupled`)
apply their decisive logic — the **signal gates** — inside the dual-ML recombine stage. Concretely:

- The gate logic lives in `recombine_signals()` / `_recombine_dual_ml_signals()` /
  `_exit_force_mask()` / `_force_healthy_mask()` in `stock_ml/src/pipeline/experiment.py`.
- `recombine_signals()` only runs for `_RECOMBINE_STRATEGIES`, and it **requires the ML
  `score` / `exit_score` columns** to compute the per-symbol causal z-scores it gates on.
- The alternative `model_mode="rule_only"` path uses `generate_signals_from_predictions()` and
  **does NOT pass through `recombine_signals()`** → it cannot reuse the gates (upleg/downleg/
  nonbull) at all.
- The dispatch in `run_experiment()` **always calls `model.fit()`** for recombine strategies —
  there is no config switch to skip ML training.
- `engine_config` mixes two unrelated concerns. ~18 recombine-level keys are manually stripped
  before constructing `EngineConfig` (the `engine_cfg.pop(...)` block in `run_experiment()`):
  `entry_gate`, `entry_xs_mom_pct`, `exit_gate`, `entry_raw_threshold`, `entry_z_low_threshold`,
  `exit_force_gate`, `exit_force_gate_nonbull`, `exit_force_suppress`, `nonbull_ma_win`,
  `nonbull_persist`, `entry_skip_nonbull_persist`, `regime_index_symbol`, `early_entry_reversal`,
  `top_reversal_exit`, `entry_zX_floor`, `entry_rollover_exit`, `entry_ensemble`, `exit_ensemble`.

**Evidence (research session 2026-06-14/15):** the champion's edge is entirely rule-based —
bypassing both ML heads (`entry_threshold ≤ −3` + dormant exit head) reproduces the result
deterministically (`n2_v15_et30`, comp 435.5, seed-independent). Measured entry-score IC is +0.13
but redundant with the gates and non-stationary; the exit head IC ≈ 0. A genuine rule-only model
(`n3_rule_champ`, comp 428.5, no `.fit()`) was only possible after adding the `rule_only_no_ml`
flag — proving the gates are not separable from the ML pipeline today.

This is a **flexibility / separation-of-concerns** gap, not a correctness bug. The system works
as designed; the design simply assumed every model is an ML model.

## 2. Goals / Non-Goals

**Goals**
- A signal source (ML, rule, hybrid, constant) and the signal gates are **independent**.
- A rule-only or hybrid model can natively use the same gates as the ML champion — no flag hack.
- `engine_config` keys are split into a clear schema: *signal-gate* config vs *engine/execution*
  config (no manual `pop()` list).
- **Zero behavioural change** for every existing model (golden parity).

**Non-Goals**
- No new alpha / no model changes. This is a pure structural refactor.
- Not re-deriving the alpha/portfolio/execution tiers from scratch — align with that effort,
  don't duplicate it.
- Not removing ML — ML stays as one (optional) signal source.

## 3. Target Architecture

Standard backtest pipeline = three decoupled stages:

```
  ┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
  │  SIGNAL src │ --> │  SIGNAL GATES     │ --> │  EXECUTION ENGINE   │
  │ (ML | rule  │     │ (regime/momentum/ │     │ (fills, exits,      │
  │  | hybrid)  │     │  force-exit/raw)  │     │  costs, occupancy)  │
  └─────────────┘     └──────────────────┘     └────────────────────┘
        score/signal        buy/sell mask            trades + metrics
```

- **Signal src** produces a base buy/sell intent + optional `score`. ML is one implementation;
  a rule (or constant "always-buy") is another. Today this is the per-fold dispatch.
- **Signal gates** is a NEW shared module that takes the base signals + causal OHLCV and applies
  the entry/exit gates. Callable by ANY source. This is the logic currently trapped in
  `recombine_signals()`.
- **Engine** is unchanged: `EngineConfig` keeps only execution-level params (overext, trailing,
  market gate, pullback fill, cooldown, hold limits, hard stop, exit_priority).

## 4. Phased Plan

### Phase 0 — Bridge (DONE, behind a flag)
- Added `engine_config.rule_only_no_ml` dispatch branch (skip `.fit()`, emit constant score so
  recombine still runs its gates) + stripped the flag in the `engine_cfg.pop` block.
- Lets a rule-only model run today. **Temporary** — superseded by Phase 1. Keep until parity-tested.
- Files: `stock_ml/src/pipeline/experiment.py`. Tests: `test_pipeline_smoke.py` (10/10 pass).

### Phase 1 — Extract a shared `signals/gates.py` (RECOMMENDED FIRST REAL STEP)
- Create `stock_ml/src/signals/gates.py` exposing pure functions:
  - `apply_entry_gates(signals, gate_cfg) -> buy_mask` — moves: `entry_gate`, `entry_xs_mom_pct`,
    `entry_raw_threshold`, `entry_z_low_threshold`, `entry_zX_floor`, `entry_rollover_exit`,
    `entry_skip_nonbull_persist`, `early_entry_reversal`, the z-threshold/EMA logic.
  - `apply_exit_force_gates(signals, gate_cfg) -> sell_mask` — moves: `exit_force_gate`,
    `exit_force_gate_nonbull`, `exit_force_suppress`, `nonbull_ma_win`, `nonbull_persist`,
    `top_reversal_exit`, plus `_exit_force_mask` / `_force_healthy_mask`.
- `recombine_signals()` becomes a thin caller of these (ML path unchanged in behaviour).
- The Phase-0 rule path calls the SAME functions instead of relying on the recombine internals.
- **Critical:** move by *extraction* (cut/paste into functions), not rewrite, to preserve byte
  identity. Pin causal-z window (252) / `min_periods` (60) semantics exactly — see §6 warmup note.
- Files: new `src/signals/gates.py`; refactor `experiment.py`. Effort: ~2–3 days. Risk: medium.

### Phase 2 — Config schema split
- Introduce a typed `GateConfig` (dataclass / pydantic) holding the ~18 gate keys; `EngineConfig`
  holds only execution keys. Templates carry both explicitly; delete the manual `engine_cfg.pop`
  block.
- Add a one-time migration/validator that splits legacy `engine_config` blobs into the two.
- Files: `src/signals/gates.py`, `src/backtest/engine.py`, template loader, a migration.
  Effort: ~1–2 days. Risk: low–medium (mechanical).

### Phase 3 — (Optional, long-term) Engine-native gates
- Move execution-relevant gates fully into `EngineConfig` so the engine applies all gates for any
  signal source, eliminating the signal/engine split. **Only do this folded into the existing
  alpha→portfolio→execution refactor**, not standalone. Effort: ~1 week. Risk: high.

## 5. Golden Parity Contract (MANDATORY for every phase)

No refactor merges unless **every existing model reproduces its exact composite + trade count**.

- Build `stock_ml/tests/test_gate_parity.py`: for a frozen set of ~6 templates (the champion
  `n2_v5_ox12`, `n2_v15_et30`, `n3_rule_champ`, a hybrid, a pure-ML, a rule model), assert the
  post-refactor composite == the recorded baseline (±0.0) on seed 42.
- Snapshot baselines BEFORE Phase 1 (a JSON of `{template: (composite, trades, config_hash)}`).
- CI gate: parity test must pass. Any drift → block merge, investigate, or revert.

## 6. Known subtlety — the z-warmup

The recombine z-score uses `min_periods=60`, so the first ~60 test bars per symbol are NaN →
not bought. This silently skips early-2020 (the COVID-crash window) for z-based models, worth
~+7 composite vs a raw/rule path that trades from bar 1 (verified: the entire `et30` vs
`n3_rule_champ` gap is 51 trades in 2020 only). When unifying paths, decide explicitly:
- (a) expose it as a first-class **`min_history_bars` entry gate** (legitimate "enough data"
  rule), applied uniformly to ML and rule — recommended; OR
- (b) document that raw/rule paths trade from bar 1 and accept the difference.
Either way it must be a *declared* rule, not an accidental z side-effect.

## 7. Verification Checklist
- [ ] `pytest stock_ml/tests/test_pipeline_smoke.py` green (baseline 10/10).
- [ ] `pytest stock_ml/tests/test_gate_parity.py` green (composite drift = 0 for the frozen set).
- [ ] Re-run top-5 leaderboard templates; `config_hash` + composite unchanged.
- [ ] `n3_rule_champ` (rule path) == `n2_v15_et30` (ML-bypassed path) after the `min_history_bars`
      gate is unified.
- [ ] No `engine_cfg.pop(...)` block remains after Phase 2.

## 8. Risks & Rollback
- **Behaviour drift** (highest): mitigated by the parity test + extract-don't-rewrite discipline.
- **Hidden coupling** (entry_ensemble / exit_ensemble / early_entry_reversal read other state):
  extract these last, with their own parity cases.
- **Rollback:** each phase is a separate PR behind the parity gate; revert the PR. Phase 0's flag
  stays until Phase 1 is parity-proven, then is removed in the same PR.

## 9. Sequencing & Effort
| Phase | Scope | Effort | Risk | Gate |
|------|-------|--------|------|------|
| 0 | `rule_only_no_ml` bridge (done) | — | low | smoke pass |
| 1 | extract `signals/gates.py` | 2–3 d | med | parity test |
| 2 | `GateConfig` schema split | 1–2 d | low–med | parity test |
| 3 | engine-native gates (optional) | ~1 w | high | fold into a/p/e refactor |

**Recommendation:** ship Phase 1 + Phase 2 (the real decoupling, ~1 week incl. parity tests).
Defer Phase 3 into the alpha/portfolio/execution refactor. Remove the Phase-0 flag once Phase 1
proves parity.

## 10. Open Questions
- Adopt `min_history_bars` as a declared gate (unifies the z-warmup)? (recommended: yes)
- Where does `GateConfig` live relative to the planned alpha/portfolio tiers — is the gate stage
  part of "alpha post-processing" or "portfolio construction"? (align before Phase 2)
- Keep `model_mode="rule_only_no_ml"` as a named mode, or express rule-only purely via a rule
  entry component once gates are decoupled? (prefer the latter post-Phase-1)
