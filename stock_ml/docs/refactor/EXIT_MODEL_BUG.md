# Exit Model Bug — discovered 2026-04-27

## TL;DR

**Tất cả 11 champion (v22, v32, v34, v35b, v37a, v37a_exit, v37d, v39d, v42_a, v19_3, rule) đều
trained Model B (exit model) tốn ~30-40s/fold nhưng output `y_pred_exit` KHÔNG bao giờ được
backtest function sử dụng.** Số trade có `exit_reason='model_b_exit'` = 0 trên 100% champions
trong golden baseline.

CHAMPION_VERSIONS.md mô tả v37a_exit, v42_a là "exit_model thực sự active" — sai. Cả 2 đều
không có model_b_exit.

## Evidence

```bash
$ python -c "import pandas as pd
... for c in ['v22','v32','v34','v35b','v37a','v37a_exit','v37d','v39d','v42_a','v19_3','rule']:
...     df = pd.read_csv(f'tests/regression/golden/trades_{c}.csv')
...     print(c, (df['exit_reason']=='model_b_exit').sum())"
v22 0
v32 0
v34 0
v35b 0
v37a 0
v37a_exit 0
v37d 0
v39d 0
v42_a 0
v19_3 0
rule 0
```

## Root cause

[run_pipeline.py:354-367](../../run_pipeline.py#L354-L367):

```python
for item in prediction_cache:
    y_pred_eff = ...
    extra_kwargs = {}
    y_pred_exit = item.get("y_pred_exit")
    if y_pred_exit is not None and "y_pred_exit" in sig_params:  # ← SILENT GATE
        extra_kwargs["y_pred_exit"] = y_pred_exit
    r = backtest_fn(
        y_pred_eff, item["returns"],
        item["sym_test_df"], item["feature_cols"],
        **mod_kwargs, **extra_kwargs,
    )
```

Gate `"y_pred_exit" in sig_params` chỉ pass khi backtest function có khai báo tham số
`y_pred_exit` RÕ RÀNG trong signature. Nhưng:

| Wrapper | Signature | y_pred_exit trong sig? | Forward `**kwargs` to engine? |
|---------|-----------|----------------------:|-------------------------------:|
| `backtest_v22` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v32` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v34` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v35b` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v37a` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v37d` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v39d` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v42` | `(y_pred, returns, df_test, feature_cols, **kwargs)` | ❌ | ✅ |
| `backtest_v19_3` (legacy) | `(y_pred, returns, df_test, feature_cols, ...)` | ❌ | ❌ |
| `backtest_unified` (engine) | `(y_pred, returns, df_test, feature_cols, y_pred_exit=None, **config)` | ✅ | — |

→ Wrapper modern dùng `**kwargs` cho flexibility nhưng signature gate ở pipeline check
explicit only → False → `y_pred_exit` bị drop ngay tại pipeline level → engine nhận `None` →
code path `model_b_exit` ở [engine.py:737-740](../../src/backtest/engine.py#L737-L740) không
bao giờ trigger.

## Impact

1. **Score không bị inflate** — kết quả golden phản ánh đúng performance của entry model + non-ML fusion strategies (fast_exit_loss, signal_hard_cap, v32_hap_preempt, peak_protect_ema, …). Không có "data leak" hay "đo nhầm".

2. **Tài liệu CHAMPION_VERSIONS.md cần sửa** — không có version nào "exit_model thực sự active". Cả 11 đều training Model B nhưng vứt output đi.

3. **Lãng phí compute** — mỗi pipeline run tốn ~30-40s/fold × 5 fold × 7 group training = ~1000s wasted training Model B (cho dataset hiện tại).

4. **Tiềm năng cải thiện**:
   - Nếu Model B có predictive power → fix bug có thể cải thiện score thêm
   - Nếu Model B = noise → bỏ training để tiết kiệm compute

## Fix plan

**Không fix ngay**. Quyết định: document bây giờ, fix ở **Phase 2** (fusion stack refactor).
Lý do:
- Golden baseline hiện tại phản ánh đúng behavior pre-refactor (kể cả bug). Refactor mục
  tiêu là "không thay đổi behavior" → fix bug = thay đổi behavior → cần golden mới.
- Phase 2 sẽ thiết kế lại fusion stack với `ml_exit_model` là 1 strategy explicit. Lúc đó
  mới có infrastructure để test "Model B có giá trị không" trên đa dạng tổ hợp.

## Fix sketch (cho Phase 2)

Option A — pipeline-level:

```python
# run_pipeline.py:361 — softer gate
if y_pred_exit is not None and (
    "y_pred_exit" in sig_params or
    any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
):
    extra_kwargs["y_pred_exit"] = y_pred_exit
```

→ Pass nếu wrapper có `**kwargs` (flow vào engine qua kwargs forwarding). v19_3 vẫn skip
(không có **kwargs).

Option B — Phase 2 architecture: bỏ qua gate này hoàn toàn. `BacktestStrategy` mới có
explicit `y_pred_exit` parameter trong base class.

## Tasks for Phase 2

- [ ] Implement `ml_exit_model` fusion strategy
- [ ] Audit Model B `y_pred_exit` distribution: how often is `y_pred_exit == 1`? Class
      imbalance? Pure noise?
- [ ] Decide: keep Model B training or drop it
- [ ] If keep: regenerate golden post-fix, label as "v2 baseline"
- [ ] Update CHAMPION_VERSIONS.md to remove "exit model active" claim
