# Champion Versions — 11 versions to port to new architecture

## Tiêu chí chọn

1. **Diversity**: đại diện cho mỗi nhóm chiến lược (legacy / mid / modern)
2. **Score relevance**: top performers hoặc benchmark được tham khảo nhiều
3. **Architectural breadth**: cover tất cả feature_set + target combinations
4. **Special cases**: có version đặc thù (GRU, dual-target, exit_model thật)

## Danh sách 11 champion

| # | Version | Score | Feature | Target | Model | Exit | Lý do giữ |
|---|---------|------:|---------|--------|-------|------|-----------|
| 1 | **v22** | 686.3 | leading_v2 | trend_regime | LightGBM | none | Top score overall, đại diện legacy `experiments/run_v22_final.py` |
| 2 | **v32** | 353.4 | leading_v3 | early_wave | LightGBM | none | Đại diện `leading_v3` family + sophisticated exit logic (HAP, SED, SHEF) |
| 3 | **v34** | 598.4 | leading_v4 | early_wave | LightGBM | none | Baseline cho `leading_v4` family (HA features), tiền đề cho v35-v39 |
| 4 | **v35b** | 603.7 | leading_v4 | early_wave | LightGBM | yes | Engine reform (single-bar signal, rule_override) — đại diện đột phá engine |
| 5 | **v37a** | 603.4 | leading_v4 | early_wave | LightGBM | yes | Per-profile dispatch (bank/momentum/etc) — đại diện symbol-profile pattern |
| 6 | **v37a_exit** | TBD | leading_v4 | early_wave_dual | LightGBM | **yes (active)** | Exit model **thực sự hoạt động** — duy nhất có Model B impact thực |
| 7 | **v37d** | 315.6 | leading_v4 | early_wave | **GRU** | yes | Đại diện non-LightGBM path (PyTorch sequence model) |
| 8 | **v39d** | 611.7 | leading_v4 | early_wave | LightGBM | yes | Top trong v39 family + per-symbol rule-exit hybrid (12 stable-trend symbols) |
| 9 | **v42_a** | TBD | leading_v4 | early_wave_dual | LightGBM | **yes (active)** | Exit model mới nhất (fw=15), engine V37a + Model B override |
| 10 | **v19_3** | 307.4 | leading | trend_regime | LightGBM | none | Legacy boundary case — đại diện `src/strategies/legacy.py` (binary position sizing) |
| 11 | **rule** | — | — | — | — | — | Non-ML baseline (rule-based) — bắt buộc giữ để so sánh |

## Vùng coverage

### Feature sets covered
- ✅ `leading` (v19_3)
- ✅ `leading_v2` (v22)
- ✅ `leading_v3` (v32)
- ✅ `leading_v4` (v34, v35b, v37a, v37a_exit, v37d, v39d, v42_a)

### Targets covered
- ✅ `trend_regime` (v22, v19_3)
- ✅ `early_wave` (v32, v34, v35b, v37a, v37d, v39d)
- ✅ `early_wave_dual` (v37a_exit, v42_a)

### Model types covered
- ✅ LightGBM (8 versions)
- ✅ GRU (v37d)
- ⚠️ XGBoost / CatBoost / RandomForest: chưa champion, sẽ thêm khi research

### Exit model coverage
> **⚠️ Bug discovered 2026-04-27**: tất cả 11 champion (kể cả v37a_exit và v42_a) đều có
> `exit_reason='model_b_exit'` count = 0 trong golden. Exit model trained xong nhưng pipeline
> drop output trước khi vào backtest. Chi tiết: [EXIT_MODEL_BUG.md](EXIT_MODEL_BUG.md). Fix ở
> Phase 2 khi rebuild fusion stack.

- exit_model trained NHƯNG output bị drop ở pipeline gate (v22, v32, v34, v35b, v37a, v37a_exit, v37d, v39d, v42_a, v19_3)
- Không train exit_model: rule

### Fusion patterns covered
- ✅ Simple ML-only (v22, v19_3)
- ✅ ML + rule ensemble (v34, v35b)
- ✅ Per-profile dispatch (v37a)
- ✅ Per-symbol routing (v39d)
- ✅ Anti-fomo + co-pilot exit (v38d patterns — covered by v34's lineage)
- ✅ HAP preempt + SED + SHEF (v32)

## Versions retired (49 versions)

Chia theo nhóm:

### Group A — Variants of champions (15 versions)
Khác champion về 1 vài flag, không thêm insight quan trọng:
- **v23, v24, v25, v26, v27, v28**: variants of v22 (cùng leading_v2, trend_regime). v22 đã đại diện.
- **v29, v30, v31, v33**: variants of v32 (cùng leading_v3, early_wave). v32 đã đại diện top features.
- **v36a, v36b, v36c**: variants of v34 (rule_override variants). v34 + v35b đã đại diện.
- **v35a, v35c**: variants of v35b. v35b đã đại diện.

### Group B — Failed experiments (8 versions)
Có `retired_reason` rõ ràng, không cần port:
- **v37b, v37c**: retired (dual-head + per-profile threshold không vượt v34)
- **v33**: identical to v32 (retired_reason)
- **v23**: superseded by V24
- **v25**: superseded by V26
- **v18, v19, v19_1, v19_2, v19_4, v20, v21**: legacy retired

### Group C — Old versions không còn relevance (16 versions)
- **v11, v12, v13, v14, v15, v16, v17**: pre-v18 era
- **v38b, v38c, v38d, v38b2, v38b3, v38c2, v38d2, v38e, v38bc, v38bd, v38cd, v38bcd**: V38 family ablation. Insights đã merge vào v34/v37a.

### Group D — Dashboard versions (10 versions)
Có trades_vXX.csv nhưng không trong models.yaml:
- **v40, v41, v43, v44, v45, v46, v47, v48, v49**: experimental, results-only
- **v22_full**: full-symbol variant of v22

→ KHÔNG port qua kiến trúc mới. Vẫn giữ trades CSV trong `results/` cho dashboard.

## Strategy: legacy adapter

49 retired versions sẽ chạy qua **legacy adapter** trong giai đoạn refactor:

```
legacy/
├── README.md                # Liệt kê 49 versions + reason for retirement
├── adapter.py                # Wraps old functions to new pipeline
├── strategies/               # Original code (frozen)
│   ├── run_v23_optimal.py
│   ├── run_v24.py
│   └── ...
└── configs/                  # Frozen old YAML entries
    ├── v23.yaml
    └── ...
```

**Khi nào cần re-run version legacy**:
1. So sánh kết quả nghiên cứu mới vs lịch sử
2. Verify regression (ngoài 11 champion)
3. Sau này quyết định promote 1 legacy lên champion

**Cú pháp**:
```bash
python -m stock_ml run legacy/v25 --device gpu
```

## Future promotion process

Nếu sau này research mới phát hiện 1 retired version có insight quan trọng:

1. Run legacy adapter để re-validate kết quả
2. Migration tool: `python -m stock_ml migrate-legacy v25` → tự động tạo `champions/v25.yaml`
3. Manual review: kiểm tra fusion strategies map đúng
4. Add to regression test
5. Promote vào `champions/` folder

Mất ~30 phút thay vì viết lại từ đầu.

## Quy trình tạo Golden Baseline

Trước khi viết refactor code:

```bash
# 1. Đảm bảo seed cố định
# Sửa src/models/registry.py: random_state=42

# 2. Set GRU deterministic
# Sửa GRU model: torch.manual_seed(42), cudnn.deterministic=True

# 3. Run 11 champions với --force
python run_pipeline.py --version v22 --compare v32,v34,v35b,v37a,v37a_exit,v37d,v39d,v42_a,v19_3,rule --device gpu --force

# 4. Lưu golden
mkdir tests/regression/golden
cp results/trades_v22.csv tests/regression/golden/
cp results/trades_v32.csv tests/regression/golden/
# ... 11 files

# 5. Hash từng file
cd tests/regression/golden
sha256sum trades_*.csv > checksums.txt

# 6. Run lại để verify reproducibility
cd ../../..
python run_pipeline.py --version v22 --compare ... --force
diff results/trades_v22.csv tests/regression/golden/trades_v22.csv
# Phải giống y hệt
```

## Acceptance criteria cho từng champion port

Một champion được coi là "ported" khi:

1. ✅ File YAML mới ở `config/experiments/champions/<name>.yaml`
2. ✅ Run qua pipeline mới: `python -m stock_ml run champions/<name>`
3. ✅ Output `results/trades_<name>.csv` match golden 100% (hash giống nhau)
4. ✅ Composite score match golden (tolerance 0.0)
5. ✅ Có unit test ở `tests/regression/test_champion_<name>.py`
6. ✅ Document mapping: cũ → mới (params nào → fusion strategy nào)

Champion FAIL acceptance criteria → không merge, tiếp tục debug.
