# Feature Store + Expression DSL — Thiết kế & Các bước triển khai

> **Trạng thái**: DONE (đã implement Phase 1–6) — quyết định kiến trúc 2026-06-01.
> Thay thế hệ thống feature hiện tại đang phân mảnh ở 3 nơi không đồng bộ.
> Tài liệu này là nguồn chân lý (SoT) cho việc triển khai. Cập nhật `Trạng thái` từng phase khi hoàn thành.
>
> 🆕 **Cập nhật 2026-07-31 — HỒ SƠ THIẾT KẾ (đã xong), không phải hướng dẫn vận hành.** SoT thật của feature là `src/features/catalog.py`. ⚠️ Script `seed_features.py` (§3.4, §6 bước 2) **đã mất** — chưa từng được git track, chỉ còn `.pyc`; **mọi lệnh `python -m stock_ml.scripts.seed_features` KHÔNG còn chạy được**. Bảng mirror `feature_def`/`feature_set` là tùy chọn (read-only cho API/UI), không phải bước bắt buộc.

---

## 0. Bối cảnh & Mục tiêu

### Hiện trạng (vấn đề cần giải)

Feature set hiện được định nghĩa ở **3 nơi lệch nhau**:

| Nơi | File | Set có | Vấn đề |
|---|---|---|---|
| Python registry (runtime thật) | `stock_ml/src/features/registry.py` | `basic_v1`, `leading_v2`, `leading_v3` | builder monolithic, list cột cứng |
| YAML config (mồ côi, không code nào load) | `stock_ml/config/feature_sets/*.yaml` | + `leading`, `leading_v4`, `leading_deriv` | "blocks" chỉ là comment, không phải code |
| DB catalog (API/UI đọc) | bảng `feature_set_catalog` | `basic_v1/v2/v3` | đếm cột sai (5/35/57), cột rỗng `[]` |

Hệ quả: số cột lệch (basic_v1 thực 8 ≠ catalog 5; leading_v3 thực 54 ≠ docstring 58 ≠ catalog 57); `leading_v4/leading/leading_deriv` chọn là **crash** (`apply_features` KeyError); feature dùng chung **bị lưu lặp** giữa các set cache; cross-sectional có **bfill leakage** (lấp warmup bằng giá trị tương lai).

### Mục tiêu (3 fork đã chốt)

1. **Engine = Expression DSL kiểu Qlib** — feature là công thức chuỗi, có engine parse → AST → eval với computation graph chia sẻ sub-expression.
2. **Storage = feature store per-feature** — mỗi feature materialize 1 lần, content-addressed; set = JOIN các cột member → 0 lặp.
3. **Rollout = big-bang** — thay registry+YAML+catalog cũ bằng DB-driven resolver, regenerate golden, xoá legacy sau khi parity test pass.

### Nguyên tắc

> Feature = **công thức** (lưu DB), không phải code. Giá trị materialize **1 lần/feature** (content-addressed). Feature set = **danh sách tham chiếu**. → chia sẻ feature là miễn phí.

---

## 1. Kiến trúc đích (3 lớp)

```
┌─ Definition (DB) ───────────┐   ┌─ Compute (engine) ──────────┐   ┌─ Storage (feature store) ──┐
│ feature_def   (expr, kind)  │   │ Parser → AST                 │   │ store/features/<expr_hash>/ │
│ feature_dep   (DAG)         │ → │ Operator registry            │ → │   <data_version>.parquet     │
│ feature_set                 │   │ Resolver (topo-sort, cache)  │   │ [symbol, date, value]        │
│ feature_set_member (M:N)    │   │ groupby(symbol|date|join)    │   │ content-addressed → dedup    │
│ feature_materialization     │   └──────────────────────────────┘   └──────────────────────────────┘
└─────────────────────────────┘
```

---

## 2. Đặc tả DSL

### 2.1 Cú pháp

- `$field` — raw OHLCV (`$open $high $low $close $volume`) hoặc trường ngoài (`$market_close`, `$sector`).
- `#feature_name` — tham chiếu feature khác đã định nghĩa (tạo cạnh DAG). (Cho phép viết trực tiếp tên đã đăng ký.)
- Literal số: `14`, `2.0`, `1e-8`.
- Toán tử: `+ - * /`, so sánh `> < >= <= == !=` (trả bool→float).
- Gọi hàm: `Func(arg1, arg2, ...)`, hỗ trợ kwarg dạng `by=$sector`.

### 2.2 Ví dụ (port từ feature hiện tại)

```
ret_5d          = $close / Ref($close, 5) - 1
sma_20_ratio    = $close / Mean($close, 20) - 1
rsi_14          = RSI($close, 14)
macd_hist       = MACD($close, 12, 26, 9).hist
atr_14_ratio    = ATR($high, $low, $close, 14) / $close
bb_pct_20       = Bollinger($close, 20, 2).pct
momentum_rank   = CSRank(#ret_20d)
return_vs_sector = #ret_20d - CSGroupMedian(#ret_20d, by=$sector)
market_trend    = (($market_close > Mean($market_close, 200))) * 1.0
```

### 2.3 Operator registry (mỗi op: `name`, `arity`, `axis`)

| Nhóm | Toán tử | Axis |
|---|---|---|
| Elementwise | `+ - * /`, `Abs Sign Log Clip Max Min`, so sánh | none |
| Time-series (per-symbol) | `Ref Mean Std Sum Max Min Delta Pct Quantile EMA TsRank Corr` | groupby(symbol) |
| Indicator built-in | `RSI ADX ATR MACD MFI OBV Bollinger ROC` (smoothing Wilder đúng) | groupby(symbol) |
| Cross-sectional (per-date) | `CSRank CSZScore CSMean CSStd CSGroupMedian(by=)` | groupby(date) |

> **Quyết định**: indicator phức tạp (RSI/ADX/MFI/MACD/ATR/Bollinger) là **op built-in**, KHÔNG phân rã ra DSL thuần — vì smoothing Wilder + ADX lồng nhau khó viết đúng. Qlib cũng làm vậy. Parity test (Phase 5) bảo chứng số khớp builder cũ.

### 2.4 Ba `kind` ↔ ngữ cảnh eval

| `kind` | Định nghĩa | Eval | Yêu cầu data |
|---|---|---|---|
| `per_symbol` | "công thức suy ra được" từ chuỗi của chính mã | `groupby(symbol)` | chỉ OHLCV mã đó |
| `cross_sectional` | "cần cả vũ trụ tại mỗi ngày" | `groupby(date)` | toàn bộ universe |
| `market` | so với chuỗi ngoài (index) | join `$market_*` theo date | index series |

`kind` được **suy ra tự động** từ op trong AST (op cross-sectional/market → set kind tương ứng), lưu vào `feature_def.kind` để query nhanh. → đây chính là phần trả lời "tách non-cross vs cần cả chuỗi": nó là metadata, không phải tách thủ công.

---

## 3. DB Schema (chuẩn hoá)

```sql
-- Định nghĩa nguyên tử (1 feature = 1 dòng)
feature_def(
  id PK, name UNIQUE, expr TEXT, kind ENUM(per_symbol|cross_sectional|market),
  output_dtype, description, version INT, expr_hash CHAR(40),
  is_active BOOL, created_at, updated_at)

-- DAG, trích từ parser — cho topo-sort + impact analysis
feature_dep(
  feature_id FK, depends_on_feature_id FK NULL, depends_on_raw VARCHAR NULL)  -- 'close' | feature khác

-- Bộ feature có tên
feature_set(id PK, name UNIQUE, version INT, description, is_active, created_at)

-- M:N  ← DEDUP ĐỊNH NGHĨA nằm ở đây
feature_set_member(feature_set_id FK, feature_id FK, position INT,
  UNIQUE(feature_set_id, feature_id))

-- Index của feature store  ← DEDUP GIÁ TRỊ
feature_materialization(
  feature_id FK, expr_hash, data_version, storage_uri,
  rows INT, engine_version, computed_at,
  UNIQUE(feature_id, expr_hash, data_version))
```

- Bỏ `column_count` cứng — đếm bằng `COUNT(member)` (diệt nguồn lệch 5/8/54/57/58).
- `expr_hash` = `sha1(canonical_AST + hash đệ quy các dep + engine_version)`. Đổi 1 op ⇒ hash đổi ⇒ downstream tự invalidate.
- `data_version` = fingerprint `(universe + timeframe + khoảng ngày + nguồn data)` — tái dùng cơ chế ở `stock_ml/src/cache/feature_cache.py`.
- FK đổi: `strategy_templates.feature_set_id` → `feature_set.id` (RESTRICT).

---

## 4. Feature store (giá trị)

- **Metadata** (5 bảng §3): DB SQLAlchemy hiện tại (SQLite/Postgres).
- **Giá trị**: parquet content-addressed: `results/cache/features/store/<expr_hash>/<data_version>.parquet`, schema `[symbol, date, value]`.
  - `rsi_14` trên cùng universe/khoảng = **1 file vật lý**, mọi set trỏ tới.
  - Set matrix = DuckDB JOIN các file member trên `(symbol, date)` — lazy, không nhân bản cột.
- So hiện trạng (`cache/features/leading_v2/*.parquet` 35 cột + `leading_v3/*.parquet` 54 cột → 35 cột chung lưu 2 lần): mới = **0 lặp**.

---

## 5. Các bước triển khai (phased build, big-bang cutover)

> Mỗi phase có cổng nghiệm thu. KHÔNG xoá legacy cho đến Phase 5 (parity pass).

### Phase 1 — DSL engine (offline, chưa đụng DB/pipeline) · Trạng thái: ✅

| # | Việc | File |
|---|---|---|
| 1.1 | Lexer + parser → AST | `stock_ml/src/features/dsl/parser.py` (mới) |
| 1.2 | Node types (Field, Const, BinOp, UnaryOp, Call) | `stock_ml/src/features/dsl/ast.py` (mới) |
| 1.3 | Operator registry (elementwise + ts + indicator + cross-sectional) | `stock_ml/src/features/dsl/ops.py` (mới) |
| 1.4 | Evaluator (eval AST trên DataFrame, dispatch theo axis) | `stock_ml/src/features/dsl/engine.py` (mới) |
| 1.5 | Suy ra `kind` + trích dependency từ AST | `stock_ml/src/features/dsl/engine.py` |
| 1.6 | `canonicalize()` + `expr_hash()` | `stock_ml/src/features/dsl/hashing.py` (mới) |
| 1.7 | Unit test parser/ops + golden cho từng op | `stock_ml/tests/features/test_dsl_*.py` (mới) |

**Nghiệm thu**: `pytest stock_ml/tests/features/ -q` xanh; eval `RSI($close,14)` khớp `leading_v2._rsi` (tol 1e-9).

### Phase 2 — Feature store · Trạng thái: ✅

| # | Việc | File |
|---|---|---|
| 2.1 | `FeatureStore`: load/save parquet content-addressed | `stock_ml/src/features/store.py` (mới) |
| 2.2 | `data_version` fingerprint (universe+timeframe+range+source) | `stock_ml/src/features/store.py` |
| 2.3 | Atomic write (tempfile→replace) + đọc lazy DuckDB | `stock_ml/src/features/store.py` |
| 2.4 | Test save/load + hit/miss | `stock_ml/tests/features/test_store.py` (mới) |

**Nghiệm thu**: ghi `rsi_14` 1 lần, 2 lần load là cache hit; file đặt đúng `store/<expr_hash>/<data_version>.parquet`.

### Phase 3 — DB schema + repo + seed · Trạng thái: ✅

| # | Việc | File |
|---|---|---|
| 3.1 | SQLAlchemy models (5 bảng §3) | `stock_ml/db/models/feature.py` (mới), export ở `db/models/__init__.py` |
| 3.2 | Alembic migration: tạo 5 bảng, **drop** `feature_set_catalog`, repoint FK templates | `stock_ml/db/migrations/versions/00XX_feature_store.py` (mới) |
| 3.3 | Repos: `FeatureDefRepository`, `FeatureSetRepository` | `stock_ml/db/repositories/feature_repo.py` (mới) |
| 3.4 | Seed: viết expr DSL cho **toàn bộ** feature leading_v2/v3 + dựng các set (`basic_v1`, `leading_v2`, `leading_v3`, `leading_deriv`, `leading_v4`) | `stock_ml/scripts/seed_features.py` (mới) |

**Nghiệm thu**: `alembic upgrade head` ok; `seed_features.py` chạy xong; `SELECT COUNT(*) FROM feature_set_member` khớp số cột mong đợi.

### Phase 4 — Resolver + wire pipeline · Trạng thái: ✅

| # | Việc | File |
|---|---|---|
| 4.1 | `FeatureResolver.materialize_set(set_name, ctx)`: load members → DAG → topo-sort → eval theo kind → store cache → JOIN matrix | `stock_ml/src/features/resolver.py` (mới) |
| 4.2 | Thay `apply_features`/`get_feature_cols` bằng resolver | `stock_ml/src/pipeline/experiment.py` (≈ dòng 740-794) |
| 4.3 | **Bỏ** `_FEATURE_SET_RANK` + `_pick_richest_feature_set` (thừa) | `stock_ml/src/pipeline/experiment.py` (≈ 655-671) |
| 4.4 | Per-slot entry/exit resolve = union member, materialize 1 lần | `stock_ml/src/pipeline/experiment.py` |

**Nghiệm thu**: chạy 1 backtest mẫu qua resolver không lỗi; feature dùng chung giữa entry/exit set chỉ tính 1 lần (log cache hit).

### Phase 5 — Parity gate + dọn legacy · Trạng thái: ✅

| # | Việc | File |
|---|---|---|
| 5.1 | Parity test: output resolver == builder cũ (tol float) trên symbols/ngày golden, **cho mỗi set** | `stock_ml/tests/features/test_parity.py` (mới) |
| 5.2 | Sửa **bfill leakage**: cross-sectional rank để NaN warmup, fail-loud (không bfill) | trong DSL op `CSRank` (§2.3) |
| 5.3 | Sau khi parity pass: xoá `registry.py`, `basic.py`, `leading_v2.py`, `leading_v3.py`, `config/feature_sets/*.yaml`, `init_feature_sets()` trong `import_yaml_templates.py` | nhiều file |
| 5.4 | Regenerate golden + cập nhật baseline snapshot test | `stock_ml/tests/test_baseline_snapshot.py` |

**Nghiệm thu**: parity test xanh TRƯỚC khi xoá; sau khi xoá toàn bộ `pytest` xanh; golden regenerate có review.

### Phase 6 — API + UI navbar · Trạng thái: ✅

#### 6a. API router mới `/api/v1/features`

| # | Việc | File |
|---|---|---|
| 6a.1 | Router `features` + endpoints (xem §6 dưới) | `stock_ml/api/routes/features.py` (mới) |
| 6a.2 | Đăng ký import | `stock_ml/api/routes/__init__.py` |
| 6a.3 | `app.include_router(routes.features.router, tags=["features"])` | `stock_ml/api/main.py` (sau dòng 78) |

Endpoints (**read-only** — `catalog.py` là SoT, không tạo feature qua API):
```
GET  /api/v1/features/definitions            # list (filter kind, search name) → name, kind, version, used_by_count
GET  /api/v1/features/definitions/{id}       # detail: expr, deps, sets dùng nó, trạng thái materialize
POST /api/v1/features/validate               # parse 1 expr (stateless) → {valid, kind, deps, error} cho UI check
GET  /api/v1/features/sets                   # list set + member_count
GET  /api/v1/features/sets/{id}              # members (theo position)
GET  /api/v1/features/materializations       # trạng thái feature store (đã/chưa materialize)
```
> **SoT một chiều**: định nghĩa feature viết trong code `src/features/catalog.py`; `seed_features.py` chiếu một chiều xuống DB (`feature_def`/`feature_set`) làm mirror read-only cho API/UI; resolver đọc thẳng catalog. Không có endpoint ghi → DB không thể lệch khỏi catalog backtest đang dùng. (Đây là chuẩn các tổ chức algo: alpha/feature định nghĩa trong version control để reproducibility.)

#### 6b. Trang UI + gắn navbar

| # | Việc | File |
|---|---|---|
| 6b.1 | Trang `Feature Store`: tab **Features** (bảng: name, kind badge, expr, version, used-by) + tab **Sets** (bảng: name, member_count, mở xem members) + form tạo feature có **live validate** gọi `POST /validate` | `stock_ml/dashboard/feature-store.html` (mới) |
| 6b.2 | Include chuẩn: `<script src="js/nav.js"></script>` + `<script src="api-config.js"></script>` (theo mẫu `universe.html`) | `feature-store.html` |
| 6b.3 | **Gắn navbar**: thêm vào mảng `PAGES` | `stock_ml/dashboard/js/nav.js` |

Thêm vào `js/nav.js` (sau dòng `template-builder.html`):
```js
{ href: 'feature-store.html', label: '🧬 Feature Store' },
```

**Nghiệm thu**: mở `feature-store.html`, navbar hiện tab "🧬 Feature Store" active; bảng load từ `/api/v1/features/definitions`; gõ expr sai → UI báo lỗi từ `/validate`.

---

## 6. Migration & Validation tổng

1. `alembic upgrade head` (Phase 3) — tạo bảng, drop `feature_set_catalog`, repoint FK.
2. `python -m stock_ml.scripts.seed_features` — nạp feature_def + set.
3. Parity test (Phase 5) là **cổng chặn** trước khi xoá builder cũ.
4. Regenerate golden (đã được chấp nhận trong quyết định refactor trước).
5. Smoke: `pytest stock_ml/tests/test_pipeline_smoke.py` + 1 backtest thật.

---

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Indicator built-in port sai (Wilder smoothing) | Parity test per-op + per-set, tol float, **trước** khi xoá legacy |
| Big-bang vỡ pipeline 1 nhịp | Build theo phase, chỉ cutover ở Phase 5 sau khi parity xanh |
| `expr_hash` không canonical (a+b ≠ b+a) | Chuẩn hoá AST (whitespace + commutative) trước hash |
| Cross-sectional phụ thuộc universe | `data_version` PHẢI gồm universe; fail-loud nếu thiếu mã |
| Leakage tái diễn | Cấm bfill/ffill xuyên tương lai trong op; để NaN warmup, fail-loud |

---

**Liên quan**: `docs/STORAGE_ARCHITECTURE_REFACTOR.md` (Postgres meta + DuckDB values), `docs/EXTENSIBILITY_GUIDE.md` §2 (sẽ thay), `docs/ARCHITECTURE.md` (Feature Pipeline).

**Last Updated**: 2026-06-02 · **Status**: DONE (Phase 1–6 implemented + post-review hardening, 216 tests green)

> **Ghi chú triển khai** (lệch nhỏ so với bản thiết kế):
> - `leading_v2` thực tế có **36** feature (docstring cũ ghi 35 — chính là lỗi đếm refactor này sửa); `leading_v3` = **55**.
> - **SoT = `stock_ml/src/features/catalog.py` (code).** Resolver đọc trực tiếp; `seed_features.py` chiếu một chiều xuống DB làm mirror read-only cho API/UI. **Quyết định (2026-06-02): giữ code là SoT** (chuẩn tổ chức algo — định nghĩa trong version control để reproducibility) và **bỏ đường ghi** (`POST /definitions`, `POST /sets`) + form tạo ở UI để DB không thể lệch khỏi catalog.
> - Market/sector features (`leading_v3/v4`): pipeline tự dựng `sector_map` ([sectors.py](../stock_ml/src/features/sectors.py)) + **equal-weight market index** ([market.py](../stock_ml/src/features/market.py), thay hành vi cũ điền 0) khi `resolver.required_raw_inputs()` báo set cần; thiếu thì **fail-loud**. `leading_v2`/`basic_v1` không bị ảnh hưởng.
> - **Cache reproducibility**: `data_version` gồm `content_fingerprint` (nội dung OHLCV/market/sector) + `engine_code_fingerprint` (hash mã `dsl/*.py`) → sửa dữ liệu hoặc operator đều tự invalidate store.
> - `init_feature_sets()` trong `import_yaml_templates.py` đổi thành lookup theo tên (không xoá hẳn). Các script ad-hoc ở root `scripts/` (create_macd_template, seed_db_macd, migrate_tech_rules) tham chiếu symbol cũ — để nguyên, ngoài phạm vi.
