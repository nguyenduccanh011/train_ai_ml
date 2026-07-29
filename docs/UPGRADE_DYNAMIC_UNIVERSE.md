# Nâng cấp Engine: chạy Universe Động point-in-time trong 1 run

> Mục tiêu: model universe-động (top-N thanh khoản point-in-time, khác symbols mỗi fold)
> chạy trong **1 run walk-forward liền mạch** — thay vì 6 run per-year ghép offline.
> Ngày lập: 2026-07-28. **Bổ sung 2026-07-29: §7 — 4 yêu cầu phía ĐĂNG KÝ** (bắt buộc đọc
> cùng 5 bước engine; đăng ký overlay xem thêm
> [refactor/PORTFOLIO_REGISTRATION_PIPELINE.md](refactor/PORTFOLIO_REGISTRATION_PIPELINE.md)).

---

## 1. Vì sao cần

**Hiện trạng**: `run_template` nhận **1 danh sách symbols cố định** cho cả run.
- Universe động → phải chạy 6 run riêng (`_dyn900_2020..2025`) rồi ghép trades offline + overlay portfolio.
- Hệ quả: KHÔNG map vào 1 champion row leaderboard; `score_nav_leaderboard.py` chấm per-fold không phản ánh model ghép.

**Sau nâng cấp**: 1 run 2020-2025, mỗi fold tự chọn universe point-in-time → đăng ký leaderboard chuẩn như champion 61-mã.

**Bằng chứng cần point-in-time** (không phải tĩnh): static-900 (universe cố định ADV toàn kỳ) CAGR 301% vs dynamic-900 (point-in-time) 205% — chênh 96pp là **look-ahead bias** (static biết trước mã nào thanh khoản tương lai). Nâng cấp phải giữ point-in-time để số hợp lệ.

---

## 2. Kiến trúc hiện tại (điểm chạm)

```
run_template.py
  → get_pipeline_symbols()            # resolve 1 danh sách symbols cố định
  → run_experiment(cfg, symbols)
       → loader.load_many(requested)  # load OHLCV cho symbols cố định  [experiment.py:3206]
       → build_feature_frame(...)      # feature + target
       → splitter.split(df)            # walk-forward, mask THEO THỜI GIAN  [splitter.py:135-144]
            for w in windows():
              train_mask = time in [train_start, train_end)   # KHÔNG lọc universe
              test_mask  = time in [test_start, test_end)
       → _load_xsec_features(duck=full-panel)  # cross-sectional rank vs TOÀN panel [experiment.py:1681]
```

**3 chỗ cần universe-aware**: (a) resolve universe per-fold, (b) mask train/test theo universe fold, (c) cross-sectional rank theo universe fold.

---

## 3. Kế hoạch sửa (5 bước)

### Bước 1 — Config: `universe_policy` thay symbols cố định

**File**: `stock_ml/src/pipeline/experiment.py` (`ExperimentConfig`, dòng ~39-372) + `strategy_templates.universe_slug`.

Thêm trường:
```python
universe_policy: dict | None = None
# vd: {"mode": "dynamic_topn", "n": 900, "metric": "adv",
#      "lookback": "prior_year", "min_sessions": 100,
#      "exclude": ["index", "derivative"]}
```
- `None` → giữ hành vi cũ (symbols cố định), **backward-compatible**.
- Khi set → engine resolve universe per-fold (bước 2).

### Bước 2 — Resolver universe point-in-time

**File mới**: `stock_ml/src/data/universe_resolver.py`

```python
def resolve_universe(policy: dict, test_year: int, duck: str) -> list[str]:
    """top-N tradable equities by prior-year ADV. Causal (chỉ dùng data < test_year).
    Loại index + derivative. KHÔNG ép champion (fully dynamic)."""
    # với dynamic_topn + lookback=prior_year:
    #   ADV = avg(volume*close) trong năm (test_year - 1)
    #   filter: count(*) >= min_sessions, not is_nonstock(symbol)
    #   return top-N by ADV
```
Port logic từ `_dynuniverse_clean.py::uni()` (đã kiểm chứng clean, causal).

### Bước 3 — Splitter universe-aware

**File**: `stock_ml/src/data/splitter.py` (`YearSplitter.split`, dòng 135-144).

Thêm param + filter:
```python
def split(self, df, date_col="date", universe_policy=None, duck=None):
    for w in self.windows():
        uni = resolve_universe(universe_policy, w.test_year, duck) if universe_policy else None
        train_mask = (dates >= w.train_start) & (dates < w.train_end)
        test_mask  = (dates >= w.test_start)  & (dates < w.test_end)
        if uni is not None:
            sym_in = df["symbol"].isin(uni)
            train_mask &= sym_in; test_mask &= sym_in   # CHỈ universe của fold
        yield w, df.loc[train_mask], df.loc[test_mask]
```
**Data load**: `load_many` phải load **union mọi fold-universe** (để có đủ bars), splitter mask xuống fold. Sửa `experiment.py:3206` để `requested = union các resolve_universe(year) ∀ year`.

### Bước 4 — Cross-sectional rank theo universe fold

**File**: `experiment.py` — `_load_xsec_features` (1681), `_load_market_breadth` (1641), `_load_regime_index` (1625).

Hiện rank trên TOÀN panel (`SELECT ... FROM ohlcv`, không filter). Với universe động phải rank trên **universe của fold đang xử lý** để nhất quán train↔serving:
```python
def _load_xsec_features(metrics, duck=..., universe_by_year=None):
    # nếu universe_by_year set: rank riêng từng năm trên đúng universe năm đó
    #   piv_year = piv[cols ∈ universe_by_year[year]]  cho mỗi năm
    #   rank(axis=1) trên piv_year
```
**Rủi ro**: đây là chỗ tinh tế nhất — conviction (cs5_ma50) là rank cross-sectional; nếu panel đổi giữa train và serving → conviction lệch. Phải đảm bảo serving dùng **cùng universe policy**.

### Bước 5 — Score NAV cho run động

**File**: `stock_ml/scripts/ops/score_nav_leaderboard.py`.

Hiện chấm per-run K25 equal-weight. Với run động 1-liền-mạch, nó chấm được ngay (run đã gồm mọi năm). Nhưng để phản ánh **overlay production** (K10 + conviction + SKIP causal), cần thêm chế độ chấm overlay — hoặc ghi 2 cột: `cagr_standard` (K25 fair) + `cagr_overlay` (K10 production).

---

## 4. Thứ tự thực hiện + test

| Thứ tự | Bước | Test bắt buộc |
|---|---|---|
| 1 | Bước 2 (resolver) — độc lập, dễ test | Unit: `resolve_universe(2024)` = top-N ADV 2023, causal, không index/deriv |
| 2 | Bước 1 (config) — backward-compat | `universe_policy=None` → snapshot 61-mã byte-identical |
| 3 | Bước 3 (splitter) | Snapshot: run universe cố định qua path mới = kết quả cũ |
| 4 | Bước 4 (xsec) — RỦI RO CAO | So conviction per-fold vs `_dynuniverse` cũ; verify byte-close |
| 5 | Bước 5 (score) | Chấm run động = số overlay đã đo (dynamic-900 CAGR 205%) |

**Nguyên tắc**: mỗi bước có **snapshot guard** — `universe_policy=None` phải cho kết quả byte-identical với engine hiện tại (không phá champion 61-mã / dl63size). Xem cách làm ở `tests/test_baseline_snapshot.py`.

---

## 5. Rủi ro & lưu ý

- **Bước 4 (cross-sectional) là điểm nguy hiểm nhất**: đổi cách rank có thể dịch mọi conviction → phá số đã verify. Phải test kỹ, so với `_dyn900` cũ (đã có kết quả tham chiếu: signal 3.07%, CAGR overlay 205%).
- **Data union**: load union mọi fold-universe có thể lớn (900+ mã × 8 năm). Kiểm memory (dataset static-900 = 2.66M bars OK).
- **Backward-compat tuyệt đối**: `universe_policy=None` = hành vi cũ. Champion 61-mã / dl63size KHÔNG được đổi 1 trade.
- **Determinism**: giữ (LGBM `deterministic:True`); universe resolver phải deterministic (cùng year → cùng symbols).
- **Ước lượng**: ~1-2 ngày code + test. Bước 1-3 dễ (~0.5 ngày), bước 4 khó (~1 ngày do test cross-sectional).

---

## 6. Tham chiếu (đã kiểm chứng)

- Logic universe clean causal: `_dynuniverse_clean.py::uni()` (root repo) — port sang bước 2.
- Kết quả dynamic-900 mục tiêu tái tạo: CAGR 205.6% / DD −17.6% (K10, full-price, NAV≤1).
- Snapshot pattern: `stock_ml/tests/test_baseline_snapshot.py`.
- Cross-sectional hiện tại: `experiment.py:1681` `_load_xsec_features` (docstring: "RANK vs ALL symbols").

---

## 7. Bổ sung 2026-07-29 — 4 yêu cầu phía ĐĂNG KÝ (để lên leaderboard chuẩn)

5 bước §3 mới lo phía ENGINE. Để một run universe-động đăng ký chính thức (leaderboard +
tab danh mục + tái lập được), cần thêm:

### 7.1 Universe = POLICY có danh tính + snapshot vật chất hóa từng fold
- `strategy_templates.universe_slug` trỏ vào một POLICY (row trong `universe_sets` với
  `selector` JSON = `{mode, n, metric, lookback, min_sessions, exclude}`), KHÔNG phải danh
  sách mã cứng.
- Mỗi run: sau khi resolver chạy, **materialize danh sách đã resolve của TỪNG fold** vào
  `universe_versions`/`universe_symbols` (key theo `(slug, fold_year)`) + hash danh sách.
  Lý do: data sau này được vá CA/refill → ADV đổi → resolve lại có thể lệch mã; snapshot làm
  run tái lập byte-stable và audit được "fold 2023 gồm đúng những mã nào". Serving đọc CÙNG
  snapshot (hoặc resolve live với policy y hệt — causal nên hợp lệ, nhưng phải log hash để đối chiếu).

### 7.2 Universe PHẢI vào cache-key (bug đã từng dính — bắt buộc)
- Sự cố thật 2026-07-26: universe không nằm trong cache-key → run PIT **ghi đè cache của base
  61-mã** (phải khôi phục). Feature/prediction cache key = hash(config + universe policy +
  resolved fold lists). Test guard: chạy run 61-mã rồi run dyn200 rồi CHẠY LẠI run 61-mã —
  kết quả byte-identical lần đầu.

### 7.3 Chốt tường minh chính sách panel cross-sectional (2 tầng, train↔serving khớp)
- Có 2 tầng rank chịu ảnh hưởng panel: **Stage-1** (`_load_xsec_features` — RS ranks vào model)
  và **Stage-2** (conviction `cs5_ma50` trong `stock_ml/portfolio` — `ctx.market_frame`).
- Hai lựa chọn hợp lệ: rank trên **universe của fold** (§3 bước 4 đề xuất) hay trên
  **full-panel** (dyn400 trước đây cố ý chọn full-488 và serving khớp). Không hard-code —
  thành field template `xsec_panel: fold_universe | full`, ghi vào cả
  `overlay_config_hash` (doc PIPELINE), và train↔serving BẮT BUỘC cùng giá trị.
- Đã đo: panel là siêu-tham-số (CAGR dịch hàng chục pp khi nới panel); SKIP gate causal
  panel-adaptive trong overlay được thiết kế để hấp thụ — nhưng chỉ khi policy nhất quán.

### 7.4 Hiển thị & so sánh trên leaderboard
- **ĐÍNH CHÍNH fairness (user, 2026-07-29)**: phán quyết cũ "mở universe = ăn gian" chỉ đúng
  thời xếp hạng theo tổng-%-cộng-dồn-lệnh. Bảng nay xếp theo **CAGR trên NAV, tổng vốn ≤ 1,
  K-slot** → universe rộng không cộng return máy móc, chỉ mở tập lựa chọn cạnh tranh cùng vốn
  = so sánh CÔNG BẰNG mọi cỡ universe trên cột NAV (`cagr_overlay`/`cagr_nav`). KHÔNG cần
  phân lớp xếp hạng.
- Còn lại 2 ghi chú: (a) cột linear cũ (`pnl_pct`/`total_pnl`/composite) vẫn nhạy số-mã —
  đừng dùng so chéo universe; (b) hiện `universe_slug`/`n_symbols` trên bảng làm metadata
  DIỄN GIẢI risk-profile (universe rộng → beta mid/small, DD sâu hơn — đọc kèm Calmar/DD),
  không phải rào so sánh.

**Ước lượng thêm cho §7**: ~0.5-1 ngày (7.1 nửa ngày; 7.2 vài giờ nhưng test cẩn thận;
7.3 là quyết định + field; 7.4 chỉ hiển thị). Thứ tự khuyến nghị: làm SAU B1-B5 của
PORTFOLIO_REGISTRATION_PIPELINE để `register_overlay.py` có sẵn chỗ ghi config hash.
