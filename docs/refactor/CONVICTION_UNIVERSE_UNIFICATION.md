# Conviction / Market-Panel Universe Unification (fail-loud, single source)

> Kế hoạch chi tiết cho một re-baseline có kiểm soát: thống nhất universe của tầng
> cross-sectional (conviction / breadth / xsec) về MỘT nguồn point-in-time từ
> sieutinhieu, thay hành vi "silent fallback 0.5 / −9.9" bằng **fail-loud** đúng hợp
> đồng dữ liệu của repo (`require_no_nan`). Áp cho **backtest + overlay + serving**.
> Được phép re-baseline (đổi số champion/khách) — có chủ đích, có golden re-pin.

Trạng thái: DRAFT — chờ chốt các "Open decisions" §7 trước khi merge phần đổi-số.

---

## 1. Vấn đề (một câu)

Tầng cross-sectional (conviction `cs5_ma50`, market-breadth, xsec-rank) được tính trên
**universe không thống nhất** (research 1357 / serving-pin 488 / ohlcv 718–901 /
sieutinhieu ~1600), và khi một mã VẮNG khỏi panel thì code **âm thầm bịa số trung tính
0.5 / −9.9** thay vì báo lỗi — trái với hợp đồng fail-loud `require_no_nan` của chính repo.
Hệ quả: rank cross-sectional phụ thuộc panel nào được nạp; run rộng bị ~25–50% conviction
GIẢ; backtest ≠ serving ngay cả với champion.

## 2. Bằng chứng (đo được)

- Số mã: `market_data/market.duckdb`=**1357**, `market_golden_pin_20260729.duckdb`=**488**,
  `ohlcv_golden_pin_*.db`=**718–901**, sieutinhieu `fetch_ohlcv`=full (~1600, *"Fail loud on any miss"*).
- Run rộng (dyn900, 1161 mã traded): **49.6%** lệnh conv=0.5 giả; trong 23.893 lệnh sống tới
  sim có **25% (5.974)** chạy trên conv giả. Run hẹp (⊆488): **0%** giả.
- `prio=−9.9`: 93% với run nhỏ (train<100/năm) — meta-priority + preempt **trơ** với run nhỏ.

## 3. Catalog vấn đề (xếp theo mức độ)

### A. Silent fallback — vi phạm hợp đồng fail-loud (CAO)
- `stock_ml/portfolio/api.py:70,74,81,119` — `CSm.get((sym,date), 0.5)` conviction bịa khi mã ∉ panel.
- `stock_ml/portfolio/api.py:120` — `pm.get((sym,ed), -9.9)` priority bịa.
- `stock_ml/portfolio/gates.py:45,50` — `cm.get(...,0.5)` → làm phồng μ của cổng SKIP thích nghi.
- `stock_ml/src/pipeline/experiment.py:~3235` — `feat[...] = feat["date"].map(br)...fillna(0.5)` breadth bịa.
- HỢP LỆ (giữ): `panel.py:45,49` warmup-NaN→0.5 của mã CÓ mặt; `api.py:105` skip-nếu-∉-price-panel;
  per-year off/thr defaults khi <30 mẫu.

### B. Universe không thống nhất (CAO)
- 3 kho khác coverage; overlay đọc kho đứng riêng thay vì nguồn sieutinhieu.
- `experiment.py:_load_market_breadth` / `_load_xsec_features` đọc `market.duckdb` (nay 1357)
  nhưng docstring/comment ghi "488" → **feature train ≠ feature serving** khi 2 kho lệch.
- Engine `entry_csr` (rank score theo cohort-ngày) ≠ overlay `cs5_ma50` (rank kỹ thuật theo
  panel) — hai "conviction" khác NGHĨA cùng chia sẻ slot; cần tách bạch tên/tài liệu.

### C. Thiếu kiểm tra & định danh (TRUNG BÌNH)
- `stock_ml/portfolio/context.py DuckDBContext` không assert `market_syms ⊇ traded_syms`
  và `market_syms` vs `price_syms` cùng phủ.
- `overlay_config_hash` (db/overlay_scoring.py) **không gồm** định danh panel/universe/date_hi →
  đổi kho KHÔNG invalidate cache/score (nợ mới, phải sửa).
- `db/overlay_scoring.py:23-31` hardcode path serving + pin ngày trong code Python.

### D. Không tài liệu (THẤP nhưng quan trọng)
- Vì sao market.duckdb 1357 (tích lũy lịch sử?) vs serving 488 (lọc thanh khoản?) — **không ghi ở đâu**.
- Build path market.duckdb (`refetch_adjusted.py`→`back_adjust.py`) không nêu ai ghi file cuối.

## 4. Thiết kế đích (clean & chuẩn)

**Nguyên tắc 1 — MỘT universe point-in-time từ sieutinhieu.** Tầng cross-sectional (breadth,
xsec, conviction cs5_ma50) tính trên **cùng** universe point-in-time do sieutinhieu cấp
(matched-ADTV + sessions_traded, đã có ở `universe_resolver`). Cache duckdb chỉ là bản sao
được phép stale; khi thiếu → refetch từ sieutinhieu, **không bịa**.

**Nguyên tắc 2 — fail-loud thay silent.** Mã traded mà VẮNG khỏi panel conviction ⇒ `raise`
(kèm danh sách mã + ngày) như `require_no_nan`, để chuẩn bị data tại nguồn. Giữ `0.5` **chỉ**
cho warmup-NaN của mã CÓ mặt. `prio` thiếu do năm chưa đủ mẫu = *bản chất* → không raise, nhưng
**đo & phơi** `prio_miss_frac`.

**Nguyên tắc 3 — panel là một phần định danh.** `overlay_config_hash` gồm `{universe_slug,
panel_symbol_count, panel_content_hash, date_hi}`; đổi panel ⇒ đổi hash ⇒ tự re-score.

**Nguyên tắc 4 — context tự kiểm.** `DuckDBContext` (và context serving) assert
`market_syms ⊇ price_syms ⊇ traded_syms` lúc dựng panel; lệch ⇒ raise.

**Nguyên tắc 5 — backtest ≡ serving ≡ board.** Cùng nguồn universe + cùng hàm panel; khác nhau
chỉ ở "as_of/date_hi". Bỏ 488-pin làm nguồn conviction; pin chỉ dùng cho GIÁ (NAV mark) nếu cần
đóng băng, nhưng phải phủ ≥ universe.

## 5. Kế hoạch triển khai — repo train_ai_ml

Thứ tự (mỗi bước tự kiểm, golden-guard):

1. **Instrument + fail-loud (không đổi số champion).** `run_portfolio` trả thêm
   `conv_miss_frac`/`prio_miss_frac`/`n_offpanel`; thêm cờ `strict_panel` (mặc định True ở
   backtest/board): mã ∉ conviction panel ⇒ raise. Champion 61⊆panel ⇒ 0 off-panel ⇒ golden
   byte-exact (verify `RUN_PORTFOLIO_GOLDEN=1`).
2. **Panel = full universe.** Đổi `OVERLAY_MARKET_DB` mặc định sang kho phủ-đủ (1357 hoặc build
   full từ sieutinhieu); `DuckDBContext` assert coverage. Bỏ path hardcode → config/env + validate.
3. **config_hash gồm panel identity** (Nguyên tắc 3). Migration nhẹ: thêm cột
   `overlay_panel_hash` vào `leaderboard_nav` (hoặc gộp vào overlay_config_hash).
4. **Dọn engine feature universe.** `_load_market_breadth`/`_load_xsec_features`: sửa docstring
   "488"→động; nhận `universe`/`as_of` tường minh; fail-loud khi mã traded thiếu breadth.
5. **Gates dùng conviction thật.** Sau (1)(2), `cm` không còn 0.5 giả ⇒ μ cổng SKIP trung thực;
   không cần sửa gates.py ngoài việc bỏ default (đổi `.get(...,0.5)`→`cm[key]`).
6. **Re-pin golden có chủ đích.** Chạy champion dưới panel mới → số mới (khác 131.4%); cập nhật
   `stock_ml/tests/goldens/champion_prod_overlay.json` trong commit riêng, ghi lý do (universe
   re-baseline), KHÔNG nới assert.
7. **Re-baseline board.** Xóa cache score cũ (hash đổi) → chạy lại `register_overlay` toàn bộ.

## 6. Handoff — repo serving (stock-serving)

Serving giữ **cùng nguyên tắc**; cần bên đó tự review + triển khai:

1. **Nguồn conviction = universe point-in-time**, không phải `market_golden_pin` 488. Serving
   `portfolio/core.py` (bản gốc mà `stock_ml.portfolio` port từ đó) phải nạp market panel phủ
   **đủ universe đang phục vụ** (khớp `tiers.yaml`), refetch từ sieutinhieu khi thiếu.
2. **Fail-loud**: mã trong sổ khách mà thiếu breadth data ⇒ báo lỗi vận hành (không bịa 0.5).
   Kiểm tra tại boot cycle + mỗi tier cycle.
3. **Đồng bộ định nghĩa panel với train**: cùng hàm `build_market_panel` (đã ở wheel
   `stock_ml.portfolio`), cùng danh sách CS4+atrpct+dist_ma50, cùng universe as_of.
4. **Rebuild market store phủ đủ**: kho breadth serving nâng từ 488 → phủ ≥ universe khách; ghi
   `symbol_count`+`as_of` vào header để audit.
5. **Re-pin serving numbers**: sau đổi panel, số 6 sổ khách đổi — cập nhật có chủ đích, thông báo.
6. **Kiểm parity backtest≡serving** trên champion dưới panel mới: overlay(train) == core(serving).

## 7. Open decisions (chốt trước khi đụng số)

- **D1 — Universe conviction = "full market" hay "universe của run"?** Đề xuất: **full
  point-in-time market** (rank vs toàn thị trường) — nhất quán, ổn định. (Ảnh hưởng: rank champion đổi.)
- **D2 — Kho phủ-đủ**: dùng `market_data/market.duckdb` (1357) ngay, hay build full từ
  sieutinhieu (1600, point-in-time chuẩn)? Đề xuất: 1357 trước (nhanh), lộ trình sieutinhieu-full.
- **D3 — Serving đổi đồng thời hay sau?** Đề xuất: train re-pin trước (có golden), serving theo
  sau bằng tài liệu này để parity vẫn giữ.

## 8. Bất biến kiểm thử

- Golden champion: byte-exact TRƯỚC bước 6 (fail-loud không đổi số champion vì 0 off-panel), rồi
  re-pin CÓ CHỦ ĐÍCH ở bước 6.
- `conv_miss_frac==0` cho mọi run sau khi panel phủ đủ + strict_panel (nếu >0 ⇒ raise, lộ mã thiếu).
- Determinism: cùng panel + cùng data ⇒ số byte-identical qua replay.

---
Liên quan: [[silent-data-corruption-classes]], [[data-source-of-truth-sieutinhieu-report-not-patch]],
[[cagr-board-overlay-unified]], `docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md`,
`docs/UPGRADE_DYNAMIC_UNIVERSE.md` (§13.9 universe từ sieutinhieu).
