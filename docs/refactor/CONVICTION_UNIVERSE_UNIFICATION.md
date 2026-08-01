# Conviction / Market-Panel Universe Unification (fail-loud, single declared artifact)

> Re-baseline có kiểm soát: tầng cross-sectional (conviction `cs5_ma50` / breadth) phải chạy
> trên MỘT panel được KHAI BÁO (rule + as_of + price-vintage sha), lọc asset_type=stock, mask
> theo bar-khớp-lệnh, và **fail-loud** khi mã traded VẮNG khỏi panel — thay hành vi silent 0.5.
> Áp cho backtest + overlay + serving. **Được phép đổi số champion/khách** (có chủ đích).
>
> **Bản này đã hợp nhất review từ repo serving (01/08).** Sửa nhiều điểm sai của bản draft đầu.
> Trạng thái: các số/địa chỉ tham chiếu theo TÊN HÀM (số dòng trôi vì block QC đã merge).

---

## 0. TRẠNG THÁI HIỆN TẠI (không phải quyết định tương lai)

- **Serving đã đi trước, re-baseline ĐÃ XẢY RA.** `serving/runner.py` đọc `market_data/market.duckdb`;
  file đó nhảy **488 → 1494** lúc 01/08 14:19. ⇒ 6 sổ khách chu kỳ hôm nay **đã chạy trên panel
  mới**, trong khi board train vẫn pin 488 — một re-baseline **chưa re-pin, chưa thông báo**.
- **Serving đã khai báo panel chuẩn**: `serving/panel.py` khai panel bằng `rule + as_of + sha`
  (vd `cab0626a…`), union-anchor theo năm, materialize từ store, ghi sidecar `panel.json` với
  `symbol_count / as_of / store_fingerprint`, `missing_from_store=0`. Đã xong §handoff mục 1/3/4.
- **HAI file vật lý cùng đường dẫn tương đối** `market_data/market.duckdb`: train=1357,
  serving=1494 (chung 1294, riêng-serving 200, riêng-train 63). ⇒ **KHÔNG thể "đổi default
  OVERLAY_MARKET_DB" để đạt parity** — phải chọn **một artifact + một declaration** (sidecar của
  serving) và train đọc đúng file/sha đó rồi **assert sha**.

## 1. Vấn đề (đã sửa)

Tầng cross-sectional tính rank trên panel **không được khai báo/không lọc**: (a) mã traded vắng
panel → **bịa 0.5** (đúng 1 chỗ, xem §3); (b) panel rộng lẫn **22.8% bar ma** (vol=0, giá phẳng)
và **mã index/phái sinh/ETF** → mẫu số rank ô nhiễm; (c) hai kho cùng danh sách mã nhưng **khác
vintage back-adjust** → conviction vẫn khác. Trái hợp đồng fail-loud `require_no_nan` + CLAUDE.md
§4.1 (cấm trộn asset_type khi gộp/rank).

## 2. Bằng chứng (đo 01/08)

- Kho: train `market.duckdb`=**1357** (rìa rách: chỉ 265/1357 có bar 2026-07-31), serving=**1494**
  (942/1494 có bar 31/07); serving-pin cũ=488; ohlcv price-pin=901; sieutinhieu=full, fail-loud.
- **Ghost bar** (vol=0 từ 2020): pin488=1.5% → panel~1494=**22.8–24.4%** (bar phẳng 3.8%→37.9%).
- **Asset pollution** (train pin): **17 mã** VNINDEX/HNX30/VN30F1M/2M/E1VFVN30 + 12×FUE*.
- **Vintage giá lệch** trên mã chung: DBC 2020 tới 1.09%, HDG 0.20%, 604/3069 bar ≥2024 lệch.
- conv=0.5 (run rộng dyn900 dưới pin488): 49.6% lệnh; dưới panel-1357 conv_miss=0 **NHƯNG** là
  "false victory" — số giả chuyển thành ghost-bar 22.8% trong mẫu số rank.

## 3. Catalog vấn đề (đã sửa theo review)

### A. Silent fallback — CHỈ 1 chỗ sống (không phải 5)
- **Chỗ DUY NHẤT bịa số**: `build_market_panel`→`api.py CSm.get((sym, sigd), 0.5)` (điểm dựng `cm`).
- `cm` dựng từ chính `rw` ⇒ mọi `cm.get(key, 0.5)` sau đó (2 chỗ trong `api.py` + 2 trong
  `gates.py`) là **default CHẾT — luôn trúng key**. "Dọn 5 chỗ" là ảo giác đã-fix.
- `panel.py` cs5_ma50 NaN→0.5 = **warmup HỢP LỆ** (mã có mặt, lịch sử ngắn), giữ.

### B. Hai loại miss — phải TÁCH trước khi bật raise
- Key = `(symbol, signal_date)`. Miss vì **mã vắng panel** = lỗi data → **raise**. Miss vì **mã
  có nhưng thiếu đúng ngày sigd** (halt/ngừng GD) = **bản chất → KHÔNG raise**.
- QC hiện tại (`n_conv_miss`) **gộp cả hai** ⇒ strict_panel sẽ báo động giả. Tách 2 counter trước.

### C. Ghost-bar + asset_type ô nhiễm mẫu số (rủi ro lớn nhất của D1)
- ~1/4 mẫu số là bar không khớp lệnh, mang cùng cụm giả (dist20low=0, ret20=0, rsi14≈50).
  `piv.notna()` KHÔNG cứu (ghost bar có giá, không NaN). Phải **mask theo BAR khớp lệnh
  (volume>0 / traded_value>0) TRƯỚC khi rank**, trong `build_market_panel` (wheel dùng chung).
- Panel phải **lọc asset_type=stock tường minh** (loại index/deriv/ETF) và ghi rule.

### D. Parity không đủ nếu chỉ so membership — phải so VINTAGE GIÁ
- Nguyên tắc "backtest≡serving" phải yêu cầu **cùng store_fingerprint (sha vintage giá)**, không
  chỉ cùng danh sách mã. Serving đã ghi `store_fingerprint` trong sidecar — train phải assert nó.

### E. Định danh & lưu vết (nợ)
- `overlay_config_hash` chưa gồm panel-id; `conv_miss_frac` mới chỉ print, **chưa lưu**
  `leaderboard_nav` ⇒ không truy được dòng board nào chấm bằng conviction giả. Thêm cột
  `conv_miss_frac` + `overlay_panel_hash`.
- `DuckDBContext` không assert market⊇price⊇traded + cùng vintage.

## 4. Thiết kế đích (5 nguyên tắc — khung giữ nguyên, cơ chế sửa)

1. **MỘT artifact panel được KHAI BÁO** (không phải "đổi default path"): rule + as_of +
   store_fingerprint, ghi sidecar `panel.json` (serving đã có). Train **đọc đúng artifact đó**,
   assert sha. sieutinhieu = nguồn; store = cache khai báo, thiếu → refetch, không bịa.
2. **Fail-loud thay silent** — nhưng CHỈ cho "mã vắng panel" (loại B-raise); "thiếu ngày sigd"
   (halt) giữ default hợp lệ. `prio` thiếu = bản chất → chỉ phơi `prio_miss_frac`.
3. **Panel sạch trước khi rank**: mask bar-khớp-lệnh + lọc asset_type=stock, trong
   `build_market_panel` (wheel) ⇒ cả 2 repo hưởng.
4. **config_hash gồm panel-id** = `{rule, as_of, store_fingerprint, symbol_count}`.
5. **Parity = cùng artifact + cùng vintage**, không chỉ cùng mã. Context assert coverage + sha.

## 5. Kế hoạch train (thứ tự ĐÃ SỬA)

1. ✅ **Instrument (đã merge)**: `run_portfolio` trả `conv_miss_frac/prio_miss_frac/n_offpanel`
   (`api.py`, cuối dict). *Golden byte-exact — đã verify seed 42.*
2. **Tách 2 counter** (mã-vắng vs thiếu-ngày) — chưa raise. `strict_panel=False` mặc định +
   cảnh báo theo ngưỡng.
3. **`build_market_panel`**: mask bar-khớp-lệnh (volume>0) + lọc asset_type=stock TRƯỚC rank.
   ⚠️ Đổi số champion NGAY CẢ trên pin488 (pin488 có 1.5% ghost) ⇒ đây là **thay đổi logic có
   chủ đích** → re-pin golden numbers (cả 2 impl đổi cùng nhau, parity giữ) — KHÁC re-pin vì
   đổi store. Đo `n_offpanel` của 3 seed fixture TRƯỚC.
4. **Adopt panel artifact của serving**: train đọc store + sidecar serving khai (không dùng file
   1357 rìa-rách riêng). `DuckDBContext`/context board assert `store_fingerprint == sidecar.sha`
   + market⊇price⊇traded.
5. **config_hash + persist**: thêm `overlay_panel_hash`; lưu `conv_miss_frac` vào `leaderboard_nav`
   (migration nhẹ, cột mới). Đổi hash ⇒ re-score.
6. **Đo lại SKIP mu**: `skip_mu_ref=0.644` hiệu chỉnh cho panel ~200 (`constants.py`) với
   `skip_gain=2.0`. Panel ~1494 + bỏ 0.5 giả **dịch mu** ⇒ đo mu theo năm trước/sau; **không**
   kết luận sớm "gates.py không cần sửa".
7. **Bật `strict_panel=True`** cho board/backtest CHỈ SAU khi (4) đạt `n_offpanel==0`. Rồi
   **re-score board** dưới panel khai báo.

## 6. Serving — trạng thái + việc còn lại (handoff, đã cập nhật)

Đã XONG (serving): (1) declaration rule+as_of+sha, (3) `build_market_panel` dùng chung wheel,
(4) kho phủ đủ + `symbol_count/as_of/store_fingerprint` sidecar, `missing_from_store=0`.

Còn phối hợp:
- **Chia sẻ artifact panel** (store + `panel.json`) để train đọc cùng file/sha (§0).
- **Đưa mask-bar + lọc asset_type vào `build_market_panel` (wheel)** — thống nhất 1 lần cho cả 2.
  Serving hiện lọc asset_type=stock nhưng **ETF vẫn lọt**; bar-mask cần xác nhận đã có chưa.
- **Đồng bộ vintage giá**: train + serving cùng `store_fingerprint`; nếu lệch → chọn 1 vintage chuẩn.
- **Thông báo re-baseline khách** (đã xảy ra từ chu kỳ 01/08) — số 6 sổ đã đổi.

## 7. Open decisions (đã sửa theo thực tế)

- **D1 (full-market conviction)**: GIỮ, nhưng "full" = **panel khai báo sạch** (stock-only,
  bar-masked), KHÔNG phải "toàn bộ mã trong duckdb".
- **D2**: ~~"dùng 1357 train cho nhanh"~~ **BỎ** (rìa rách, ghost, asset pollution). ⇒ **dùng thẳng
  panel serving đã khai** (artifact + sidecar sha).
- **D3**: ~~"train trước serving sau"~~ **vượt bởi thực tế** — serving đã đổi. Nay: **reconcile về
  1 artifact**, train bắt kịp, re-pin số board/khách có thông báo.

## 8. Bất biến test (đã sửa)

- **Golden = neo PARITY hai implementation trên snapshot 488 đóng băng** (`test_portfolio_golden.py`
  hardcode `market_golden_pin_20260729.duckdb`). **KHÔNG re-pin golden vì đổi OVERLAY_MARKET_DB**
  (board's store) — golden không đụng tới nó. Golden CHỈ đổi khi **logic `build_market_panel` đổi**
  (bar-mask §5.3) — khi đó re-pin numbers có chủ đích, parity vẫn giữ.
- Trước khi bật strict_panel: đo `n_offpanel` của 3 seed fixture = 0 (nếu >0 golden sẽ raise).
- Sau panel sạch: đo lại phân phối `conv` trước/sau (ghost-bar removal) + `skip mu` theo năm.
- Determinism: cùng artifact + cùng vintage ⇒ số byte-identical qua replay.

---
Liên quan: [[silent-data-corruption-classes]], [[data-source-of-truth-sieutinhieu-report-not-patch]],
[[cagr-board-overlay-unified]], `docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md`, CLAUDE.md §4.1.
