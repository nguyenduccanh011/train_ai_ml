# Nhiệm vụ cho repo SERVING — conviction/panel unification

> Gửi từ train_ai_ml. Bối cảnh đầy đủ: `docs/refactor/CONVICTION_UNIVERSE_UNIFICATION.md`.
> Mục tiêu: backtest ≡ serving ≡ board dùng **một artifact panel khai báo** (rule+as_of+
> **store_fingerprint**), panel **sạch** (mask bar-khớp-lệnh + chỉ stock), **fail-loud** khi mã
> traded vắng panel. Serving đã làm phần declaration/kho phủ đủ — dưới đây là phần CÒN LẠI +
> đồng bộ với thay đổi wheel bên train.

## Trạng thái ghi nhận (serving đã xong)
- Declaration panel `rule + as_of + sha` (union-anchor theo năm), materialize từ store.
- `build_market_panel` dùng chung wheel `stock_ml.portfolio`.
- Sidecar `panel.json`: `symbol_count / as_of / store_fingerprint`, `missing_from_store=0`.

## T1 — Bump wheel (BẮT BUỘC, chặn mọi thứ khác)
- **Việc**: nâng `stock_ml_core` lên bản có ghost-bar mask (commit train `8eb20825`).
  `build_market_panel` nay **NaN-mask bar `volume<=0` trước khi rank** và **yêu cầu cột `volume`**.
- **Kiểm serving**: `ctx.market_frame(...)` phải trả cột `volume`; nếu thiếu → hàm **raise**
  (đúng fail-loud). Chạy 1 tier cycle xác nhận không raise.
- **Nghiệm thu**: serving materialize panel qua wheel mới, không lỗi; số 1 sổ khách đổi nhẹ do
  bar-mask (ghi lại delta).

## T2 — Lọc asset_type: loại NỐT index/derivative/ETF/fund-cert
- **Vấn đề**: review thấy panel còn lọt **17 mã** phi-cổ-phiếu: `VNINDEX, HNX30, VN30F1M, VN30F2M,
  E1VFVN30` + 12× `FUE*`. `asset_type='stock'` hiện chưa chặn ETF/CCQ.
- **Việc**: mở rộng rule declaration để loại tường minh: index (VNINDEX/HNX30/…), phái sinh
  (VN30F*), ETF/fund-cert (`FUE*`, `E1VFVN*`). Ghi **filter rule** vào `panel.json`.
- **Nghiệm thu**: `SELECT DISTINCT symbol` trong panel materialize KHÔNG còn 17 mã trên;
  `panel.json.filter_rule` mô tả rõ; CLAUDE.md §4.1 (không trộn asset_type khi rank) thỏa.

## T3 — Chia sẻ artifact cho train (để đọc chung 1 file/sha)
- **Việc**: publish (đường dẫn/URL ổn định) **store panel + `panel.json`** để train đọc **đúng
  file** và `assert store_fingerprint == panel.json.sha`. Không để "hai file cùng tên khác nội
  dung" (train=1357 rìa-rách vs serving=1494) như hiện tại.
- **Nghiệm thu**: train chạy overlay với artifact serving, `n_offpanel==0`, sha khớp.

## T4 — Chốt MỘT vintage giá (store_fingerprint), không chỉ danh sách mã
- **Vấn đề**: cùng mã nhưng **vintage back-adjust khác** ⇒ conviction khác (đo: DBC 2020 lệch
  1.09%, HDG 0.20%, 604/3069 bar ≥2024 lệch). Membership khớp là **chưa đủ**.
- **Việc**: chọn 1 vintage giá chuẩn (điểm as_of), ghi `store_fingerprint`; train + serving cùng
  fingerprint. Nếu lệch → refetch back-adjusted từ sieutinhieu cho đồng bộ.
- **Nghiệm thu**: `store_fingerprint` train == serving; parity champion (T6) byte-đủ-gần.

## T5 — Bật fail-loud (strict_panel) phía serving
- **Việc**: khi materialize/serve, mã trong sổ khách (tiers.yaml) mà **vắng panel** ⇒ báo lỗi
  vận hành (không bịa 0.5). Kiểm ở boot cycle + mỗi tier cycle. Wheel mới đã có
  `PortfolioConstants.strict_panel` (mặc định False) — serving set **True**.
- **Nghiệm thu**: cố tình bỏ 1 mã khỏi panel ⇒ tier cycle raise rõ mã thiếu; panel đủ ⇒ chạy sạch.

## T6 — Parity champion sau khi panel sạch
- **Việc**: chạy champion dưới panel serving (đã T1-T4) và so với module train
  (`run_portfolio`) trên cùng artifact/sha.
- **Nghiệm thu**: T+2 CAGR/NAV/DD serving == train (khớp tới chữ số như golden). Ghi vào
  `PORTFOLIO_LAYER_UNIFICATION`/golden serving.

## T7 — Thông báo + ghi nhận re-baseline khách (ĐÃ XẢY RA)
- **Bối cảnh**: `market.duckdb` nhảy 488→1494 lúc 01/08 14:19 ⇒ 6 sổ khách chu kỳ hôm nay **đã
  đổi số** dưới panel mới (chưa thông báo).
- **Việc**: chốt số mới sau khi panel SẠCH (T1-T5, vì bar-mask + bỏ ETF sẽ dịch tiếp), rồi
  thông báo khách + cập nhật bảng số chính thức.
- **Nghiệm thu**: bảng 6 sổ có số mới + ngày + lý do (panel unification), có sidecar sha truy vết.

## Thứ tự đề xuất
T1 → T2 → T4 (đồng bộ vintage) → T3 (share artifact) → train ráp + T6 parity → T5 bật strict →
T7 thông báo. T1 chặn tất cả; T6 là cổng nghiệm thu chung hai repo.

## Bất biến chung (cả hai repo)
- `build_market_panel` = hàm DUY NHẤT (wheel) — không nhân bản logic panel ở serving.
- Panel = stock-only + bar-masked + 1 vintage (fingerprint). Thiếu data ⇒ fail-loud, refetch
  sieutinhieu, KHÔNG bịa.
- Số khách/board đổi là **re-baseline có chủ đích** — luôn kèm sha + ngày + lý do.
