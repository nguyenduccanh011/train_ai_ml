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

---

## Trả lời từ SERVING — 2026-08-01 (T1/T2/T4/T5/T6 xong, T3 = artifact dưới đây, T7 đã chốt số)

### Artifact dùng chung (T3) — train đọc ĐÚNG 3 file này, đừng dựng bản sao
```
C:/Users/DUC CANH PC/Desktop/stock-serving/market_data/market.duckdb
C:/Users/DUC CANH PC/Desktop/stock-serving/market_data/market.duckdb.panel.json
C:/Users/DUC CANH PC/Desktop/stock-serving/market_data/market.duckdb.panel_symbols.txt
```
| khoá | giá trị |
|---|---|
| `n_symbols` | **1477** |
| `symbols_sha256` | `0d239e6128cc9c955590f3083c471f166add71be69b777b7f6fd98f50f0825a5` |
| `panel_fingerprint` | `4421162:2026-07-31:58109396.84:1654095309512` |
| `store_fingerprint` | giống hệt trên (T4: **một** vintage giá, không có nguồn thứ hai) |
| `filter` | `{asset_type: stock, sessions: 250, min_sessions_traded: 100, exclude_icb: [8995, 8985]}` |
| `max_date` | 2026-07-31 |

`panel_fingerprint` = `rows:max_date:Σclose:Σvolume` trên `timeframe='1D'` — **cùng công thức**
với `OhlcvStore.fingerprint`, tính được từ DuckDB một mình, không cần store/mạng. Cổng kiểm:
`python -m serving.panel verify` (bắt cả lệch membership lẫn **lệch vintage giá ở cùng danh
sách mã**) — serving chạy nó tự động trong scheduler ngay sau `panel refresh`.

⚠️ **Đừng dùng `market_data/market.duckdb` của train làm nguồn conviction.** Đo 01/08: 1357 mã,
rìa gần đây rách (**265**/1357 mã có bar ngày 31/07, serving 942/1477), có **VNINDEX, HNX30,
VN30F1M, VN30F2M** trong mẫu số xếp hạng, và giá khác vintage (DBC 2020 lệch 1,09%, HDG 0,20%,
604/3069 bar ≥2024 lệch).

### T6 — parity champion trên đúng artifact trên: **PASS**
Fixtures golden của train, `PortfolioConstants(tplus=2)` (causal default), `date_hi=2026-07-31`:

| seed | NAV | CAGR | maxDD | conv_mu | n_offpanel | conv_miss_frac |
|---|---|---|---|---|---|---|
| 42 | ×183.533611 | 120.9136% | −12.7852% | 0.671884 | 0 | 0.0 |
| 21 | ×168.454221 | 118.0523% | −14.9001% | 0.672577 | 0 | 0.0 |
| 123 | ×156.650897 | 115.6568% | −14.8857% | 0.672606 | 0 | 0.0 |

`stock_ml.portfolio.run_portfolio` == `serving.portfolio.core.run_portfolio` khớp tới 6 chữ số
thập phân cả 3 seed. **Không so trực tiếp với golden 139,7%**: golden pin panel 488 +
`date_hi=2026-07-08`; đây là panel 1477 + `date_hi=2026-07-31`.

### T2 — lọc bằng ICB của chính nguồn, KHÔNG bằng tên mã
Nguồn xếp ETF/CCQ vào `asset_type='stock'`; phân biệt được bằng **ICB 8995** (equity investment
instruments) / **8985** (non-equity). Trên panel serving trúng **đúng 17 mã**
(`E1VFVN30` + 12×`FUE*` + `FUCTVGF1/2`, `FUCVREIT`, `FUETCC50`), **không bắt nhầm** mã thật.
Regex tên là bẫy: `^E1` nuốt luôn **E12** (một doanh nghiệp xây dựng UPCoM). Panel serving
không có VNINDEX/HNX30/VN30F* — 17 mã trong catalog của các anh là bảng của **train**.

### 🔴 Việc CHO TRAIN: `universe_resolver` dính đúng lỗi này
Universe giao dịch của bundle dyn **cũng chứa CCQ**: `dyn300` 3 mã (`E1VFVN30`, `FUESSV50`,
`FUEVFVND`), `dyn900` **15 mã**. Sổ live đã **nắm thật** (E1VFVN30 ở 4 sổ khách;
FUEKIV30/FUEMAV30/FUEKIVFS/FUEDCMID/FUESSV30 ở dyn900) — tức model đang mua **rổ của chính
universe mình**. Serving chặn tạm ở `serving/portfolio/core.py::_drop_non_stock_trades` (lý do
`not_stock`, lấy danh sách từ `excluded_icb` của panel → một định nghĩa duy nhất): 60 leg bị bỏ
ở base dyn300, 283 ở dyn900. **Bản sửa gốc là ở `universe_resolver` của train**, lần re-pin
universe đầu năm, bằng cùng luật ICB.

### T5 — strict_panel BẬT ở serving
`serving/portfolio/core.py::_strict` ép `strict_panel=True` cho mọi sổ (thoát hiểm sự cố:
`PANEL_STRICT=0`). Chu kỳ 7 tier ngày 01/08 chạy **sạch, không raise** — panel 1477 phủ đủ mọi
universe sau khi bỏ CCQ.

### T1 — wheel 0.4.4 + delta ĐO ĐƯỢC (không "nhẹ")
Build từ `8eb20825`+`1166fdfd`, pin ở `requirements.txt` + `Dockerfile` + `dist/` (cổng
`serving/deploy.py` xác nhận 3 chỗ khớp). Riêng ghost-bar mask (cùng panel 1494, cùng store,
cùng `date_hi`) làm `conv_mu` dyn300 **0,6978 → 0,6809** và NAV YTD 7 sổ **giảm 8,2–21,9%**.
Cộng cả bước bỏ CCQ, tổng delta **−2,6% … −28,5%**. Bảng đầy đủ: serving
`DEPLOY_DYN_TIERS.md` §7.7; số cũ lưu ở `backups/panel_rebaseline_20260801/`.

### Ghi chú kỹ thuật cho wheel (chưa chặn gì)
Ghost-mask đặt `cs5_ma50 = NaN` cho bar ma, rồi `panel.py:45` đổi NaN → **0.5** qua nhánh
warmup. Nên leg nào có **bar ngày tín hiệu là bar ma** vẫn nhận 0.5 giả mà `n_offpanel` **không
đếm** — cùng lớp lỗi đang dọn, chỉ khác cửa. Rủi ro thực tế nhỏ (universe dyn300 chỉ **0,32%**
bar ma năm 2026 so với 6,52% của cả panel), nhưng nên tách nhánh warmup-NaN khỏi ghost-NaN ở
bản wheel sau.
