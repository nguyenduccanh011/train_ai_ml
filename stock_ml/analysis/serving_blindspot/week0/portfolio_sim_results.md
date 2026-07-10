# Kết quả mô phỏng danh mục MTM hàng ngày — Champion vs Pyramid (kịch bản vốn thực)

Script: `f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/week0/portfolio_sim.py`
Dữ liệu: 1384 lệnh champion (`trades_n2_2643_wavestruct_la05_lamp02.csv`), 524 lệnh có add (`trades_w0_pyr_u10_r02.csv`, weight=2.0), giá đóng cửa từ `"C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"`, lịch giao dịch 2020-01-02 → 2026-07-08 (union 61 mã).

## Công thức / quy tắc mô phỏng
- NAV₀ = 1.0, lãi kép hoàn toàn. Lệnh BASE mới chiếm `f = 1/K` NAV hiện tại (K = số slot). Nếu tiền mặt tự do < size → BỎ QUA lệnh (đếm).
- Đường giá mỗi lệnh được neo 2 đầu: `shares = invested / entry_price_csv`; tỷ lệ hiệu chỉnh nội suy tuyến tính theo chỉ số bar giữa `ratio0 = entry_price_csv / close_db(bar_vào)` và `ratio1 = entry_price_csv·(1+pnl_net) / close_db(bar_ra)`. Nhờ đó pnl thực hiện khi thoát = `invested·(1+pnl_net_CSV)` đúng tuyệt đối (trung hòa lệch giá raw/điều-chỉnh ~1.4%). Thiếu giá ngày nào → giữ giá trị gần nhất.
- ADD (pyramid): tại bar giao dịch thứ 3 sau ngày vào (lịch từng mã), size = ĐÚNG bằng vốn base của lệnh đó; giá vào = close đã neo × 1.0015; pnl add = `pnl_cand − pnl_champ` (đã net). Cấp vốn: tiền mặt trước, sau đó margin (nếu cho phép) tới trần; không đủ → bỏ qua add (đếm).
- MARGIN: trần = 50% NAV hiện tại. Lãi 15%/năm cộng dồn HÀNG NGÀY trên dư nợ: `lãi = dư_nợ × 0.15/365 × số_ngày_lịch giữa 2 phiên`. Lãi trừ vào tiền mặt (âm → chuyển thành margin trong trần; vượt trần → chặn mở lệnh mới). Mọi dòng tiền vào (thoát lệnh) trả margin TRƯỚC.
- NAV MTM = tiền mặt + Σ giá trị vị thế − dư nợ margin, ghi nhận từng ngày.
- CAGR = `NAV_cuối^(1/số_năm) − 1`, số_năm = số ngày lịch/365.25 ≈ 6.51 năm. MaxDD trên chuỗi NAV ngày. Lợi nhuận theo năm = NAV cuối năm / NAV cuối năm trước − 1 (2026 chỉ tới 08/07).

## Bảng kết quả (NAV₀ = 100%)

| Config | K | NAV cuối | Tổng LN | CAGR | MaxDD | Lãi vay (%NAV₀ / %LN) | Base bị bỏ | Add chạy/bỏ | Margin đỉnh %NAV | Ngày chạm trần | Ngày xấu nhất |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CHAMPION | 25 | 14.056 | +1305.6% | 50.05% | −13.76% | 0 / 0 | 356 | 0/0 | 0% | 0 | −6.10% (09/03/2026) |
| PYR_AGGR_MARGIN | 25 | 16.835 | +1583.5% | 54.26% | −18.51% | 10.96% / 0.69% | 521 | 280/0 | 37.8% | 0 | −5.99% (09/03/2026) |
| PYR_AGGR_CASHONLY | 25 | 14.659 | +1365.9% | 51.02% | −17.87% | 0 / 0 | 481 | 211/84 | 0% | 0 | −5.77% (18/08/2023) |
| PYR_DEFENSIVE | 25 | 12.929 | +1192.9% | 48.14% | −14.99% | 0 / 0 | 304 | 295/82 | 0% | 0 | −5.85% (09/03/2026) |
| CHAMPION | 30 | 12.274 | +1127.4% | 46.96% | −13.35% | 0 / 0 | 257 | 0/0 | 0% | 0 | −5.60% (09/03/2026) |
| PYR_AGGR_MARGIN | 30 | 14.473 | +1347.3% | 50.72% | −16.11% | 8.85% / 0.66% | 444 | 317/0 | 35.9% | 0 | −6.09% (09/03/2026) |
| PYR_AGGR_CASHONLY | 30 | 13.524 | +1252.4% | 49.16% | −16.35% | 0 / 0 | 402 | 242/93 | 0% | 0 | −5.78% (09/03/2026) |
| PYR_DEFENSIVE | 30 | 11.206 | +1020.6% | 44.92% | −13.94% | 0 / 0 | 238 | 344/58 | 0% | 0 | −5.50% (18/08/2023) |

## Lợi nhuận theo năm (%)

| Config | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (tới 08/07) |
|---|---|---|---|---|---|---|---|
| CHAMPION K25 | +90.1 | +119.1 | +31.3 | +33.6 | +21.2 | +44.9 | +9.5 |
| PYR_AGGR_MARGIN K25 | +107.3 | +110.9 | +32.5 | +32.9 | +26.2 | +48.2 | +17.0 |
| PYR_AGGR_CASHONLY K25 | +96.8 | +108.8 | +29.4 | +31.3 | +23.1 | +48.5 | +14.9 |
| PYR_DEFENSIVE K25 | +89.0 | +99.4 | +25.6 | +36.3 | +22.0 | +48.1 | +10.9 |
| CHAMPION K30 | +84.7 | +106.3 | +28.9 | +35.1 | +21.2 | +44.5 | +5.5 |
| PYR_AGGR_MARGIN K30 | +96.4 | +91.0 | +29.3 | +38.4 | +25.6 | +51.5 | +13.3 |
| PYR_AGGR_CASHONLY K30 | +93.3 | +102.8 | +26.6 | +32.1 | +23.2 | +47.2 | +13.8 |
| PYR_DEFENSIVE K30 | +84.9 | +95.9 | +23.2 | +35.8 | +18.3 | +45.1 | +7.7 |

## Kiểm tra sanity (đã chạy)
- (a) CHAMPION K=25: NAV cuối 14.06 (+1306%). Khung unit: +127u trên trung bình ~26.4 unit ≈ +481% KHÔNG kép; với lãi kép 6.5 năm và 356/1384 lệnh bị bỏ do hết slot, ln(14.06)=2.64 so với cận trên Σf·pnl = 127/25 = 5.08 (nếu không bỏ lệnh nào, không tiền mặt nhàn rỗi) → cùng bậc độ lớn, hợp lý.
- (b) Pnl thực hiện mỗi lệnh trong sim / vốn đầu tư == pnl_net CSV: sai số tuyệt đối lớn nhất ≤ 8.9e-16 trên cả 8 config (khớp máy tính dấu phẩy động).
- (c) Không có ngày nào |biến động NAV| > 15% ở bất kỳ config nào (đếm = 0); ngày xấu nhất −6.10%.
- 524/524 cờ add hợp lệ (bar add < bar thoát); adds bị bỏ chỉ do thiếu tiền mặt (cash-only/defensive), config margin không bỏ add nào và không ngày nào chạm trần 50%.

## Kết luận cho 2 kịch bản vốn
1. **NAV cố định 100% (không margin):** PYR_AGGR_CASHONLY K25 tốt nhất về lợi nhuận (CAGR 51.0% vs champion 50.0%) nhưng DD sâu hơn (−17.9% vs −13.8%) — lợi thế mỏng vì 84 add + 125 base thêm bị bỏ do kẹt tiền mặt. PYR_DEFENSIVE (2/3 size) KÉM hơn champion (48.1% vs 50.0% ở K25) — giảm size base 33% để dành chỗ cho add không bù lại được.
2. **Có margin +50% NAV, lãi 15%/năm:** PYR_AGGR_MARGIN thắng rõ: K25 CAGR 54.3% (+4.2 điểm so champion), tổng lãi vay chỉ 10.96% NAV₀ = 0.69% tổng lợi nhuận; margin đỉnh 37.8% NAV, không bao giờ chạm trần. Đổi lại MaxDD tăng −13.8% → −18.5%.

## Caveat
- Không mô phỏng force-liquidation của công ty chứng khoán (nhưng margin đỉnh chỉ ~38% NAV, dưới trần 50%).
- Giá db là raw (không hồi tố); neo 2 đầu vào entry/exit CSV nên pnl thực hiện chính xác, nhưng đường MTM GIỮA kỳ nắm giữ có thể lệch nhẹ quanh ngày chia tách/cổ tức → MaxDD trong-lệnh có sai số nhỏ.
- Bỏ lệnh khi hết slot/tiền theo thứ tự thời gian + alphabet mã — thực tế có thể chọn lệnh khác → kết quả phụ thuộc quy tắc chọn.
- Add bị bỏ (cash-only) được bỏ toàn phần, không cấp vốn một phần.
- 2026 là năm chưa hoàn chỉnh (tới 08/07).
