# HỒ SƠ CÔNG TỐ — AUDIT KHUNG ĐO NAV (r2_nav.py / pr2_04_navsim) CHO SO SÁNH R2-vs-gb_x08
Ngày: 2026-07-10. Scripts: `na_00_recon.py` … `na_06_final_scen.py` (thư mục này). Sim audit `na_navlib.py` = bản sao ngữ nghĩa r2_nav.py, **tái lập đúng từng chữ số** cả 7 anchor đã công bố (c2 K25 ×16.34/−13.63, K22 ×18.03/−15.41, K23 ×17.14/−14.77, base K25 ×14.78, K20 ×17.70, gb K25 ×14.44/−15.31, c2 f22 ×3.80) trước khi bật bất kỳ công tắc nào — mọi kết luận dưới đây là delta sạch trên cùng một máy.

## PHÁN QUYẾT: KHUNG ĐO **THIÊN VỊ HỆ QUAY VÒNG NHANH (R2)** — 3 kênh, tổng độ lớn đủ để co lợi thế headline +24.9% xuống +6..+14%, và về 0 nếu vận hành không ứng trước tiền bán
Kết luận §B R2_ROUND2 ("c2 đè bẹp gb +19..+25% tại matched-DD") KHÔNG đứng vững ở dạng phát biểu đó. Phần alpha thật còn lại sau khi gỡ bias: **+6..+14% (full), +3..+6% (f22)** ở kịch bản vận hành khả thi — dương nhưng mỏng, và **f22 nằm trong tầm noise tie-break (±5%)**.

---

## 1. COST SWEEP + PHÍ HÒA VỐN

**Phát hiện nền: cost model nhúng thực tế là 0.6% roundtrip hiệu dụng, KHÔNG phải 0.7% như narrative.** Recon (na_00) xác nhận trên 100% của 2082+2012+1378 lệnh: `pnl_pct = exit_price/entry_price − 1 − 0.004` (max err 9e-16); slippage 0.1%/chiều đã nhân trong giá fill (engine.py CostModel). Tổng hiệu dụng = 0.2% slippage + 0.4% phí/thuế ≈ **0.6%**. Khung đo lạc quan hơn 0.1đ so với chính niềm tin của nó. Cost KHÔNG double-count trong NAV sim (sim chỉ dùng net) — vô tội ở khoản này.

Sweep (na_01, tái tạo net từ giá raw, fee 0.4% giữ, slippage biến thiên; R0.6 khớp anchor y hệt):

| hệ (turnover/năm) | R0.6 | R0.7 | R0.9 | R1.1 | R1.3 | R1.5 |
|---|---|---|---|---|---|---|
| c2 K22 (10.8×) | ×18.03 | ×16.75 | ×14.45 | ×12.47 | ×10.92 | ×9.39 |
| c2 K23 (10.6×) | ×17.14 | ×15.94 | ×13.88 | ×12.01 | ×10.39 | ×9.05 |
| c2 K25 (10.2×) | ×16.34 | ×15.23 | ×13.25 | ×11.53 | ×10.02 | ×8.72 |
| base K20 (11.2×) | ×17.70 | ×16.29 | ×14.01 | ×12.11 | ×10.32 | ×8.74 |
| base K25 (10.0×) | ×14.78 | ×13.74 | ×12.00 | ×10.48 | ×9.15 | ×7.99 |
| **gb K25 (6.6×)** | ×14.44 | ×13.86 | ×12.65 | ×11.66 | ×10.74 | ×9.90 |

Độ nhạy: R2 mất ~**−7%/0.1đ phí**, gb chỉ ~**−4%/0.1đ** (turnover 10.8 vs 6.6×/năm). MaxDD c2 cũng phình theo phí (−15.4→−17.4%) trong khi gb đứng im — matched-DD point trôi bất lợi cho R2 khi phí tăng.

**Phí hòa vốn của lợi thế R2-vs-gb** (nội suy, tie-break alphabet gốc):
- c2 K22: **R ≈ 1.35%** (f22 1.40%); c2 K23: 1.20% (f22 1.26%); c2 K25: 1.06% (f22 1.25%); base K20: 1.20%; base K25: 0.68% (f22 chết trên toàn dải).

**0.7% là lạc quan hay bảo thủ cho VN 2026?** Hai hệ CÙNG kiểu fill: entry = pullback-limit passive (slippage ≈ 0, na_04), exit ≈ 100% tại close/ATC (giá ATC = giá close, slippage danh nghĩa 0.1% là hợp lý cho vốn cá nhân). Phí retail 2026: commission 0.10-0.15%/chiều + thuế bán 0.1% → roundtrip thực **0.5-0.6% (vốn nhỏ) / 0.8-1.1% (vốn lớn, impact midcap)**. Kết luận: 0.6% CHẤP NHẬN ĐƯỢC cho scale cá nhân, nhưng **biên an toàn của R2 chỉ ~0.7đ phí** (chết ở 1.35%) trong khi gb chịu được vô hạn trong dải khảo sát — lợi thế R2 là lợi thế CÓ ĐIỀU KIỆN scale.

## 2. SETTLEMENT T+2.5 — TỘI NẶNG NHẤT

r2_nav.py **KHÔNG model settlement**: exit xử lý TRƯỚC entry cùng ngày (dòng 87-104), tiền bán xài lại NGAY trong phiên. VN thực tế: tiền bán về T+2 (chiều). Hệ hold-7d tái vào liên tục được bơm vòng quay ảo. Định lượng (na_02, tiền treo vẫn tính NAV, chỉ không xài được):

| hệ | lag0 (gốc) | lag1 | lag2 (thực tế) | lag3 |
|---|---|---|---|---|
| c2 K22 | ×18.03 | −5.5% | **−21.5%** (×14.15) | −25.3% |
| c2 K25 | ×16.34 | −2.7% | **−16.1%** | −23.7% |
| base K20 | ×17.70 | −7.5% | −22.9% | −29.6% |
| base K25 | ×14.78 | −1.9% | −11.4% | −22.7% |
| **gb K25** | ×14.44 | −6.4% | **−8.8%** (×13.17) | −16.5% |

Delta c2K22-vs-gb: lag0 **+24.9%** → lag2 **+7.4%** (f22: +12.1% → +4.4%). **Hơn 2/3 lợi thế headline là tiền-về-tức-thì.** Lối thoát thực tế: ứng trước tiền bán (~0.0375%/ngày ×2 ≈ +0.08% phí/vòng) ≈ kịch bản "R0.7 lag0" → delta +20.8%/+10.2% — sống, nhưng phải ghi rõ vào cost model, không được miễn phí như hiện tại.

## 3. SLOT-ASSIGNMENT BIAS — TỘI THỨ HAI

Tie-break tín hiệu cùng ngày = **sort alphabet symbol** (r2_nav.py dòng 59), quan trọng vì cash-constrained (skip_cash 324-509). Xáo 20 seed (na_03):

| hệ | alphabet | shuffle mean±sd | span | alphabet vs mean |
|---|---|---|---|---|
| c2 K22 full | ×18.03 | ×16.70±0.72 | **16.5%** | **+8.0%** (≈max của 20 lần xáo: 18.11) |
| c2 K25 full | ×16.34 | ×15.76±0.48 | 11.0% | +3.6% |
| c2 K22 f22 | ×3.92 | ×3.81±0.11 | 10.8% | +2.8% |
| base K25 | ×14.78 | ×14.90±0.41 | 11.5% | −0.8% |
| gb K25 | ×14.44 | ×14.22±0.42 | 12.2% | +1.6% |

Không có lookahead, nhưng con số công bố của c2 K22 (điểm "vùng ngọt" của R2_ROUND2) là **cú rút thăm alphabet gần đỉnh phân phối**. Dải ±8% quanh mean → **mọi chênh lệch <±5% giữa các variant/K trong R2_ROUND1/2 là noise tie-break** (bảng frontier §B so nhau 0-2% ở matched-DD = không phân biệt được). MaxDD bất biến theo seed (−15.4% mọi lần) — DD đo tin được, NAV điểm thì không. Chuẩn sửa: báo **shuffle-mean ± sd**, không báo alphabet.

## 4. FILL/PRICE — TRẮNG ÁN (có ghi chú)

na_00/na_04: (a) pnl_pct khớp giá từng lệnh 100%, KHÔNG double-count cost; (b) entry_raw nằm trong [low,high] ngày entry ~91% cả hai hệ, median mua 0.22% DƯỚI close, chỉ 3-4% khớp đúng đáy ngày — passive limit thực tế, không bottom-tick; (c) exit_raw == close ±0.1% ở ~89% cả hai hệ, MỌI loại exit (signal/overext/trail/dtstop đều fill tại close — R2 không được ưu ái giá stop trong ngày); (d) ~9-11% lệnh lệch >0.1% (p95 ~ +5%) = mismatch adjustment (cổ tức/điều chỉnh) giữa data engine (duckdb) và ohlcv.db của audit — **đối xứng y hệt giữa c2 (90.9%) và gb (90.2%)**, chỉ ảnh hưởng mark trong-hold (nội suy ratio0→ratio1 nuốt mất), không ảnh hưởng NAV cuối. Ghi chú duy nhất: 48.5% exit của R2 là overext/trail/dtstop — nếu serving thực thi bằng stop-market trong phiên thay vì chờ close thì giá thật sẽ KHÁC backtest; đây là rủi ro triển khai, không phải bias khung đo.

## 5. GIẢI PHẪU ĐÁY 2025-04 — TRẮNG ÁN VỀ MAY MẮN, có caveat cấu trúc

Episode (na_05, c2 K22): peak 2025-01-02 → trough 2025-04-09 (−15.28%), phục hồi 2025-05-13. Sự kiện = tariff crash 04-03..04-09. Sự thật quan trọng: **cả hai hệ vào crash gần như TRỐNG SỔ** (04-02: c2 = 2 vị thế/9% exposure, gb = 3/14%) — DD không đến từ ôm lệnh xuyên crash mà từ **bắt dao rơi**: c2 lên 50% exposure ngày 04-03 và 98% đúng đáy 04-09; gb y hệt (53%/94%). Các lệnh bắt đáy sau đó dương (c2 +5.0%/24 lệnh, gb +12.9%/21). Cùng cửa sổ 03-01..05-15: c2 −14.51% vs gb −12.79% — **c2 rơi SÂU HƠN gb trong chính episode này**; MaxDD toàn cục của gb (−15.31%) nằm ở 2021-02-01, KHÁC episode. Trống-sổ-trước-crash là nhân quả (gate upleg_abovema20 chặn entry suốt downtrend tháng 3), không phải may mắn sổ; và không bất đối xứng giữa hai hệ. Caveat: MaxDD của MỌI điểm K đều neo trên một episode có cấu trúc "trống sổ khi crash nổ" — crash nổ giữa lúc full-book (kiểu 2022) sẽ sâu hơn mọi con số DD trong bảng frontier; DD frontier là lower-bound lạc quan, cho CẢ hai hệ.

## 6. FROM-2022 — CÔNG BẰNG, nhưng thiếu đối chứng gb

Code: `_f22` lọc trades entry ≥2022-01-01, khởi động 100% tiền mặt, calendar cắt từ 2022 — áp dụng y hệt mọi hệ, không thừa kế vị thế → công bằng. Tội nhỏ: R2_ROUND2 so f22 của c2 với base mà **không có gb f22**. Bổ sung (na_01): **gb K25 f22 = ×3.50** → c2 K22 f22 3.92 = +12.1%, c2 K25 3.80 = +8.6% (alphabet); shuffle-mean chỉ còn **+8.2%/+2.5%** — sát noise.

## KỊCH BẢN VẬN HÀNH TRUNG THỰC (na_06: shuffle-mean 20 seed × phí × lag)

| kịch bản | c2 K22 vs gb (full) | c2 K25 vs gb (full) | c2 K22 vs gb (f22) |
|---|---|---|---|
| R0.6 lag0 — khung gốc | +17.4% | +10.9% | +8.2% |
| **R0.7 lag0 — ứng trước tiền bán (khả thi nhất)** | **+13.6%** | +7.6% | **+6.4%** |
| R0.7 lag2 — không ứng | +1.3% | −0.9% | −1.4% |
| R0.9 lag0 — phí dày | +6.2% | +1.4% | +3.1% |
| R0.9 lag2 — xấu nhất | −4.4% | −6.0% | −4.1% |

## KẾT LUẬN + SỬA

1. **Khung đo THIÊN VỊ turnover-cao**, hướng: bơm R2. Ba kênh: settlement lag0 (lớn nhất, −17.5đ delta), tie-break alphabet may cho c2 K22 (+8%), cost 0.6% thay vì 0.7% danh nghĩa (nhỏ, chung).
2. **Lợi thế R2-vs-gb thật** sau gỡ bias, tại vận hành khả thi (ứng trước tiền bán, phí retail): **+13.6% full / +6.4% f22 tại K22** — vẫn dương nhưng KHÔNG phải +19..+25%; f22 +6.4% chỉ ~1.7 sd noise tie-break. Không ứng trước tiền bán → lợi thế = 0.
3. **Biên an toàn phí**: lợi thế chết tại R≈1.35% (K22); mỗi +0.1đ phí ăn ~3đ lợi thế. Chỉ an toàn ở scale vốn cá nhân.
4. Sửa khung trước mọi quyết định promote: (a) thêm `settle_lag=2` HOẶC cộng phí ứng trước ~0.08%/vòng vào cost model; (b) báo NAV shuffle-mean±sd thay cho alphabet; (c) sửa narrative "0.7%" → 0.6% hiệu dụng; (d) chạy lại frontier §B R2_ROUND2 trên khung đã sửa — các kết luận matched-DD 0-2% hiện tại đều dưới noise floor.
