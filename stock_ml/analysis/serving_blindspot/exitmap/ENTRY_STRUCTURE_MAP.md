# ENTRY STRUCTURE MAP — giải phẫu cấu trúc entry + hình thái PnL của gb_x08 (blind-spot map v2)

*2026-07-10. Offline thuần, không đốt run nào. Nguồn: Postgres `run_trades` run_id `template/gb_x08-32a8dfee`
(1378 lệnh, 2020-04-07 → 2026-06-16, 61 mã, Σu = +128.50, WR 57.0%, mean +9.32%/lệnh) + OHLCV
`stock-serving\data\ohlcv.db`. Proxy tín hiệu cho mục 4: frame serving 2643 (`signals.csv`/`unfilled_signals.csv`,
79% trades trùng gb_x08 — caveat ghi rõ). Scripts: `em_00_dump.py` … `em_04_exante.py`; dữ liệu enrich:
`gbx08_enriched_em.csv`. Đơn vị u = pnl fraction (1u = 100% trên 1 vị thế).*

*Lưu ý: các file `em01_dump.py`/`em02_enrich.py`/`em03_tables.py`/`gbx08_enriched.csv` trong cùng thư mục
là của một tiến trình phân tích song song khác, không thuộc chuỗi này.*

---

## 1. Anatomy lệnh thua lớn — "knife sau run-up nóng" KHÔNG tồn tại theo dạng user sợ

**Top-50 lệnh thua nặng nhất = −6.50u** (5.1% tổng u dương ròng). Phân bố năm: 2020:1, 2021:7, **2022:13**,
2023:7, 2024:4, **2025:10, 2026:8** — dày lên ở năm bear/sideways, không dồn cụm 1 năm.

**Kiểm tra chữ ký knife (median top-50 losers vs winners vs toàn mẫu):**

| feature | losers50 | winners50 | ALL |
|---|---|---|---|
| runup20 trước tín hiệu | +5.1% | +8.2% | +4.2% |
| runup60 | +7.2% | +12.1% | +4.3% |
| dist MA20 tại fill | −0.7% | +1.3% | −0.8% |
| dist MA60 tại fill | −1.4% | +3.0% | −0.7% |
| snr mã tại fill | 0.026 | 0.092 | 0.000 |
| vol_shock tại fill | 1.29× | 1.08× | 1.11× |
| fill_age (bar) | 3.5 | 3.0 | 6.0 |
| depth10 sau fill | **−14.8%** | −1.3% | −3.2% |
| MFE | +3.0% | +115.8% | +8.1% |
| hold | 6.5 | 89.5 | 14 |

- Percentile của losers trên phân phối toàn mẫu: runup20 median-pctile **0.54**, runup60 **0.58**, dist_ma20 0.51,
  vol_shock 0.62, snr 0.55 → **không có feature ex-ante nào tách losers khỏi đám đông**. Chỉ 5/50 losers có
  runup20 >p80 (kỳ vọng ngẫu nhiên = 10). Run-up nóng KHÔNG phải điềm knife.
- Ngược hẳn trực giác sợ hãi: **quintile runup20 cao nhất là cohort TỐT NHẤT** (Q4: mean +16.7%, WR 69.6%;
  Q0 giảm-sâu: +11.5%, WR 63.4%; tệ nhất là vùng giữa Q1: +3.9%). Interaction "hot + fill nhanh" (runup20≥p80
  & fill_age≤2, n=84): mean **+26.6%, WR 75.0%, min −16.2%** — cohort tốt nhất toàn map.
- Chữ ký THỰC của losers (mô tả, không gate được): **fill rất nhanh (33/50 khớp ngày 1-5 của window, median 3.5
  bar) → giá tiếp tục rơi thêm −14.8% trong 10 bar sau fill → không bao giờ có MFE (median +3%) → signal-exit
  cắt sau ~6.5 phiên.** Tức knife là "rơi-xuyên-limit", chỉ nhận diện được SAU fill (depth10), không trước.
  13/50 losers từng có MFE ≥+5% trước khi chết (giveback nhỏ); 5/50 là fill muộn ≥d16.

**Tail:** lệnh ≤−15% chỉ **9 lệnh, −1.49u = 5.8% tổng u âm = 1.2% tổng u ròng**; ≤−20%: đúng 1 lệnh (−0.20u,
loss sâu nhất −25.5%). Mở rộng ≤−10%: 52 lệnh, −6.70u = 26.2% u âm = 5.2% u ròng. **Phân phối loss thin-tail —
exit-signal hiện tại đã chặn knife rất tốt.** Nỗi sợ knife-tail về cơ bản được giải tỏa bằng số; trần u của mọi
can thiệp tail-side chỉ ~6.7u lý thuyết (thực tế thấp hơn nhiều vì cắt sớm cũng giết lệnh sẽ hồi).

## 2. Chất lượng fill theo vị trí trong window 40

| bucket | n | Σu | mean | WR | hold med | MFE med | u/slot-day |
|---|---|---|---|---|---|---|---|
| ngày 1-5 | 670 | 80.02 | **+11.9%** | 57.9% | 23 | 13.7% | 0.0033 |
| ngày 6-15 | 409 | 32.34 | +7.9% | 55.3% | 11 | 7.0% | 0.0028 |
| ngày 16-40 | 299 | 16.14 | +5.4% | 57.5% | 4 | 4.8% | 0.0030 |

Fine-grain: (0,2]=+14.7% → (2,5]=+9.3% → (5,10]=+8.5% → (25,40]=+5.9% — **per-trade edge giảm đơn điệu theo
tuổi lệnh treo, gần một nửa khi fill muộn**. Bền qua năm (trừ 2025: d6-15 nhỉnh hơn; 2024 phẳng ~0 cả 3 bucket).

**NHƯNG fill muộn KHÔNG phải cohort xấu:** vẫn +16.1u, WR ngang (57.5%), hold ngắn hơn nhiều (4 vs 23 phiên)
nên **hiệu suất trên slot-day gần như phẳng (0.0033 / 0.0028 / 0.0030)**. Lệnh muộn ăn ít nhưng chiếm slot ít.
→ Lever "depth/conviction taper theo tuổi lệnh treo" có cơ sở per-trade nhưng **kỳ vọng u thấp và rủi ro
occupancy-reshuffle giống hệt lý do bot_shallow/deepen chết**. Xếp hạng thấp (xem §6).

## 3. Cấu trúc xổ số của PnL

- **Theo lệnh: nửa-xổ-số.** Top 1% lệnh = 16.6% u; top 5% = **47.3%**; top 10% = 71.6%; top 20% = 98.2%
  (⇒ 80% lệnh còn lại cộng lại ≈ 0). 87 lệnh ≥+50% đóng 70.3u (55%); 16 lệnh ≥+100% đóng 23.4u.
  Gross+ 154.1u / gross− 25.6u.
- **Theo mã: mặt bằng RỘNG, không sống nhờ mega.** Top-6/61 mã chỉ chiếm 19.3% tổng u. Chỉ **1/61 mã âm ròng**
  (SAB −0.42u, WR 23%, âm 6/7 năm — mã "chết lặp lại" duy nhất, u quá nhỏ để đáng làm gì). FRT/BSR/LPB dương
  cả 7/7 năm. Không tồn tại cụm mã-độc cần đặc trưng hóa.
- **Gini theo năm:** 2020: 0.450 → 2022: 0.298 → **2024: 0.218** → 2025: 0.416. Năm xấu concentration
  giảm vì mất đuôi phải (không phải vì mặt bằng dày lên): 2024 top-10% lệnh = 148.6% u năm (mặt bằng 90% còn
  lại ÂM ròng); 2026H1 tổng âm. **Kết luận: model sống nhờ đuôi phải của mặt-bằng-rộng; năm nào thị trường
  không cấp đuôi phải thì mặt bằng ≈ hòa-vốn-trừ-phí.**

## 4. Slot utilization & dead-time (slot = theo-mã, max concurrent quan sát 58/61)

% mã-ngày theo trạng thái (61 mã × lịch; proxy tín hiệu = frame 2643, 79% trùng — số tuyệt đối có nhiễu,
hình dạng đáng tin):

| năm | có vị thế | idle: limit treo (đói fill) | idle: có signal 40-bar nhưng hết treo | idle: KHÔNG tín hiệu |
|---|---|---|---|---|
| 2020 | 60.6% | 17.2% | 13.5% | 8.7% |
| 2021 | 60.4% | 12.4% | 18.0% | 9.2% |
| 2022 | 31.7% | 16.7% | 34.1% | 17.5% |
| 2023 | 45.8% | 18.9% | 20.7% | 14.5% |
| 2024 | 38.6% | 21.2% | 20.9% | 19.4% |
| 2025 | 41.7% | 19.7% | 20.7% | 17.9% |
| 2026H1 | 31.0% | 15.2% | 33.4% | 20.4% |

- Occupancy sụt hẳn sau 2021: 60% → 31-46%. Đáy 20-ngày: 10/2022, 5/2022, 8/2024 (~6% occupancy).
  Toàn kỳ chỉ đúng 1 ngày occ=0 (2022-05-06) — hệ không bao giờ "chết hẳn", chỉ mỏng đi.
- **Chẩn đoán 2024-26: "đói tín hiệu (mới)" > "đói fill".** Đói fill (limit đang treo không khớp — chính là
  cohort RUNAWAY đã giải phẫu) ổn định 15-21% mọi năm, KHÔNG phình ra ở năm xấu. Cái phình là hai cột phải:
  signal cũ đã dùng xong/hết hạn treo + không có signal nào trong 40 bar = **40-54% mã-ngày 2024-26** so với
  ~22-27% thời 2020-21. Điểm mù lớn hơn ở năm gần là **coverage/tần suất tín hiệu**, không phải limit quá sâu.

## 5. Hình thái theo năm gần — model già đi hay thiếu sóng?

| năm | n | Σu | mean/lệnh | WR | hold med | MFE med | PF | lệnh ≥+30% | market ret20 mean (universe) |
|---|---|---|---|---|---|---|---|---|---|
| 2020 | 180 | 45.44 | +25.2% | 76.1% | 35 | 23.1% | 29.8 | 52 | +6.5% |
| 2021 | 183 | 26.44 | +14.5% | 66.7% | 38 | 18.2% | 9.5 | 32 | +6.9% |
| 2022 | 277 | 11.34 | +4.1% | 56.0% | 8 | 7.9% | 3.0 | 17 | −3.1% |
| 2023 | 223 | 18.04 | +8.1% | 58.7% | 12 | 8.5% | 5.2 | 21 | +2.6% |
| 2024 | 192 | 2.35 | +1.2% | 47.4% | 6 | 4.5% | 1.7 | 3 | +1.3% |
| 2025 | 215 | 25.42 | +11.8% | 53.5% | 9 | 5.8% | 7.4 | 40 | +1.6% |
| 2026H1 | 108 | **−0.55** | **−0.5%** | **32.4%** | 9.5 | 5.2% | **0.86** | 5 | −0.5% |

- Per-trade edge mỏng đi rõ so với 2020-21 (25%→~10%), nhưng **2025 là phản chứng của giả thuyết "model già"**:
  market proxy 2025 (+1.6%) gần bằng 2024 (+1.3%) mà model in +25.4u nhờ 40 lệnh ≥+30% — khi có sóng cục bộ,
  máy bắt được ngay. Edge đồng pha chặt với universe ret20 (thứ hạng năm gần trùng khớp).
- Điểm đáng lo THẬT: **2026H1 là kỳ âm đầu tiên** (WR 32.4%, PF 0.86) và WR trượt gần đơn điệu 76→32% qua 7 năm
  — floor được exit bảo vệ (không âm sâu) nhưng chi phí cơ hội của năm sideways ngày càng nguyên chất: mặt bằng
  lệnh về 0, sống hoàn toàn nhờ đuôi phải mà 2024/2026 thị trường không cấp.
- Kết luận trung thực: **chưa đủ bằng chứng "model già đi"; đủ bằng chứng "edge = f(sóng rộng của universe)"**
  và cường độ f không suy giảm (2025). Rủi ro là chuỗi năm không-sóng, không phải decay tham số.

## 6. Bản đồ điểm-giới-hạn + lever hợp lệ (xếp hạng)

| # | điểm giới hạn | u ước lượng | lever đề xuất (hợp lệ) | trạng thái |
|---|---|---|---|---|
| 1 | **Mặt bằng lệnh mỏng: 80% lệnh ≈ 0u; cohort snr/dist-MA thấp kéo trung bình xuống** (ex-ante tại bar signal: snr_sig Q4 = +15.8%/WR 68.8% vs Q1 +5.5%; dma20_sig Q4 = +17.5%/WR 77.2% vs Q0 +5.5%; snr_sig Q4 dương cả 2024 +3.7% lẫn 2026 +4.6% — cohort duy nhất dương 2026) | dịch chuyển nội bộ 3-8u/kỳ nếu nghiêng vốn | **Conviction-scaling liên tục theo snr_sym + dist_MA20 tại bar signal** — điều hòa depth/notional, KHÔNG gate (SELECTION WALL không bị vi phạm: không cắt lệnh nào). Knob conviction-scaling ĐÃ CÓ trong engine; cần thêm input feature mới (per-symbol SNR ex-ante) = knob mới nhỏ | thiết kế sẵn, chưa code |
| 2 | **Đói tín hiệu năm gần: 40-54% mã-ngày 2024-26 không có signal mới/limit treo** (§4) — chi phí cơ hội lớn nhất hệ | trần lý thuyết lớn (occupancy 31%→45% ~ +30-50% số lệnh năm gần) nhưng CHƯA có cơ chế | trục nghiên cứu coverage: vì sao frame im lặng ở 1/5 mã-ngày (universe? ngưỡng ensemble OR-union?). KHÔNG phải lever chỉnh knob — cần dòng nghiên cứu riêng, ngoài scope vòng này | mở |
| 3 | **Tail loss ≤−10%: 52 lệnh −6.7u, chữ ký chỉ hiện SAU fill (depth10 −14.8%, MFE≈0)** | trần 6.7u, thực thu ước ≤2-3u | exit-side liên tục: khuếch đại exit_score/giảm ngưỡng defer cho vị thế **chưa từng có MFE** và đang âm sâu (điều hòa liên tục theo MFE-to-date — cùng họ với exit_snr_defer_min_giveback hiện có, KHÔNG phải stop cơ học vì vẫn đi qua signal-exit). Cần knob mới exit-side | thiết kế, cần cẩn trọng ranh giới "stop cơ học" |
| 4 | Per-trade edge giảm ~50% theo tuổi lệnh treo (§2) nhưng u/slot-day phẳng | ±2-5u, dễ âm vì reshuffle | conviction-taper theo fill-age (liên tục). **Khuyến nghị: KHÔNG ưu tiên** — cùng cơ chế chết của bot_shallow/deepen | xếp cuối |
| 5 | SAB âm ròng 6/7 năm | −0.4u | không đáng hành động (mô tả để hoàn chỉnh bản đồ) | đóng |

**Không chạy leaderboard run nào vòng này.** Lever #1 là ứng viên duy nhất đáng probe vòng sau (offline sim
trước bằng chính `gbx08_enriched_em.csv` + reshuffle-check occupancy như protocol runaway).
