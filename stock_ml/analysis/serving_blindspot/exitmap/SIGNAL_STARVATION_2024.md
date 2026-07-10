# SIGNAL STARVATION 2024-26 — "thị trường hết sóng" hay "model mù sóng"?

*2026-07-10. Offline thuần, không đốt run nào. Nguồn: OHLCV `stock-serving\data\ohlcv.db` (universe top150
của frame 2643), buy-signals `signals.csv` (frame serving 2643, 79% trades trùng gb_x08 — proxy, caveat như
ENTRY_STRUCTURE_MAP §4), trades thật Postgres `template/gb_x08-32a8dfee` (1378 lệnh), scores mọi bar từ
`prediction_history.parquet` của bundle champion (234,747 bar × 5 head + exit). Scripts: `ss_00_waves.py`
(kiểm kê sóng) → `ss_01_catch.py` (tỉ lệ bắt) → `ss_02_autopsy.py` (giải phẫu sóng mù) → `ss_03_ceiling.py`
(trần thu hoạch); dữ liệu: `ss_waves_w1/w2.csv`, `ss_catch_w1/w2.csv`, `ss_autopsy_w1.csv`.*

**Câu hỏi (từ phát hiện #4 của ENTRY_STRUCTURE_MAP):** 40-54% mã-ngày 2024-26 không có tín hiệu mới —
vì thị trường hết sóng, hay vì model mù sóng? Nếu mù: near-miss (điều hòa được) hay true-silence (cần head mới)?

---

## 1. Kiểm kê sóng độc lập với model

Hai định nghĩa cơ học, chỉ dùng OHLCV, non-overlap greedy per symbol:

- **W1 "dip-rally"**: chân sóng = trailing-21-bar low của `low` (refine argmin [t, t+3]),
  `max(close[t+1..t+40])/close[chân] − 1 ≥ 15%` — cùng họ định nghĩa wavestart, horizon 40 bar khớp
  pullback window của champion.
- **W2 "reclaim-MA20"**: close cắt lên MA20 sau ≥5 bar liên tục dưới, forward max 40 bar ≥ +10%.

| năm | W1 n | W1 gain TB | W2 n | W2 gain TB | ghi chú |
|---|---|---|---|---|---|
| 2019 | 183 | 22.7% | 209 | 20.6% | 137/150 mã có dữ liệu |
| 2020 | 379 | 28.0% | 379 | 30.0% | |
| 2021 | 407 | 34.1% | 429 | 30.3% | |
| 2022 | 508 | 26.2% | 374 | 24.9% | bear: nhiều đáy-nảy 15% |
| 2023 | 311 | 24.3% | 366 | 18.6% | |
| **2024** | **257** | 23.0% | **258** | 18.2% | **−35% vs mean 2020-21 (393)** |
| 2025 | 366 | 26.7% | 334 | 23.0% | ≈ mức 2020 |
| 2026H1 | 111 (~222 ann.) | 22.0% | 91 | 19.4% | ~−45% vs 2020-21 (annualized) |

→ **"Ít sóng thật" đúng cho 2024 (−35%) và 2026H1 (−45%), nhưng KHÔNG đúng cho 2025** (366 ≈ 379 của 2020).
Biên độ sóng cũng mỏng đi ở 2024/2026 (gain TB 22-23% vs 28-34% thời 2020-21).

## 2. Tỉ lệ bắt sóng của model theo năm — bảng trung tâm (W1)

`sig%` = sóng có ≥1 buy-signal (frame 2643) trong ±10 bar quanh chân; `trd%` = sóng có ≥1 trade gb_x08
(entry_signal_date trong [chân−10, peak], mỗi trade gán 1 sóng); `u/sóng-trade` = u thực trung bình trên
sóng có trade.

| năm | n sóng | sig% ±10bar | trd% | u tổng trên sóng-trade | u/sóng-trade |
|---|---|---|---|---|---|
| 2020 | 379 | 59.1%* | 27.7% | +30.96 | 0.29 |
| 2021 | 407 | 79.6% | 28.7% | +20.34 | 0.17 |
| 2022 | 508 | 60.0% | 32.1% | +12.13 | 0.07 |
| 2023 | 311 | 73.0% | 24.8% | +13.92 | 0.18 |
| **2024** | 257 | **78.2%** | 25.7% | **+2.00** | **0.03** |
| 2025 | 366 | 66.7% | 29.2% | +22.38 | 0.21 |
| 2026H1 | 111 | 59.5% | **35.1%** | +0.90 | 0.02 |

*\*2020 thấp giả tạo: z-score cần warmup 60 bar → frame câm đến ~T4/2020, đúng lúc đáy COVID. Blind 2020
thực tế thấp hơn nhiều con số 40.9%.* W2 cùng hình dạng (sig% 78-88% mọi năm, 2024 = 83.0%).

**Kết luận mục 2: tỉ lệ bắt KHÔNG suy giảm ở 2024-26.** 2024 là năm bắt-signal TỐT NHẤT lịch sử (78.2%,
ngang 2021); trd% 2026H1 cao nhất (35.1%). Đói-tín-hiệu mã-ngày của §4 ENTRY_STRUCTURE_MAP phân rã thành:
(i) **ít sóng hơn** (−35%/−45%), (ii) **hold ngắn hơn** (6-9 bar vs 35-38 của 2020-21 → chuỗi có-vị-thế co lại,
nhả mã-ngày "trống" dù vẫn bắt đủ sóng), (iii) **u/sóng-bắt-được sập** (0.03/0.02 vs 0.17-0.29) — tức cái
thiếu ở 2024/26 không phải TẦN SUẤT bắt mà là THU HOẠCH trên mỗi lần bắt.

## 3. Autopsy 223 sóng-bị-mù 2024-26H1 (W1, không signal ±10 bar) — near-miss hay true-silence?

Tái lập đúng chain buy của champion từ scores mọi bar (khớp signals.csv **99.0/99.2%** hai chiều):
`buy = (zE>−1.9 & upleg_abovema20) ∪ z2>0.9 ∪ z3>0.7 ∪ z4>0.7 ∪ z5>0.7`, sell thắng buy cùng bar.

| phân loại (ưu tiên trên xuống) | n | % | định nghĩa |
|---|---|---|---|
| **SELL-VETO** | **218** | **97.8%** | điều kiện buy ĐÃ bật trong ±10 bar nhưng bar đó sell (downleg12 force-gate / belowma20p2 / lowbreadth / z-exit) đè chết |
| NEAR-MISS | 3 | 1.3% | max ensemble margin ∈ [−0.25, 0) |
| GATE-BLOCKED | 2 | 0.9% | zE qua ngưỡng nhưng gate đóng suốt cửa sổ, ensemble im |
| TRUE-SILENCE | 0 | 0.0% | mọi head thấp hẳn |

Ensemble margin (max z − threshold trong cửa sổ) của sóng mù: **median +1.34, p10 +0.05, p90 +7.54** — head
không những không im mà **kêu rất to trên ngưỡng**; 2024-26 không có sóng nào true-silence. Phân bố này giết
cả hai hướng cứu đã đặt cược ở đề bài: **không có đất cho điều-hòa-threshold** (near-miss 1.3%; nhất quán quy
luật "threshold tuning chưa bao giờ >+1.5") **và không cần head family mới** (không tồn tại true-silence để học).
Cái chặn duy nhất là **veto kiến trúc tại chân sóng**: sell-precedence + downleg force-gate — đúng vùng
"chân sóng dưới MA20" mà tuyến wavestart đã đóng sau 4 vòng falsification (1-slot).

Chú ý: SELL-VETO ở chân sóng là hằng số kiến trúc mọi năm (2020: 19.5%, 2022: 38.0%, 2025: 33.1%, 2026H1:
40.5% số sóng) — không phải bệnh mới của 2024-26. Và 83/223 sóng mù (37%) **vẫn có trade** nhờ limit treo từ
signal trước đó hoặc signal muộn (+6.33u) — mù-±10-bar ≠ mất trắng.

**Đặc điểm sóng mù vs sóng bắt được (2024-26H1, median):** giống nhau về thanh khoản (ADV20 ~0.05 tỷ, và
61% vs 64% ngoài nhóm 61 mã từng trade — không phải chuyện mã nhỏ/lạ); khác ở HÌNH THÁI: mù mọc từ hố sâu hơn
(drawdown trước −21.2% vs −14.3%), leo chậm hơn (26 vs 19 bar tới +15%), 100% chân dưới MA20, gate
upleg+MA20 mở muộn hơn (12 vs 8 bar sau chân — quá cửa sổ ±10). Tức sóng mù = **sóng-đáy-hố-sâu reclaim chậm**,
model chỉ "mù" trong đúng pha mà kiến trúc CẤM nó nhìn.

## 4. Trần thu hoạch (`ss_03_ceiling.py`)

Giả lập: phát tín hiệu tại bar buy-recon đầu tiên trong ±10 quanh chân sóng chưa-có-trade, treo limit
pullback 4.5%/40 bar như thường (không đổi cơ chế fill), giá trị = (peak/fill − 1) × deflator-hiệu-suất-thực
của champion cùng năm (deflator = Σu thực / Σ tiềm năng foot-to-peak trên các sóng ĐÃ trade — đã hàm chứa
exit sớm, giveback, cost, WR năm đó).

- Fill-rate qua pullback 4.5% cao: ~80% (152/191 sóng 2024; 226/259 sóng 2025) — **fill không phải nút cổ chai**, khớp phát hiện "đói fill ổn định" của map trước.
- Deflator theo năm: 2020: 1.12, 2021: 0.54, 2022: 0.30, 2023: 0.79, **2024: 0.14**, 2025: 0.87, **2026H1: 0.10** — hiệu suất chuyển-sóng-thành-u là cái sập ở năm chết, không phải coverage.

| năm | n sóng chưa-trade fill được | Σ tiềm năng fill-to-peak | trần u @100% bắt | @50% | @25% |
|---|---|---|---|---|---|
| 2024 | 152 | 28.8 | **+4.1** | +2.1 | +1.0 |
| 2025 | 226 | 39.7 | +34.4* | +17.2 | +8.6 |
| 2026H1 | 55 | 8.0 | **+0.8** | +0.4 | +0.2 |

*\*Trần 2025 phồng: deflator 0.87 vì trade 2025 giữ vượt peak-40-bar của sóng; và 100% capture của các chân
sóng đang-downleg là fantasy — chính cohort này là hố knife mà 4 vòng wavestart đã chết.*

**So với noise seed ±2-3 composite:** năm CHẾT (2024/2026H1) — trần +4.1u và +0.8u NGAY CẢ KHI bắt 100% sóng
mù, tức mức thực tế (≤25-50%) chìm trong noise. Năm SỐNG (2025) trần đáng kể trên giấy, nhưng 2025 champion
đã in +25.4u — không phải năm cần cứu.

## 5. Verdict

**"Thị trường hết sóng + sóng còn lại nghèo; model vô tội ở tầng coverage."**

1. 2024/2026H1 ít sóng thật (−35%/−45%) và sóng mỏng hơn; 2025 đủ sóng và model ăn ngay (+25.4u) — khớp
   kết luận §5 map trước: edge = f(sóng), cường độ f không suy giảm.
2. Tỉ lệ bắt sóng 2024-26 KHÔNG thấp hơn 2020-23 (2024 cao nhất lịch sử). Occupancy thấp năm gần =
   ít sóng × hold ngắn, không phải catch-rate.
3. Sóng-bị-mù: 97.8% SELL-VETO với score TRÊN ngưỡng (margin median +1.34) — **không phải near-miss (1.3%),
   không phải true-silence (0%)**. Cả hai hướng cứu dự kiến (điều hòa ensemble / head family mới học regime
   2022+) đều **không có đối tượng để cứu**.
4. **KHÔNG đáng mở chiến dịch coverage** theo trục union-thêm-head, hạ threshold, hay head mới: trần năm-chết
   ≤ +4u @100% capture (dưới noise ở mức capture thực). Điểm #2 của ENTRY_STRUCTURE_MAP §6 ("trục nghiên cứu
   coverage — mở") có thể **ĐÓNG** với dữ liệu này.
5. Con đường duy nhất còn u thật ở cohort này là gỡ **veto kiến trúc chân sóng** — tức tuyến wavestart/
   multi-position đã được xử: đóng ở 1-slot, chỉ mở lại qua cổng multi-position (quyết định 2026-07-09,
   ngoài scope vòng này). Trong khung 1-slot hiện tại, lever duy nhất đáng probe vẫn là #1 conviction-scaling
   (map trước), không phải coverage.

**Caveat:** signals = frame 2643 (79% trùng gb_x08); tái lập chain khớp 99%; trade→sóng gán first-match;
deflator là proxy thô (per-year, gộp multi-trade/sóng); sóng 2019 đếm được nhưng không đo catch (frame bắt đầu
2020 + warmup z).
