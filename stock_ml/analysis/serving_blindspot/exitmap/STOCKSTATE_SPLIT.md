# PHÂN RÃ QUẦN THỂ THEO TRẠNG THÁI CỔ PHIẾU — mkt_drop suppress & struct-donch80

Ngày lập: 2026-07-10. Scripts: `em10_stockstate.py` (đo discriminator tại bar can thiệp đầu tiên +
grid ranh giới theo giả thuyết), `em10b_flipscan.py` (grid CẢ HAI CHIỀU — falsification đầy đủ).
Output: `stockstate_mkt.csv`, `stockstate_struct.csv`. Input tái dùng: `pdr_forensic_scan.csv`
(blocker=mkt_drop, bar = sw_date đầu tiên bị nuốt), `pdr_structtrail_cf.csv` (bar = cf_date %-trail fire).

**Giả thuyết kiểm định (từ vụ PDR):** 2 cơ chế MÙ TRẠNG THÁI CỔ PHIẾU — suppress/struct-trail chỉ
đúng cho cú rơi beta, sai khi trend riêng của mã ĐÃ GÃY (dưới MA20, SNR âm, sell active lâu, rơi
idiosyncratic). Kỳ vọng: ranh giới "trạng thái gãy" cắt ≥70% u-phá, mất ≤20% u-cứu.

## 1. Hai quần thể (≥2022, delta = pnl_thực − pnl_counterfactual; CỨU = delta>0, PHÁ = delta<0)

- **mkt_drop suppress**: n=266, u-cứu +9.68 (156 lệnh), u-phá −3.35 (109 lệnh).
- **struct-donch80 vs %-trail**: n=161, u-cứu +14.58 (91), u-phá −5.32 (70).

Discriminator đo tại bar can thiệp (chỉ dữ liệu tới bar đó): ma20_rel, ma60_rel (close/MA−1),
blw20_run (bar liên tiếp < MA20), snr21 mã, sell_run (bar force-sell liên tiếp, proxy),
rs5 (ret5 mã − ret5 VNI), idio5 (ret5 − beta120×ret5 VNI), age, gain, giveback_e.

### Median CỨU vs PHÁ — phân phối CHỒNG NHAU và ĐẢO HƯỚNG so với giả thuyết

| Cơ chế | Biến | CỨU (median) | PHÁ (median) | Giả thuyết đòi |
|---|---|---|---|---|
| mkt_drop | ma20_rel | **−5.3%** | −3.9% | PHÁ phải gãy sâu hơn — NGƯỢC |
| mkt_drop | snr21 | **−0.095** | −0.022 | PHÁ phải âm hơn — NGƯỢC |
| mkt_drop | sell_run | 1 (q75=3) | 1 (q75=1) | PHÁ phải dài hơn — NGƯỢC |
| mkt_drop | idio5 | −0.010 | −0.004 | ~trùng |
| struct | ma20_rel | +1.5% | +2.0% | trùng (fire bar luôn sau đỉnh gain lớn, mã còn trên MA) |
| struct | snr21 | +0.232 | +0.207 | trùng |

### Bucket theo u (mkt_drop, ≥2022) — suppress hoạt động TỐT NHẤT chính ở vùng "trạng thái gãy"

| Bucket | n | u-cứu | u-phá | NET |
|---|---|---|---|---|
| ma20_rel ≤ −8% (gãy sâu nhất) | 52 | +2.88 | −0.39 | **+2.49** |
| ma20_rel (−8%, −4%] | 105 | +2.90 | −1.16 | +1.74 |
| ma20_rel (−4%, 0] | 92 | +3.12 | −1.24 | +1.88 |
| ma20_rel (0, +4%] | 9 | +0.74 | −0.16 | +0.58 |
| ma20_rel > +4% | 8 | +0.03 | −0.39 | **−0.36** (vùng âm duy nhất, quá nhỏ) |
| snr21 ≤ −0.2 | 50 | +2.42 | −0.24 | +2.19 |
| snr21 > +0.2 | 11 | +0.32 | −0.42 | −0.10 |
| sell_run > 3 bar (kiểu PDR) | 32 | +2.23 | −0.18 | **+2.05** |
| rs5 ≤ −4% (rơi idiosyncratic) | 65 | +2.09 | −0.45 | +1.63 |

Struct tương tự: net dương lớn nhất ở rs5 ≤ −4% (+6.52) và sell_run=0 (+9.11); vùng âm duy nhất
rs5 > 0 (−0.59). Tức **washout càng sâu/idiosyncratic thì cơ chế bảo vệ càng ĐÚNG** — cổ phiếu gãy
sâu dưới MA20 với sell active nhiều phiên trong cú sập thị trường đa số HỒI (mean-reversion), đúng
thiết kế washout-protection. PDR 4/2024 là đuôi 6% hiếm, không đại diện quần thể.

## 2. Grid ranh giới — CẢ HAI CHIỀU (em10 + em10b), tiêu chí cắt ≥70% u-phá / mất ≤20% u-cứu

### mkt_drop (u-phá tổng 3.35)
- Chiều giả thuyết (flag khi gãy): **TẤT CẢ net ≤ +0.017u**. Ví dụ ma20_rel<−4% net −0.7;
  sell_run≥5 net −1.11; idio5<−0.04 net −0.69; snr21<−0.4 net −0.38. Kép tốt nhất
  snr21<−0.4 & rs5<−0.08: net +0.017 (n=2).
- Chiều ngược (flag khi còn khỏe): đơn tốt nhất snr21≥0.2 net +0.10 (pha_cov 12.5%);
  kép tốt nhất **ma20_rel≥+2% & snr21≥0.1: net +0.32u, pha_cov 10.7%, cuu_loss 0.3%, không đè mega**.
  → xa vô vọng so với 70%: 89% u-phá nằm TRỘN trong vùng cứu, không mặt phân tách.

### struct-donch80 (u-phá tổng 5.32)
- Chiều giả thuyết: gần như rỗng — tại fire bar mã hầu hết còn trên MA20/MA60, SNR dương
  (%-trail chỉ fire sau gain ≥27%). Best net +0.07 (n=1).
- Chiều ngược: đơn tốt nhất **rs5≥0: net +0.59u, pha_cov 30.9%, cuu_loss 7.2% nhưng đè 1 mega**;
  kép rs5≥0 & giveback_e<8%: net +0.45, pha_cov 10.2%. Cũng xa tiêu chí.

Mega-guard: các cú cứu lớn (VIC 7/2025 +1.18, GEX 4/2025 +1.10, LPB 10/2023 +0.88, SBT +0.64,
BSR +0.60, STB +0.56, PDR 3/2023 +0.54, DGC +0.39, VCI +0.22…) đều nằm rải khắp vùng "trạng thái
gãy" — mọi ranh giới đủ rộng để chạm 70% u-phá đều chém vào chúng trước.

## 3. Counterfactual first-order (bước 2)

| Cơ chế | Ranh giới tốt nhất | Δnet ≥2022 | Ngưỡng protocol |
|---|---|---|---|
| mkt_drop | ma20_rel≥+2% & snr21≥0.1 (chiều NGƯỢC giả thuyết) | **+0.32u** | +1.5u → FAIL |
| struct-donch80 | rs5≥0 tại fire bar | **+0.59u** (đè 1 mega) | +1.5u → FAIL |

→ **DỪNG theo protocol trước bước 3.** Không tạo knob, không probe `sv2_`, không chạy seed/regime.
(Walk-CF per-bar đầy đủ không cần thiết: walk chỉ có thể làm exit MUỘN hơn first-order → net chỉ
giảm thêm về phía cơ chế hiện trạng, không cứu được con số +0.32/+0.59.)

## 4. VERDICT — FIX CHẾT, giả thuyết trung tâm bị FALSIFY (đảo hướng)

1. Hai cơ chế **không mù trạng thái cổ phiếu theo hướng có hại**: chính vùng "trend mã đã gãy"
   (dưới MA20 sâu, SNR âm, sell-run dài, rơi idiosyncratic) là nơi suppress/struct-trail kiếm
   nhiều u nhất (+2.0 ÷ +2.5u mỗi lát) và u-phá ở đó gần bằng 0. Washout sâu đa số hồi — kể cả
   khi là cú rơi riêng của mã.
2. Vùng net âm duy nhất là trạng thái KHỎE (ma20_rel>+4%, snr21>+0.2, rs5>0) nhưng tổng chỉ
   −0.3 ÷ −0.6u trên 4 năm — dưới ngưỡng hành động, và ranh giới chiều đó cũng chỉ gom được
   10–31% u-phá.
3. Con số quyết định: best counterfactual +0.32u (mkt_drop) / +0.59u (struct) ≥2022, so với yêu
   cầu +1.5u. Khoảng 89% (mkt) / 69% (struct) u-phá không tách được khỏi u-cứu bằng bất kỳ
   trạng-thái-cổ-phiếu đơn/kép nào tại bar can thiệp.
4. Khớp với PDR_FORENSIC §4: chi phí kiểu PDR là giá đã tính trong thiết kế. Tuyến
   "stock-state veto cho exit suppress/trailing" **ĐÓNG**. Nếu còn muốn giảm đuôi, hướng duy nhất
   chưa thử là thông tin NGOÀI trạng thái giá của mã (ví dụ chất lượng breadth-hồi-phục của chính
   cú washout), nhưng chưa có bằng chứng nào từ dữ liệu này ủng hộ.

Hạn chế: proxy sell = force-gates (cận dưới n bị nuốt); struct-CF xấp xỉ em09 (bỏ overext-arm 4%);
first-order không mô phỏng slot/re-entry. Không hạn chế nào đủ lật dấu kết quả (khoảng cách tới
ngưỡng là 3–5 lần).
