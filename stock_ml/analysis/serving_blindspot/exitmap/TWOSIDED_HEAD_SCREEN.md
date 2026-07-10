# SCREEN EXIT HEAD HAI CHIỀU (remaining upside/downside) — HỒ SƠ ĐÓNG LANE

Ngày: 2026-07-10. Scripts: `ts01_dataset.py` (dataset position-bar), `ts02_screen.py`
(folds + IC + decision proxy + cohorts), `ts03_extra.py` (biến thể decision bổ sung).
Dữ liệu: `ts_bars.parquet` (68,131 bar; 39,506 in-position có label), `ts_preds.parquet`,
`ts_sim.parquet`, `ts_sim_extra.parquet`, `ts_screen_results.json`. Offline thuần.

**Câu hỏi quyết định** (đặt trước mọi engine work): head ML dự đoán HAI CHIỀU
(rem_MFE = upside còn lại, rem_MAE = downside còn lại) có nhìn thấy gì mà force-gates
KHÔNG thấy không — và cái thấy thêm đó có đổi được thành quyết định exit tốt hơn không?

## 0. Thiết kế (leakage discipline chuẩn screen_ic.py)

- **Dataset**: mỗi (lệnh gb_x08 s42 × bar mở, entry+1 → decision bar) + 20 bar extension
  sau exit (chỉ dùng cho sim, không train). Label ex-post trên OHLCV close:
  `rem_MFE/rem_MAE = max/min(close[t+1..W])/close_t − 1`, W = min(exit+20, t+40).
- **Features 3 tầng**: GATE (12 = ĐÚNG input force-gates, parity em02 100% cả 3 flag tại
  decision bar: leg12/leg6 zigzag, bma20p2, nonbull EW<MA35p2, breadth MA50 + lowbreadth,
  drop5_z + mkt_drop, SNR20, f_dl12/f_nb/f_lb) → GATEPOS (+gain, peak, giveback, age,
  bars_since_peak) → FULL (+19 features `exit_vol_downpress`, port đúng DSL catalog.py:
  ATR Wilder, Bollinger n20k2, TsRank/CSRank/Quantile theo ops.py; CSRank trong universe
  61 mã; market_* trên EW index của chính universe theo market.py).
- **Folds OOS**: train = bar có label-window KẾT THÚC trước 1/1/y (purge overlap),
  test = bar năm y, y ∈ 2022..2026H1. LightGBM (400 trees, lr 0.05, leaves 31) × 2 target × 3 set.
- **Null**: 200 lần circular-shift label TRONG từng lệnh (giữ autocorrelation + giữ nguyên
  identity lệnh) — null này CÓ tâm khác 0: nó đo "biết lệnh nào nhưng không biết timing",
  tức phần vượt null = thông tin timing thật trong đời lệnh.
- **PnL convention sim**: fill close(bar quyết định + 1), trừ chi phí round-trip implied
  per-trade sao cho sim-exit-tại-decision-bar-thật == pnl_pct ghi sổ (khớp exact).

## 1. IC OOS theo fold — pooled Spearman (gate / gatepos / FULL), resid = IC của rank(pred_full) sau khi trừ rank-OLS trên pred_gatepos

### rem_MFE (upside còn lại)

| fold | n_test | gate | gatepos | FULL | resid (incremental) | null band (shift) | vượt null? |
|---|---|---|---|---|---|---|---|
| 2022 | 4,272 | 0.120 | 0.152 | **0.220** | +0.168 | (0.071, 0.158) | **CÓ** |
| 2023 | 6,529 | 0.026 | 0.014 | 0.081 | +0.085 | (0.107, 0.199) | KHÔNG (dưới cả band) |
| 2024 | 5,453 | −0.038 | 0.081 | 0.049 | +0.009 | (0.027, 0.101) | KHÔNG |
| 2025 | 5,911 | 0.021 | 0.040 | 0.063 | +0.049 | (0.153, 0.243) | KHÔNG (dưới band) |
| 2026H1 | 1,827 | 0.078 | 0.166 | 0.190 | +0.108 | (0.166, 0.252) | KHÔNG (trong band) |

Within-trade IC (timing trong đời lệnh, FULL): 2022 **+0.10** nhưng 2023..2026 =
**−0.05 / −0.08 / −0.23 / −0.15** — chiều upside không chỉ yếu mà **đảo dấu** OOS từ 2023:
model càng nói "còn upside" thì upside thực còn lại càng ít. AUC(rem_MFE≥5%) FULL:
0.61/0.52/0.53/0.58/0.59.

### rem_MAE (downside còn lại)

| fold | gate | gatepos | FULL | resid (incremental) | null band | vượt null? |
|---|---|---|---|---|---|---|
| 2022 | 0.217 | 0.235 | **0.272** | +0.164 | (0.074, 0.157) | **CÓ** |
| 2023 | −0.028 | 0.017 | −0.052 | −0.087 | (−0.048, 0.050) | ÂM (sai chiều) |
| 2024 | 0.010 | 0.129 | **0.268** | **+0.254** | (−0.051, 0.084) | **CÓ** |
| 2025 | −0.017 | 0.049 | **0.130** | +0.138 | (−0.135, −0.009) | **CÓ** |
| 2026H1 | 0.003 | 0.094 | 0.118 | +0.074 | (−0.076, 0.019) | **CÓ** |

Within-trade IC FULL: +0.28/+0.09/+0.13/+0.16/+0.16 — dương mọi năm.
AUC(rem_MAE≤−5%) FULL: 0.63/0.47/**0.61**/0.59/0.54 (gate-only ~0.50 ở 2024/2025).

**Trả lời nửa đầu câu hỏi quyết định: CÓ — nhưng chỉ MỘT chiều.** Chiều *downside còn lại*
có thông tin thật ngoài force-gates (4/5 fold vượt shift-null, resid-IC tới +0.25 năm 2024,
gate-only gần 0 cùng năm — đúng chỗ force-rules độc quyền). Chiều *upside còn lại* — chiều
cần để cứu sold-then-rallied — KHÔNG có: 1/5 fold vượt null (chỉ 2022), timing đảo dấu từ 2023.

## 2. Decision-value proxy — thoát khi pred_up < k·pred_down (quét k), same-trade, entry ≥2022 (1,014 lệnh)

d_pnl = pnl_sim − pnl_act (u); psd = pnl per-slot-day. EARLY = chỉ được bán sớm hơn exit thật;
REPLACE = thay hẳn (được giữ tới +20 bar sau exit thật). Trích k tốt nhất mỗi nhóm
(bảng đầy đủ: `ts_sim.parquet` / json):

| slice | set | biến thể tốt nhất | d_pnl | psd_act → psd_sim |
|---|---|---|---|---|
| ≥2022 | FULL | early k=0.5 | **−14.6u** | 0.00232 → 0.00243 |
| ≥2022 | FULL | replace k=0.5 | −31.9u | 0.00232 → 0.00082 |
| ≥2022 | gatepos | early k=0.5 | −20.1u | 0.00232 → 0.00238 |
| 2024+ | FULL | early k=0.5 | **−11.3u** | 0.00237 → 0.00226 |
| 2024+ | FULL | replace k=0.5 | −14.0u | 0.00237 → 0.00123 |

Mọi ô của lưới 2 set × 6 k × 2 biến thể × 2 slice đều **d_pnl ÂM** (−11 → −45u). psd_sim chỉ
vượt psd_act ở vài ô early-≥2022 nhờ CẮT NGẮN hold (bán sớm, pnl sập, mẫu số ngày co lại) —
đổi thành tiền thật đòi entry mới cùng chất lượng lấp slot, mâu thuẫn trực tiếp
SIGNAL_STARVATION_2024; ở 2024+ psd_sim thua luôn cả psd_act.

Biến thể bổ sung (ts03_extra.py), cô lập từng hướng:

- **EXTEND-ONLY** (giữ nguyên exit thật, chỉ hoãn khi model nói "còn upside" — test trực tiếp
  hướng cứu sold-then-rallied): ≥2022 FULL −20.3/−18.4/−15.9/**−14.6u** (k=0.75/1/1.5/2);
  2024+ −2.6/−1.7/−0.9/**−0.3u** — tiệm cận 0 từ phía ÂM khi k→∞ (= càng ít nghe model càng
  đỡ lỗ); psd_sim < psd_act ở mọi ô (chưa tính knock-on chiếm slot — caveat chỉ làm tệ thêm).
- **ABS-DOWNSIDE** (bán sớm khi pred_rem_MAE ≤ −5/−8/−12%, bỏ qua chiều upside — test xem
  IC downside có tự đổi thành tiền không): ≥2022 −40.8/−31.9/−20.1u; 2024+ −30.9/−24.6/−13.7u.
  IC downside +0.25 **không đổi được thành exit tốt hơn**: bar mà model thấy "downside lớn"
  chính là giữa pullback — bán đó là bán đáy, đúng cơ chế đã đo ở 7 lever exit cơ học.

## 3. Hai cohort đau đã biết (chuẩn "đúng chỗ đau")

**Sold-then-rallied** (decision bar thật ≥2022, n=1,042, trong đó rallied=449):
P(model nói "còn upside" tại bar exit thật | rallied) = **0.753** vs | không-rallied = **0.747**
(FULL, k=1) — không phân biệt được; AUC(margin up−down vs rallied) = **0.518** (gatepos 0.546).
Model nói "giữ" gần như VÔ ĐIỀU KIỆN ở bar exit (75% cả hai nhóm) — tái tạo prior
"exit của force-gate thường sớm", không tách được lệnh NÀO sẽ chạy tiếp.

**Suppress-victims PDR-type** (pdr_forensic_scan, sw_date ≥2022, n=348; victim = delta<0,
n=142): P(model bắn "hết upside, bán ngay" trong 5 bar từ sw_date | victim) = **0.303** vs
| non-victim = **0.301** (FULL); nhóm victim nặng ≥5% (n=31): 0.290. Không phân biệt —
head hai chiều KHÔNG biết cửa sổ suppress nào đáng bán, cửa sổ nào nên để suppress cứu.

## 4. VERDICT: LANE CHẾT — đóng, không code head/knob nào

1. **Incremental có thật nhưng lệch chiều**: chỉ rem_MAE (downside) vượt gate-baseline
   ngoài null (4/5 fold, resid-IC +0.07..+0.25); rem_MFE (upside — chiều duy nhất có thể
   trả tiền mới, vì "bán sớm hơn" đã bão hòa) KHÔNG vượt null 4/5 fold và timing đảo dấu
   từ 2023. "Head hai chiều" thực chất chỉ là "head một chiều downside" — mà chiều đó
   force-gates tuy không *dự đoán* nhưng đã *hành động* đủ tốt.
2. **Decision proxy âm toàn lưới**: 2 set × 6 k × 2 biến thể × 2 slice + extend-only +
   abs-downside = **0/60+ cấu hình dương**; tốt nhất là −0.26u (extend k=2, 2024+) — đúng
   nghĩa "càng tắt model càng gần 0". Khớp quy luật EXIT_ATTRIBUTION_MAP: mọi biến thể
   bán-sớm âm toàn dải, mọi biến thể giữ-lâu chỉ dương nhờ 2020–21 (ở đây loại hẳn 2020–21
   khỏi sim nên hiện nguyên hình âm).
3. **Cả hai cohort đau đều không tách được** (rallied AUC 0.518; PDR fire-rate victim
   30.3% vs non-victim 30.1%) → taxonomy exit nhiều loại trên nền tín hiệu này vô nghĩa,
   như tiêu chí đã đặt trước.
4. Đối chiếu 16 thế hệ: lane "ML-score vào quyết định thuần giá" từng trả tiền ở ENTRY;
   ở EXIT, sau hồ sơ này + DERIV_EXIT_SCREEN + 7 lever cơ học, kết luận nhất quán:
   **exit-timing của gb_x08 đã bão hòa với mọi thông tin trong dữ liệu hiện có** — kể cả
   khi cho ML nhìn thẳng label hai chiều ex-post ngay trên chính các lệnh của hệ.

Caveat ghi trung thực: (a) sim first-order, chưa tính knock-on slot cho các biến thể
giữ-lâu — chỉ làm verdict âm THÊM; (b) fold 2026H1 nhỏ (1,827 bar); (c) CSRank tính trong
61 mã universe thay vì panel serving đầy đủ; (d) label dùng close (không high/low) đúng
spec — đổi sang high/low chỉ scale biên độ, không đổi rank. Không caveat nào đủ lật
0/60 cấu hình âm.

Hàm ý còn lại (không phải nợ nghiên cứu mới): tín hiệu downside resid-IC +0.25 (2024)
là thông tin thật nhưng chỉ có giá trị nếu có **kiến trúc dùng nó không phải bằng cách
bán sớm** — tức sizing/multi-position (giảm size thay vì đóng lệnh) — khớp hướng đã chốt
trong hồ sơ wavestart/sizing-line; KHÔNG mở lại lane exit-rule.
