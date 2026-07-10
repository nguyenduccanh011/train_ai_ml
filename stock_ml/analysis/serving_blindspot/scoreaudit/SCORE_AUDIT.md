# SCORE AUDIT — bản đồ chẩn đoán chuỗi score thô của gb_x08 (template 2783)

Ngày: 2026-07-10. Offline, không chạy leaderboard, không sửa engine.
Scripts: `sa00_dump.py` → `sa_scores.parquet` (96,951 bar × 61 mã, 2020-01-02→2026-06-16,
RAW + z-norm 252/60 + gate/force state + quyết định tái lập), `sa01_drift.py`,
`sa02_corr.py`, `sa03_calib.py`, `sa04_inertia.py` (+ CSV cùng thư mục).

## 0. Kiểm kê nguồn

| Chuỗi | Head / target | Nguồn |
|---|---|---|
| `score` (entry chính) | triple_barrier h30 pt.15 sl.08, LightGBM | `bundles/bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf/prediction_history.parquet` |
| `score2..5` (4 ensemble) | reversal h10 / continuation h10 / mfe h20 / fwd_ret_penalized h20 | như trên |
| `exit_score` | velocity_exit_regression h20/u8/volnorm40 | như trên |
| Head thay thế (mô phỏng nuốt) | reward_risk_regression h10 | `bundles/bundle_n2_3h_brk05_z07_2025-01-01_wf` (phủ 99.6% bar) |

Bundle nguồn model-side **trùng 2783** (cùng entry/exit target, 4 ensemble + threshold, seed 42,
WF 2y/1y gap 85); 2783 chỉ thêm engine knob `exit_snr_*` — không chạm score. Z-norm dùng đúng
`_causal_zscore_by_symbol` (252/60) và gate mask của `src/pipeline/experiment.py`.
Quyết định tái lập: buy_main = zE>-1.9 & upleg_abovema20; buy2..5 = z>0.9/0.7/0.7/0.7;
sell_ml = zX>2.0 & cons2_w20; sell_force = downleg12 | (belowma20p2 & nonbull35p2) | (downleg6 & breadth<0.25).

## 1. Drift & saturation theo năm (điểm bất thường nhất)

**Exit head (velocity) RAW sụp đổ 2 tầng — đây là bất thường lớn nhất toàn audit:**

| Năm | RAW mean | RAW std | RAW q95 | RAW skew | z q99 | % z>2.0 (ngưỡng sell) |
|---|---|---|---|---|---|---|
| 2023 | 2.52 | 2.15 | 7.23 | +1.74 | 4.63 | 12.17% |
| 2024 | 1.46 | 1.15 | 3.56 | +1.72 | 4.19 | 5.23% |
| 2025 | 0.60 | 0.80 | 1.64 | **−1.35** | **1.14** | **0.007%** (1 bar/năm) |
| 2026H1 | 0.64 | 0.63 | 1.55 | −1.03 | 1.74 | 0.49% |

- RAW trôi thật: mean giảm 4.2×, q95 giảm 4.4×, và **skew đảo dấu** (mất hẳn đuôi phải —
  head không còn "spike" cảnh báo). Vậy chẩn đoán head-im có **2 tầng nguyên nhân**: (a) force
  pre-emption cơ học (đã đo ở EXIT_ATTRIBUTION_MAP) VÀ (b) phân phối RAW tự sụp từ 2025.
- Z-norm 252/60 **không cứu được**: z chuẩn hóa mean/std nhưng ngưỡng 2.0 cần đuôi phải; phân
  phối 2025 lệch trái nên z q99=1.14 < 2.0 — **ngưỡng vận hành nằm ngoài toàn bộ phân phối năm**.
  Sell_ml 2025 = 1 bar duy nhất trên 15,121 bar; đối chiếu trades s42: 100% exit 2025 là force/rule.
- Z-norm chỉ giảm drift ~20–35%, không triệt: dispersion yearly-mean (đơn vị pooled-std)
  RAW→Z: score 0.53→0.35, score4 0.59→0.45, exit 0.49→0.38; std-range RAW→Z: 0.71→0.37 … 1.32→0.72.
  Tỷ lệ vượt ngưỡng vì thế vẫn dao động dữ dội theo năm (xem cột dưới) — z-norm KHÔNG phải bộ ổn định tỷ lệ.

**Tỷ lệ vượt ngưỡng theo năm (% bar)** — kênh nào cũng drift mạnh:

| Năm | main zE>-1.9 | e2>0.9 | e3>0.7 | e4>0.7 | e5>0.7 | buy_union | sell_ml |
|---|---|---|---|---|---|---|---|
| 2020 | 99.6 | 1.3 | 20.7 | 11.3 | 5.0 | 54.0 | 4.0 |
| 2021 | 99.6 | 23.6 | 39.9 | 30.9 | 43.1 | 76.1 | 2.0 |
| 2022 | 99.0 | 30.3 | 24.5 | 49.8 | 42.0 | 79.5 | 6.0 |
| 2023 | 97.5 | 3.9 | 20.4 | 4.5 | 9.9 | 56.6 | 7.8 |
| 2024 | 98.7 | 14.3 | 27.2 | 7.0 | 13.9 | 61.2 | 4.9 |
| 2025 | 98.0 | 23.9 | 33.1 | 18.2 | 44.9 | 80.7 | 0.007 |
| 2026H1 | 99.3 | 16.4 | 20.8 | 20.9 | 24.6 | 73.4 | 0.29 |

- **Kênh entry chính là pass-through**: zE>-1.9 đậu 97.5–99.6% mọi năm (ngưỡng nằm ở percentile
  0.3–3.3 của phân phối). Sức phân biệt thực của buy_main đến từ gate upleg_abovema20, không từ head.
- Saturation RAW (ngoài [q01,q99] pooled 2020-23): chỉ 2022 bão hòa đuôi trên (~3.9% mọi score,
  gấp 4 mức tham chiếu); 2024-26 KHÔNG bão hòa — vấn đề 2025 là co cụm, không phải kẹp biên.

## 2. Cấu trúc tương quan (phát hiện chính)

Corr z pooled (panel symbol-ngày): zE–z4 = **0.77**, z2–z5 = **0.80** (mọi năm 0.72–0.90) —
4 ensemble thực chất ~2.5 tín hiệu độc lập: {zE,z4} (momentum/mfe), {z2,z5} (reversal/penalized), z3 riêng.

- **Không thoái hóa thêm ở các năm gần** — ngược lại: mean corr cặp ensemble 2023 = 0.71 (đỉnh
  thoái hóa) nhưng 2025 = 0.29, 2026H1 = 0.21 (z3,z4,z5 gần trực giao). Cặp z2–z5 là dính vĩnh viễn (0.76–0.83).
- **Exit head là ảnh ngược của entry**: corr(zX, z5) = −0.67, (zX, z2) = −0.61 pooled — exit head
  nhìn phần lớn cùng một thứ với các entry head, dấu ngược. corr(zE,zX) dao động −0.06…−0.65 theo năm.
- Score vs force-state: yếu, nhưng **đảo dấu 2025-26**: corr(zX, downleg12) 2024 = +0.18 → 2025 = −0.22,
  (zX, mkt_drop_z) −0.32 → +0.24. Tức từ 2025 exit head đọc RỦI RO THẤP hơn ngay trong downleg —
  head không thể xác nhận force-gate về cấu trúc, không chỉ chậm chân.
- Overlap quyết định: khi sell_ml nổ, force đã/đang nổ cùng bar: 2020 9.8% → 2022 75.5% → 2024 94.1%
  (khớp EXIT_ATTRIBUTION_MAP); 2025-26 sell_ml gần như không tồn tại để mà bị pre-empt.

## 3. Calibration từng năm + oracle threshold (chỉ đo trần, KHÔNG đề xuất tune)

Đơn vị "excess/avoided" = tổng %-point fwd-return so với mean năm, cộng trên bar-ngày tín hiệu
(mức TÍN HIỆU, chưa qua occupancy 1-slot / một-lần-bán-mỗi-trade — trần lạc quan).

**Entry** (fwd20, `sa03_entry_oracle.csv`): điểm vận hành trôi khỏi tối ưu gần như MỌI năm nhưng
theo hướng khác nhau:
- 2024 là năm chết tín hiệu: cả 4 ensemble chọn tập DƯỚI trung bình (excess_cur: e2 −799, e3 −2795,
  e4 −1648, e5 −3405 %-pt-ngày) và **oracle cũng không cứu nổi** (max excess ≈ 0/âm với n đáng kể)
  → 2024 head không có thông tin, không phải lỗi ngưỡng.
- 2023 tương tự phía âm (e3 −3318, e5 −1619; oracle chỉ +317/+647) — 2 năm liền ensembles anti-selective.
- 2025: e2 −3702 hiện tại vs oracle +2884 (t*=2.0 thay vì 0.9) và e5 −2626 vs +1139 — ngưỡng 0.7–0.9
  đứng SAI CHỖ trên curve 2025; e3/e4 lại đang dưới tối ưu theo hướng ngược (bỏ sót +2500/+2600).
- Kênh chính zE: trần oracle rất lớn trên giấy (2020: +9800 vs hiện tại +300) nhưng vô nghĩa vận hành —
  muốn ăn phải kéo ngưỡng từ percentile ~1 lên ~50 (đổi bản chất chiến lược, đụng luật ≤+1.5).

**Exit** (fwd10 trên 42,220 bar đang-mở từ trades s42, post-gate cons2_w20, `sa03_exit_oracle.csv`):
- Decile calibration chỉ ĐƠN ĐIỆU ĐÚNG 2 năm: 2020 (d9 = −0.5% vs d0 +5.3%) và 2022 (d9 −4.3%).
  **2021/2023/2025 calibration ĐẢO** (zX cao → fwd10 CAO hơn: 2023 d9 +2.1 vs d0 +0.5; 2025 d9 +2.4
  vs d0 +0.5): bán theo zX các năm đó là bán trúng người thắng. 2024/2026H1 phẳng (~không thông tin).
- Điểm vận hành z>2.0 = percentile 94–99.98 theo năm. Avoided hiện tại: 2021 **−651** (bán bar
  fwd10 +6.0%!), 2023 +124, 2024 **−155**, 2025 +6.7 (1 bar), 2026H1 +39.
- Oracle theo-năm: 2022 t*=0.04 → +1318 (hiện tại +399); 2025 t*=0.75 → +315; 2026H1 t*≈−0.2 → +157;
  2024 t*≈0 → +81. Trần oracle-exit 2024-26H1 ≈ **+550 %-pt-ngày so với hiện tại** ở mức tín hiệu;
  quy đổi thô mỗi trade chỉ bán 1 lần (mean avoided/bar-bán ≈ 1.1–1.5%) → trần thô **≈ +1–3u/năm**,
  kèm caveat lớn: 2021/2023 oracle nói "đừng bán bằng zX" (t* ngoài phân phối, avoided ≤ 0).
- Kết luận calibration: không tồn tại MỘT ngưỡng zX đúng cho mọi năm — dấu của thông tin còn đổi
  chiều theo regime. Trần oracle không nằm ở "ngưỡng tốt hơn" mà ở "điều-hòa-liên-tục theo trạng thái"
  (lever đã treo ở operating-point retest): oracle t* nhảy 0.0↔3.3 giữa các năm liền kề.

## 4. Độ ì ống dẫn + cơ chế nuốt tín hiệu mới (câu hỏi trung tâm)

**Độ ì (σ cần cộng vào score để lật thêm X% bar-quyết-định, `sa04_inertia.csv`):**

| Kênh | lật 1% | lật 5% | Ghi chú |
|---|---|---|---|
| ensembles e2..e5 | 0.02–0.10σ | 0.07–0.40σ | mềm — mật độ z quanh ngưỡng dày |
| exit zX 2020-24 | 0.08–0.52σ | 0.35–1.31σ | trung bình |
| **exit zX 2025** | **0.86σ** | **1.25σ** | ì gấp ~10× 2020; cả phân phối nằm xa dưới 2.0 |
| entry chính zE | 0.13–0.54σ (nhiều năm không lật nổi 1%) | không thể (chỉ 0.3–3% bar dưới ngưỡng) | **đòn bẩy chết**: chỉ ≤3% quyết định có thể đổi dù dịch vô hạn |

**Nuốt tín hiệu thật** — trộn head reward_risk h10 (corr −0.56 với head cũ sau chuẩn hóa,
tức mang thông tin khác thật sự) vào exit_score rồi qua đúng ống z-norm 252/60 + thr 2.0 + cons2_w20:

| w (trọng số head mới) | % quyết định sell đổi 2025 | sell MỚI 2025 | sell MỚI 2026H1 |
|---|---|---|---|
| 0.1 | 0.026% | 4 | 27 |
| 0.2 | 0.073% | 11 | 79 |
| 0.3 | 0.185% | 28 | 129 |
| 0.5 | 1.83% | 277 | 185 |
| head mới đứng MỘT MÌNH | — | 1,042 (7.4% bar, z q99=6.9) | 204 |

→ **Ống dẫn nuốt thật**: head mới tự nó tạo 1,042 tín hiệu sell 2025 (nó CÓ đuôi phải ở regime mới),
nhưng trộn w=0.3 vào head cũ chỉ còn 28 (**giữ lại 2.7%**); w=0.5 còn 27%. Cơ chế: head cũ 2025
co cụm lệch trái, kéo blend về giữa phân phối, z-norm tái chuẩn hóa xong ngưỡng 2.0 vẫn ngoài đuôi.
(Hàng 2020 trong `sa04_blend.csv` là artifact warmup expanding-60, bỏ qua.)
Đối chứng nhiễu độc lập σ=1: w=0.1/0.2/0.3 chỉ đổi 0.33/0.75/1.29% bar (8/19/33% số sell) —
một tín hiệu IC nhỏ trộn kiểu additive gần như không lay được lớp quyết định ở w hợp lý.

## 5. Verdict masker

**MASKER THẬT (kèm số):**
1. **Sụp phân phối RAW của exit head từ 2025** — mean 2.52→0.60, q95 7.2→1.6, skew +1.7→−1.35;
   z q99 1.14 < ngưỡng 2.0 → sell_ml 2025 = 1 bar. Tầng nguyên nhân THỨ HAI bên dưới force
   pre-emption; kể cả bỏ force-gate, ML exit 2025-26 vẫn câm. Nuốt luôn head mới trộn vào
   (giữ 2.7% tín hiệu ở w=0.3). Đây là masker chính user nghi — **xác nhận**.
2. **Ngưỡng entry chính -1.9 là đòn bẩy chết** — pass 97.5–99.6% mọi năm; mọi cải thiện head
   triple-barrier bị lớp này nuốt sạch (tối đa ~3% quyết định có thể đổi). Sức phân biệt buy_main
   thực tế = gate upleg_abovema20. Label/head entry mới đi qua kênh chính sẽ vô hình — **xác nhận**.
3. **Calibration exit đảo chiều theo regime** (2021/2023/2025 zX cao = fwd cao) — không ngưỡng tĩnh
   nào đúng; oracle t* nhảy 0.0↔3.3. Masker dạng "sai dấu theo năm", không sửa được bằng threshold.
4. **Cặp ensemble z2–z5 trùng lặp vĩnh viễn** (corr 0.76–0.90) — 4 kênh trả phí vận hành của ~2.5
   tín hiệu; 2023-24 cả cụm anti-selective đồng loạt (excess âm 800–3400 %-pt-ngày/kênh).

**MASKER VÔ TỘI (đã nghi nhưng số minh oan):**
- **Z-norm 252/60 KHÔNG khuếch đại drift** — giảm dispersion mean 20–35% và std-range 25–45% ở mọi
  chuỗi; nó chỉ không đủ mạnh (không ổn định hóa được tỷ lệ vượt ngưỡng khi hình dạng phân phối đổi).
- **Saturation biên**: không có kẹp min/max ở 2024-26 (chỉ 2022 bão hòa đuôi trên ~3.9%).
- **Thoái hóa ensemble ở năm gần**: không — corr cặp 2025-26 thấp nhất lịch sử (0.21–0.29);
  độ ì kênh ensemble cũng thấp (0.02–0.1σ lật 1%) — tín hiệu entry mới đi qua NGÕ ENSEMBLE RIÊNG
  (kênh mới + ngưỡng riêng, kiểu OR-union hiện tại) sẽ KHÔNG bị nuốt.

**Khuyến nghị cho vòng operating-point retest đang chạy song song:**
- Lever điều-hòa-liên-tục cho exit đáng mở: trần oracle theo-năm ≈ +1–3u/năm thô cho 2024-26H1,
  nhưng điều kiện tiên quyết là xử lý tầng-1 (head chết): head mới (vd reward_risk h10 — tự nó
  có đuôi phải 2025-26) phải vào như **KÊNH SELL ĐỘC LẬP OR-union với ngưỡng riêng** (như kiến trúc
  exit_ensemble/`exit2_z_threshold` đã có sẵn trong recombine), tuyệt đối không blend vào exit_score cũ.
- Không tốn thêm vòng nào cho: tune ngưỡng zE kênh chính (đòn bẩy chết), tune ngưỡng tĩnh zX
  (calibration đảo chiều theo năm), hay "sửa" z-norm window (vô tội).
