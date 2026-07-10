# SIM CHUẨN v2 + FRONTIER CHÍNH THỨC (thay §B R2_ROUND2 — bản trên khung thiên vị)
Ngày: 2026-07-10. Scripts: `nh_nav2.py` (sim v2), `nh_01_frontier.py` (frontier + bảng quyết định, log `nh_01_out.txt`, số liệu `nh_frontier_results.csv`), `nh_02_r2b.py` (điểm r2b của worker, log `nh_02_out.txt`).

## 0. Sim v2 là gì + VERIFY

`nh_nav2.py` = na_navlib (đã tái lập từng chữ số 7 anchor trong audit) + 3 sửa bias theo NAV_FRAMEWORK_AUDIT.md:
- **(a) settle_lag** (mặc định 2): tiền bán về sau 2 phiên làm việc (VN T+2, tiền về chiều T+2); tiền treo vẫn tính NAV, không xài được để entry. Cấu hình 0/1/2.
- **(b) advance_fee** (tùy chọn, mặc định 0.08%/vòng khi bật): ứng trước tiền bán — proceeds xài ngay, trả phí trên proceeds mỗi vòng (~0.0375%/ngày × ~2 ngày).
- **(c) shuffle-mean**: mỗi cấu hình 20 permutation tie-break (`shuffle_stats`), báo **mean±sd** — không dùng alphabet đơn lẻ.
- Cost R tham số (mặc định 0.6% = cost nhúng hiệu dụng thực; tái tạo net từ giá raw, fee 0.4% giữ, slippage biến thiên — y hệt na_01).

**VERIFY (bắt buộc trước khi tin số nào bên dưới)**: chế độ legacy (lag0, không ứng, alphabet, R0.6) tái lập **9/9 anchor đúng từng chữ số**: c2 K22 ×18.03/−15.41, K23 ×17.14/−14.77, K25 ×16.34/−13.63, base K20 ×17.70, K25 ×14.78, gb K25 ×14.44/−15.31, c2 f22 K25 ×3.80 / K22 ×3.92, gb f22 ×3.50. Ngoài ra trades CSV của worker r2b khớp NAV họ công bố (oxtrail04 ×16.76, p40_45 ×16.98) trên cùng máy → mọi delta dưới đây là delta sạch.

## 1. FRONTIER CHÍNH THỨC (shuffle-mean±sd 20 perm, R0.6) — THAY BẢNG §B CŨ

### Chế độ lag2 KHÔNG ứng trước tiền bán
| điểm | full mean±sd | MaxDD (worst seed) | f22 mean±sd |
|---|---|---|---|
| c2 K20 | ×13.82±0.51 | −16.89% | ×3.39±0.10 |
| c2 K22 | ×13.58±0.68 | −15.41% | ×3.35±0.09 |
| c2 K23 | ×13.71±0.60 | −14.77% | ×3.32±0.10 |
| c2 K25 | ×13.26±0.41 | −13.63% | ×3.24±0.07 |
| base K20 | ×13.41±0.57 | −15.21% | ×3.35±0.10 |
| base K25 | ×12.65±0.38 | −12.25% | ×3.20±0.05 |
| **gb K25** | ×13.03±0.42 | −15.42% (−15.70%) | ×3.34±0.08 |

### Chế độ lag2 + ứng trước 0.08%/vòng
| điểm | full mean±sd | MaxDD (worst seed) | f22 mean±sd |
|---|---|---|---|
| c2 K20 | ×16.09±0.85 | −17.05% | ×3.76±0.14 |
| c2 K22 | ×15.69±0.70 | −15.56% | ×3.67±0.11 |
| c2 K23 | ×15.58±0.50 | −14.91% | ×3.59±0.10 |
| c2 K25 | ×14.90±0.46 | −13.76% | ×3.47±0.08 |
| base K20 | ×15.20±0.61 | −15.33% | ×3.51±0.12 |
| base K25 | ×14.11±0.38 | −12.35% | ×3.31±0.06 |
| **gb K25** | ×13.78±0.42 | −15.28% (−15.61%) | ×3.43±0.08 |

Đọc frontier:
- **Không ứng trước tiền bán → tuyến R2 CHẾT vs gb.** Điểm tốt nhất (c2 K22) chỉ +4.2%±6.2 full, f22 **+0.1%±3.6** — mọi con số trong ±1sd noise; K25 còn âm f22 (−3.0%). Bảng §B cũ ("+19..+25% tại matched-DD") không sống sót một điểm nào ở chế độ này.
- **Có ứng trước**: matched-DD tại trần gb (−15.3..−15.6%): c2 K22 ×15.69 vs gb ×13.78 = **+13.9%**, f22 **+6.7%** — khớp kịch bản sơ bộ audit (+13.6/+6.4). Tại trần DD thấp (−13.8%): c2 K25 chỉ +8.2% full, **+1.0% f22 (noise)**.
- MaxDD của gb giờ CŨNG dao động theo seed (−15.28 mean, −15.70 worst) — trần matched-DD nên lấy theo worst-seed, bất lợi thêm chút cho narrative "gb DD ngang c2 K22".
- c2-vs-base ở matched-DD (K22 vs base K20): +3.2% full / +4.6% f22 (adv) — dưới hoặc sát 1sd; kết luận cũ "c2 trội base ở mọi mức DD f22" teo lại còn "nhỉnh trong noise".

## 2. BẢNG QUYẾT ĐỊNH — 3 kịch bản vận hành (c2 vs gb K25, delta mean ± sd lan truyền)

| kịch bản | full K22 | full K25 | f22 K22 | f22 K25 |
|---|---|---|---|---|
| **Cá nhân retail R0.6 + ứng trước** | **+13.9%±6.2** | +8.2%±4.7 | **+6.7%±4.0** | +1.0%±3.3 |
| R0.6 KHÔNG ứng trước | +4.2%±6.2 | +1.8%±4.6 | +0.1%±3.6 | −3.0%±3.3 |
| Vốn lớn R0.9 + ứng trước | +3.5%±5.3 | −0.9%±4.2 | +1.4%±3.8 | −3.0%±3.3 |

Noise-adjusted: chỉ ô đậm vượt ~1.7-2.2sd; toàn bộ hàng "không ứng" và "vốn lớn" ≤ 1sd = **không phân biệt được với gb**.

## 3. ĐIỂM MỚI CỦA WORKER r2b QUA SIM v2 (nh_02)

Hai điểm tốt nhất từ CSV r2b (MD §A/§C của họ chưa chốt lúc chạy): `oxtrail04` (combo trail, ×16.76/f22 ×3.89 khung cũ — điểm duy nhất thắng c2 cả hai frame) và `p40_45` (đỉnh plateau full ×16.98, f22 khung cũ ×3.79 — tự tính, không hơn tâm).

| điểm (R0.6) | chế độ | full ±sd (DD) | vs gb | f22 ±sd | vs gb |
|---|---|---|---|---|---|
| oxtrail04 K25 | no-adv | ×13.84±0.44 (−13.6%) | +6.2%±4.8 | ×3.41±0.08 | +2.2%±3.5 |
| **oxtrail04 K25** | **adv** | **×15.75±0.54 (−13.7%)** | **+14.4%±5.3** | **×3.60±0.07** | **+5.0%±3.3** |
| oxtrail04 K22 | adv | ×15.90±0.82 (−15.5%) | +15.4%±6.9 | ×3.69±0.13 | +7.4%±4.7 |
| p40_45 K22 | adv | ×15.97±0.90 (−15.6%) | +15.9%±7.4 | ×3.71±0.07 | +8.0%±3.2 |
| p40_45 K25 | adv | ×15.03±0.50 (−13.8%) | +9.1%±4.9 | ×3.54±0.07 | +3.1%±3.2 |

Điểm đáng giá nhất toàn cuộc: **oxtrail04 K25 + advance** — ăn gb +14.4% full (2.7sd) với MaxDD **thấp hơn gb 1.6đ** (−13.7 vs −15.3%), f22 +5.0% (1.5sd). Nó dominate c2 K25 (cùng DD: +14.4 vs +8.2 full; +5.0 vs +1.0 f22). p40_45 chỉ đẹp ở K22 (DD ngang gb) và là đỉnh-gai full-frame, f22 khung cũ không vượt tâm — đúng nghi ngờ plateau.

## 4. VERDICT

1. **Lợi thế R2-vs-gb chỉ tồn tại khi CÓ ứng trước tiền bán và phí retail ≤0.6%.** Không ứng: chết toàn tuyến (delta ≤ +4.2%, <1sd). Vốn lớn (R0.9): chết kể cả có ứng.
2. Trong điều kiện sống (retail + advance): **+13.9%±6.2 full / +6.7%±4.0 f22** tại c2 K22 (matched-DD với gb); full vượt ~2.2sd = tín hiệu thật nhưng mỏng; **f22 chỉ ~1.7sd — chưa đạt chuẩn 2sd**, phải coi là "nghiêng về dương" chứ không phải chứng minh.
3. **Điểm vận hành khuyến nghị nếu đi tiếp tuyến R2: r2b_oxtrail04 K25 + ứng trước tiền bán** — lợi thế full +14.4%±5.3 (2.7sd, khỏe nhất bảng) tại DD thấp hơn gb 1.6đ; f22 +5.0%±3.3. Nếu chấp nhận DD ngang gb: oxtrail04/p40_45 K22 (+15..+16% full, f22 +7..+8% ~1.6-2.5sd).
4. Ghi vào cost model serving nếu promote: phí ứng trước 0.08%/vòng là ĐIỀU KIỆN TỒN TẠI của hệ, không phải tùy chọn; và mọi số công bố từ nay = shuffle-mean±sd trên sim v2 (`nh_nav2.py`), không dùng r2_nav.py alphabet/lag0 nữa.
