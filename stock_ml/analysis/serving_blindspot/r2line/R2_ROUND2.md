# TUYẾN R2 — VÒNG 2 (plateau + matched-DD frontier + winner-riding combo)
Ngày: 2026-07-10. Prefix template/run vòng này: `r2b_` (id DB mới, không đụng r2_* vòng 1 hay canonical t2005/t2835).

Khung: y hệt vòng 1 — clone fc_rule2, seed 42 deterministic, NAV sim `r2_nav.py` K25 100% NAV; chống-ghost = NAV-từ-2022 (`*_f22`). Ứng viên vào vòng: ★ r2_c2_pb40snr (pb 4.0%/50 + snr 0.8/20/g0.12/gb0.08) ×16.34/53.55%/−13.63%, f22 ×3.80.

## B. Matched-MaxDD frontier K × (c2 vs base) — KHÔNG cần run mới, chỉ đổi K trong NAV sim

| K | c2 NAV | c2 MaxDD | c2 f22 | base NAV | base MaxDD | base f22 |
|---|---|---|---|---|---|---|
| 15 | ×15.45 | −20.69% | ×3.60 | ×15.77 | −19.34% | — |
| 18 | ×17.72 | −18.62% | ×3.97 | ×**18.14** | −16.84% | ×3.78 |
| 20 | ×17.82 | −16.89% | ×**4.06** | ×17.70 | −15.21% | ×3.85 |
| 22 | ×**18.03** | −15.41% | ×3.92 | ×16.16 | −13.87% | ×3.75 |
| 23 | ×17.14 | −14.77% | ×3.83 | — | — | — |
| 25 | ×16.34 | −13.63% | ×3.80 | ×14.78 | −12.25% | ×3.48 |
| 28 | ×15.08 | −12.21% | ×3.49 | — | — | — |
| 30 | ×14.22 | −11.42% | ×3.34 | — | — | — |

(MaxDD mọi điểm cùng episode 2025-04-09 — không có tail mới xuất hiện khi đổi K.)

**NAV tại matched-MaxDD:**
- **Trần −15.31% (mức gb_x08)**: c2 K23 ×17.14/−14.77 (f22 ×3.83); c2 K22 ×18.03 nhưng −15.41 lố trần 0.10đ. Base K20 ×17.70/−15.21 (f22 ×3.85). → **cả hai đè bẹp gb ×14.44 (+19..+25%)**; c2-vs-base ở trần này gần như HÒA full-frame (nội suy c2 tại −15.2 ≈ ×17.6-18.0), f22 c2 nhỉnh hơn ở K22 (3.92 vs 3.85).
- **Trần −12.25% (mức base K25)**: c2 K28 ×15.08/−12.21 vs base K25 ×14.78 (**+2.0%**); f22 3.49 vs 3.48 — parity.

**Đọc trung thực**: lợi thế headline của c2 tại K25 (+10.6% NAV so base) chủ yếu vì K25 là điểm LỆCH đỉnh của base frontier (đỉnh base full-frame ở K18 ×18.14 — nhưng tựa 2021 +158%, f22 chỉ 3.78). Ở matched-DD, c2 chỉ hơn base 0-2% full-frame; **cái c2 giữ được là frontier f22 trội hơn base ở MỌI mức DD đo được** (4.06/3.92/3.83/3.80 vs 3.85/3.75/3.48) — alpha thật sau 2022, không phải ghost. Trục K vẫn là đòn bẩy risk thuần: chọn điểm vận hành theo khẩu vị DD; **vùng ngọt của c2 là K20-K23** (NAV ×17.1-18.0, DD −14.8..−16.9%, f22 3.83-4.06, uw ~101d).

## A. Plateau vs gai quanh đỉnh 4.0/50 (nền snr g12)
(đang chạy — r2b_p{38,39,40,41,42}_{45,50,55}, 10 điểm mới + tâm c2)

## C. Winner-riding combo overext × snr-window
(đang chạy — r2b_ox{16,18}_w{20,40}, r2b_ox12_w40, r2b_oxtrail04; điều kiện sống: NAV-từ-2022)

## D. Tổng hợp
(chờ A+C)
