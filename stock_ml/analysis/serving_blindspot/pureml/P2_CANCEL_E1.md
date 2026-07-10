# P2_CANCEL_E1 — Oracle ceiling + anatomy cho cancel-policy sổ lệnh treo — **KILL**

*2026-07-10. Offline thuần (0 run engine). Scripts: `p2_01_dataset.py` → `p2_05_binary.py`;
data: `p2_orders.parquet` (11,395 limit đã đặt), `p2_bars.parquet` (332,049 bar-treo).
Nguồn: `events.csv` (filled 3,813 + missed 7,582 = **11,395 khớp NEW_FAMILY_OPTIONS P2**),
`trades_raw.csv`, ohlcv store serving (RAW price, cùng nguồn `02_enrich.py`), flag 13 lệnh
spans_bad_adjustment loại khỏi frame feature. Frame = serving 2643 (3,781 fill đóng, Σ+287.6u,
718 big_win @ pnl≥0.15 — report cũ ghi 717 do đếm strict), KHÔNG phải gb_x08 1378 lệnh —
caveat quen thuộc 79% trùng. Đơn vị u = pnl fraction/lệnh, slot giải phóng = 0 (bảo thủ).*

**Verdict một dòng: oracle ceiling ≥2022 = +25.4u (vượt kill bar +3u) NHƯNG hoàn toàn
không truy cập được — mọi model/threshold ex-ante đều ÂM ở mọi độ sâu cancel, kể cả tại
ràng buộc recall big_win 99%. Đóng P2, không xây E2.**

---

## 1. Cơ chế bị giới hạn cứng ngay ở dữ liệu: 471 fill wait=1 KHÔNG cancel được

Lệnh đặt ở close bar signal; fill sớm nhất = bar kế tiếp. Với fill wait=1 **không tồn tại
bar thông tin mới nào** giữa đặt và fill → "cancel" các lệnh này = không đặt = entry-gate
(SELECTION WALL, đã có án). Cơ chế cancel thật chỉ áp cho fill wait≥2.
Fill wait=1 ≥2022 = 266 lệnh **+18.9u dương ròng mọi năm trừ 2026 (−0.9u)** — cohort fill-ngay
là cohort TỐT (khớp ENTRY_STRUCTURE_MAP §2: ngày 1-5 = +11.9%/lệnh, tốt nhất).

## 2. Oracle ceiling (kill/go) — PASS hình thức

Hủy đúng decile tệ nhất theo pnl thực (perfect foresight), per-year:

| ≥2022 | n fill | cancel | u cứu |
|---|---|---|---|
| A. ALL fills (gate-contaminated) | 2,687 | 268 | **+29.2u** |
| B. wait≥2 — trần THẬT của cơ chế | 2,421 | 242 | **+25.4u** (2022: +8.3, 2023: +5.1, 2024: +4.1, 2025: +4.7, 2026: +3.1) |

- Kill bar NEW_FAMILY_OPTIONS (< +3u → đóng): **vượt** → được phép đi tiếp sang separation.
- Trần tuyệt đối (hủy mọi fill lỗ): +53.7u/1,215 lệnh (50.2% số fill!). Knife ≤−15% ≥2022
  wait≥2: chỉ **29 lệnh = −5.19u** — đúng cảnh báo "trần knife thấp" trong option file.
- Decile oracle toàn lệnh pnl ∈ [−24.8%, −7.1%] — muốn ăn trần phải bắn trúng cohort −8..−25%.

## 3. Separation ex-ante tại decision bar — KHÔNG TỒN TẠI (đây là chỗ chết)

Frame A (lạc quan nhất cho policy): fills wait≥2, feature tại **bar cuối trước fill**
(cơ hội hủy cuối cùng, chỉ dùng dữ liệu ≤ close bar đó; policy per-bar thật còn không được
biết "fill sắp xảy ra" nên mọi số dưới đây là CẬN TRÊN). n=3,314; loser(≤−5%) 528; big_win 573.

AUC per-feature ≥2022 (loser vs rest | loser vs big_win — cột sau mới là cột sống còn):

| feature | AUC l\|rest | AUC l\|bw | đọc |
|---|---|---|---|
| speed (rơi/bar về limit) | 0.448 | 0.485 | **ĐẢO chiều giả thuyết** |
| ret5 / ret3 | 0.568 / 0.544 | 0.477 / 0.480 | tách loser khỏi MID, không khỏi big_win |
| gap_open / red_streak / vol_shock | 0.537 / 0.482 / 0.516 | ~0.49-0.54 | nhiễu |
| m_z20 / m_breadth (index sập) | 0.502 / 0.532 | **0.390 / 0.394** | PHẢN-tách: breadth thấp = big_win NHIỀU HƠN |
| sig_score | 0.604 | 0.545 | chỉ tách loser khỏi mid (mid ≈ 0u — vô giá trị) |
| age lệnh treo | 0.416 | 0.516 | loser fill nhanh, big_win cũng fill nhanh |

**Giả thuyết tự nhiên "rơi NHANH về limit = dao" — BÁC bằng số:** quintile speed rơi nhanh
nhất (−1.36%/bar) ≥2022: mean **+5.74%/lệnh, bigwin-rate 16.3%, Σ+27.8u = quintile TỐT NHẤT**;
quintile rơi chậm Q3 mới là tệ nhất (+2.5%). Top-50 losers khớp ngày 1-5 là thật, nhưng
big_win cũng khớp ngày 1-5 (median age loser 4 = big_win 4): tốc độ rơi là tính chất của
CƠ HỘI, không phải của dao. Tương tự: bảng wait-to-fill ≥2022 — wait=2 mean +6.8% (cao nhất),
wait 21-40 +2.1% (thấp nhất): fill nhanh TỐT, đơn điệu ngược nỗi sợ. Gap-down ≤−2% tại
decision bar: mean +7.0% vs +4.3% không-gap. Index z quintile sập nhất (−2.17): +2.8%, vẫn dương.

Loser và big_win là **cặp song sinh ex-ante** — mọi hướng chém đều chém cả hai. Khớp độc lập
với ENTRY_STRUCTURE_MAP §1 trên gb_x08 ("không feature ex-ante nào tách losers; knife =
rơi-xuyên-limit, chỉ nhận diện được SAU fill qua depth10") — giờ xác nhận thêm lần nữa trên
frame 2643 với bộ feature pending-book chuyên dụng (speed/gap/tuổi lệnh/index-z/breadth).

## 4. Model walk-forward (LGBM, purge exit<fold, 3 seeds, folds 2022-2026)

**(a) Regression pnl, decile dự-đoán-xấu per-year (con số E1 chỉ định):**

| year | cancel | u cứu | mean pnl lệnh hủy | big_win hủy |
|---|---|---|---|---|
| 2022 | 60 | −0.18u | +0.3% | 2 |
| 2023 | 56 | −0.53u | +0.9% | 4 |
| 2024 | 49 | −0.18u | +0.4% | 2 |
| 2025 | 47 | −1.14u | +2.4% | 5 |
| 2026 | 28 | −0.06u | +0.2% | 2 |
| **Σ≥2022** | 240 | **−2.09u** | | **15/316** |

Decile dự-đoán-xấu nhất có **mean pnl DƯƠNG mọi năm** — hủy nó là đốt tiền. Rank-IC(pred,pnl)
per-year +0.06..+0.31: model CÓ kỹ năng xếp hạng, nhưng kỹ năng nằm ở tách mid-vs-winner;
đuôi trái không tách được, và 1 big_win +42% trong cohort hủy nuốt sạch 5 loser −8%.

**(b) Precision @ recall big_win ≥99% (pooled ≥2022, cho phép hủy nhầm 3/316):**

| objective | cancel | u cứu | precision | nền |
|---|---|---|---|---|
| regression pnl | 55 (2.3%) | **−0.35u** | loser 16.4% | 16.5% — **zero enrichment** |
| binary loser ≤−5% | 21 | **−1.22u** | 9.5% | 16.5% — DƯỚI nền |
| binary knife ≤−15% | 31 | **−2.17u** | **0.000** (0 knife trúng) | 1.2% |

**(c) Sweep độ sâu không ràng buộc (regression):** 2%→30% cancel: −0.12u → −11.52u —
**âm đơn điệu ở MỌI độ sâu**, không tồn tại điểm ngọt. Binary loser decile: −6.84u;
knife-targeted decile: −10.85u (2023 một mình −8.6u vì hốt 11 big_win, 0 knife).
Feature quan trọng nhất model học: m_breadth/m_ret5/m_z20 — tức model bị hút vào
"index đang sập thì hủy", chính xác là anti-signal (mục 3).

## 5. Đối chiếu confirm_reversal — cùng mộ, khác quan tài

- **Cơ chế khác thật**: confirm_reversal trì hoãn fill chờ xác nhận (vẫn vào, basis đắt hơn);
  cancel bỏ hẳn lệnh (không vào). Về surface án: đúng là chưa có án trực tiếp — E1 này là án đó.
- **Nhưng chết cùng một nguyên nhân gốc**: cả hai đều từ chối fill trong điều kiện "giá đang
  rơi xấu về limit". LINE_A: confirm_reversal −28..−49 vì alpha champion = cái đệm giá 4.5%
  — từ chối đúng lúc đệm mở ra là vứt alpha. E1 đo được phiên bản cancel của cùng sự thật:
  cohort "rơi nhanh/gap/index sập" mà mọi discriminator trỏ vào chính là nơi big_win được
  mua rẻ (speed Q0 = +27.8u, breadth thấp = bigwin-rate cao hơn).
- **Tự bác như đề bài yêu cầu**: discriminator "tốt nhất" mà model tìm ra (m_breadth, m_ret5,
  ret5) khi đóng thành ngưỡng cancel thì hành vi kinh tế **tương đương confirm_reversal rời
  rạc hóa** (không mua khi thị trường đang rơi) — họ lever đã có án tử. Không có discriminator
  nào còn lại KHÁC họ đó mà mang AUC l|bw > 0.55.

## 6. Verdict & điều kiện mở lại

**KILL P2 tại E1 — không xây E2 (`pending_cancel_*`), không đốt run leaderboard nào.**
Chuỗi logic: oracle +25.4u ≥2022 (vượt bar +3u, trần có thật) → nhưng thông tin phân biệt
loser/big_win **không tồn tại trong daily bar trước fill** (AUC ~0.5, đảo chiều ở speed/breadth)
→ mọi policy hiện thực hóa đều âm (−0.35u tại recall99; âm mọi độ sâu, mọi objective, cả 5 năm
test) → giá trị kỳ vọng E2 < 0 trước cả chi phí parity. Kết quả âm này ĐỘC LẬP xác nhận
"chữ ký knife chỉ hiện SAU fill" (ENTRY_STRUCTURE_MAP) và "hard-stop sau fill ~0" (DEEP_STOP)
— ba mũi khác nhau cùng chạm một bức tường: **daily OHLCV không chứa cảnh báo dao trước fill.**

Điều kiện duy nhất mở lại P2: **data intraday (P3)** — "cancel trong phiên khi thấy nến sàn
đang hình thành + breadth intraday sập" đổi tiền đề thông tin của án này (giống án exit-daily
chỉ mở lại được bằng P3). Nếu P3 probe cho ≥4 năm lịch sử, P2-intraday xứng đáng một E1 mới
với đúng harness này (p2_01 nhận thêm bar intraday là chạy lại được).

*Caveat trung thực: (1) frame = serving 2643, không phải gb_x08 — nhưng cả hai frame đã cho
cùng kết luận anatomy độc lập; (2) market proxy = median EW 193 mã store (store không có
VNINDEX) — m_z20/m_breadth vẫn là feature MẠNH NHẤT model chọn nên proxy không phải nút chặn;
(3) oracle không tính slot giải phóng (nếu tính, oracle tăng nhưng phần hiện thực hóa vẫn 0
vì separation = 0).*
