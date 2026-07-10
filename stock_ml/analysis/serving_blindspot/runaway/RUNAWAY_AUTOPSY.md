# NGHỊCH LÝ RUNAWAY — AUTOPSY COHORT THUẦN (2026-07-09)

Giai đoạn 1 offline, KHÔNG đốt run leaderboard. Frame: serving bundle 2643 wavestruct
(= champion, parity wheel-engine đã xác nhận ở 01_generate.py; 3.813 trades, 106.120 buy-signal,
2020-01→2026-07). Cost model tái lập từ engine, verify 20/20 trades (rw_00_check.py):
entry fill = px×1.0015, exit signal = sell-bar t → fill close[t+1]×0.9985, pnl = ratio −1 −0.004.
Đơn vị u = tổng pnl fraction (1u = 100%/1 vị thế), cùng quy ước e_sim.py. Exit stack champion
xấp xỉ bằng signal-exit thuần (thực tế 3.773/3.813 = 99% exit 'signal'; trailing/overext 8 lệnh bỏ qua).
Scripts: rw_00_check.py → rw_04_recapture.py, CSV cùng thư mục.

## 1. Định lượng cohort (rw_01)

- Buy-signal toàn frame: **106.120**; unfilled thuần (limit treo 40 bar không khớp): **7.318 (6,9%)**.
- **RUNAWAY thuần** (close sau 40 bar > close tín hiệu): **6.987 = 6,6% tổng tín hiệu, 95,5% unfilled**.
  Gần như MỌI lệnh treo là lệnh treo trên sóng lên — đúng trực giác của nghịch lý.
- Forward return từ close tín hiệu (không phí): fwd21 **+12,5%**, fwd40 **+18,4%**, fwd60 **+21,4%**
  (median +9,4/+13,8/+16,8%). Non-runaway unfilled (331): fwd40 −1,2%.
- Theo năm (n / fwd40): 2020: 1515/+22,7% · 2021: 1027/+26,5% · 2022: 510/+19,5% ·
  2023: 1238/+13,9% · 2024: 1374/+12,3% · 2025: 1117/+17,7% · 2026: 206/+16,4%. Không phải hiện tượng 1 regime.

## 2. Giả lập tận dụng at-market (mua close[i+1] riêng cohort) — phân rã 3 kênh chi phí

Sequential 1-slot/mã: 6.987 signal → chỉ **1.186 lệnh takeover** (5.801 bị chính takeover trước đó
của cùng mã chiếm slot — cohort cực kỳ chồng chéo trên cùng con sóng).

Kết quả per-scheme, slot + occupancy nhất quán theo từng exit (rw_02):

| exit scheme | taken | pnl (u) | core bị chặn | foregone (u) | **NET (u)** | hold med | WR |
|---|---|---|---|---|---|---|---|
| champion (sell-head) | 1.186 | +190,2 | 1.114 | +196,9 | **−6,7** | 39 | 0,82 |
| ride21 | 1.168 | +137,0 | 1.150 | +199,0 | **−62,1** | 21 | 0,90 |
| ride40 | 1.083 | +194,6 | 1.305 | +204,1 | **−9,5** | 40 | 0,94 |
| trail10 (peak-close −10%) | 1.009 | +205,8 | 1.317 | +202,6 | **+3,2** | 51 | 0,83 |

Phân rã 3 kênh (trên exit champion, cohort taken):
- **(a) Mất đệm basis 4,5%: −81,2u.** Cùng exit, fill tại limit (counterfactual không tồn tại) = +271,4u
  vs at-market +190,2u. Đúng bài học lineA: alpha nằm ở cái đệm giá.
- **(b) Occupancy: −196,9u — kênh ăn thịt chính.** 1.114 core trades bị chặn, **+17,67%/lệnh
  (pool champion chỉ +7,58%)** — lệnh bị chặn là lệnh PREMIUM vì chúng nằm trên chính con sóng runaway.
- **(c) Exit-mismatch: −36,8u** (champion-exit +190,2 vs trail10 +227,0 cùng entry; sell-head cắt runner
  sớm hơn trailing ~10%). Nhỏ nhất trong 3 kênh, và có sửa cũng vô ích: trail10 net chỉ +3,2u / 6,5 năm
  / 1.009 lệnh ≈ 0 (noise), vì hold dài hơn lại chặn thêm core (blocked 196,9→202,6u tự bù trừ).
- Per-year NET: không năm nào dương có nghĩa ở mọi scheme (champ: 2020 −16,8; các năm khác −1,8..+6,5).

## 3. Separator ex-ante (rw_02) — KHÔNG TỒN TẠI

Feature tại bar tín hiệu: score/score2-5/exit_score/entry_csr + dist_MA20, ret5, ret21, vol20,
runup_lo20, snr21. IC(rank, net-after-occ) tất cả ≤ |0,062|. Best quintile: score5 Q4 +3,9u (n=237,
p_perm=0,018) — không monotone (Q5 lại −0,9), cỡ hiệu ứng ~4u = mức nhiễu, và với ~65 phép thử
(13 feature × 5 quintile) p=0,012–0,036 là đúng kỳ vọng của null. **Không có ngưỡng nào tách được
sub-cohort at-market net dương sau occupancy đáng tin.**

## 4. Cơ chế ex-ante "state-transition takeover" (phase-2 fidelity, rw_03) — ÂM SÂU MỌI X

Trigger đúng như knob dự kiến: limit treo, close vượt signal_close×(1+X) TRƯỚC khi khớp/hết hạn
→ hủy limit, mua market bar kế, exit champion. Kế toán marginal vs reality (lệnh lẽ-ra-khớp bị hủy
= mất trade thật; blocked chỉ tính phần giữ slot VƯỢT reality; KHÔNG cộng lại slot được giải phóng
sớm — bias nhẹ chống takeover, chỉ ảnh hưởng nhánh cancel_fill):

| X | fires | hủy-fill-thật | pure-add | tw_pnl (u) | pnl bị hủy (u) | blocked (u) | **DELTA (u)** |
|---|---|---|---|---|---|---|---|
| 3% | 2.009 | 832 | 1.177 | +98,5 | +16,7 | +199,4 | **−117,6** |
| 5% | 1.702 | 564 | 1.138 | +84,2 | +8,3 | +199,5 | **−123,6** |
| 8% | 1.373 | 326 | 1.047 | +66,0 | +2,3 | +197,0 | **−133,4** |

- **Cả 2 nhánh đều âm, mọi X, mọi năm** (7/7 năm âm ở cả 3 X). Nhánh add (X=5%): −79,6u;
  nhánh cancel_fill: −44,0u — takeover trade trên lệnh lẽ-ra-khớp có tw_pnl **−35,7u** (mua cú pop +5%
  của mã sau đó vẫn chỉnh về −4,5% = mua đúng đỉnh whipsaw, rồi bị sell-head xả).
- X càng cao càng TỆ (monotone −117,6→−133,4): filter cross-X không lọc được gì, chỉ trì hoãn entry
  lên basis đắt hơn. Y hệt cấu trúc kết quả bs_/bch_: ngoại suy về 0 chỉ khi knob off.
- Cross-check frame leaderboard: 79% trades gb_x08 (1.088/1.378) trùng symbol+entry_date với core
  serving — cấu trúc occupancy chuyển thẳng sang frame leaderboard, không cần đốt run để xác nhận âm.

## 5. Đáp án nghịch lý — "tại sao tận dụng runaway thì lỗ" (rw_04)

**Champion KHÔNG bỏ lỡ sóng runaway — nó monetize sóng đó bằng LỆNH LIMIT KẾ TIẾP:**
- **99% (6.940/6.987)** runaway signal có core fill kế tiếp cùng mã, median chỉ **20 ngày** sau
  (p25/p75 = 11/41); 97% trong ≤90 ngày.
- Fill kế tiếp đó ăn **+18,81%/lệnh — gấp 2,5 lần pool champion (+7,58%)**, với basis median chỉ
  **+2,3% trên close của tín hiệu runaway** (tín hiệu mới giữa sóng + đệm −4,5% khi sóng nghỉ).
  (Thống kê theo signal, một fill có thể được đếm cho nhiều signal chồng nhau; bản dedup chính là
  1.114 blocked trades +17,67%/lệnh ở mục 2.)
- Takeover at-market vì vậy không "nhặt tiền rơi": nó **thay thế đúng những fill premium đó bằng
  entry đắt hơn ~4,5–7%** trên cùng con sóng. Kênh occupancy (−197u) không phải side-effect —
  nó CHÍNH LÀ giá trị mà cơ chế hiện tại đã thu được từ runaway wave.

## VERDICT — DỪNG, KHÔNG VIẾT ENGINE CODE (điều kiện bước 7 thỏa)

1. Không tồn tại sub-cohort at-market net dương sau occupancy (best-case trail10 +3,2u/6,5 năm = noise;
   separator null; cơ chế ex-ante −117..−133u, âm 7/7 năm, monotone tệ theo X).
2. Nghịch lý runaway ĐÓNG VĨNH VIỄN bằng số: limit treo trên sóng lên không phải lệnh bị lỡ mà là
   **bộ lọc thời điểm** — sóng đủ khỏe sẽ tự sinh tín hiệu mới và fill −4,5% ở nhịp nghỉ (99%, 20 ngày,
   +18,8%/lệnh). Mọi cơ chế chen ngang (takeover/at-market/reprice) chỉ mua lại chính sóng đó ở basis
   tệ hơn và chiếm slot của fill premium. Nhất quán với chuỗi kết quả A2 (−237), bs_ (−27..−50),
   bch_ (−3..−47), F1 (−14,7).
3. Không sweep rw_, không clone 2783, không knob mới. gb_x08 giữ nguyên top-1.

Caveat trung thực: (i) cohort định nghĩa ex-post (cần biết trước limit không khớp + giá lên) — mọi số
mục 1-3 là CẬN TRÊN lạc quan cho cơ chế thực thi được, và cận trên đó đã ≈ 0; (ii) occupancy first-order
(không sim chuỗi domino re-entry/cooldown), cùng phương pháp với e_sim/bch_ đã được chấp nhận;
(iii) exit champion xấp xỉ signal-exit (99% thực tế); (iv) không cộng slot giải phóng sớm ở nhánh
cancel_fill — nhánh add một mình đã −62..−105u nên verdict không đổi.
