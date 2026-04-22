# V17 vs Rule Deep Analysis (Run date: 2026-04-20)

## 1) Tổng quan hiệu năng (14 mã)

- V17 tổng: **+1654.8%**
- Rule tổng: **+1982.7%**
- Chênh lệch: **-327.9%** (V17 thấp hơn Rule)
- Số mã V17 > Rule: **4/14** (`HPG`, `VIC`, `REE`, `VNM`)
- Số mã V17 < Rule: **10/14**

Nguồn dữ liệu: `stock_ml/visualization/data/index.json` sau khi chạy `python stock_ml/export_v17_comparison.py`.

## 2) AAV chi tiết (mã trọng tâm)

- V17: **+171.7%** (26 trades, WR 42.3%, PF 2.93)
- Rule: **+318.5%** (36 trades, WR 44.4%, PF 2.91)
- Gap: **-146.8%**

### 2.1 Các nhịp Rule kiếm tốt nhưng V17 không bắt được / bắt kém

1. `2024-04-23 -> 2024-06-04`: Rule **+58.47%**, V17 **không có lệnh overlap** (miss hoàn toàn).
2. `2020-11-24 -> 2021-01-07`: Rule **+79.11%**, V17 chỉ overlap lệnh `2020-12-29 -> 2020-12-31` **+1.37%**.
3. `2021-08-04 -> 2021-10-18`: Rule **+127.36%**, V17 overlap `2021-08-24 -> 2021-10-13` **+61.80%**.
4. `2024-10-29 -> 2024-12-25`: Rule **+28.81%**, V17 overlap `2024-11-29 -> 2024-12-31` **+7.35%**.

### 2.2 Giao dịch nhiễu / không hiệu quả của V17 trên AAV

Các lệnh lỗ ngắn hạn (`hold <= 10 ngày`) nổi bật:

- `2023-02-20 -> 2023-03-03`: **-13.33%** (`signal`)
- `2022-01-06 -> 2022-01-19`: **-13.05%** (`signal`)
- `2023-02-02 -> 2023-02-15`: **-12.24%** (`signal`)
- `2023-06-23 -> 2023-07-06`: **-11.11%** (`signal`)
- `2024-07-29 -> 2024-08-09`: **-9.38%** (`signal`)

Tổng quan exit reason của AAV (V17):

- `signal`: 18 lệnh, tổng PnL **-18.9%**
- `peak_protect_dist`: 2 lệnh, tổng PnL **+93.5%**
- `trailing_stop`: 2 lệnh, tổng PnL **+63.5%**
- `peak_protect_ema`: 1 lệnh, tổng PnL **+28.9%**

=> Lợi nhuận AAV đến từ ít lệnh trend lớn; phần `signal` đang bào mòn mạnh.

### 2.3 Root cause cụ thể cho nhịp miss lớn AAV (23/04/2024 -> 04/06/2024)

Trong cửa sổ này, model vẫn dự báo `pred=1` nhiều phiên nhưng bị chặn bởi filter entry của V17:

- Block lặp lại mạnh nhất: **`ret_5d > 5%` (`ret5_hot`)**
- Kèm theo một số ngày bị **`dp_too_low`** (`dist_to_resistance < 0.025` + score chưa đủ cao).

Ví dụ:

- `2024-04-23`: `pred=1`, bị block bởi `dp_too_low`, `ret5_hot`
- `2024-05-09`: `pred=1`, trend mạnh nhưng vẫn bị block bởi `ret5_hot`
- `2024-05-30`: `pred=1`, tiếp tục bị block bởi `ret5_hot`

Kết luận: bộ lọc anti-chasing đang quá cứng ở pha breakout tăng mạnh của AAV.

## 3) Các mã khác có vấn đề tương tự (V17 < Rule)

### DGC (gap -108.6%)

- Rule tốt hơn rõ ở các nhịp:
  - `2020-04-09 -> 2020-06-24`: Rule +70.18% vs V17 overlap +37.78%
  - `2021-02-08 -> 2021-03-22`: Rule +26.81% vs V17 overlap -5.59%
- Mẫu lỗi: vào lệnh vẫn có nhưng exit/đảo tín hiệu khiến payoff thấp hoặc âm ngay trong trend Rule đang lời.

### AAS (gap -70.9%)

- Gap lớn tại:
  - `2021-05-21 -> 2021-07-06`: Rule +57.95% vs V17 +16.70%
  - `2020-12-21 -> 2021-01-25`: Rule +49.36% vs V17 +9.81%
  - `2023-08-31 -> 2023-09-21`: Rule +2.14% vs V17 -23.62%
- Mẫu lỗi: giữ trend chưa đủ dài và có các lệnh `signal` âm lớn.

### MBB, TCB, SSI

- MBB: nhịp `2023-12-06 -> 2024-03-08` Rule +29.86% vs V17 +6.19%
- TCB: nhịp `2020-04-10 -> 2020-06-11` Rule +24.28% vs V17 -3.57%
- SSI: nhịp `2022-11-17 -> 2022-12-21` Rule +21.70% vs V17 -4.91%

Mẫu chung: V17 thường không miss hoàn toàn quá nhiều nhịp dương, nhưng nhiều lần **tham gia kém hiệu quả / ra sớm / vào lệch nhịp**, khiến net payoff thấp hơn Rule.

## 4) Điểm tốt của V17 hiện tại

1. So với V16, V17 vẫn cải thiện tổng mạnh: **+79.1%** (từ +1575.7% lên +1654.8%).
2. Các module bảo vệ trend winner tốt:
   - `peak_protect_dist` và `trailing_stop` tạo phần lớn lợi nhuận dương.
3. Một số mã V17 thắng Rule rõ (HPG, VIC, VNM): khả năng né các nhịp xấu của Rule tốt hơn ở một số regime.

## 5) Điểm yếu cốt lõi của V17

1. **Entry filter quá chặt ở breakout mạnh**  
   `ret_5d > 5%` + `dp` gate làm bỏ lỡ các trend tăng nhanh (AAV là case rõ nhất).

2. **`signal` exit đang gây âm ròng**  
   Toàn bộ 14 mã: `signal` = 302 lệnh, tổng PnL **-48.6%**.

3. **Payoff asymmetry thua Rule ở nhiều mã**  
   Rule giữ được một số swing lớn hơn, còn V17 thường thu ngắn hoặc vào muộn.

## 6) Đề xuất cải tiến để lợi nhuận “đột phát” hơn nhưng vẫn ổn định

### Ưu tiên 1: Nới anti-chasing theo regime (thay vì block cứng)

- Hiện tại: `ret_5d > 5%` là chặn cứng.
- Đề xuất:
  - Nếu `trend == strong` và (`breakout_setup_score >= 3` hoặc `vol_surge_ratio > 1.5`) thì cho phép vào dù `ret_5d` cao.
  - Chuyển từ hard-block sang **position scaling**: `ret_5d` càng cao thì giảm size, không tắt tín hiệu hoàn toàn.

Kỳ vọng: giảm miss các nhịp kiểu AAV 2024-04/06.

### Ưu tiên 2: Exit `signal` theo quality score (không chỉ confirm bars)

- Tách `signal` exit thành 2 lớp:
  - Exit ngay nếu có cụm bearish mạnh (MACD hist âm sâu + close < MA20 + volume phân phối).
  - Nếu bearish yếu, giữ lệnh thêm với trailing mềm.
- Thêm rule “recovery bar grace” 1-2 bar cho lệnh đang lời > X%.

Kỳ vọng: giảm các lệnh âm ngắn hạn và tránh bị rung khỏi trend.

### Ưu tiên 3: Meta policy cho symbol “high-beta/choppy”

- Phân nhóm mã theo biến động/chop (ví dụ dựa trên `bb_width_percentile`, whipsaw rate lịch sử).
- Với nhóm high-beta (AAV/AAS dạng này): giảm độ cứng của `dp` và `ret5` block, tăng trọng breakout quality.

Kỳ vọng: tăng payoff trên nhóm penny/high-vol mà không làm vỡ risk toàn cục.

### Ưu tiên 4: Objective tuning theo payoff, không chỉ WR

- Khi tune threshold module, tối ưu mục tiêu đa tiêu chí:
  - Max `total_pnl`
  - Min `signal_exit_negative_pnl`
  - Min `missed_positive_rule_windows`
  - Ràng buộc `max_drawdown`/`max_loss`

## 7) Kế hoạch triển khai thực nghiệm ngắn

1. Thêm logging lý do block entry từng bar (đặc biệt `ret5_hot`, `dp_too_low`, `anti_chop`, `bear_defense`).
2. Chạy ablation 3 biến thể:
   - V17 + adaptive ret5 gate
   - V17 + signal-exit quality gate
   - V17 + cả hai
3. Đánh giá riêng các mã gap âm lớn (`AAV`, `DGC`, `AAS`, `MBB`, `TCB`, `SSI`) trước khi rollout full universe.

