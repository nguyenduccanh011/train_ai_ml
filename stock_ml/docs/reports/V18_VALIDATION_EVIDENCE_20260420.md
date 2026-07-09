# V18 Validation Evidence (2026-04-20)

## 1) Phát hiện lỗi so sánh ban đầu

Trong `run_v18_compare.py` bản đầu, cấu hình gắn nhãn `V17` đã gọi nhầm `backtest_v18`, dẫn đến kết quả `V17 == V18` giả tạo.

Đã sửa:

- Import `backtest_v17` và cho `run_test(..., backtest_fn=...)` chọn đúng engine.
- Cấu hình `V17` chạy `backtest_v17`, cấu hình `V18` chạy `backtest_v18`.

## 2) Kết quả sau khi sửa comparator (full run)

Nguồn: `python stock_ml/run_v18_compare.py`

- `V11`: **+1350.8%**
- `V17`: **+1654.6%**
- `V18`: **+1831.5%**
- `Rule`: **+1982.6%**

Kết luận:

- V18 **đã vượt V17**: `+176.9%`
- V18 **đã vượt V11**: `+480.7%`
- V18 vẫn **chưa vượt Rule**: `-151.2%`

## 3) Bằng chứng mạnh hơn: same-pred comparison

Để loại bỏ nhiễu do train lại model, đã chạy so sánh `backtest_v17` vs `backtest_v18` trên **cùng y_pred** mỗi window/symbol.

Kết quả:

- V17: `n=418`, `tot=+1654.63%`, `PF=2.529`
- V18: `n=422`, `tot=+1831.48%`, `PF=2.738`
- Delta: `+176.85%`, thêm `+4` trades

=> Cải thiện của V18 là **thực** (không phải do may rủi train khác seed).

## 4) Evidence thay đổi có hiệu lực thật (counter nội bộ V18)

Tổng cộng toàn danh mục:

- `n_v18_relaxed_ret5_entries`: **112**
- `n_v18_relaxed_dp_entries`: **135**
- `n_v18_signal_quality_saves`: **49**

Diễn giải:

- Relax `ret_5d` và `dp` đã mở được nhiều entry vốn bị block ở V17.
- Quality check mới ở `signal exit` đã giữ lại thêm các lệnh mà logic cũ sẽ thoát.

## 5) Mã hưởng lợi / bị ảnh hưởng

Top tăng (V18 - V17):

- `AAV`: `+81.0%`
- `VIC`: `+57.0%`
- `DGC`: `+21.0%`
- `SSI`: `+20.3%`
- `AAS`: `+19.1%`
- `TCB`: `+17.4%`

Top giảm:

- `REE`: `-29.8%`
- `VNM`: `-10.3%`
- `FPT`: `-4.9%`
- `BID`: `-3.1%`

## 6) Exit reason evidence

So với V17, V18 có dịch chuyển:

- `peak_protect_dist`: `+3` lệnh
- `peak_protect_ema`: `+2` lệnh
- `stop_loss`: `-2` lệnh
- `signal`: `-1` lệnh

=> V18 đẩy được một phần giao dịch từ nhóm thoát kém chất lượng sang nhóm chốt lời theo trend.

## 7) Kết luận kỹ thuật

1. V18 cải thiện rõ ràng và có bằng chứng định lượng.
2. Khoảng cách còn lại so với Rule là ~`151%`, tập trung ở một số mã chưa tối ưu (đặc biệt các mã bị giảm như `REE`, `VNM`).
3. Thay đổi V18 đã “có tác dụng thật”, không phải thay đổi chết.

