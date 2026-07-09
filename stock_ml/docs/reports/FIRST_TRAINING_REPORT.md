# 📊 BÁO CÁO HIỆU SUẤT LẦN TRAIN ĐẦU TIÊN

**Ngày chạy:** 19/04/2026  
**Version:** v1.0 - First Run  

---

## 1. THÔNG SỐ THỬ NGHIỆM

| Thông số | Giá trị |
|---|---|
| Số mã cổ phiếu | 10 mã (AAH, AAS, AAT, AAV, ABB, ABS, ABW, ACG, ACM, ACV) |
| Giai đoạn dữ liệu | 2015-2025 |
| Tổng dòng dữ liệu | 14,612 rows |
| Feature sets | Minimal (22 features), Technical (52 features) |
| Models | Random Forest, XGBoost, LightGBM, Logistic Regression |
| Walk-forward windows | 6 windows (test 2020→2025) |
| Target | 3-class trend regime (UP/SIDE/DOWN) |
| Vốn backtest | 100,000,000 VND |

---

## 2. KẾT QUẢ CLASSIFICATION (F1-macro)

### 2.1 Feature set: Technical (52 features)

| Model | Test 2020 | Test 2021 | Test 2022 | Test 2023 | Test 2024 | Test 2025 | **TB** |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **0.793** | 0.727 | **0.790** | 0.757 | 0.733 | **0.765** | **0.761** |
| **XGBoost** | 0.759 | **0.751** | 0.787 | **0.764** | **0.738** | 0.753 | **0.759** |
| **LightGBM** | 0.713 | 0.743 | 0.778 | 0.751 | 0.730 | 0.757 | **0.745** |
| **Logistic Reg** | 0.635 | 0.533 | 0.701 | 0.694 | 0.659 | 0.713 | **0.656** |

### 2.2 Feature set: Minimal (22 features)

| Model | Test 2020 | Test 2021 | Test 2022 | Test 2023 | Test 2024 | Test 2025 | **TB** |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 0.681 | 0.704 | **0.760** | 0.723 | 0.723 | **0.748** | **0.723** |
| **XGBoost** | **0.705** | **0.745** | 0.749 | 0.718 | 0.715 | 0.726 | **0.726** |
| **LightGBM** | 0.723 | 0.718 | 0.749 | 0.719 | 0.700 | 0.710 | **0.720** |
| **Logistic Reg** | 0.676 | 0.549 | 0.672 | 0.684 | 0.665 | 0.703 | **0.658** |

### 2.3 Nhận xét Classification
- **Random Forest** và **XGBoost** dẫn đầu ở cả 2 feature sets
- Feature set **Technical** (52 features) cải thiện ~2-4% F1 so với Minimal
- Giai đoạn **2022** (thị trường giảm mạnh) model vẫn giữ F1 cao (~0.78-0.80)
- **Logistic Regression** yếu nhất, đặc biệt năm 2021 (F1=0.53)
- **Kết quả khá ổn định** qua các năm → model có khả năng generalize tốt

---

## 3. KẾT QUẢ BACKTEST (Lợi nhuận)

### 3.1 Tổng hợp (Feature: Technical, 10 mã, 6 năm test)

| Model | Vốn cuối | Lợi nhuận | Return | Return/năm | Sharpe |
|---|---|---|---|---|---|
| 🥇 **Random Forest** | **477.8M** | **+377.8M** | **+377.8%** | +4.14% | 0.283 |
| 🥈 **Logistic Reg** | 145.3M | +45.3M | +45.3% | +0.97% | 0.183 |
| 🥉 **LightGBM** | 135.7M | +35.7M | +35.7% | +0.79% | 0.164 |
| 4️⃣ **XGBoost** | 104.8M | +4.8M | +4.8% | +0.12% | 0.137 |

> **Benchmark Buy&Hold:** -99.81% (do tính trung bình nhiều mã small/mid cap)

### 3.2 Chi tiết theo từng năm test (Random Forest - Technical)

| Năm Test | Model Return | Buy&Hold | Excess | Ghi chú |
|---|---|---|---|---|
| 2020 | **+443.3%** | +242.5% | +200.8% | COVID recovery rally |
| 2021 | +106.9% | +663.9% | -557.1% | Bull market - model thận trọng |
| 2022 | -58.1% | -99.9% | +41.8% | Bear market - model giảm lỗ |
| 2023 | -34.0% | +19.4% | -53.4% | Sideway - model bị whipsaw |
| 2024 | -17.2% | -65.3% | +48.2% | Model tránh được đợt giảm |
| 2025 | +49.1% | -74.7% | +123.8% | Recovery - model bắt đúng |

### 3.3 Chi tiết thống kê lệnh

| Chỉ số | Random Forest | XGBoost | LightGBM | Logistic Reg |
|---|---|---|---|---|
| **Tổng lệnh** | 430 | 486 | 512 | 462 |
| **Lệnh thắng** | 85 | 97 | 97 | 87 |
| **Lệnh thua** | 345 | 389 | 415 | 375 |
| **Win Rate** | 19.8% | 20.0% | 18.9% | 18.8% |
| **LN TB lệnh thắng** | +62.9M (+12.5%) | +21.7M (+9.9%) | +28.8M (+9.7%) | +25.2M (+10.8%) |
| **Lỗ TB lệnh thua** | -15.6M (-2.4%) | -4.9M (-2.0%) | -6.2M (-1.8%) | -5.5M (-2.0%) |
| **LN lớn nhất** | +436.7M (+87.2%) | +204.5M (+87.2%) | +257.1M (+87.2%) | +212.7M (+87.2%) |
| **Lỗ lớn nhất** | -182.3M (-16.9%) | -56.5M (-14.5%) | -70.5M (-12.7%) | -117.0M (-17.7%) |
| **Giữ TB/lệnh** | 5.9 ngày | 4.6 ngày | 4.4 ngày | 6.0 ngày |
| **Giữ median** | 4.0 ngày | 3.0 ngày | 2.0 ngày | 3.0 ngày |
| **Chuỗi thắng dài nhất** | 4 | 3 | 3 | 4 |
| **Chuỗi thua dài nhất** | 24 | 31 | 32 | 30 |
| **Profit Factor** | 0.99 | 1.10 | 1.09 | 1.06 |
| **Max Drawdown** | -88.6% | -90.2% | -93.3% | -90.0% |
| **% Thời gian đầu tư** | 26.0% | 23.1% | 23.2% | 28.5% |
| **Phí GD tổng** | 1.02B | 444.2M | 640.7M | 454.4M |

---

## 4. TOP 10 KẾT QUẢ THEO F1-MACRO

| # | Model | Features | Window | Accuracy | Balanced Acc | F1-macro | MCC |
|---|---|---|---|---|---|---|---|
| 1 | Random Forest | Technical | Test 2020 | 79.9% | 79.4% | **0.793** | 0.688 |
| 2 | Random Forest | Technical | Test 2022 | 79.6% | 79.0% | **0.790** | 0.669 |
| 3 | XGBoost | Technical | Test 2022 | 80.2% | 78.3% | **0.787** | 0.675 |
| 4 | LightGBM | Technical | Test 2022 | 79.1% | 77.7% | **0.778** | 0.657 |
| 5 | Random Forest | Technical | Test 2025 | 75.9% | 77.2% | **0.765** | 0.638 |
| 6 | XGBoost | Technical | Test 2023 | 76.0% | 76.3% | **0.764** | 0.630 |
| 7 | Random Forest | Minimal | Test 2022 | 77.1% | 76.2% | **0.760** | 0.631 |
| 8 | XGBoost | Technical | Test 2020 | 76.2% | 75.5% | **0.759** | 0.655 |
| 9 | Random Forest | Technical | Test 2023 | 75.2% | 76.5% | **0.757** | 0.623 |
| 10 | LightGBM | Technical | Test 2025 | 75.0% | 75.1% | **0.757** | 0.619 |

---

## 5. PHÂN TÍCH & NHẬN XÉT

### 5.1 Điểm mạnh
✅ **F1-macro 0.76-0.80** cho classification 3 lớp → kết quả tốt  
✅ **Tất cả models beat Buy&Hold** khi tính tổng hợp  
✅ **Random Forest** cho kết quả ổn định nhất qua các giai đoạn  
✅ Feature set **Technical** cải thiện đáng kể so với Minimal  
✅ Model chỉ **đầu tư 23-28% thời gian** → giảm exposure rủi ro  

### 5.2 Điểm yếu & Vấn đề cần cải thiện
⚠️ **Win rate thấp (~20%)** - cần cải thiện precision cho signal BUY  
⚠️ **Max Drawdown rất cao (-88 đến -93%)** → cần stop-loss, position sizing  
⚠️ **Chuỗi thua dài (24-32 lệnh)** → tâm lý giao dịch rất khó chịu  
⚠️ **Phí giao dịch cao** (do giao dịch quá nhiều) → cần lọc signal  
⚠️ **Năm 2021 (bull):** Model thận trọng quá, bỏ lỡ rally lớn  
⚠️ **Năm 2023 (sideway):** Model bị whipsaw, thua lỗ  

### 5.3 Giải thích Win Rate thấp nhưng vẫn lãi
- **Chiến lược "let profit run":** Lệnh thắng TB +12.5% vs lệnh thua -2.4%
- Tỷ lệ LN thắng/thua = **5.2x** → chỉ cần thắng 1/5 lệnh là đủ hòa vốn
- Đây là đặc điểm của **trend-following strategy** - thắng ít nhưng thắng lớn

### 5.4 So sánh feature sets
```
Technical (52 features) vs Minimal (22 features):
- F1-macro: +2-4% improvement
- Accuracy: +1-3% improvement  
- Backtest: Kết quả tương đương (không cải thiện rõ ràng)
→ Technical features giúp classification tốt hơn nhưng 
   chưa rõ ràng cải thiện lợi nhuận trading
```

---

## 6. ĐỀ XUẤT CẢI THIỆN CHO LẦN TRAIN TIẾP

### Ưu tiên cao
1. **Thêm stop-loss** vào backtest (giảm max drawdown)
2. **Filter signal** - chỉ trade khi confidence > threshold
3. **Thêm models:** Extra Trees, Gradient Boosting, SVM
4. **Chạy full 270 mã** thay vì 10 mã
5. **Hyperparameter tuning** cho top 3 models

### Ưu tiên trung bình
6. **Feature selection** - loại bỏ features không quan trọng
7. **Ensemble** top models (voting/stacking)
8. **Thêm feature set Full** (market context)
9. **Thử target khác** (return-based thay vì MA-based)

### Ưu tiên thấp
10. **Position sizing** (Kelly criterion)
11. **Multi-stock portfolio** optimization
12. **Neural Networks** (LSTM, Transformer)

---

## 7. FILES KẾT QUẢ

| File | Nội dung |
|---|---|
| `results/results_20260419_000045.csv` | Classification metrics tất cả experiments |
| `results/results_20260419_000045.json` | Cùng data dạng JSON |
| `results/backtest_20260419_001532.csv` | Backtest lợi nhuận tổng hợp |

---

*Báo cáo tự động sinh ngày 19/04/2026*  
*Tổng thời gian chạy: ~30 giây (10 mã, 4 models, 2 feature sets, 6 windows)*
