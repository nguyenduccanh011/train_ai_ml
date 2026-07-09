# 🔍 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH & ĐỀ XUẤT CẢI THIỆN

**Ngày:** 19/04/2026  
**Phiên bản:** v1.0  

---

## 📊 1. TỔNG QUAN KẾT QUẢ HIỆN TẠI

### 1.1 Classification Metrics
| Model | Feature Set | F1-Macro (Best) | Accuracy (Best) | 
|---|---|---|---|
| Random Forest | Technical | 0.801 | 79.9% |
| XGBoost | Technical | ~0.76 | ~76% |
| LightGBM | Technical | ~0.76 | ~76% |
| Logistic Regression | Technical | ~0.70 | ~70% |

### 1.2 Backtest Metrics (Technical features, 10 cổ phiếu, 100M VND)
| Model | Total Return | Sharpe | Max Drawdown | Win Rate | Trades |
|---|---|---|---|---|---|
| Random Forest | +377.8% | 0.283 | **-88.6%** | 47.6% | 860 |
| Logistic Regression | +45.3% | 0.183 | **-90.0%** | 48.1% | 924 |
| LightGBM | +35.7% | 0.164 | **-93.3%** | 46.3% | 1024 |
| XGBoost | +4.8% | 0.137 | **-92.2%** | 46.0% | 972 |
| Buy & Hold | **-99.8%** | - | - | - | - |

---

## 🚨 2. CÁC VẤN ĐỀ NGHIÊM TRỌNG

### 🔴 Vấn đề 1: Max Drawdown cực kỳ cao (-88% đến -93%)
**Mức độ: NGHIÊM TRỌNG**

- Drawdown -88% có nghĩa là vốn 100M bị giảm còn 12M tại thời điểm tệ nhất
- Bất kỳ trader/quỹ nào cũng sẽ **bị liquidate** trước khi hồi phục
- Nguyên nhân: **Không có quản lý rủi ro** (stop-loss, position sizing, maximum drawdown limit)
- Chiến lược hiện tại: all-in khi predict UPTREND, all-out khi không → quá rủi ro

### 🔴 Vấn đề 2: Buy & Hold = -99.8% là BẤT THƯỜNG
**Mức độ: NGHIÊM TRỌNG**

- Benchmark Buy & Hold -99.8% là **không hợp lý** cho 10 cổ phiếu VN 2020-2025
- VNIndex tăng ~40% từ 2020-2025, nhiều blue-chip tăng mạnh
- **Nguyên nhân có thể:**
  - Bug trong cách tính Buy & Hold (cộng returns thay vì compounding?)
  - Returns data bị tính sai (có thể dùng log returns nhưng compound theo arithmetic)
  - 10 mã được chọn tình cờ đều giảm (rất unlikely)
  - **Data leakage** hoặc shift sai khiến returns bị lệch

### 🔴 Vấn đề 3: Win Rate chỉ ~47% cho bài toán 3 lớp
**Mức độ: NGHIÊM TRỌNG**

- Classification accuracy 77-80% nhưng win rate giao dịch chỉ 47%
- **Gap lớn giữa accuracy và profitability** → mô hình đúng ở class dễ (SIDEWAYS) nhưng sai ở class quan trọng (UPTREND)
- F1-macro cao nhưng **hit_rate_buy** (precision khi predict BUY) có thể rất thấp

### 🟡 Vấn đề 4: Sharpe Ratio quá thấp
**Mức độ: TRUNG BÌNH**

- Sharpe 0.14-0.28 rất thấp (quỹ tốt cần > 1.0, chấp nhận được > 0.5)
- Returns có variance cực cao → chiến lược không ổn định

### 🟡 Vấn đề 5: Target Definition có thể không tối ưu
**Mức độ: TRUNG BÌNH**

- `dual_ma` trend regime dùng SMA20 và SMA50 crossover → lagging indicator
- Target bị **shift -1** (dự đoán ngày mai) nhưng trend regime thay đổi chậm
- Có thể gây **look-ahead bias** nếu dual_ma dùng future prices để xác định regime

### 🟡 Vấn đề 6: Không có Feature Selection / Importance Analysis
**Mức độ: TRUNG BÌNH**

- Technical feature set có ~50+ features → nhiều features có thể là noise
- Không có feature importance tracking → không biết features nào thực sự hữu ích
- Có thể bị **curse of dimensionality** với dataset nhỏ

### 🟡 Vấn đề 7: Không có Hyperparameter Tuning
**Mức độ: TRUNG BÌNH**

- Tất cả models dùng **default parameters**
- RandomForest: n_estimators=200, max_depth=15 → có thể overfit
- Không có cross-validation trong training window

---

## 💡 3. ĐỀ XUẤT CẢI THIỆN (Theo thứ tự ưu tiên)

### 🏆 Ưu tiên 1: Sửa Bug Buy & Hold và Kiểm tra Data Pipeline
```
Hành động:
1. Verify returns calculation trong backtest.py
2. Kiểm tra Buy & Hold formula (phải là compounding, không phải sum)
3. Spot-check 2-3 mã cổ phiếu bằng tay so với giá thực tế
4. Đảm bảo không có look-ahead bias trong target generation
```

### 🏆 Ưu tiên 2: Thêm Risk Management vào Backtest
```python
# Cần thêm:
- Stop-loss per trade: -5% đến -7%
- Maximum portfolio drawdown limit: -20% → giảm position size
- Position sizing: Kelly criterion hoặc fixed fractional (2-5% risk per trade)
- Trailing stop: lock profits khi giá tăng
```

### 🏆 Ưu tiên 3: Cải thiện Target Definition
```
Phương án A: Forward Return Classification
- Target = 1 nếu return 5 ngày tới > +2%
- Target = 0 nếu -2% < return < +2% 
- Target = -1 nếu return < -2%
→ Trực tiếp hơn, ít lag hơn dual_ma

Phương án B: Hybrid Target
- Kết hợp trend regime + forward return confirmation
- UPTREND chỉ khi cả trend regime = UP VÀ forward return > 0
```

### 🏆 Ưu tiên 4: Feature Engineering nâng cao
```
Thêm features:
1. Market regime: VNIndex trend, market breadth (% cổ phiếu trên SMA200)
2. Sector rotation: sector relative strength
3. Volatility regime: VIX-equivalent, realized vol percentile
4. Price patterns: support/resistance levels, chart patterns
5. Order flow: bid-ask imbalance, large trade detection
6. Calendar: day of week, month effects, earnings season

Loại bỏ features noise:
- Dùng feature_importances_ từ RF/XGB
- Mutual Information test
- Recursive Feature Elimination (RFE)
- Loại bỏ features có correlation > 0.95
```

### 🏆 Ưu tiên 5: Hyperparameter Tuning
```python
# Dùng TimeSeriesSplit + Optuna/RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Ví dụ cho Random Forest:
param_space = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_leaf': [5, 10, 20, 50],
    'max_features': ['sqrt', 'log2', 0.3],
    'class_weight': ['balanced', 'balanced_subsample', None],
}
```

### 🏆 Ưu tiên 6: Thêm Ensemble / Stacking
```
- Ensemble predictions từ top 3 models (voting)
- Chỉ trade khi ≥ 2/3 models đồng ý UPTREND
- Meta-learner stack: dùng model predictions làm features cho meta-model
- Confidence threshold: chỉ trade khi probability > 0.6
```

### 🏆 Ưu tiên 7: Cải thiện Backtest Strategy
```
Chiến lược nâng cao:
1. Partial position: allocate theo confidence (60% confident → 60% vốn)
2. Multi-stock portfolio: diversify, không all-in 1 mã
3. Rebalance rules: weekly/monthly rebalance thay vì daily
4. Transaction cost modeling chi tiết hơn (slippage, market impact)
5. Out-of-sample validation: giữ 2024-2025 hoàn toàn out-of-sample
```

---

## 📋 4. ROADMAP CẢI THIỆN

### Phase 1: Bug Fix & Data Validation (1-2 ngày)
- [ ] Kiểm tra và sửa Buy & Hold calculation
- [ ] Validate returns data với giá thực tế
- [ ] Kiểm tra look-ahead bias trong target
- [ ] Thêm data quality checks (null, outliers)

### Phase 2: Risk Management (2-3 ngày)
- [ ] Implement stop-loss trong backtest
- [ ] Implement position sizing (fixed fractional)
- [ ] Thêm maximum drawdown limit
- [ ] Re-run backtest và so sánh

### Phase 3: Model Improvement (3-5 ngày)
- [ ] Feature importance analysis
- [ ] Feature selection (loại bỏ noise)
- [ ] Hyperparameter tuning với TimeSeriesSplit
- [ ] Thử target definition mới (forward return)
- [ ] Implement ensemble/voting classifier

### Phase 4: Advanced Strategy (5-7 ngày)
- [ ] Confidence-based position sizing
- [ ] Multi-stock portfolio optimization
- [ ] Sector/market regime filtering
- [ ] Walk-forward optimization
- [ ] Comprehensive out-of-sample test

---

## 📈 5. KỲ VỌNG SAU CẢI THIỆN

| Metric | Hiện tại | Mục tiêu Phase 2 | Mục tiêu Phase 4 |
|---|---|---|---|
| Max Drawdown | -88% → -93% | < -30% | < -20% |
| Sharpe Ratio | 0.14 - 0.28 | > 0.5 | > 1.0 |
| Win Rate | 47% | > 52% | > 55% |
| Annual Return | 0.8% - 4.1% | > 10% | > 15% |
| Calmar Ratio | < 0.05 | > 0.3 | > 0.5 |

---

## 🎯 6. KẾT LUẬN

Mô hình hiện tại có **classification performance khá** (F1 ~0.77-0.80) nhưng **trading performance rất kém** do:

1. **Thiếu risk management** → drawdown thảm họa
2. **Bug tiềm ẩn** trong data pipeline (Buy & Hold -99.8%)
3. **Gap giữa accuracy và profitability** → cần cải thiện target definition
4. **Chiến lược giao dịch quá đơn giản** (all-in/all-out)

**Ưu tiên #1 tuyệt đối: Sửa data pipeline bugs trước khi làm bất kỳ điều gì khác.**
Nếu Buy & Hold = -99.8% là sai, thì toàn bộ kết quả backtest đều không đáng tin cậy.
