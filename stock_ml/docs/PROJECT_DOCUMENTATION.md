# 📋 VN STOCK ML - TÀI LIỆU DỰ ÁN

## 1. MỤC ĐÍCH DỰ ÁN

Xây dựng hệ thống **thử nghiệm có hệ thống** (systematic experimentation) nhiều mô hình Machine Learning trên dữ liệu chứng khoán Việt Nam (2015-2025), nhằm:

1. **Tìm ra mô hình tốt nhất** cho dự đoán xu hướng giá cổ phiếu VN
2. **So sánh công bằng** giữa nhiều thuật toán, feature sets, thời kỳ khác nhau
3. **Đánh giá hiệu quả thực tế** qua backtest mô phỏng giao dịch với phí thực tế VN
4. **Tạo nền tảng mở rộng** để dễ dàng thêm model mới, feature mới, strategy mới

---

## 2. DỮ LIỆU ĐẦU VÀO

### 2.1 Dataset chính
- **Nguồn:** `portable_data/vn_stock_ai_dataset_cleaned/`
- **Giai đoạn:** 2015 - hiện tại (04/2025)
- **Số mã:** 270 mã cổ phiếu (đã lọc từ 1,581 mã ban đầu)

### 2.2 Tiêu chí lọc dữ liệu (đã thực hiện ở bước preprocessing)
| Tiêu chí | Ngưỡng |
|---|---|
| Số ngày giao dịch tối thiểu | ≥ 250 ngày |
| Giá trị GD trung bình tối thiểu | ≥ 500,000 VND |
| Giá trị GD median tối thiểu | ≥ 100,000 VND |
| Tỷ lệ dữ liệu đầy đủ | ≥ 70% |
| Tỷ lệ ngày volume = 0 | ≤ 30% |

### 2.3 Lý do loại bỏ
- **Low coverage (<70%):** 226 mã (ACB, FPT, VNM, VCB,... - thiếu nhiều cột context)
- **Low liquidity:** 757 mã (giao dịch quá ít)
- **Too few days (<250):** 32 mã (niêm yết mới)
- **Data anomaly:** 6 mã (volume = 0 quá nhiều)
- **Non-stock:** 4 (index, futures)

### 2.4 Cấu trúc mỗi file CSV (per symbol)
Các cột OHLCV tiêu chuẩn: `date, open, high, low, close, volume, traded_value`

### 2.5 Dữ liệu context
- **HNXINDEX:** Chỉ số sàn HNX
- **HNXUPCOM:** Chỉ số sàn UPCOM
- **VN30F1M, VN30F2M:** Hợp đồng tương lai VN30

---

## 3. CÁC QUYẾT ĐỊNH THIẾT KẾ ĐÃ CHỐT

### 3.1 Bài toán (Problem Formulation)
- **Loại bài toán:** Classification 3 lớp (UPTREND / SIDEWAYS / DOWNTREND)
- **Phương pháp tạo target:** Dual Moving Average Crossover
  - MA ngắn: 10 ngày
  - MA dài: 40 ngày
  - Label: 1 (UP) khi MA10 > MA40, -1 (DOWN) khi MA10 < MA40, 0 (SIDEWAYS) khi chênh lệch < 1%

### 3.2 Data Split Strategy
- **Phương pháp:** Walk-Forward Validation (rolling window)
- **Train window:** 4 năm
- **Test window:** 1 năm
- **Gap:** 0 ngày (có thể điều chỉnh)
- **Các windows:**
  - Train 2016-2019 → Test 2020
  - Train 2017-2020 → Test 2021
  - Train 2018-2021 → Test 2022
  - Train 2019-2022 → Test 2023
  - Train 2020-2023 → Test 2024
  - Train 2021-2024 → Test 2025

> **Lý do chọn Walk-Forward:** Tránh look-ahead bias, phản ánh thực tế khi deploy model - luôn train trên quá khứ, test trên tương lai.

### 3.3 Feature Engineering - 3 cấp độ

#### Minimal (22 features)
| Nhóm | Features |
|---|---|
| Returns | return_1d, return_5d, return_10d, return_20d |
| Volatility | volatility_10d, volatility_20d, atr_14 |
| Volume | volume_ratio_5d, volume_ratio_20d |
| MA signals | sma_10, sma_20, sma_50, ema_12, ema_26 |
| Basic TA | rsi_14, macd, macd_signal |
| Price position | high_low_range, close_to_high, close_to_low |
| Day of week | day_of_week, month |

#### Technical (52 features) = Minimal + 
| Nhóm | Features bổ sung |
|---|---|
| Bollinger Bands | bb_upper, bb_lower, bb_width, bb_position |
| Stochastic | stoch_k, stoch_d |
| Williams %R | williams_r |
| CCI | cci_20 |
| MFI | mfi_14 |
| OBV | obv, obv_sma |
| ADX | adx_14, plus_di, minus_di |
| Ichimoku | tenkan, kijun, senkou_a, senkou_b |
| VWAP | vwap_ratio |
| Price gaps | gap_up, gap_down |
| Candle patterns | body_ratio, upper_shadow, lower_shadow |
| Momentum | roc_10, roc_20, tsi |
| Lag features | return_1d_lag1..lag5 |

#### Full (70+ features) = Technical + Market Context
| Nhóm | Features bổ sung |
|---|---|
| Market index | vn30_return, hnx_return, market_volatility |
| Cross-market | correlation_with_market, beta, relative_strength |

### 3.4 Models đã chọn

| Model | Loại | Cần Scaling | Lý do chọn |
|---|---|---|---|
| **Random Forest** | Tree ensemble | Không | Baseline mạnh, robust |
| **Extra Trees** | Tree ensemble | Không | So sánh với RF |
| **Gradient Boosting** | Boosting | Không | Classic boosting |
| **AdaBoost** | Boosting | Không | Simple boosting |
| **XGBoost** | Boosting | Không | State-of-the-art tabular |
| **LightGBM** | Boosting | Không | Nhanh, hiệu quả |
| **Logistic Regression** | Linear | Có (Robust) | Baseline linear |
| **SGD Classifier** | Linear | Có (Robust) | Online learning |
| **KNN** | Distance-based | Có (Robust) | Non-parametric |
| **SVM** | Kernel | Có (Robust) | Margin-based |
| **Naive Bayes** | Probabilistic | Có (Robust) | Baseline simple |

### 3.5 Metrics đánh giá

#### Classification Metrics
| Metric | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng tổng thể |
| Balanced Accuracy | Accuracy cân bằng giữa các lớp |
| F1-score (macro) | **Metric chính** - cân bằng precision/recall |
| Cohen's Kappa | Đo mức đồng thuận vượt random |
| MCC | Matthews Correlation - robust cho imbalanced |

#### Trading/Backtest Metrics
| Metric | Ý nghĩa |
|---|---|
| Total Return (%) | Lợi nhuận tổng |
| Annualized Return (%) | Lợi nhuận quy năm |
| Sharpe Ratio | Return điều chỉnh rủi ro |
| Max Drawdown (%) | Mức giảm tối đa từ đỉnh |
| Profit Factor | Tổng lãi / Tổng lỗ |
| Win Rate (%) | Tỷ lệ lệnh thắng |
| Avg Win / Avg Loss | LN trung bình lệnh thắng vs thua |
| Avg Holding Days | Số ngày giữ trung bình |
| Max Consecutive Wins/Losses | Chuỗi thắng/thua dài nhất |
| Expectancy | Kỳ vọng lợi nhuận mỗi lệnh |
| Time in Market (%) | % thời gian đầu tư |

### 3.6 Trading Strategy (Backtest)
- **Signal:** Buy khi model dự đoán UPTREND (1), bán/giữ tiền khi SIDEWAYS (0) hoặc DOWNTREND (-1)
- **Phí giao dịch:** 0.15% mỗi chiều (mua + bán)
- **Thuế bán:** 0.1%
- **Vốn ban đầu:** 100,000,000 VND (100M)
- **Position sizing:** All-in (100% vốn mỗi lệnh) - có thể cải thiện sau

---

## 4. CẤU TRÚC DỰ ÁN

```
stock_ml/
├── run.py                      # Entry point - chạy experiment grid
├── run_backtest.py             # Entry point - chạy backtest lợi nhuận
├── requirements.txt            # Dependencies
├── config/
│   ├── base.yaml               # Config mặc định
│   └── features/
│       └── technical.yaml      # Feature config
├── src/
│   ├── __init__.py
│   ├── pipeline.py             # Orchestrator chính
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Load CSV data
│   │   ├── splitter.py         # Walk-forward split
│   │   └── target.py           # Tạo target labels
│   ├── features/
│   │   ├── __init__.py
│   │   └── engine.py           # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   └── registry.py         # Model registry
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # Classification metrics
│       └── backtest.py         # Trading backtest
├── results/                    # Kết quả tự động lưu
│   ├── results_*.csv
│   ├── results_*.json
│   └── backtest_*.csv
└── docs/
    ├── PROJECT_DOCUMENTATION.md
    └── FIRST_TRAINING_REPORT.md
```

---

## 5. CÁCH SỬ DỤNG

### Quick test (5 symbols, 2 models)
```bash
python run.py
```

### Full experiment (all symbols, all models, all features)
```bash
python run.py --full
```

### Custom run
```bash
python run.py --symbols 20 --models lightgbm xgboost random_forest --features minimal technical
```

### Backtest lợi nhuận
```bash
python run_backtest.py --symbols 10 --features technical
python run_backtest.py --full --capital 200000000
```

---

## 6. HƯỚNG PHÁT TRIỂN TIẾP THEO

### Giai đoạn 2 (dự kiến)
- [ ] Thêm **CatBoost** (đã có sẵn trong registry)
- [ ] Thêm **Neural Networks** (LSTM, Transformer)
- [ ] **Hyperparameter tuning** (Optuna/GridSearch)
- [ ] **Feature selection** (importance-based, RFE)
- [ ] **Ensemble methods** (Stacking, Blending top models)

### Giai đoạn 3 (dự kiến)
- [ ] **Portfolio optimization** (multi-stock selection)
- [ ] **Position sizing** (Kelly criterion, risk parity)
- [ ] **Real-time prediction** pipeline
- [ ] **Dashboard** visualization (Streamlit/Gradio)
- [ ] **Thêm features** từ sentiment, macro data

---

## 7. GHI CHÚ KỸ THUẬT

### Xử lý XGBoost labels
XGBoost yêu cầu labels bắt đầu từ 0. Do target có giá trị {-1, 0, 1}, cần offset +1 khi train và -1 khi predict.

### Xử lý NaN/Inf
Tất cả NaN và Inf trong features được thay thế bằng 0 trước khi train (`np.nan_to_num`).

### Scaling
Chỉ áp dụng `RobustScaler` cho models cần scaling (Linear, SVM, KNN). Tree-based models không cần scaling.

### Walk-Forward vs Time-Series Split
Chọn Walk-Forward (rolling) thay vì expanding window để:
- Train size ổn định → model comparison công bằng hơn
- Tránh data quá cũ ảnh hưởng (market regime changes)

---

*Tài liệu cập nhật: 19/04/2026*
*Version: 1.0*
