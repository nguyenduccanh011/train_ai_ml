# Fix: Compound Return Calculation

## Vấn đề

Hệ thống đang tính **tổng đơn giản** (simple sum) thay vì **compound return** (lợi nhuận gộp) cho:
- `total_pnl_pct`: Tổng lợi nhuận
- `max_drawdown`: Drawdown tối đa
- `yearly_consistency`: Độ ổn định theo năm

### Ví dụ minh họa

Giả sử có 3 trades:
- Trade 1: +10%
- Trade 2: +10%
- Trade 3: +10%

**Cách tính SAI (simple sum):**
```
Total PnL = 10% + 10% + 10% = 30%
```

**Cách tính ĐÚNG (compound):**
```
Equity = 1.0
After trade 1: 1.0 × 1.10 = 1.10 (+10%)
After trade 2: 1.10 × 1.10 = 1.21 (+21%)
After trade 3: 1.21 × 1.10 = 1.331 (+33.1%)

Total PnL = 33.1%
```

### Tại sao sai số lớn?

Với nhiều trades, sai số tích lũy rất nhanh:
- 100 trades × 5% mỗi trade:
  - Simple sum: 500%
  - Compound: 13,050% (!)
  
Đây là lý do bạn thấy:
- Tiền: 46,900% (simple sum - SAI)
- %: 187.5% (có thể là compound - ĐÚNG)

## Các file đã sửa

### 1. `src/export/unified_export.py`

**Function `compute_stats()`:**

```python
# TRƯỚC (SAI):
"total_pnl_pct": round(sum(pnls), 1),

# SAU (ĐÚNG):
cumulative_multiplier = 1.0
for pnl in pnls:
    cumulative_multiplier *= (1 + pnl / 100)
total_pnl_compound = (cumulative_multiplier - 1) * 100

"total_pnl_pct": round(total_pnl_compound, 1),  # Compound return
"total_pnl_simple": round(sum(pnls), 1),  # Simple sum (for reference)
```

### 2. `src/evaluation/scoring.py`

**Function `calc_metrics()`:**
- Tương tự như trên, sửa `total_pnl` từ simple sum sang compound

**Function `calc_mdd_per_symbol()`:**

```python
# TRƯỚC (SAI):
equity = np.cumsum(pnls)  # Simple cumulative sum

# SAU (ĐÚNG):
equity = np.zeros(len(pnls))
cumulative = 1.0
for i, pnl in enumerate(pnls):
    cumulative *= (1 + pnl / 100)
    equity[i] = (cumulative - 1) * 100
```

**Function `calc_max_drawdown()`:**
- Tương tự như `calc_mdd_per_symbol()`

**Function `calc_yearly_consistency()`:**
- Thay vì cộng đơn giản `sym_yr[sym][yr] += t["pnl_pct"]`
- Tính compound return cho mỗi (symbol, year)

## Impact

### Metrics sẽ thay đổi:

1. **total_pnl_pct**: Giảm đáng kể (từ 46,900% → ~187.5%)
   - Giá trị mới chính xác hơn
   - Phản ánh lợi nhuận thực tế

2. **max_drawdown**: Có thể thay đổi
   - MDD trên compound curve khác với simple sum
   - Thường nhỏ hơn vì compound curve mượt hơn

3. **yearly_consistency**: Có thể thay đổi
   - CV tính trên compound return chính xác hơn

4. **composite_score**: Có thể thay đổi
   - Phụ thuộc vào total_pnl và mdd_per_symbol

### Backward compatibility:

- Thêm field `total_pnl_simple` để giữ giá trị cũ (for reference)
- `total_pnl_pct` giờ là compound (breaking change)

## Testing

Sau khi deploy, cần:

1. **Regenerate tất cả signal cache:**
```bash
# Clear cache
rm -rf cache/signals/*
rm -rf data/ohlcv/*

# Restart server để regenerate
python app/serve_train61_model.py
```

2. **Verify metrics:**
- Check một symbol có nhiều trades
- So sánh `total_pnl_pct` (compound) vs `total_pnl_simple` (simple sum)
- Compound phải nhỏ hơn simple sum nếu có losses

3. **Check frontend:**
- Equity curve chart đã dùng compound (line 492: `cumulativeMoney`)
- Stats panel sẽ hiển thị compound return mới

## Ví dụ thực tế

Giả sử symbol AAA có trades:
```
Trade 1: +5%
Trade 2: +3%
Trade 3: -2%
Trade 4: +4%
```

**Simple sum (SAI):**
```
Total = 5 + 3 - 2 + 4 = 10%
```

**Compound (ĐÚNG):**
```
1.0 × 1.05 × 1.03 × 0.98 × 1.04 = 1.0989
Total = 9.89%
```

Với nhiều trades, sai số sẽ rất lớn!

## Rollout Plan

1. ✅ Fix code trong `unified_export.py` và `scoring.py`
2. ⏳ Test với một vài symbols
3. ⏳ Clear cache và regenerate toàn bộ
4. ⏳ Verify metrics trên dashboard
5. ⏳ Update documentation nếu cần

## Notes

- Frontend equity chart đã dùng compound từ trước (đúng)
- Backend stats đang dùng simple sum (sai) → đã fix
- Sau khi fix, backend và frontend sẽ consistent
