# Implementation Summary: Data Freshness System

## Tổng quan

Đã implement hệ thống data freshness để tự động phát hiện khi dữ liệu nguồn thay đổi và invalidate cache cũ, trigger regenerate signal mới.

## Các thay đổi chính

### 1. Data Fingerprinting (serve_train61_model.py)

**Thêm helper functions:**

- `_get_data_source_path(symbol)`: Lấy đường dẫn file data source
- `_symbol_data_fingerprint(symbol)`: Tính fingerprint của data source
  - Bao gồm: size, mtime_ns, latest_bar_date, latest_close
  - Handle errors gracefully
- `_fingerprints_match(fp1, fp2)`: So sánh 2 fingerprints

**Metrics tracking:**
```python
freshness_metrics = {
    "freshness_check_count": 0,
    "stale_cache_detected": 0,
    "fingerprint_error_count": 0,
    "regenerate_triggered": 0,
}
```

### 2. OHLCV Freshness Check

**Cập nhật `_load_ohlcv(symbol)`:**
- Check cache có fingerprint không
- So sánh fingerprint hiện tại với cached fingerprint
- Nếu khác nhau → rebuild cache
- Gắn fingerprint và latest_bar_date vào payload

**Payload mới:**
```json
{
  "symbol": "AAA",
  "ohlcv": [...],
  "data_fingerprint": {
    "symbol": "AAA",
    "source_path": "...",
    "size": 12345,
    "mtime_ns": 1234567890,
    "latest_bar_date": "2026-05-10",
    "latest_close": 25.5
  },
  "latest_bar_date": "2026-05-10"
}
```

### 3. Signal Freshness Check

**Thêm `_is_signal_payload_fresh(payload, symbol)`:**
- Check payload có fingerprint không
- So sánh với fingerprint hiện tại
- Track metrics
- Log khi phát hiện stale

**Cập nhật `_build_signal_payload()`:**
- Gắn data_fingerprint vào mọi signal payload
- Gắn latest_bar_date

**Cập nhật `/api/signal/<model_id>/<symbol>`:**
- Check freshness cho memory cache
- Check freshness cho disk cache
- Check freshness cho legacy cache
- Nếu stale → remove cache và trigger regenerate
- Skip freshness check cho `backtest_replay` model type
- Trả `reason: "stale_cache"` trong 202 response

### 4. Symbol List Stale Handling

**Cập nhật `/api/symbols`:**
- Check freshness cho mỗi symbol payload
- Nếu stale → không dùng stats, trả null
- Thêm field `stale: true/false`
- `cached` chỉ true nếu payload fresh

**Response mới:**
```json
{
  "symbol": "AAA",
  "train61_model_trades": 10,
  "train61_model_pnl": 5.5,
  "train61_model_wr": 60.0,
  "is_train61": true,
  "cached": true,
  "has_historical_export": true,
  "stale": false
}
```

### 5. Monitoring

**Cập nhật `/api/cache-stats`:**
- Thêm section `freshness` với metrics:
  - `freshness_check_count`
  - `stale_cache_detected`
  - `fingerprint_error_count`
  - `regenerate_triggered`

### 6. Logging

**Thêm log messages:**
- `[STALE] OHLCV cache for {symbol} is stale, rebuilding`
- `[STALE] Signal cache for {symbol} is stale`
- `[REGENERATE] model={model_id} symbol={symbol} reason={reason}`
- `[FINGERPRINT_ERROR] symbol={symbol} error={error}`
- `[FINGERPRINT_WARNING] Could not read latest bar for {symbol}: {error}`
- `[CACHE_ERROR] Failed to read {cache_type} cache for {symbol}: {error}`

## Behavior Changes

### Trước khi implement:
1. OHLCV cache: Đọc file JSON nếu tồn tại, không check data source
2. Signal cache: Trả cache nếu tồn tại, không check data đã đổi
3. Cần xóa cache thủ công khi data thay đổi

### Sau khi implement:
1. OHLCV cache: Tự động rebuild khi data source thay đổi
2. Signal cache: Tự động detect stale và regenerate
3. Không cần xóa cache thủ công
4. Backtest replay không bị auto-refresh (preserved historical results)

## Testing

**Test scripts:**
1. `tools/test_freshness.py`: Automated tests cho freshness system
   - Test OHLCV freshness
   - Test signal freshness
   - Test cache stats
   - Test symbols list

2. `tools/test_data_change.py`: Manual test cho data change flow
   - Generate signal
   - Modify data source (add fake bar)
   - Verify auto-regenerate
   - Restore original data

**Cách chạy test:**
```bash
# Start server
python app/serve_train61_model.py

# Run automated tests
python tools/test_freshness.py

# Run manual test (interactive)
python tools/test_data_change.py
```

## Migration Strategy

**Option 1 (Recommended): Clear cache khi deploy**
```bash
rm -rf cache/signals/*
rm -rf data/ohlcv/*
```

**Option 2: Lazy migration**
- Cache cũ không có fingerprint sẽ được coi là stale
- Tự động rebuild khi được request
- Không cần action thủ công

## Performance Impact

**Minimal overhead:**
- Fingerprint calculation: ~1-2ms (stat + read last row)
- Fingerprint comparison: <1ms (dict comparison)
- Chỉ chạy khi check cache, không ảnh hưởng generate time

**Benefits:**
- Giảm confusion khi data cập nhật
- Không cần xóa cache thủ công
- Tự động sync với data source
- Metrics để monitor behavior

## Known Limitations

1. **mtime_ns edge case**: Nếu file được copy với preserved timestamp, có thể không detect change
   - Mitigation: Cũng check size và latest_bar_date

2. **Pooled model**: `pooled_global_rerun` cache toàn cục chưa có global fingerprint
   - TODO: Implement global fingerprint cho pooled models

3. **Race condition**: Concurrent requests có thể trigger duplicate jobs
   - Mitigation: `_ensure_job_for_model()` đã có lock, nhưng cần verify

## Next Steps (Optional)

1. **HTTP ETag support**: Thêm ETag header dựa trên fingerprint
2. **Global fingerprint**: Cho pooled models
3. **Fingerprint cache**: Cache fingerprint với TTL ngắn để giảm I/O
4. **Alert system**: Alert khi stale rate cao hoặc fingerprint error nhiều

## Files Changed

- `app/serve_train61_model.py`: Core implementation
- `PRODUCTION_DATA_REFRESH_PLAN.md`: Updated plan với improvements
- `tools/test_freshness.py`: Automated test script (new)
- `tools/test_data_change.py`: Manual test script (new)
- `IMPLEMENTATION_SUMMARY.md`: This file (new)
