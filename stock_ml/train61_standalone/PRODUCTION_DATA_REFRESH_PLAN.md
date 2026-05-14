# Kế hoạch production cho cập nhật dữ liệu mới, feature và predict

## Mục tiêu

Đảm bảo khi sang ngày giao dịch mới và dữ liệu OHLCV được bổ sung, ứng dụng Train61 standalone:

- Hiển thị nến mới trên chart.
- Tự tính lại feature khi dữ liệu nguồn thay đổi.
- Tự generate lại signal/predict khi cache cũ không còn khớp dữ liệu mới.
- Không tải lại hoặc predict lại quá mức khi dữ liệu chưa đổi.
- Giữ hiệu suất ổn khi nhiều symbol/model được truy cập.

## Hiện trạng

### Frontend

- OHLCV đã được đổi sang `cache: 'no-store'`, nên trình duyệt không giữ cache cũ.
- Signal cũng đang gọi với `cache: 'no-store'`.
- Polling on-demand đã có timeout, backoff và abort khi đổi symbol/model.

Điểm còn thiếu: frontend không biết signal cache backend có đang cũ so với dữ liệu ngày mới hay không.

### Backend OHLCV

`_load_ohlcv(symbol)` hiện ưu tiên file cache JSON nếu tồn tại:

```python
path = _base_ohlcv_path(symbol)
if path.exists():
    return _read_json(path)
```

Vấn đề: nếu dữ liệu gốc đã có nến mới nhưng JSON cache vẫn tồn tại, API `/api/ohlcv/<symbol>` có thể trả dữ liệu cũ.

### Backend signal

`/api/signal/<model_id>/<symbol>` hiện ưu tiên:

1. Memory cache `signal_results`
2. Disk cache signal
3. Legacy cache
4. Chỉ generate mới nếu không có cache hoặc có `refresh=1`

Vấn đề: nếu signal được generate hôm qua, hôm nay có nến mới thì endpoint vẫn có thể trả signal cũ.

### Feature cache

`FeatureCacheManager` đã fingerprint dữ liệu nguồn bằng `stat_size` và `st_mtime_ns` của `all_symbols/symbol=.../timeframe=.../data.csv`.

Điểm tốt: nếu code đi tới bước tính feature và file CSV đã đổi, feature cache sẽ miss và tính lại.

Điểm rủi ro: nếu signal cache cũ được trả trước, flow sẽ không đi tới bước tính feature/predict.

## Thiết kế đề xuất

### 1. Thêm data fingerprint cho mỗi symbol

Tạo helper backend để lấy fingerprint dữ liệu nguồn của một symbol:

```python
def _symbol_data_fingerprint(symbol: str) -> dict[str, Any]:
    ...
```

Thông tin nên gồm:

- `symbol`
- `source_path`
- `size`
- `mtime_ns`
- `latest_bar_date`
- `latest_close`

Ưu tiên lấy từ data source thật đang dùng bởi `DataLoader`, không chỉ từ OHLCV JSON cache.

Nếu đọc latest bar từ file nguồn quá tốn, có thể bắt đầu với `size + mtime_ns`, sau đó bổ sung `latest_bar_date`.

### 2. Gắn fingerprint vào OHLCV cache

Khi `_load_ohlcv(symbol)` build OHLCV JSON, payload nên có metadata:

```json
{
  "symbol": "AAA",
  "ohlcv": [...],
  "data_fingerprint": {...},
  "latest_bar_date": "2026-05-10"
}
```

Khi đọc cache:

- Nếu cache không có `data_fingerprint`: coi là stale và rebuild.
- Nếu fingerprint hiện tại khác fingerprint trong cache: rebuild.
- Nếu giống: trả cache.

Kết quả: `/api/ohlcv/<symbol>` tự cập nhật khi data source đổi, nhưng vẫn nhanh khi chưa đổi.

### 3. Gắn fingerprint vào signal payload

Khi generate signal trong `_generate_signal_threaded`, trước hoặc sau `_generate_signal_for_model`, lấy fingerprint hiện tại và gắn vào payload:

```python
payload["data_fingerprint"] = fingerprint
payload["latest_bar_date"] = fingerprint.get("latest_bar_date")
```

Có thể đặt trong `_build_signal_payload()` nếu truyền `realtime`/metadata vào thuận tiện hơn.

### 4. Kiểm tra signal cache stale trước khi trả cache

Tạo helper:

```python
def _is_signal_payload_fresh(payload: dict[str, Any], symbol: str) -> bool:
    ...
```

Logic:

- Nếu payload không phải dict: false.
- Nếu thiếu `data_fingerprint`: false.
- So sánh fingerprint hiện tại với fingerprint trong payload.
- Nếu khác: false.
- Nếu giống: true.

Áp dụng trong `/api/signal/<model_id>/<symbol>` trước khi trả:

- `signal_results`
- disk cache
- legacy cache

Nếu stale:

- Xoá khỏi memory cache.
- Có thể bỏ qua disk cache và submit generate job mới.
- Không nhất thiết xoá file ngay; có thể overwrite sau khi job done.

### 5. Làm mới symbol list cache khi dữ liệu đổi

`/api/symbols` hiện cache theo `model_id` trong `symbol_list_cache`.

Vấn đề: list có stats/pnl/trade count từ signal cache cũ. Nếu signal stale, list có thể hiển thị metric cũ.

Cách xử lý tối thiểu:

- Khi build row cho từng symbol, nếu payload stale thì không dùng stats từ payload.
- `cached` / `has_historical_export` có thể vẫn true, nhưng nên thêm field `stale: true` để frontend biết.

Cách tốt hơn:

- Cache symbol list theo `(model_id, global_data_version)`.
- `global_data_version` có thể hash từ danh sách file source `size + mtime_ns`.
- Khi data đổi, key đổi và list tự rebuild.

Khuyến nghị triển khai theo hướng tối thiểu trước để ít rủi ro.

### 6. Frontend hiển thị trạng thái stale/generating rõ hơn

Sau khi backend tự phát hiện stale:

- Frontend không cần gọi `refresh=1` mặc định.
- Nếu signal cache stale, `/api/signal` trả `202 generating` như hiện tại.
- Overlay `Đang sinh tín hiệu...` tiếp tục hoạt động.

Có thể bổ sung status text:

- `Đang cập nhật tín hiệu theo dữ liệu mới...`
- Nếu API trả thêm `reason: "stale_cache"`, frontend hiển thị chính xác hơn.

### 7. Không dùng `refresh=1` mặc định cho mọi request

Không nên để frontend luôn gọi `?refresh=1` vì:

- Mỗi lần click symbol sẽ generate lại dù dữ liệu không đổi.
- Tốn CPU khi nhiều user.
- Dễ tạo queue nhiều job.

Chỉ dùng `refresh=1` cho nút thủ công hoặc debug.

## Thứ tự triển khai

### Phase 1: Backend freshness cho OHLCV

1. Tạo helper lấy fingerprint data source cho symbol.
2. Sửa `_load_ohlcv()` để kiểm tra cache stale.
3. Gắn `data_fingerprint` và `latest_bar_date` vào OHLCV payload.
4. Test:
   - Gọi `/api/ohlcv/AAA` lần đầu tạo cache.
   - Sửa/append data source.
   - Gọi lại API và xác nhận latest bar đổi.

### Phase 2: Backend freshness cho signal

1. Gắn `data_fingerprint` vào signal payload khi generate.
2. Tạo `_is_signal_payload_fresh()`.
3. Áp dụng check stale cho memory cache và disk cache trong `/api/signal`.
4. Khi stale, submit generate job mới và trả `202`.
5. Test:
   - Generate signal cũ.
   - Cập nhật data source.
   - Gọi `/api/signal/<model>/<symbol>` và xác nhận không trả cache cũ.
   - Poll status tới done.
   - Xác nhận payload mới có latest bar mới.

### Phase 3: Symbol list không dùng metric stale

1. Trong `/api/symbols`, nếu payload stale thì không dùng stats cũ.
2. Trả `pnl: null`, `trade_count: null` cho metrics + thêm field `stale: true`.
3. Frontend có thể hiển thị badge `stale` hoặc đơn giản coi như chưa cached.
4. Test list sau khi data đổi.

### Phase 4: Monitoring và production readiness

1. Thêm metrics tracking cho freshness checks.
2. Thêm logging cho stale detection và regenerate events.
3. Giữ frontend `no-store` cho signal và OHLCV, nhưng backend tự cache đúng theo fingerprint.
4. Theo dõi `/api/cache-stats` khi nhiều request.
5. Verify `_ensure_job_for_model()` de-dup logic hoạt động đúng.
6. Test với concurrent requests để đảm bảo không có race condition.

## Rủi ro cần chú ý

### Data source không nằm ở `all_symbols/.../data.csv`

Feature cache đang fingerprint theo layout `all_symbols/symbol=.../timeframe=.../data.csv`. Nếu production data loader đọc nguồn khác, fingerprint phải dùng đúng nguồn đó.

### `mtime_ns` không đổi dù nội dung đổi

Hiếm nhưng có thể xảy ra khi copy file giữ timestamp. Nếu cần chắc hơn, hash tail file hoặc latest row.

### Model `pooled_global_rerun`

Model dạng pooled có cache toàn cục `pooled_global_cache`. Khi bất kỳ symbol data đổi, cache pooled có thể stale.

Cần xử lý riêng:

- Gắn global fingerprint cho toàn bộ train61 symbol set.
- Nếu global fingerprint đổi, clear/rebuild `pooled_global_cache` cho model đó.

### Backtest replay

`backtest_replay` là historical cố định, không nên auto-refresh theo data mới nếu mục tiêu là replay kết quả cũ.

**Giải pháp**: Thêm flag `skip_freshness_check` trong model config hoặc check model type để bỏ qua freshness check cho backtest.

### Race condition khi concurrent requests

Nếu 2 request cùng phát hiện stale và submit generate job, có thể tạo duplicate jobs.

**Giải pháp**: `_ensure_job_for_model()` đã có lock và check `state.status == "running"` để tránh duplicate. Cần verify logic này hoạt động đúng.

### Data source bị xóa hoặc corrupt

Fingerprint helper cần handle exception khi file không tồn tại hoặc không đọc được.

**Giải pháp**: Wrap trong try-except, trả error rõ ràng thay vì crash. Log warning để debug.

### Migration cache cũ

Cache hiện tại không có `data_fingerprint`. Khi deploy code mới:

**Option 1 (khuyến nghị)**: Clear all cache khi deploy - đơn giản, downtime ngắn
```python
# Trong deployment script
rm -rf signal_cache/*
rm -rf base_data/*
```

**Option 2**: Lazy migration - rebuild từng symbol khi được request. Code tự động handle vì cache thiếu fingerprint sẽ được coi là stale.

## Monitoring và Metrics

Thêm các metrics để theo dõi hiệu suất và phát hiện vấn đề:

### Cache metrics (đã có trong `/api/cache-stats`)
- Cache hit/miss rate cho signal_results
- Queue depth (số job đang chạy)
- Eviction count

### Freshness metrics (cần thêm)
- `freshness_check_count`: Số lần check fingerprint
- `stale_cache_detected`: Số lần phát hiện cache stale
- `fingerprint_error_count`: Số lần lỗi khi tính fingerprint
- `regenerate_triggered`: Số lần trigger regenerate do stale

### Logging
- Log khi phát hiện stale cache: `[STALE] model={model_id} symbol={symbol} reason={fingerprint_mismatch}`
- Log khi fingerprint error: `[FINGERPRINT_ERROR] symbol={symbol} error={error}`
- Log khi regenerate: `[REGENERATE] model={model_id} symbol={symbol} reason={stale_cache}`

## Tiêu chí hoàn thành

- Khi thêm nến mới vào data source, `/api/ohlcv/<symbol>` trả latest bar mới mà không cần xoá cache thủ công.
- Khi thêm nến mới, `/api/signal/<model>/<symbol>` không trả signal cũ; endpoint tự generate lại hoặc trả `202`.
- Feature cache miss khi file nguồn đổi và hit khi không đổi.
- Click qua lại cùng symbol không generate lại nếu dữ liệu không đổi.
- Frontend chart, markers, trades và status cùng phản ánh latest bar mới.
- Metrics và logs cho phép monitor freshness behavior trong production.

## Kiểm thử đề xuất

### Manual test

1. Start server.
2. Mở app và chọn một symbol.
3. Ghi nhận latest bar, trade count, status source.
4. Append một dòng OHLCV mới vào data source của symbol đó.
5. Reload app hoặc chọn lại symbol.
6. Xác nhận:
   - Chart có nến mới.
   - Signal được regenerate hoặc polling chạy tới done.
   - `realtime.latest_bar_date` khớp ngày mới.
   - Không cần xoá cache thủ công.

### API test

1. `GET /api/ohlcv/<symbol>` trước và sau khi data đổi.
2. `GET /api/signal/<model>/<symbol>` trước và sau khi data đổi.
3. `GET /api/signal/<model>/<symbol>/status` nếu nhận `202`.
4. `GET /api/cache-stats` để kiểm tra queue/job không tăng bất thường.

### Regression test

- Data không đổi: gọi signal nhiều lần phải trả cache nhanh.
- Data đổi: signal cũ bị bỏ qua.
- Model không tồn tại: vẫn trả 400.
- Symbol không đủ dữ liệu: vẫn trả lỗi rõ ràng.
- Backtest replay không bị regenerate không mong muốn.
