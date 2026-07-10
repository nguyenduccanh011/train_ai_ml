# P3 PROBE: KHẢ THI REFETCH INTRADAY — HỒ SƠ NỬA NGÀY

Ngày: 2026-07-10. Scripts + dữ liệu thô: `p3_probe/p3_00_fetch_probe.py`, `p3_probe/p3_01_entrade_probe.py`,
`p3_probe/p3_probe_report.json`, `p3_probe/p3_entrade_report.json`, `p3_probe/p3_entrade_bars.parquet` (8.940 bar mẫu),
`p3_probe/p3_f1m_yearly_coverage.csv`. KHÔNG ghi gì vào duckdb production, KHÔNG sửa schema.

## 1. Hạ tầng fetch hiện có + chỗ bug schema

**Nguồn đang dùng: API tự host `https://sieutinhieu.vn/api/v1`** (public, KHÔNG cần key — `.env` không có
API key nào cho data; chỉ có `SIEUTINHIEU_API_BASE` optional trong train61). Ba lớp code:

- `stock_ml/scripts/ops/refetch_adjusted.py` — refetch 1D 488 mã vào `market_raw_api.duckdb` (bảng
  `ohlcv_raw` không PK). Paging limit=1000, sleep 0.1s, backoff; log cho thấy 488 mã chạy trót lọt.
- `stock_ml/scripts/ops/migrate_csv_to_duckdb.py` — **CHỖ BUG GỐC** (dòng ~88–99):

  ```sql
  CREATE TABLE IF NOT EXISTS ohlcv (
      symbol VARCHAR NOT NULL, timeframe VARCHAR NOT NULL DEFAULT '1D',
      date DATE NOT NULL, ...,
      PRIMARY KEY (symbol, timeframe, date)   -- DATE không có giờ
  )
  ```
  kèm `INSERT OR IGNORE INTO ohlcv` (dòng 66) → mọi bar intraday cùng ngày bị nuốt còn 1 bar/ngày.
  Bảng `ohlcv` hiện tại trong `market.duckdb` (rebuild qua `back_adjust.py` CTAS) không còn PK nhưng
  dữ liệu đã collapse từ trước: 1m chỉ 3.594 row / 2 symbol / 8 năm (xác nhận lại read-only hôm nay).
- `stock_ml/train61_standalone/` (provider `sieutinhieu.py` + `ingest_sieutinhieu.py`) — nhánh Postgres
  có PK **ĐÚNG** `(symbol, timeframe, timestamp, provider)`; bug chỉ nằm ở nhánh duckdb.

## 2. Kết quả fetch thử (HPG + VN30F1M/F2M, 1m/5m/1H, mốc 2018→2026)

### 2a. sieutinhieu API (nguồn hiện tại) — KHÔNG đạt

| timeframe | VN30F1M | HPG | ghi chú |
|---|---|---|---|
| 1m / 5m / 15m | **total = 0** | total = 0 | API không còn phục vụ (dù db từng ingest 1m từ 2018 — data cũ đã bị purge phía server) |
| 30m | 351 bar (~7 tuần) | — | quá nông |
| 1H | 3.503 bar, **từ 2023-09-11** (~2,8 năm) | 3.167 bar, 2023-09-12 | timestamp có giờ (UTC), nhưng < 4 năm |

### 2b. Entrade/DNSE chart API (`services.entrade.com.vn/chart-api/v2/ohlcs/{stock|derivative}`) — ĐẠT cho phái sinh

Public, không key, không auth; `resolution=1` (phút), `from/to` epoch. ~40 request probe không hề bị rate-limit
(0,04–0,5 s/request); **1 request trả nguyên 1 năm 1m (60.488 bar) trong 0,4 s**.

**VN30F1M 1m — độ sâu thực tế theo năm** (`p3_f1m_yearly_coverage.csv`):

| năm | bar | ngày | bar/ngày (med) | ghi chú |
|---|---|---|---|---|
| 2018 | 23.804 | 98 | 243 | bắt đầu **2018-08-13** (khớp đúng min-date trong duckdb → đây chính là nguồn gốc ingest cũ) |
| 2019–2022 | ~60,5k/năm | 249–250 | 243 | sạch; 0–1 ngày lỗi/năm |
| **2023** | 30.615 | **126** | 243 | **LỖ HỔNG: giữa 2023-03 → 2023-09 mất ~6 tháng** (T4,6,7 trống; T5 1 ngày; T8 2 ngày) |
| 2024 | 60.002 | 247 | 243 | sạch (1 ngày lẻ thiếu: 2024-03-05) |
| 2025–2026H1 | 60k + 30,4k | 249+127 | 241 | sạch (241 bar/ngày từ 2025 — đổi vi cấu trúc phiên/KRX) |

Chất lượng bar (tuần mẫu 2018/2019/2020/2022/2024/2026, lưu parquet): **09:00→14:45 gồm ATC, 243 bar/ngày
đều tăm tắp, 0 null close, 0 duplicate timestamp, 0% zero-volume**. VN30F2M 1m: phủ mọi năm 2019–2025,
117–158 bar/ngày (kém thanh khoản — chỉ có bar khi khớp lệnh), cùng lỗ 2023.

**HPG (stock) 1m: chỉ từ ~2026-04-06** (binary search; 5m/15m/60m lịch sử cũng = 0). 2026-06: 226 bar/ngày
09:15–14:45, sạch. → Entrade KHÔNG có stock intraday lịch sử.

**VN30 spot index**: thử `index/VN30`, `VNINDEX` (1m + 1D) → n=0/HTTP 400. Blocker basis-level VẪN CÒN.

### 2c. Nguồn khác (thử nhanh, đều fail từ máy này)

- VCI trading API trực tiếp: HTTP 403 (WAF). vnstock 3.2.6 (đã cài) source VCI: `KeyError('data')`;
  source TCBS: ConnectionError. Không xác thực được độ sâu stock-intraday của VCI/TCBS trong probe này.
- Chưa thử (cần đăng ký key, ngoài scope nửa ngày): **SSI FastConnect Data** (key free theo TK SSI,
  endpoint intraday-ohlc — độ sâu lịch sử cần xác minh, tương truyền ngắn), **FireAnt** (token),
  DNSE LightSpeed API (key, nhưng backend chart chính là entrade ở trên), vendor trả phí (FiinPro/Vietstock).

## 3. VERDICT

1. **Phái sinh (VN30F1M/F2M) 1m: KHẢ THI, VƯỢT ngưỡng 4 năm** — 7,9 năm (2018-08-13 → nay) qua Entrade/DNSE,
   miễn phí, không key, không rate-limit thực tế; trừ lỗ ~6 tháng giữa 2023 → **~7,3 năm dùng được**;
   riêng vùng đau exit 2024+ phủ sạch 100%. P3 GIỮ ưu tiên với kênh phái sinh.
2. **Cổ phiếu (HPG…) 1m lịch sử: KHÔNG khả thi** với mọi nguồn free xác thực được hôm nay (Entrade chỉ từ
   2026-04; sieutinhieu 1H chỉ từ 2023-09). Muốn stock intraday ≥4 năm phải qua key/phí chưa xác minh
   (SSI FastConnect, FireAnt, vendor) — hạ ưu tiên theo đúng tiêu chí P3.
3. Bug schema là lỗi ingestion phía duckdb (`migrate_csv_to_duckdb.py`), KHÔNG phải lỗi nguồn; nhánh
   train61/Postgres đã đúng từ đầu.

## 4. THIẾT KẾ SỬA (chỉ thiết kế — chưa thực thi)

### 4a. Schema

KHÔNG sửa bảng `ohlcv` production (daily pipeline đang phụ thuộc, PK theo date là đúng cho 1D).
Tạo **DB riêng** `market_data/market_intraday.duckdb`, bảng riêng:

```sql
CREATE TABLE ohlcv_intraday (
    symbol    VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,          -- chỉ '1m'; 5m/15m/1H resample cục bộ, không fetch riêng
    ts        TIMESTAMP NOT NULL,        -- UTC (nguồn trả epoch); giờ VN = ts + 7h
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
    source    VARCHAR DEFAULT 'entrade',
    PRIMARY KEY (symbol, timeframe, ts)  -- cột time nằm TRONG PK — sửa đúng lỗi gốc
);
CREATE TABLE _fetch_manifest (symbol VARCHAR, year INTEGER, nbars INTEGER,
    ts_min TIMESTAMP, ts_max TIMESTAMP, status VARCHAR, PRIMARY KEY (symbol, year));
```

### 4b. Kế hoạch refetch — Phase A (phái sinh, đề xuất duyệt)

| hạng mục | ước lượng |
|---|---|
| Phạm vi | VN30F1M + VN30F2M × 1m × 2018-08-13 → nay |
| Số request | 1 request/năm/mã ≈ **18 request** (paging không cần — 1 năm về trọn gói) |
| Thời gian chạy | **< 2 phút** kể cả sleep 0,5 s/request + verify |
| Dung lượng | F1M ~447k bar + F2M ~280k bar ≈ **~730k row ≈ 20–30 MB** duckdb |
| Rate limit | không quan sát thấy; vẫn sleep 0,5 s + retry/backoff như `refetch_adjusted.py` |
| Verify sau fetch | bar/ngày == 243 (241 từ 2025) cho F1M; liệt kê ngày thiếu; đối chiếu close 1m cuối ngày vs close 1D trong `ohlcv` production (chỉ đọc) |
| Lỗ 2023 | chấp nhận (ngoài vùng đau 2024+); nếu cần vá: SSI FastConnect/FireAnt sau khi có key |
| Công | 1 script ~100 dòng theo khuôn `refetch_adjusted.py`, < 1 giờ |

### 4c. Phase B (tùy chọn, KHÔNG thuộc P3)

- **Stocks forward-accumulation**: cron ngày kéo 1m Entrade cho universe 488 mã từ nay (chỉ có từ 2026-04)
  — ~110k row/ngày, ~27M row/năm ≈ 300–500 MB/năm; chỉ có giá trị cho tương lai (4 năm nữa mới đủ lịch sử).
- **VN30 spot**: chưa có nguồn — blocker basis-level giữ nguyên như DERIV_EXIT_SCREEN.
