"""
Tiền xử lý dữ liệu chứng khoán Việt Nam để train AI.
Loại bỏ mã cổ phiếu không đạt tiêu chuẩn:
1. Thanh khoản quá thấp (avg traded value < threshold)
2. Dữ liệu quá ít (row_count < threshold)
3. Coverage ratio quá thấp (nhiều ngày thiếu dữ liệu)
4. Dữ liệu bất thường (giá âm, volume âm, giá = 0, spike bất thường)
5. Loại bỏ index và futures (không phải cổ phiếu thực)
"""

import os
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime

DATA_ROOT = "portable_data/vn_stock_ai_dataset"
ALL_SYMBOLS_DIR = os.path.join(DATA_ROOT, "all_symbols")
METADATA_DIR = os.path.join(ALL_SYMBOLS_DIR, "metadata")
OUTPUT_DIR = "portable_data/vn_stock_ai_dataset_cleaned"

# === THRESHOLDS ===
MIN_TRADING_DAYS = 250            # Ít nhất ~1 năm giao dịch
MIN_AVG_TRADED_VALUE = 500_000    # 500K VND/ngày trung bình (rất thấp = không thanh khoản)
MIN_MEDIAN_TRADED_VALUE = 100_000 # Median traded value tối thiểu
MIN_COVERAGE_RATIO = 0.7          # Ít nhất 70% ngày có dữ liệu
MAX_DAILY_RETURN = 5.0            # 500% trong 1 ngày = bất thường (ceiling/floor VN ~7% nhưng có điều chỉnh)
MAX_ZERO_VOLUME_RATIO = 0.3      # Tối đa 30% ngày volume = 0

def load_symbols_metadata():
    """Load symbols.csv"""
    symbols = {}
    with open(os.path.join(METADATA_DIR, "symbols.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbols[row["symbol"]] = {
                "exchange": row["exchange"],
                "row_count": int(row["row_count"]),
                "avg_daily_traded_value": float(row["avg_daily_traded_value"]),
                "median_daily_traded_value": float(row["median_daily_traded_value"]),
                "latest_close": float(row["latest_close"]) if row["latest_close"] else 0,
                "first_timestamp": row["first_timestamp"],
                "last_timestamp": row["last_timestamp"],
            }
    return symbols

def load_coverage_metadata():
    """Load coverage.csv và tính lại coverage_ratio chính xác.
    
    BUG FIX: File coverage.csv gốc tính expected_sessions sai cho sàn HOSE 
    (inflate ~1.56x so với HNX/UPCOM cùng khoảng thời gian).
    Điều này khiến tất cả blue-chip HOSE (ACB, FPT, VNM, VCB, HPG...) bị loại nhầm.
    
    Giải pháp: Tính lại expected_sessions dựa trên ~252 phiên/năm (trading calendar thực tế).
    """
    coverage = {}
    with open(os.path.join(METADATA_DIR, "coverage.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["symbol"]
            observed = int(row["observed_sessions"])
            original_expected = int(row["expected_sessions"])
            original_ratio = float(row["coverage_ratio"])
            
            # Tính lại expected_sessions từ date range
            first_ts = row["first_timestamp"]
            last_ts = row["last_timestamp"]
            try:
                # Parse timestamp (format: 2015-01-05 00:00:00+00:00)
                first_date = datetime.strptime(first_ts[:10], "%Y-%m-%d")
                last_date = datetime.strptime(last_ts[:10], "%Y-%m-%d")
                calendar_days = (last_date - first_date).days + 1
                # Ước tính số phiên giao dịch: ~252 phiên/365 ngày
                recalc_expected = int(calendar_days * 252 / 365)
                recalc_ratio = observed / recalc_expected if recalc_expected > 0 else 0
            except (ValueError, ZeroDivisionError):
                recalc_expected = original_expected
                recalc_ratio = original_ratio
            
            coverage[symbol] = {
                "coverage_ratio": recalc_ratio,
                "coverage_ratio_original": original_ratio,
                "observed_sessions": observed,
                "expected_sessions_original": original_expected,
                "expected_sessions_recalc": recalc_expected,
            }
    return coverage

def check_data_anomalies(symbol):
    """Check individual stock data for anomalies"""
    data_path = os.path.join(ALL_SYMBOLS_DIR, f"symbol={symbol}", "timeframe=1D", "data.csv")
    if not os.path.exists(data_path):
        return {"exists": False}
    
    issues = []
    total_rows = 0
    zero_volume_days = 0
    zero_close_days = 0
    negative_values = 0
    extreme_returns = 0
    prev_close = None
    
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            try:
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
                v = float(row["volume"])
                
                # Check negative values
                if o < 0 or h < 0 or l < 0 or c < 0 or v < 0:
                    negative_values += 1
                
                # Check zero close
                if c == 0:
                    zero_close_days += 1
                
                # Check zero volume
                if v == 0:
                    zero_volume_days += 1
                
                # Check OHLC consistency (high >= low, high >= open/close, low <= open/close)
                if h < l:
                    issues.append(f"high < low on row {total_rows}")
                
                # Check extreme returns
                if prev_close and prev_close > 0 and c > 0:
                    ret = abs(c / prev_close - 1)
                    if ret > MAX_DAILY_RETURN:
                        extreme_returns += 1
                
                prev_close = c
                
            except (ValueError, KeyError):
                issues.append(f"Parse error on row {total_rows}")
    
    zero_vol_ratio = zero_volume_days / total_rows if total_rows > 0 else 1
    
    return {
        "exists": True,
        "total_rows": total_rows,
        "zero_volume_days": zero_volume_days,
        "zero_volume_ratio": zero_vol_ratio,
        "zero_close_days": zero_close_days,
        "negative_values": negative_values,
        "extreme_returns": extreme_returns,
        "ohlc_issues": len([i for i in issues if "high < low" in i]),
    }

def main():
    print("=" * 70)
    print("TIỀN XỬ LÝ DỮ LIỆU CHỨNG KHOÁN VIỆT NAM CHO AI TRAINING")
    print("=" * 70)
    
    # Load metadata
    print("\n[1] Loading metadata...")
    symbols = load_symbols_metadata()
    coverage = load_coverage_metadata()
    print(f"  Tổng số mã: {len(symbols)}")
    
    # Track removal reasons
    removed = defaultdict(list)  # reason -> [symbols]
    kept = []
    
    all_symbols = sorted(symbols.keys())
    
    # === FILTER 1: Remove non-stock symbols (index, futures) ===
    print("\n[2] Lọc bỏ index và futures...")
    non_stock = []
    KNOWN_NON_STOCK = {"HNXINDEX", "HNXUPCOM", "VN30F1M", "VN30F2M", "VNINDEX", "VN30"}
    for sym in all_symbols:
        data_path = os.path.join(ALL_SYMBOLS_DIR, f"symbol={sym}", "timeframe=1D", "data.csv")
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row and row.get("asset_type") in ("index", "derivative", "futures"):
                    non_stock.append(sym)
                elif sym in KNOWN_NON_STOCK:
                    non_stock.append(sym)
    
    # Deduplicate
    non_stock = sorted(set(non_stock))
    
    for sym in non_stock:
        removed["non_stock (index/futures)"].append(sym)
    
    candidates = [s for s in all_symbols if s not in non_stock]
    print(f"  Loại bỏ {len(non_stock)} mã non-stock khỏi danh sách cổ phiếu: {non_stock}")
    print(f"  → Sẽ giữ lại làm CONTEXT FEATURES (chỉ số thị trường)")
    
    # === FILTER 2: Minimum trading days ===
    print(f"\n[3] Lọc mã có ít hơn {MIN_TRADING_DAYS} ngày giao dịch...")
    too_few = [s for s in candidates if symbols[s]["row_count"] < MIN_TRADING_DAYS]
    for sym in too_few:
        removed[f"too_few_days (<{MIN_TRADING_DAYS})"].append(sym)
    candidates = [s for s in candidates if s not in too_few]
    print(f"  Loại bỏ {len(too_few)} mã")
    
    # === FILTER 3: Liquidity ===
    print(f"\n[4] Lọc mã thanh khoản quá thấp...")
    low_liq = [s for s in candidates 
                if symbols[s]["avg_daily_traded_value"] < MIN_AVG_TRADED_VALUE
                or symbols[s]["median_daily_traded_value"] < MIN_MEDIAN_TRADED_VALUE]
    for sym in low_liq:
        removed["low_liquidity"].append(sym)
    candidates = [s for s in candidates if s not in low_liq]
    print(f"  Loại bỏ {len(low_liq)} mã")
    
    # === FILTER 4: Coverage ratio ===
    print(f"\n[5] Lọc mã có coverage ratio < {MIN_COVERAGE_RATIO}...")
    low_cov = [s for s in candidates 
                if s in coverage and coverage[s]["coverage_ratio"] < MIN_COVERAGE_RATIO]
    for sym in low_cov:
        removed[f"low_coverage (<{MIN_COVERAGE_RATIO})"].append(sym)
    candidates = [s for s in candidates if s not in low_cov]
    print(f"  Loại bỏ {len(low_cov)} mã")
    
    # === FILTER 5: Data anomalies (scan actual CSV) ===
    print(f"\n[6] Kiểm tra dữ liệu bất thường (quét file CSV)...")
    anomaly_removed = []
    total = len(candidates)
    for i, sym in enumerate(candidates):
        if (i + 1) % 100 == 0:
            print(f"  Đang kiểm tra {i+1}/{total}...")
        
        result = check_data_anomalies(sym)
        
        if not result["exists"]:
            removed["no_data_file"].append(sym)
            anomaly_removed.append(sym)
            continue
        
        reasons = []
        if result["zero_volume_ratio"] > MAX_ZERO_VOLUME_RATIO:
            reasons.append(f"zero_vol={result['zero_volume_ratio']:.1%}")
        if result["negative_values"] > 0:
            reasons.append(f"negative_vals={result['negative_values']}")
        if result["zero_close_days"] > 5:
            reasons.append(f"zero_close={result['zero_close_days']}")
        
        if reasons:
            removed[f"data_anomaly"].append(f"{sym} ({', '.join(reasons)})")
            anomaly_removed.append(sym)
    
    candidates = [s for s in candidates if s not in anomaly_removed]
    print(f"  Loại bỏ {len(anomaly_removed)} mã có dữ liệu bất thường")
    
    # === RESULTS ===
    print("\n" + "=" * 70)
    print("KẾT QUẢ TIỀN XỬ LÝ")
    print("=" * 70)
    print(f"\n  Tổng mã ban đầu:    {len(all_symbols)}")
    print(f"  Tổng mã bị loại:    {len(all_symbols) - len(candidates)}")
    print(f"  Tổng mã còn lại:    {len(candidates)}")
    
    print("\n  Chi tiết lý do loại bỏ:")
    for reason, syms in sorted(removed.items()):
        print(f"    - {reason}: {len(syms)} mã")
    
    # === COPY CLEAN DATA ===
    print(f"\n[7] Copy dữ liệu sạch sang {OUTPUT_DIR}...")
    os.makedirs(os.path.join(OUTPUT_DIR, "all_symbols", "metadata"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "context_features"), exist_ok=True)
    
    # Copy context features (index/futures - dùng làm market context cho AI)
    print(f"  Copy {len(non_stock)} context features (index/futures)...")
    for sym in non_stock:
        src = os.path.join(ALL_SYMBOLS_DIR, f"symbol={sym}")
        dst = os.path.join(OUTPUT_DIR, "context_features", f"symbol={sym}")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    
    # Copy clean stock symbols
    print(f"  Copy {len(candidates)} mã cổ phiếu sạch...")
    for sym in candidates:
        src = os.path.join(ALL_SYMBOLS_DIR, f"symbol={sym}")
        dst = os.path.join(OUTPUT_DIR, "all_symbols", f"symbol={sym}")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    
    # Save filter report
    report = {
        "generated_at": datetime.now().isoformat(),
        "original_count": len(all_symbols),
        "removed_count": len(all_symbols) - len(candidates),
        "remaining_count": len(candidates),
        "thresholds": {
            "min_trading_days": MIN_TRADING_DAYS,
            "min_avg_traded_value": MIN_AVG_TRADED_VALUE,
            "min_median_traded_value": MIN_MEDIAN_TRADED_VALUE,
            "min_coverage_ratio": MIN_COVERAGE_RATIO,
            "max_zero_volume_ratio": MAX_ZERO_VOLUME_RATIO,
        },
        "removed_by_reason": {k: v for k, v in sorted(removed.items())},
        "kept_symbols": sorted(candidates),
    }
    
    report_path = os.path.join(OUTPUT_DIR, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save clean symbols list
    symbols_list_path = os.path.join(OUTPUT_DIR, "clean_symbols.txt")
    with open(symbols_list_path, "w", encoding="utf-8") as f:
        for sym in sorted(candidates):
            f.write(f"{sym}\n")
    
    print(f"\n  ✅ Report saved: {report_path}")
    print(f"  ✅ Clean symbols list: {symbols_list_path}")
    print(f"  ✅ Clean data copied to: {OUTPUT_DIR}")
    print(f"\n  Danh sách {len(candidates)} mã đạt chuẩn đã sẵn sàng cho AI training!")

if __name__ == "__main__":
    main()
