"""
Manual test script for data change detection.

This script helps you manually test the data freshness flow:
1. Generate signal for a symbol
2. Modify the data source (add a new bar)
3. Request signal again - should detect stale and regenerate

Usage:
1. Start the server: python app/serve_train61_model.py
2. Run this script: python tools/test_data_change.py
3. Follow the prompts
"""

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "http://localhost:5000"
DATA_DIR = Path(__file__).parent.parent / "data" / "vn_stock_ai_dataset_cleaned"


def list_available_symbols():
    """List symbols that have data files."""
    symbols = []
    if DATA_DIR.exists():
        for symbol_dir in DATA_DIR.glob("symbol=*"):
            symbol = symbol_dir.name.replace("symbol=", "")
            data_file = symbol_dir / "timeframe=1D" / "data.csv"
            if data_file.exists():
                symbols.append(symbol)
    return sorted(symbols)


def get_signal(symbol: str, model_id: str = "train61_model"):
    """Get signal for a symbol, polling if needed."""
    print(f"\nFetching signal for {symbol}...")
    resp = requests.get(f"{BASE_URL}/api/signal/{model_id}/{symbol}")

    if resp.status_code == 202:
        print("Signal generating (202)...")
        # Poll for completion
        for i in range(60):
            time.sleep(2)
            status_resp = requests.get(f"{BASE_URL}/api/signal/{model_id}/{symbol}/status")
            status_data = status_resp.json()
            status = status_data.get("status")
            print(f"  Poll {i + 1}: status={status}")

            if status == "done":
                # Get the actual signal
                resp = requests.get(f"{BASE_URL}/api/signal/{model_id}/{symbol}")
                break
            elif status == "error":
                print(f"  Error: {status_data.get('error')}")
                return None

    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Failed to get signal: {resp.status_code}")
        return None


def show_signal_info(data: dict):
    """Display signal information."""
    if not data:
        return

    print("\nSignal info:")
    print(f"  Symbol: {data.get('symbol')}")
    print(f"  Model ID: {data.get('model_id')}")
    print(f"  Latest bar date: {data.get('latest_bar_date')}")
    print(f"  Source: {data.get('source')}")

    fp = data.get("data_fingerprint")
    if fp:
        print("  Fingerprint:")
        print(f"    - Size: {fp.get('size')}")
        print(f"    - Mtime: {fp.get('mtime_ns')}")
        print(f"    - Latest bar: {fp.get('latest_bar_date')}")
        print(f"    - Latest close: {fp.get('latest_close')}")


def backup_data_file(symbol: str) -> Path | None:
    """Backup the data file for a symbol."""
    data_file = DATA_DIR / f"symbol={symbol}" / "timeframe=1D" / "data.csv"
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return None

    backup_file = data_file.with_suffix(".csv.backup")
    import shutil

    shutil.copy2(data_file, backup_file)
    print(f"Backed up to: {backup_file}")
    return backup_file


def restore_data_file(symbol: str):
    """Restore the data file from backup."""
    data_file = DATA_DIR / f"symbol={symbol}" / "timeframe=1D" / "data.csv"
    backup_file = data_file.with_suffix(".csv.backup")

    if backup_file.exists():
        import shutil

        shutil.copy2(backup_file, data_file)
        print(f"Restored from backup: {backup_file}")
        backup_file.unlink()
    else:
        print("No backup file found")


def add_fake_bar(symbol: str):
    """Add a fake bar to the data file."""
    data_file = DATA_DIR / f"symbol={symbol}" / "timeframe=1D" / "data.csv"
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return False

    # Read existing data
    df = pd.read_csv(data_file)
    print(f"\nOriginal data: {len(df)} rows")
    print(f"Last row:\n{df.tail(1)}")

    # Get last row
    last_row = df.iloc[-1].copy()

    # Create new row with next date
    last_date = pd.Timestamp(last_row["date"] if "date" in df.columns else last_row["timestamp"])
    new_date = last_date + pd.Timedelta(days=1)

    new_row = last_row.copy()
    date_col = "date" if "date" in df.columns else "timestamp"
    new_row[date_col] = new_date.strftime("%Y-%m-%d")

    # Slightly modify prices
    new_row["close"] = float(last_row["close"]) * 1.01
    new_row["high"] = float(last_row["high"]) * 1.01
    new_row["low"] = float(last_row["low"]) * 0.99
    new_row["open"] = float(last_row["open"])

    # Append new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save
    df.to_csv(data_file, index=False)
    print(f"\nAdded fake bar: {new_date.strftime('%Y-%m-%d')}")
    print(f"New data: {len(df)} rows")

    return True


def main():
    print("=" * 70)
    print("Manual Data Change Test")
    print("=" * 70)

    # Check server
    try:
        resp = requests.get(f"{BASE_URL}/api/models", timeout=5)
        if resp.status_code != 200:
            print(f"\n❌ Server not responding: {resp.status_code}")
            return
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("Start server: python app/serve_train61_model.py")
        return

    # List available symbols
    symbols = list_available_symbols()
    if not symbols:
        print("\n❌ No symbols found in data directory")
        return

    print(f"\nAvailable symbols: {', '.join(symbols[:10])}...")
    symbol = input(f"\nEnter symbol to test (default: {symbols[0]}): ").strip().upper()
    if not symbol:
        symbol = symbols[0]

    if symbol not in symbols:
        print(f"❌ Symbol {symbol} not found")
        return

    print(f"\n{'=' * 70}")
    print(f"Testing with symbol: {symbol}")
    print(f"{'=' * 70}")

    # Step 1: Get initial signal
    print("\n[Step 1] Get initial signal")
    signal1 = get_signal(symbol)
    if not signal1:
        print("❌ Failed to get initial signal")
        return
    show_signal_info(signal1)

    # Step 2: Backup and modify data
    print("\n[Step 2] Modify data source")
    backup_file = backup_data_file(symbol)
    if not backup_file:
        return

    if not add_fake_bar(symbol):
        restore_data_file(symbol)
        return

    print("\n⚠ Data file modified. The cache should now be stale.")

    # Step 3: Get signal again - should detect stale
    input("\nPress Enter to fetch signal again (should detect stale and regenerate)...")

    print("\n[Step 3] Get signal after data change")
    signal2 = get_signal(symbol)
    if not signal2:
        print("❌ Failed to get signal after data change")
        restore_data_file(symbol)
        return

    show_signal_info(signal2)

    # Compare
    print("\n[Step 4] Compare signals")
    print(
        f"  Latest bar date changed: {signal1.get('latest_bar_date')} → {signal2.get('latest_bar_date')}"
    )

    fp1 = signal1.get("data_fingerprint", {})
    fp2 = signal2.get("data_fingerprint", {})
    print(f"  Size changed: {fp1.get('size')} → {fp2.get('size')}")
    print(f"  Mtime changed: {fp1.get('mtime_ns')} → {fp2.get('mtime_ns')}")

    if signal2.get("latest_bar_date") != signal1.get("latest_bar_date"):
        print("\n✓ Success! Signal was regenerated with new data")
    else:
        print("\n⚠ Warning: Latest bar date did not change")

    # Step 5: Check cache stats
    print("\n[Step 5] Check cache stats")
    resp = requests.get(f"{BASE_URL}/api/cache-stats")
    if resp.status_code == 200:
        stats = resp.json()
        freshness = stats.get("freshness", {})
        print(f"  Freshness checks: {freshness.get('freshness_check_count')}")
        print(f"  Stale detected: {freshness.get('stale_cache_detected')}")
        print(f"  Regenerate triggered: {freshness.get('regenerate_triggered')}")

    # Restore
    print("\n[Step 6] Restore original data")
    restore_data_file(symbol)
    print("✓ Data restored")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
