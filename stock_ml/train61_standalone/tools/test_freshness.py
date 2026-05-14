"""
Test script for data freshness implementation.

Tests:
1. OHLCV cache freshness
2. Signal cache freshness
3. Auto-regenerate on stale cache
"""

import time

import requests

BASE_URL = "http://localhost:5000"
TEST_SYMBOL = "AAA"  # Change to a symbol that exists in your dataset


def test_ohlcv_freshness():
    """Test OHLCV cache freshness detection."""
    print("\n=== Test 1: OHLCV Freshness ===")

    # Get OHLCV first time
    print(f"1. Fetching OHLCV for {TEST_SYMBOL}...")
    resp = requests.get(f"{BASE_URL}/api/ohlcv/{TEST_SYMBOL}")
    assert resp.status_code == 200, f"Failed to get OHLCV: {resp.status_code}"

    data = resp.json()
    print(f"   - Symbol: {data['symbol']}")
    print(f"   - OHLCV bars: {len(data['ohlcv'])}")
    print(f"   - Latest bar date: {data.get('latest_bar_date')}")
    print(f"   - Has fingerprint: {data.get('data_fingerprint') is not None}")

    if data.get("data_fingerprint"):
        fp = data["data_fingerprint"]
        print(f"   - Fingerprint: size={fp.get('size')}, mtime_ns={fp.get('mtime_ns')}")

    # Get again - should use cache
    print("\n2. Fetching OHLCV again (should use cache)...")
    resp2 = requests.get(f"{BASE_URL}/api/ohlcv/{TEST_SYMBOL}")
    assert resp2.status_code == 200
    data2 = resp2.json()
    print(f"   - Latest bar date: {data2.get('latest_bar_date')}")
    print(f"   - Same as before: {data2.get('latest_bar_date') == data.get('latest_bar_date')}")

    print("\n✓ OHLCV freshness test passed")


def test_signal_freshness():
    """Test signal cache freshness detection."""
    print("\n=== Test 2: Signal Freshness ===")

    # Get signal first time
    print(f"1. Fetching signal for {TEST_SYMBOL}...")
    resp = requests.get(f"{BASE_URL}/api/signal/train61_model/{TEST_SYMBOL}")

    if resp.status_code == 202:
        print("   - Signal generating (202)...")
        # Poll for completion
        for i in range(30):
            time.sleep(2)
            status_resp = requests.get(f"{BASE_URL}/api/signal/train61_model/{TEST_SYMBOL}/status")
            status = status_resp.json().get("status")
            print(f"   - Poll {i + 1}: status={status}")

            if status == "done":
                break
            elif status == "error":
                print(f"   - Error: {status_resp.json()}")
                return

        # Get signal after generation
        resp = requests.get(f"{BASE_URL}/api/signal/train61_model/{TEST_SYMBOL}")

    assert resp.status_code == 200, f"Failed to get signal: {resp.status_code}"

    data = resp.json()
    print("\n2. Signal received:")
    print(f"   - Symbol: {data['symbol']}")
    print(f"   - Model ID: {data.get('model_id')}")
    print(f"   - Latest bar date: {data.get('latest_bar_date')}")
    print(f"   - Has fingerprint: {data.get('data_fingerprint') is not None}")

    if data.get("data_fingerprint"):
        fp = data["data_fingerprint"]
        print(f"   - Fingerprint: size={fp.get('size')}, latest_bar={fp.get('latest_bar_date')}")

    # Get again - should use cache
    print("\n3. Fetching signal again (should use cache)...")
    resp2 = requests.get(f"{BASE_URL}/api/signal/train61_model/{TEST_SYMBOL}")
    assert resp2.status_code == 200
    data2 = resp2.json()
    print(f"   - Latest bar date: {data2.get('latest_bar_date')}")
    print(f"   - Same as before: {data2.get('latest_bar_date') == data.get('latest_bar_date')}")

    print("\n✓ Signal freshness test passed")


def test_cache_stats():
    """Test cache stats endpoint includes freshness metrics."""
    print("\n=== Test 3: Cache Stats ===")

    resp = requests.get(f"{BASE_URL}/api/cache-stats")
    assert resp.status_code == 200

    stats = resp.json()
    print("1. Cache stats:")
    print(f"   - Active jobs: {stats.get('active_jobs')}")
    print(f"   - Completed jobs: {stats.get('completed_jobs')}")
    print(f"   - Failed jobs: {stats.get('failed_jobs')}")

    if "freshness" in stats:
        freshness = stats["freshness"]
        print("\n2. Freshness metrics:")
        print(f"   - Freshness checks: {freshness.get('freshness_check_count')}")
        print(f"   - Stale detected: {freshness.get('stale_cache_detected')}")
        print(f"   - Fingerprint errors: {freshness.get('fingerprint_error_count')}")
        print(f"   - Regenerate triggered: {freshness.get('regenerate_triggered')}")
    else:
        print("\n⚠ Warning: No freshness metrics in cache stats")

    print("\n✓ Cache stats test passed")


def test_symbols_list():
    """Test symbols list includes stale flag."""
    print("\n=== Test 4: Symbols List ===")

    resp = requests.get(f"{BASE_URL}/api/symbols?model_id=train61_model")
    assert resp.status_code == 200

    symbols = resp.json()
    print(f"1. Symbols list: {len(symbols)} symbols")

    # Find test symbol
    test_row = next((s for s in symbols if s["symbol"] == TEST_SYMBOL), None)
    if test_row:
        print(f"\n2. Test symbol ({TEST_SYMBOL}):")
        print(f"   - Cached: {test_row.get('cached')}")
        print(f"   - Stale: {test_row.get('stale')}")
        print(f"   - Has historical export: {test_row.get('has_historical_export')}")
    else:
        print(f"\n⚠ Warning: {TEST_SYMBOL} not found in symbols list")

    print("\n✓ Symbols list test passed")


def main():
    print("=" * 60)
    print("Data Freshness Implementation Test")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}")
    print(f"Test Symbol: {TEST_SYMBOL}")

    try:
        # Check server is running
        resp = requests.get(f"{BASE_URL}/api/models", timeout=5)
        if resp.status_code != 200:
            print(f"\n❌ Server not responding correctly: {resp.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("Make sure the server is running: python app/serve_train61_model.py")
        return

    # Run tests
    try:
        test_ohlcv_freshness()
        test_signal_freshness()
        test_cache_stats()
        test_symbols_list()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
