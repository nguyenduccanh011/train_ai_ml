#!/usr/bin/env python3
"""
Performance testing script for Train61 Standalone Server.

Tests:
1. Cache hit rates
2. Thread pool limits
3. Memory usage
4. Response times
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "http://127.0.0.1:5012"


def test_cache_stats():
    """Test cache statistics endpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: Cache Statistics")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/api/cache-stats")
    if resp.status_code != 200:
        print(f"❌ Failed: {resp.status_code}")
        return

    stats = resp.json()
    print("\n📊 Cache Stats:")
    for cache in stats["caches"]:
        print(f"\n  {cache['name']}:")
        print(f"    Size: {cache['size']}/{cache['max_size']}")
        print(f"    Hit rate: {cache['hit_rate_pct']}%")
        print(f"    Hits: {cache['hits']}, Misses: {cache['misses']}")
        print(f"    Evictions: {cache['evictions']}")

    print("\n  Jobs:")
    print(f"    Active: {stats['active_jobs']}")
    print(f"    Completed: {stats['completed_jobs']}")
    print(f"    Failed: {stats['failed_jobs']}")
    print(f"    Queue size: {stats['executor_queue_size']}")

    print("\n✅ Cache stats endpoint working")


def test_model_preloading():
    """Test that models are pre-loaded (fast first request)."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Pre-loading")
    print("=" * 60)

    # Get model info (should be instant if pre-loaded)
    start = time.time()
    resp = requests.get(f"{BASE_URL}/api/model-info?model_id=train61_pooled")
    elapsed = time.time() - start

    if resp.status_code == 200:
        info = resp.json()
        print(f"\n✅ Model info retrieved in {elapsed * 1000:.0f}ms")
        print(f"   Model: {info.get('model_id')}")
        print(f"   Symbols: {info.get('train_symbol_count')}")
        print(f"   Mode: {info.get('mode')}")

        if elapsed < 0.5:
            print("   ✅ Fast response = model was pre-loaded")
        else:
            print("   ⚠️  Slow response = model may not be pre-loaded")
    else:
        print(f"❌ Failed: {resp.status_code}")


def test_symbol_list_caching():
    """Test symbol list caching per model."""
    print("\n" + "=" * 60)
    print("TEST 3: Symbol List Caching")
    print("=" * 60)

    model_id = "train61_pooled"

    # First request (cache miss)
    start = time.time()
    resp1 = requests.get(f"{BASE_URL}/api/symbols?model_id={model_id}")
    time1 = time.time() - start

    if resp1.status_code != 200:
        print(f"❌ Failed: {resp1.status_code}")
        return

    # Second request (cache hit)
    start = time.time()
    requests.get(f"{BASE_URL}/api/symbols?model_id={model_id}")
    time2 = time.time() - start

    print(f"\n  First request:  {time1 * 1000:.0f}ms (cache miss)")
    print(f"  Second request: {time2 * 1000:.0f}ms (cache hit)")

    speedup = time1 / time2 if time2 > 0 else 0
    print(f"  Speedup: {speedup:.1f}x")

    if speedup > 5:
        print("  ✅ Symbol list caching working well")
    elif speedup > 2:
        print("  ⚠️  Some caching benefit, but could be better")
    else:
        print("  ❌ Caching may not be working")


def test_thread_pool_limit():
    """Test that thread pool limits concurrent requests."""
    print("\n" + "=" * 60)
    print("TEST 4: Thread Pool Limit")
    print("=" * 60)

    symbols = ["VNM", "VIC", "VHM", "HPG", "TCB", "VCB", "BID", "CTG"]
    model_id = "train61_pooled"

    print(f"\n  Sending {len(symbols)} concurrent requests...")

    def fetch_signal(symbol):
        start = time.time()
        resp = requests.get(f"{BASE_URL}/api/signal/{model_id}/{symbol}")
        elapsed = time.time() - start
        return symbol, resp.status_code, elapsed

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_signal, sym) for sym in symbols]
        results = [f.result() for f in as_completed(futures)]

    # Check cache stats to see if jobs were queued
    resp = requests.get(f"{BASE_URL}/api/cache-stats")
    stats = resp.json()

    print("\n  Results:")
    for symbol, status, elapsed in sorted(results):
        status_icon = "✅" if status in [200, 202] else "❌"
        print(f"    {status_icon} {symbol}: {status} ({elapsed * 1000:.0f}ms)")

    print(f"\n  Active jobs: {stats['active_jobs']}")
    print(f"  Queue size: {stats['executor_queue_size']}")

    if stats["active_jobs"] <= 4:
        print("  ✅ Thread pool limit working (max 4 concurrent)")
    else:
        print(f"  ⚠️  More than 4 active jobs: {stats['active_jobs']}")


def test_lru_eviction():
    """Test LRU cache eviction policy."""
    print("\n" + "=" * 60)
    print("TEST 5: LRU Cache Eviction")
    print("=" * 60)

    # Clear cache first
    requests.post(f"{BASE_URL}/api/cache-clear")
    print("  Cache cleared")

    # Get initial stats
    resp = requests.get(f"{BASE_URL}/api/cache-stats")
    resp.json()

    # Load multiple models to trigger eviction (artifact cache max_size=4)
    models = ["train61_pooled", "top1_on_demand", "top1_fold_chain_no_context"]

    for model_id in models:
        print(f"\n  Loading model: {model_id}")
        resp = requests.get(f"{BASE_URL}/api/model-info?model_id={model_id}")
        if resp.status_code == 200:
            print("    ✅ Loaded")

    # Check final stats
    resp = requests.get(f"{BASE_URL}/api/cache-stats")
    final_stats = resp.json()

    artifact_cache = next(c for c in final_stats["caches"] if c["name"] == "artifact")

    print("\n  Artifact cache:")
    print(f"    Size: {artifact_cache['size']}/{artifact_cache['max_size']}")
    print(f"    Evictions: {artifact_cache['evictions']}")

    if artifact_cache["size"] <= artifact_cache["max_size"]:
        print("    ✅ Cache size within limit")
    else:
        print("    ❌ Cache size exceeded limit")


def main():
    print("\n" + "=" * 60)
    print("TRAIN61 PERFORMANCE TEST SUITE")
    print("=" * 60)
    print(f"Target: {BASE_URL}")

    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=2)
        print("✅ Server is running")
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Start it first:")
        print("   python app/serve_train61_model.py")
        return

    # Run tests
    test_cache_stats()
    test_model_preloading()
    test_symbol_list_caching()
    test_thread_pool_limit()
    test_lru_eviction()

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
