"""Check cache stats and test a non-pooled model."""

import requests

BASE_URL = "http://127.0.0.1:5012"

# Check cache stats
print("=" * 60)
print("Cache Stats")
print("=" * 60)
resp = requests.get(f"{BASE_URL}/api/cache-stats")
if resp.status_code == 200:
    stats = resp.json()
    print(f"Active jobs: {stats.get('active_jobs')}")
    print(f"Completed jobs: {stats.get('completed_jobs')}")
    print("\nFreshness metrics:")
    freshness = stats.get("freshness", {})
    for key, val in freshness.items():
        print(f"  {key}: {val}")

    print(f"\nPooled cache: {stats['caches'][3]}")

# Test with a non-pooled model
print("\n" + "=" * 60)
print("Testing non-pooled model: top1_fold_chain_no_context")
print("=" * 60)

resp = requests.get(f"{BASE_URL}/api/signal/top1_fold_chain_no_context/AAV")
if resp.status_code == 202:
    print("Generating signal...")
    import time

    time.sleep(5)
    resp = requests.get(f"{BASE_URL}/api/signal/top1_fold_chain_no_context/AAV")

if resp.status_code == 200:
    data = resp.json()
    stats = data.get("top1_fold_chain_no_context_stats", {})

    print(f"\nSymbol: {data.get('symbol')}")
    print(f"Total trades: {stats.get('total_trades')}")
    print(f"Win rate: {stats.get('win_rate')}%")
    print(f"Avg PnL: {stats.get('avg_pnl_pct')}%")
    print()
    print(f"Total PnL (compound): {stats.get('total_pnl_pct')}%")
    print(f"Total PnL (simple): {stats.get('total_pnl_simple')}%")
    print()
    print(f"Latest bar: {data.get('latest_bar_date')}")
    print(f"Has fingerprint: {data.get('data_fingerprint') is not None}")

    if data.get("data_fingerprint"):
        fp = data["data_fingerprint"]
        print(f"Fingerprint size: {fp.get('size')}")
        print(f"Fingerprint latest_bar: {fp.get('latest_bar_date')}")
else:
    print(f"Error: {resp.status_code}")
