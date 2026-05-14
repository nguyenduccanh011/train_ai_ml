"""Quick test to check compound return fix."""

import requests

BASE_URL = "http://127.0.0.1:5012"

# Get a symbol with many trades
print("Fetching AAV signal...")
resp = requests.get(f"{BASE_URL}/api/signal/train61_pooled/AAV")

if resp.status_code == 202:
    print("Signal generating... waiting...")
    import time

    time.sleep(5)
    resp = requests.get(f"{BASE_URL}/api/signal/train61_pooled/AAV")

if resp.status_code == 200:
    data = resp.json()
    stats = data.get("train61_pooled_stats", {})

    print("\n" + "=" * 60)
    print(f"Symbol: {data.get('symbol')}")
    print("=" * 60)
    print(f"Total trades:        {stats.get('total_trades')}")
    print(f"Win rate:            {stats.get('win_rate')}%")
    print(f"Avg PnL per trade:   {stats.get('avg_pnl_pct')}%")
    print()
    print(f"Total PnL (compound): {stats.get('total_pnl_pct')}%  <- CORRECT")
    print(f"Total PnL (simple):   {stats.get('total_pnl_simple')}%  <- OLD (for reference)")
    print()

    if stats.get("total_pnl_simple") and stats.get("total_pnl_pct"):
        ratio = stats["total_pnl_simple"] / stats["total_pnl_pct"]
        print(f"Ratio (simple/compound): {ratio:.2f}x")
        print(f"Difference: {stats['total_pnl_simple'] - stats['total_pnl_pct']:.1f}%")

    print()
    print(f"Latest bar date:     {data.get('latest_bar_date')}")
    print(f"Has fingerprint:     {data.get('data_fingerprint') is not None}")
    print(f"Stale:               {data.get('stale', False)}")
    print("=" * 60)
else:
    print(f"Error: {resp.status_code}")
    print(resp.text[:500])
