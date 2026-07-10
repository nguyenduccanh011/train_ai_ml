#!/bin/bash
# Health check script for stock_ml API

API_URL="http://localhost:8000/health"
MAX_RETRIES=3
RETRY_DELAY=5

for i in $(seq 1 $MAX_RETRIES); do
    response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

    if [ $response -eq 200 ]; then
        echo "[OK] API is healthy (HTTP $response)"
        exit 0
    fi

    echo "[WARN] Attempt $i/$MAX_RETRIES failed (HTTP $response)"
    sleep $RETRY_DELAY
done

echo "[ERROR] API health check failed after $MAX_RETRIES retries"
exit 1
