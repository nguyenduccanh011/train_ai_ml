#!/bin/bash
# Test template-based submission flow (DB-First Phase 0-3)
#
# Prerequisites:
#   1. DB is initialized with migrations
#   2. API server is running: python -m stock_ml.api.main
#   3. At least one feature set and target exist in DB

set -e

API_BASE="http://localhost:8000/api"
MARKET="vn_stock"

echo "=== Template Submission Flow Test ==="
echo ""

# Step 1: Create Entry Rule Component
echo "Step 1: Creating entry rule component..."
ENTRY_RESPONSE=$(curl -s -X POST "$API_BASE/model-library/components" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "macd_ma20_entry_test",
    "role": "entry",
    "algorithm": "rule",
    "componentType": "rule",
    "params": {
      "conditions": [
        {"feature": "macd_hist", "op": ">", "value": 0},
        {"feature": "sma_20_ratio", "op": "<", "value": 1.0},
        {"feature": "close_to_open", "op": ">", "value": 1.0}
      ],
      "logic": "AND",
      "score_feature": "macd_hist"
    },
    "description": "Test: MACD HIS > 0, MA20 < C, C > O"
  }')

ENTRY_ID=$(echo "$ENTRY_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)
echo "✓ Entry component created: ID $ENTRY_ID"
echo ""

# Step 2: Create Exit Rule Component
echo "Step 2: Creating exit rule component..."
EXIT_RESPONSE=$(curl -s -X POST "$API_BASE/model-library/components" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "macd_ma20_exit_test",
    "role": "exit",
    "algorithm": "rule",
    "componentType": "rule",
    "params": {
      "conditions": [
        {"feature": "macd_hist", "op": "<", "value": 0},
        {"feature": "sma_20_ratio", "op": ">", "value": 1.0},
        {"feature": "close_to_open", "op": "<", "value": 1.0}
      ],
      "logic": "AND",
      "score_feature": "macd_hist"
    },
    "description": "Test: MACD HIS < 0, C < MA20, C < O"
  }')

EXIT_ID=$(echo "$EXIT_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)
echo "✓ Exit component created: ID $EXIT_ID"
echo ""

# Step 3: Get Feature Set ID
echo "Step 3: Retrieving feature set catalog..."
FEATURE_SETS=$(curl -s -X GET "$API_BASE/model-library/feature-sets")
FEATURE_SET_ID=$(echo "$FEATURE_SETS" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)
echo "✓ Feature set ID: $FEATURE_SET_ID"
echo ""

# Step 4: Get Target ID
echo "Step 4: Retrieving target catalog..."
TARGETS=$(curl -s -X GET "$API_BASE/model-library/targets?type=trend_regime")
TARGET_ID=$(echo "$TARGETS" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)
echo "✓ Target ID: $TARGET_ID"
echo ""

# Step 5: Create Template
echo "Step 5: Creating strategy template..."
TEMPLATE_RESPONSE=$(curl -s -X POST "$API_BASE/templates" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"macd_ma20_test_$(date +%s)\",
    \"strategy\": \"rule_only\",
    \"market\": \"$MARKET\",
    \"description\": \"Test template for rule-based MACD + MA20\",
    \"hypothesis\": \"MACD histogram crossover with MA20 filter\",
    \"featureSetId\": $FEATURE_SET_ID,
    \"targetId\": $TARGET_ID,
    \"direction\": \"long\",
    \"signalMode\": \"entry_first\",
    \"signalThreshold\": 0.0,
    \"componentSlots\": [
      {
        \"slotType\": \"entry\",
        \"ruleComponentId\": $ENTRY_ID,
        \"mlComponentId\": null
      },
      {
        \"slotType\": \"exit\",
        \"ruleComponentId\": $EXIT_ID,
        \"mlComponentId\": null
      }
    ],
    \"splitConfig\": {
      \"type\": \"walk_forward_year\",
      \"train_years\": 1,
      \"test_years\": 1,
      \"gap_days\": 25,
      \"first_test_year\": 2023,
      \"last_test_year\": 2024
    },
    \"engineConfig\": {
      \"max_hold_bars\": 20,
      \"min_hold_bars\": 1,
      \"hard_stop_pct\": -0.08,
      \"costs\": {
        \"commission\": 0.0015,
        \"tax\": 0.001,
        \"slippage\": 0.0015
      }
    },
    \"seed\": 42
  }")

TEMPLATE_ID=$(echo "$TEMPLATE_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)
echo "✓ Template created: ID $TEMPLATE_ID"
echo ""

# Step 6: Get Template Details
echo "Step 6: Retrieving template details..."
TEMPLATE_DETAILS=$(curl -s -X GET "$API_BASE/templates/$TEMPLATE_ID")
echo "$TEMPLATE_DETAILS" | python3 -m json.tool | head -20
echo "..."
echo ""

# Step 7: Submit Experiment
echo "Step 7: Submitting template for execution..."
SUBMIT_RESPONSE=$(curl -s -X POST "$API_BASE/templates/$TEMPLATE_ID/submit" \
  -H "Content-Type: application/json" \
  -d '{}')

JOB_ID=$(echo "$SUBMIT_RESPONSE" | grep -o '"job_id":"[^"]*' | cut -d'"' -f4)
echo "✓ Experiment submitted!"
echo "  Job ID: $JOB_ID"
echo ""

# Step 8: Monitor Execution
echo "Step 8: Monitoring execution..."
LOG_PATH=$(echo "$SUBMIT_RESPONSE" | grep -o '"log_path":"[^"]*' | cut -d'"' -f4)
if [ -f "$LOG_PATH" ]; then
  echo "  Log file: $LOG_PATH"
  echo ""
  echo "  (First 10 lines of log):"
  head -10 "$LOG_PATH" || echo "  [log not written yet - experiment may still be queuing]"
else
  echo "  [Log file not yet created - waiting for subprocess to start]"
fi
echo ""

echo "=== Test Complete ==="
echo ""
echo "Next steps:"
echo "1. Monitor execution: tail -f $LOG_PATH"
echo "2. Check template runs: curl $API_BASE/templates/$TEMPLATE_ID/runs"
echo "3. View leaderboard results once complete"
echo ""
