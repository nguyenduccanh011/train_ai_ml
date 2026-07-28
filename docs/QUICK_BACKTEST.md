# Quick Backtest Guide - Run Strategy to Leaderboard

Bạn đã có strategy template, bây giờ chạy backtest để đưa lên leaderboard.

---

## **3 Cách Chạy Backtest**

### **Cách 1: UI (Template Explorer) - ⭐ Dễ Nhất**

1. Mở dashboard:
   ```
   http://localhost:8000/dashboard/template-explorer.html
   ```

2. Tìm template của bạn (ví dụ: `macd_ma20_rule_v1`)

3. Nhấn button **"Submit"** (xanh lá)

4. Confirm popup → Hệ thống sẽ queue experiment

5. Nhận Job ID:
   ```
   Job ID: tmpl_1_macd_ma20_rule_v1
   Log: logs/tmpl_1_macd_ma20_rule_v1.log
   ```

---

### **Cách 2: cURL (API)**

```bash
curl -X POST http://localhost:8000/api/v1/templates/1/submit \
  -H "Content-Type: application/json" \
  -d '{
    "override_seed": 42
  }'
```

**Response:**
```json
{
  "job_id": "tmpl_1_macd_ma20_rule_v1",
  "template_id": 1,
  "template_name": "macd_ma20_rule_v1",
  "status": "queued",
  "log_path": "logs/tmpl_1_macd_ma20_rule_v1.log"
}
```

---

### **Cách 3: CLI (Command Line)**

```bash
python -m stock_ml.scripts.run_template \
  --template-id 1 \
  --seed 42 \
  --out results/
```

**Output:**
```
[OK] Loaded template 1: macd_ma20_rule_v1
[OK] Resolved symbols: 30 symbols
[OK] Experiment completed
[OK] Upserted 30 results to leaderboard
```

---

## **Monitor Execution**

### **1. View Log File**
```bash
# Real-time log
tail -f logs/tmpl_1_macd_ma20_rule_v1.log

# First 20 lines
head -20 logs/tmpl_1_macd_ma20_rule_v1.log
```

**Expected output:**
```
[macd_ma20_rule_v1] loading 30 symbols from /data/vn_stock
  [debug] Resolved to: /data/vn_stock/all_symbols.duckdb
  [debug] Using DuckDB loader
  [debug] Found 30 symbols
✓ Executing walk_forward_year: year 2020 (train: 2018-2019, test: 2020)
  ├─ Training entry model (lightgbm)
  ├─ Training exit model (lightgbm)
  ├─ Backtesting
  └─ Results: 30 symbols processed

✓ Experiment completed
```

### **2. Check Job Status**
```bash
# List running jobs
curl http://localhost:8000/api/v1/jobs

# Get specific job
curl http://localhost:8000/api/v1/jobs/tmpl_1_macd_ma20_rule_v1
```

### **3. Check Template Runs**
```bash
# List all runs from this template
curl http://localhost:8000/api/v1/templates/1/runs
```

**Response:**
```json
[
  {
    "runId": "run_0001",
    "state": "completed",
    "composite_score": 87.5,
    "total_pnl": 12500,
    "sharpe": 1.8,
    "max_drawdown": -8.5,
    "n_symbols": 30,
    "generated_at": "2026-05-31T10:30:00"
  }
]
```

---

## **View Results on Leaderboard**

### **1. Open Leaderboard**
```
http://localhost:8000/dashboard/leaderboard.html
```

### **2. Filter by Template**

Search for your template name, hoặc filter:
```
?template_id=1
?market=vn_stock
?strategy=rule_only
```

### **3. Results Display**

Bảng hiển thị:
```
Run ID          Template         Market      Strategy    PnL      Sharpe   MaxDD   Trades  Status
─────────────────────────────────────────────────────────────────────────────────────────────────
run_0001        macd_ma20_v1     vn_stock    rule_only   +12500   1.80     -8.5%   245     ✓ PASS
```

### **4. Click for Details**
- PnL breakdown per year
- Trade statistics
- Symbol performance
- Feature importance (if ML)
- Risk metrics

---

## **Timeline**

| Duration | What Happens |
|----------|--------------|
| **Immediately** | Job queued, log file created |
| **1-5 mins** | Experiment loading data + training |
| **5-15 mins** | Walk-forward backtesting (depends on symbols count) |
| **After complete** | Results upserted to leaderboard |

---

## **Example: Full Workflow**

### **Step 1: Create Rule Components**
```bash
python scripts/create_macd_rule_components.py
# Output: Entry component ID: 1, Exit component ID: 2
```

### **Step 2: Create Template**
```bash
python scripts/create_macd_template.py
# Output: Template ID: 1
```

### **Step 3: Submit via UI**
```
1. Open: http://localhost:8000/dashboard/template-explorer.html
2. Find: "macd_ma20_rule_v1"
3. Click: "Submit" button
4. Copy: Job ID from response
```

### **Step 4: Monitor**
```bash
# Watch logs
tail -f logs/tmpl_1_macd_ma20_rule_v1.log

# Check every 2 mins
watch -n 2 "curl -s http://localhost:8000/api/v1/templates/1/runs | jq '.[-1]'"
```

### **Step 5: View Results**
```bash
# When complete, open leaderboard
open http://localhost:8000/dashboard/leaderboard.html

# Filter
?template_id=1
```

---

## **Troubleshooting**

### **❌ "Template not found"**
```bash
# Check template exists
curl http://localhost:8000/api/v1/templates/1

# If not, create it
python scripts/create_macd_template.py
```

### **❌ "No symbols found"**
```bash
# Check market data exists
ls -la /data/vn_stock/

# Check leaderboard can load
curl http://localhost:8000/leaderboard?market=vn_stock | head -20
```

### **❌ Backtest still running after 30 mins**
```bash
# Check log for errors
tail -100 logs/tmpl_1_*.log | grep -i error

# Check system resources
ps aux | grep run_template
top -p $(pgrep -f run_template)
```

### **❌ Results not in leaderboard**
```bash
# Check if experiment completed
curl http://localhost:8000/api/v1/templates/1/runs

# If status = "completed", check DB
sqlite3 stock_ml.db "SELECT * FROM leaderboard_runs WHERE template_id=1;"
```

---

## **Performance Tips**

### **Faster Backtest**
1. **Reduce symbols count**
   ```yaml
   universe:
     mode: explicit
     explicit_list: [AAA, SSI, VNM, HPG]  # 4 symbols = faster
   ```

2. **Shorter time period**
   ```yaml
   split:
     first_test_year: 2023  # Skip old years
     last_test_year: 2024
   ```

3. **Fewer seeds**
   ```yaml
   validation:
     n_seeds: 1  # Default is 1, sufficient for first run
   ```

### **Parallel Execution**
Sequential (1 template mỗi lần) qua `stock_ml/scripts/run_template.py --id <template_id>`.
(CLI `run_experiments` của hệ YAML cũ đã bị xóa.)

---

## **Expected Results**

### **Good Strategy** (Leaderboard Score > 75)
- Sharpe > 1.0
- Max Drawdown < -15%
- Winning Rate > 50%
- Trade Count > 100

### **Acceptable Strategy** (Score 60-75)
- Sharpe > 0.5
- Max Drawdown < -25%
- Winning Rate > 40%
- Trade Count > 50

### **Needs Improvement** (Score < 60)
- Sharpe < 0.5
- Max Drawdown > -40%
- Too few trades
- Overfitted or underfitted

---

## **Next Steps**

After first backtest:

1. **Analyze Results**
   - Check leaderboard details
   - Review per-symbol performance
   - Identify weak areas

2. **Iterate**
   - Adjust rule conditions
   - Try different features
   - Optimize entry/exit signals

3. **Compare**
   - Create variants (v2, v3)
   - Run A/B comparison
   - Track improvements

4. **Production**
   - When satisfied with results
   - Consider deployment constraints
   - Plan live trading

---

## **API Reference**

```bash
# Submit template
POST /api/v1/templates/{id}/submit

# List templates
GET /api/v1/templates?market=vn_stock

# Get template runs
GET /api/v1/templates/{id}/runs

# Get leaderboard
GET /leaderboard?market=vn_stock&template_id={id}

# Get job status
GET /api/jobs/{job_id}

# List all jobs
GET /api/jobs
```

---

**Đơn giản thôi: Create → Submit → Monitor → View! 🚀**
