# ⚡ Dual-Component Architecture — Quick Start Guide

**For:** Anyone wanting to create hybrid ML+rule strategies  
**Time:** 5 minutes to understand, 2 minutes to create your first template  
**Status:** ✅ Ready to use

---

## 🎯 What is Dual-Component?

**Simple explanation:**
```
Old way: Choose EITHER ML (LightGBM) OR Rule (technical conditions)

New way: Use BOTH
  └─ ML predicts return (LightGBM)
  └─ Rule checks conditions (macd > 0, volume > avg)
  └─ Final signal = ML signal IF rule passes (stricter entry)
```

**Visual:**
```
Entry Signal Flow:
  
  LightGBM → Predicts +0.5% return
      ↓
  Rule Filter → Check: macd > 0? YES, volume > avg? YES
      ↓
  Result → SIGNAL = 1 (BUY)

If rule fails:
  
  LightGBM → Predicts +0.5% return
      ↓
  Rule Filter → Check: macd > 0? NO
      ↓
  Result → SIGNAL = 0 (HOLD - vetoed)
```

---

## 🚀 Create Your First Hybrid Strategy (2 minutes)

### Method 1: Via UI (Recommended)

**Step 1:** Open template builder
```
http://localhost:8000/dashboard/template-builder.html
```

**Step 2:** Fill in strategy details
```
Template Name:  my_first_hybrid
Market:         vn_stock
Strategy:       entry_exit_ensemble
Direction:      long
```

**Step 3:** Select components
```
Feature Set:    leading_v2
Target:         forward_return_regression (horizon 5)

[ENTRY SLOT]
  🧠 ML Algorithm:    LightGBM
  🔴 Rule Filter:     None (optional — add later)

[EXIT SLOT]
  🧠 ML Algorithm:    None
  🔴 Rule Filter:     Select technical rules
```

**Step 4:** Click "Save Template"

**Result:**
```
✅ Template saved!
Preview: "🧠 LightGBM + (no filter yet)"
```

---

### Method 2: Via API (Power Users)

```bash
curl -X POST http://localhost:8000/api/v1/templates/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_first_hybrid",
    "market": "vn_stock",
    "strategy": "entry_exit_ensemble",
    "featureSetId": 2,
    "targetId": 1,
    "entryMlComponentId": 2,
    "entryRuleComponentId": null,
    "exitMlComponentId": null,
    "exitRuleComponentId": null,
    "direction": "long",
    "signalMode": "entry_first",
    "signalThreshold": 0.0,
    "seed": 42,
    "splitConfig": {"type": "walk_forward_year"},
    "engineConfig": {
      "max_hold_bars": 20,
      "costs": {
        "commission": 0.0025,
        "tax": 0.001,
        "slippage": 0.0015
      }
    }
  }'
```

**Response:**
```json
{
  "id": 11,
  "name": "my_first_hybrid",
  "entryMlComponentId": 2,
  "entryRuleComponentId": null,
  ...success
}
```

---

### Method 3: Via Python (Developers)

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.src.pipeline.experiment import ExperimentConfig

async def test_hybrid():
    engine = create_async_engine("sqlite+aiosqlite:///results/leaderboard.db")
    async_session = sessionmaker(engine, class_=AsyncSession)
    
    async with async_session() as session:
        # Load template with dual-components
        config = await ExperimentConfig.from_template_id_async(10, session)
        
        # Check if hybrid mode detected
        entry_model = config.entry_model
        has_ml = entry_model["type"] != "none"
        has_rule = "rule_conditions" in entry_model
        
        if has_ml and has_rule:
            print(f"✅ Hybrid mode detected!")
            print(f"   ML: {entry_model['type']}")
            print(f"   Rule: {entry_model['rule_conditions']}")
        
        # Now run backtest
        from stock_ml.src.pipeline.experiment import run_experiment
        metrics = run_experiment(config, "data/", symbols, "results/")
        return metrics

asyncio.run(test_hybrid())
```

---

## 📊 Understanding the Modes

### Mode 1: ML Only (Default)
```yaml
entryMlComponentId: 2        # LightGBM
entryRuleComponentId: null   # No rule filter

Signal Generation:
  LightGBM.predict(X) > threshold? → BUY
  else → HOLD
```

### Mode 2: Rule Only
```yaml
entryMlComponentId: null     # No ML
entryRuleComponentId: 5      # MACD cross

Signal Generation:
  macd_line > 0 AND sma_ratio > 1? → BUY
  else → HOLD
```

### Mode 3: Hybrid (NEW! 🎉)
```yaml
entryMlComponentId: 2        # LightGBM
entryRuleComponentId: 5      # MACD filter

Signal Generation:
  ml_pred = LightGBM.predict(X)
  rule_valid = (macd_line > 0 AND sma_ratio > 1)
  
  IF rule_valid:
    return ml_pred > threshold? BUY : HOLD
  ELSE:
    return HOLD  # Rule veto
```

---

## 🎛️ Available Components

> **Lưu ý:** Component KHÔNG được seed sẵn theo thuật toán. Bạn tự tạo qua UI/API.
> Mặc định DB hiện chỉ có 2 rule component mẫu (`macd_ma20_entry`, `macd_ma20_exit`).
> Xem [Model Library Guide](../MODEL_LIBRARY_GUIDE.md) để biết cách tạo component.

### ML Algorithms (cho `entryMlComponentId`)
Các thuật toán hỗ trợ khi tạo ML component (`algorithm`):
```
- lightgbm        (gradient boosting)
- xgboost         (gradient boosting)
- random_forest   (ensemble)
- mlp             (neural network)
```

### Rule Filters (cho `entryRuleComponentId`)
Rule component được tạo từ các điều kiện kỹ thuật, ví dụ:
```
- MACD Cross      (momentum)
- RSI Extreme     (overbought/oversold)
- Volume Filter   (liquidity)
- ATR Filter      (volatility)
- SMA Trends      (direction)
```

**To create your own rule:**
1. Open `stock_ml/dashboard/model-library.html`
2. Add new component
3. Algorithm: `rule`
4. Role: entry (or exit/regime/size)
5. Define conditions in JSON

---

## 🧪 Verify Your Setup

### Check 1: Components Available
```bash
curl "http://localhost:8000/api/v1/model-library/components?role=entry"
# Trả về các entry component đã tạo (mặc định: macd_ma20_entry)
```

### Check 2: Feature Sets Ready
```bash
curl http://localhost:8000/api/v1/model-library/feature-sets
# Mặc định trả về: leading_v2 (chạy import_yaml_templates.py để thêm leading_v3)
```

### Check 3: Targets Available
```bash
curl http://localhost:8000/api/v1/model-library/targets
# Mặc định trả về: trend_regime
```

### Check 4: Create Test Template
```bash
# Use UI or API to create a template
# Should succeed without errors
```

---

## ⚙️ Hybrid Strategy Configuration Examples

### Example 1: Strict Entry
```json
{
  "entryMlComponentId": 2,      # LightGBM (predicts return)
  "entryRuleComponentId": 5,    # MACD > 0 (momentum filter)
  "signalThreshold": 0.01,       # Only strong signals
  
  "Result: Only enter if ML predicts >1% return AND MACD > 0"
}
```

### Example 2: Smart Exit
```json
{
  "entryMlComponentId": 2,        # LightGBM
  "entryRuleComponentId": null,   # No entry filter
  "exitMlComponentId": null,
  "exitRuleComponentId": 7,       # Exit on RSI > 70 (overbought)
  
  "Result: ML-driven entry, but exit early if overbought"
}
```

### Example 3: Dual Hybrid
```json
{
  "entryMlComponentId": 2,        # LightGBM
  "entryRuleComponentId": 5,      # MACD filter (stricter)
  "exitMlComponentId": 3,         # RF exit (separate logic)
  "exitRuleComponentId": 7,       # RSI exit (secondary check)
  
  "Result: Sophisticated ML entry + rule filter + dual exit"
}
```

---

## 🔄 Backward Compatibility

✅ **Your old strategies still work!**

**Old single-component template:**
```json
{
  "entryComponentId": 2  # Old way
}
```

**Automatic mapping:**
```
entryComponentId (old) → entryMlComponentId (new)
System auto-detects and converts
No changes needed!
```

---

## 🐛 Troubleshooting

### "No components available"
**Solution:** Import YAML templates first
```bash
python stock_ml/scripts/import_yaml_templates.py --commit
```

### "Template not found" (404)
**Solution:** Check template ID
```bash
curl http://localhost:8000/api/v1/templates/
# List all and find the ID
```

### "Invalid component_type"
**Solution:** Use exact values
```
component_type: "ml"    ✅ (lowercase)
component_type: "rule"  ✅ (lowercase)
component_type: "ML"    ❌ (wrong case)
```

### "Backtest fails with hybrid template"
**Solution:** Check that rule conditions are valid
```python
# Rule condition format:
{
  "conditions": [
    {"macd_line": ">0"},
    {"volume_sma": ">0.8"}
  ],
  "logic": "AND"  # or "OR"
}
```

---

## 📚 Learn More

For **deep understanding**, read:
- `DUAL_COMPONENT_IMPLEMENTATION_SUMMARY.md` — Full reference
- `memory/hybrid_component_architecture.md` — Why this design
- `memory/dual_component_implementation_v1.md` — Technical details

For **implementation**, check:
- `plans/piped-crunching-castle.md` — Step-by-step plan

---

## ✨ What's Next?

### Immediate (Ready Now)
- ✅ Create hybrid ML+rule strategies
- ✅ Create custom rule components (UI/API)
- ✅ Support regime/size dual-components
- ✅ Test via backtest engine
- ✅ Analyze results in leaderboard

### Future
- 💡 Live trading integration
- 💡 OR composition (either ML or rule)
- 💡 Multi-filter per slot
- 💡 Filter library & templates

---

## 🎯 One-Minute Summary

```
Before (Old):
  Strategy = ML(entry) + Rule(exit)
  Limited flexibility

After (New):
  Each slot = ML + Rule (both optional)
  Full flexibility
  Auto-detected mode
  Backward compatible
  UI-friendly dual picker

Created:
  ✅ 8 components implemented
  ✅ ~800 lines of code
  ✅ Full UI support
  ✅ Complete API
  ✅ Test infrastructure
  ✅ Comprehensive docs

Ready to:
  🚀 Create hybrid strategies NOW
  🚀 Research ML + rule combinations
  🚀 Scale with team
```

---

**🚀 You're ready! Pick Method 1/2/3 above and create your first hybrid strategy.**

