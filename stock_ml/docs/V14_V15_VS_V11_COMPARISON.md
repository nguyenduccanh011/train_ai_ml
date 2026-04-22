# V14 & V15 vs V11 (Baseline) — Detailed Comparison Report

**Generated:** 2026-04-20  
**Test Period:** 2020–2025 (Walk-Forward, 6 folds)  
**Symbols:** 14 Vietnamese stocks (ACB, FPT, HPG, SSI, VND, MBB, TCB, VNM, DGC, AAS, AAV, REE, BID, VIC)

---

## 📊 Overall Results Summary

| Version | Total PnL | Avg PnL/Symbol | Description |
|---------|-----------|----------------|-------------|
| **V11 (Baseline)** | **+3,971.6%** | **+283.7%** | Stricter entry filters + consolidation breakout |
| V15 (V14 + Mod E) | +1,556.5% | +111.2% | V14 + Secondary Breakout Scanner |
| V14 (V11 + Mod A+B) | +1,546.7% | +110.5% | V11 + V-Shape Entry + Profit-Peak Protection |
| Rule-Based | +1,982.7% | +141.6% | Simple technical rules |

### Key Finding: **V11 massively outperforms V14/V15 by ~2,415%**

---

## 📈 Per-Symbol Breakdown

| Symbol | V15 PnL | V14 PnL | V11 PnL | V11 Winner? | V15 Trades | V11 Trades |
|--------|---------|---------|---------|-------------|------------|------------|
| ACB | +64.6% | +63.3% | **+134.6%** | ✅ | 33 | 31 |
| FPT | +89.9% | +85.5% | **+425.6%** | ✅ | 28 | 26 |
| HPG | +175.7% | +175.7% | **+341.8%** | ✅ | 31 | 31 |
| SSI | +222.8% | +216.3% | **+248.0%** | ✅ | 34 | 31 |
| VND | +238.0% | +237.4% | **+391.7%** | ✅ | 35 | 32 |
| MBB | +88.4% | +91.2% | **+341.2%** | ✅ | 38 | 34 |
| TCB | -8.4% | -2.9% | **+110.3%** | ✅ | 42 | 40 |
| VNM | -0.5% | **+3.3%** | -5.8% | ❌ V14 wins | 31 | 29 |
| DGC | +153.0% | +152.9% | **+295.7%** | ✅ | 32 | 30 |
| AAS | **+182.7%** | +182.7% | +121.2% | ❌ V15 wins | 27 | 23 |
| AAV | +167.2% | +164.9% | **+262.0%** | ✅ | 30 | 23 |
| REE | +21.9% | +21.9% | **+429.5%** | ✅ | 29 | 27 |
| BID | -12.1% | -13.9% | **+498.6%** | ✅ | 39 | 39 |
| VIC | +173.3% | +168.4% | **+377.2%** | ✅ | 25 | 23 |

**V11 wins in 12/14 symbols** (86%). V14/V15 only win in VNM and AAS.

---

## 🔍 Root Cause Analysis: Why V14/V15 Underperform V11

### 1. Module A (V-Shape Entry) — HARMFUL
V14 adds V-Shape bottom detection that **bypasses** V11's safety filters:
- Disables the `ret_5d > 5%` late-entry blocker
- Disables the `drop_from_peak_20 <= -15%` falling-knife blocker
- Uses reduced position size (0.6x) but still catches many false bottoms
- **Impact:** More trades entering during dangerous conditions → lower quality entries

**Evidence:** V14 has 1-3 more trades per symbol than V11, but with significantly lower PnL. The extra entries from V-shape detection are mostly losers or small winners that dilute the portfolio.

### 2. Module B (Profit-Peak Protection) — HARMFUL
V14 adds profit-peak exit rules:
- Exits when max_profit ≥ 20% AND close < SMA10 with heavy volume
- Exits when max_profit ≥ 15% AND 2 consecutive closes < EMA8 at <75% of peak
- **Impact:** Prematurely exits winning trades that V11 would hold longer

**Evidence:** Look at FPT: V11 earns +425.6% vs V14's +85.5%. V11's simpler exit logic lets winners run much further. The profit-peak protection clips gains on the best trades, which are exactly the trades that drive total portfolio returns.

### 3. Module E (Secondary Breakout Scanner in V15) — MARGINAL
V15 adds a looser consolidation breakout with:
- Range ratio ≤ 18% (vs 15% in V11)
- Lookback 12 bars (vs 15)
- Volume threshold 1.2x (vs 1.5x)
- **Impact:** +9.8% improvement over V14, but still far below V11

**Evidence:** V15 vs V14 difference is only +9.8% across all symbols. The secondary breakout adds a few extra entries but doesn't compensate for the damage from Modules A and B.

### 4. Modules C & D (Disabled in V15) — Correctly Disabled
- Module C (Fast Loss Cut): Proved negative — cutting losses too aggressively
- Module D (Adaptive Exit Confirmation): Proved negative — reducing EXIT_CONFIRM when losing causes premature exits

---

## 💡 Key Insights

### Why V11's Simplicity Wins:
1. **Strict entry filters preserve capital**: V11's late-entry blocker and falling-knife blocker prevent entering during dangerous conditions. V14/V15 bypass these filters.
2. **Let winners run**: V11's exit logic (trailing stop + hybrid exit) allows big winners to compound. Profit-peak protection in V14/V15 clips the tail of the distribution.
3. **Fewer but better trades**: V11 averages ~29 trades/symbol vs V14/V15's ~32 trades/symbol. The extra trades in V14/V15 have negative expected value.
4. **The 80/20 rule**: A few big winning trades drive most of the portfolio return. V11's approach maximizes these big winners, while V14/V15's modifications hurt them.

### The Over-Optimization Trap:
V14/V15 represent **over-engineering** — adding complexity that looks good in theory (catch V-bottoms, protect profits) but hurts in practice because:
- V-bottom detection catches false reversals more often than true ones
- Profit protection exits before the real trend completes
- More parameters = more curve-fitting risk

---

## 📋 Version Architecture

```
V11 (Baseline) ← Best performer
├── Stricter entry quality filters
├── Consolidation breakout scanner
├── Stabilized-sideways detection
├── Late-entry blocker (ret_5d > 5%)
├── Falling-knife blocker (drop_from_peak_20 ≤ -15%)
├── Low-volume blocker
├── Bearish-candle position penalty
└── Overextended cap

V14 = V11 + Module A + Module B
├── Module A: V-Shape Entry (bypasses safety filters) ← HARMFUL
└── Module B: Profit-Peak Protection (premature exits) ← HARMFUL

V15 = V14 + Module E (+ disabled C, D)
├── Module C: Fast Loss Cut (DISABLED — negative)
├── Module D: Adaptive Exit Confirmation (DISABLED — negative)
└── Module E: Secondary Breakout Scanner (ENABLED — marginal +9.8%)
```

---

## ✅ Recommendations

1. **V11 remains the best model** — do not deploy V14 or V15
2. **Module A should be discarded** — V-shape entry bypasses critical safety filters
3. **Module B should be discarded** — profit-peak protection hurts more than it helps
4. **Module E has marginal value** — could be tested independently on V11 without A/B
5. **Future improvements** should focus on:
   - Better entry signal quality (not more entries)
   - More sophisticated trend-following exits (not earlier exits)
   - Adaptive position sizing based on signal confidence

---

## 🔗 Visualization

Open `stock_ml/visualization/index.html` in a browser to interactively compare buy/sell signals for each symbol across V15, V14, V11, and Rule-Based strategies.
