# V19.1 Deep Analysis Report - 2026/04/20

## EXECUTIVE SUMMARY

| Model | Trades | WR | AvgPnL | TotalPnL | PF | MaxLoss |
|-------|--------|-----|--------|----------|-----|---------|
| V11 | 419 | 46.8% | +3.22% | +1350.8% | 2.26 | -39.0% |
| V17 | 418 | 45.9% | +3.96% | +1654.6% | 2.53 | -23.6% |
| V19 | 427 | 44.5% | +4.37% | +1864.2% | 2.64 | -26.2% |
| **V19.1** | **419** | **44.9%** | **+4.46%** | **+1866.8%** | **2.68** | **-28.6%** |
| Rule | 585 | 40.3% | +3.39% | +1982.6% | 2.21 | -27.2% |

**V19.1 vs Rule gap: -115.8%** (Rule wins overall due to more trades & capturing more moves)

---

## V19.1 STRENGTHS

### 1. Superior Profit Factor (2.68 vs Rule's 2.21)
- V19.1 has better risk/reward per trade than Rule
- Higher avg PnL (+4.46% vs +3.39%)

### 2. Peak Protection Module (Excellent)
- `peak_protect_dist`: 32 trades, 96.9% WR, avg +32.54%, total **+1041.2%**
- `peak_protect_ema`: 9 trades, 100% WR, avg +14.44%, total **+129.9%**
- These are the #1 profit generators

### 3. Strong Trend Performance
- Strong trend entries: 203 trades, PF=3.49, total +1196.3%
- V-shape entries: 42 trades, 50% WR, +175.0% total

### 4. Symbols where V19.1 BEATS Rule
| Symbol | V19.1 | Rule | Gap |
|--------|-------|------|-----|
| VIC | +249.0% | +152.5% | **+96.5%** |
| VNM | -0.6% | -53.5% | **+52.9%** |
| HPG | +155.9% | +137.1% | **+18.8%** |
| BID | +54.2% | +42.0% | **+12.2%** |
| AAV | +323.1% | +318.5% | +4.6% |

---

## V19.1 WEAKNESSES

### 1. Signal Exit Losses (ROOT CAUSE #1)
- `signal` exit: 336 trades, WR=34.5%, total only **+95.2%** (vs -1069.4% in losses)
- This is the dominant exit type but barely profitable
- Many big losses: VND -28.6%, -26.9%; SSI -18.5%; DGC -14.3%

### 2. Problematic Symbols (V19.1 underperforms Rule)
| Symbol | V19.1 | Rule | Gap | Root Cause |
|--------|-------|------|-----|------------|
| DGC | +183.0% | +266.6% | **-83.6%** | Missed 4 profitable rule trades, 13 trades losing >3% |
| AAS | +171.1% | +253.7% | **-82.6%** | Missed 5 profitable rule trades, 12 trades losing >3% |
| VND | +253.6% | +302.1% | **-48.5%** | Huge single-trade losses (-28.6%, -26.9%) |
| SSI | +207.1% | +231.8% | **-24.6%** | Weak trend entries with big losses (-18.5%) |
| FPT | +69.8% | +89.6% | **-19.8%** | Strong trend entries still losing -10% |
| REE | +19.3% | +36.6% | **-17.4%** | 10 trades losing >3% |
| ACB | +26.2% | +43.4% | **-17.2%** | V11 (+59.8%) was actually better! |

### 3. VND Deep Dive
- **V17 had better WR (61.1%) vs V19.1 (46.9%)** on VND
- Rule captured +302.1% vs V19.1 +253.6%
- Worst trade: 2021-01-18→2021-02-01: **-28.59%** (signal exit, moderate trend)
  - This was just before the massive rally that Rule caught from 2021-02-17
- Rule captured the 2022-11-18→2022-12-23 rally (+25.6%) but V19.1 entered late and lost -9.7%

### 4. Entry in Wrong Trend Context
- Many worst trades entered in "strong" trend but still lost 8-14%
- Indicates trend detection is lagging or doesn't account for reversals

### 5. Position Size 60-80% Bucket: Worst Performance
- PF=1.94, WR=34.3% — these medium-conviction trades are the weakest

---

## CROSS-MODEL INSIGHTS

### Where V17 beats V19.1:
- **VND**: V17 +278.3% vs V19.1 +253.6% (V17 had 61.1% WR!)
- **HPG**: V17 +182.9% vs V19.1 +155.9%
- **REE**: V17 +52.3% vs V19.1 +19.3%
- **VNM**: V17 +16.1% vs V19.1 -0.6%
- **Root cause**: V17's simpler exit logic (no regime adapter) holds winners longer in certain conditions

### Where V11 beats V19.1:
- **ACB**: V11 +59.8% vs V19.1 +26.2%
- **Root cause**: V11 with more trades and higher WR (48.4%) on ACB; V19.1's filters block too many valid entries

### Where Rule beats all models:
- **DGC**: Rule +266.6% (best). Rule captures more moves with 42 trades vs V19.1's 31
- **AAS**: Rule +253.7%. Rule's simple SMA crossover catches trend starts that ML misses

---

## ROOT CAUSE ANALYSIS

### Problem 1: Signal Exit = Slow Death
The `signal` exit accounts for 336/419 trades and has only 34.5% WR. The multi-score exit confirmation system is too lenient — it keeps V19.1 in losing positions too long.

**Evidence**: VND -28.59% trade held 10 days with signal exit. The exit score threshold wasn't reached despite clearly bearish conditions.

### Problem 2: Rule Captures More Trend Starts
Rule has 585 trades vs V19.1's 419. Rule's SMA crossover is a simple but effective trend-start detector that doesn't have the complex entry filtering of V19.1.

**Evidence**: DGC, AAS, SSI — Rule enters earlier and captures the full trend move. V19.1 often enters mid-trend and gets shaken out.

### Problem 3: Strong Trend ≠ Safe Entry
Many worst trades (-10% to -28%) were entered in "strong" trend context. The trend detection uses lagging indicators (SMA20/50, MACD) which confirm trends AFTER they peak.

### Problem 4: High-Beta Symbols Need Earlier Stops
VND, SSI losses of -18% to -28% shouldn't be possible with proper risk management. The MIN_HOLD=6 days and confirm bars delay stop-losses too long.

---

## IMPROVEMENT RECOMMENDATIONS

### A. SIGNAL EXIT OVERHAUL (Highest Impact)
1. **Replace multi-score exit with MACD crossover + price structure**:
   - Exit when MACD histogram turns negative AND close < EMA8
   - No more confirm bars for losing trades (cum_ret < 0)
2. **Time-based urgency**: If hold > 15d and cum_ret < 3%, lower exit threshold by 50%
3. **Price-based hard stop for signal exits**: If cum_ret < -12%, exit immediately regardless of score

### B. ENTRY TIMING (Align with Rule)
1. **Require close > SMA20** for non-breakout entries (Rule's core signal)
2. **For high_beta symbols**: Add min 1-day confirmation after ML signal
3. **Reduce MIN_HOLD to 4 days** for entries in weak/moderate trend

### C. CAPTURE MORE MOVES (Close the 166-trade gap)
1. **Lower entry score minimum** in strong macro uptrend (SMA20 > SMA50 for >20 days)
2. **Add trend-start detector**: When SMA20 crosses above SMA50 with volume, enter even without ML signal
3. **Hybrid entry**: If Rule would enter AND ML score > 0.4, enter with full size

### D. SYMBOL-SPECIFIC FIXES
1. **VND/SSI**: Reduce max loss to -15% (add hard stop at -15% for high_beta)
2. **ACB**: Remove some entry filters (V11 without filters was better)
3. **DGC/AAS**: Allow more entries by relaxing anti-chop filter when macro trend is confirmed

### E. NEW MODEL IDEAS (V20 concepts)
1. **Ensemble confidence**: Use agreement between V11, V17, V19 as meta-signal
2. **Rule-ML hybrid**: Enter on Rule signal, size on ML confidence, exit on V19.1 logic
3. **Regime-switching model**: Train separate models for bull/bear/sideways
4. **Shorter holding in weak trends**: Target 5-8 day swing trades instead of trend-following

---

## KEY METRICS TO TARGET FOR V20

| Metric | V19.1 Current | Target |
|--------|--------------|--------|
| Total PnL | +1866.8% | >+2200% |
| Win Rate | 44.9% | >48% |
| Profit Factor | 2.68 | >3.0 |
| Max Single Loss | -28.6% | <-15% |
| Signal Exit WR | 34.5% | >42% |
| Trades | 419 | 480-520 |
