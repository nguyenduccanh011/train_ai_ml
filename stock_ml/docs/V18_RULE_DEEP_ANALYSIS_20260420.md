# V18 vs Rule Deep Analysis (Run date: 2026-04-20)

## 1) Tong quan hieu nang (14 ma)

- V18 tong: **+1831.5%** (422 trades, PF 2.74, WR 45.5%)
- Rule tong: **+1982.6%** (585 trades, PF 2.21, WR 40.3%)
- Chenh lech: **-151.2%** (V18 thap hon Rule)
- So ma V18 > Rule: **3/14**
- So ma V18 < Rule: **11/14**

## 2) TCB chi tiet (ma trong tam)

- V18: **+23.1%** (36 trades, PF 1.26)
- Rule: **+37.6%** (47 trades, PF 1.29)
- Gap: **-14.5%**

### 2.1 Nhip Rule kiem loi tot nhung V18 khong bat duoc / bat kem

```
(none)
```

### 2.2 Cac nhip Rule co overlap nhung V18 an qua it

```
entry_date  exit_date  holding_days  pnl_pct  v18_overlap_pnl  capture_gap
2020-04-10 2020-06-11            42    24.28             0.52        23.76
2020-11-13 2021-01-27            52    45.26            28.27        16.99
2020-08-12 2020-10-07            39    10.49            -1.08        11.57
```

### 2.3 Giao dich V18 khong hieu qua gay thua lo / nhieu

```
entry_date  exit_date  holding_days  pnl_pct exit_reason  rule_overlap_pnl  avoidable_gap entry_trend  entry_ret_5d  entry_dist_sma20  quick_reentry  breakout_entry
2021-01-25 2021-01-27             2    -6.39   stop_loss             45.26          51.65    moderate         -1.11              1.31          False           False
2022-10-28 2022-11-10             9    -6.76      signal              7.09          13.85        weak          6.56             -3.82          False           False
2020-08-28 2020-10-09            29    -1.08      signal             10.49          11.57      strong          6.14              7.93          False           False
2022-02-09 2022-02-24            11    -5.77      signal              2.94           8.71    moderate          3.26              6.18          False           False
2020-01-06 2020-02-03            15    -6.98      signal               NaN           6.98        weak          0.00              0.00          False           False
```

### 2.4 Counter module V18 tren TCB

- relaxed_ret5_entries: **9**
- relaxed_dp_entries: **15**
- signal_quality_saves: **4**
- chop_blocked: **0**
- bear_blocked: **3**

## 3) Cac ma khac co van de tuong tu

### DGC (gap -87.6%)
- V18 +178.9% vs Rule +266.6%
- Missed big-win proxy: +0.0%
- Undercapture gap proxy: +146.8%
- Avoidable loss proxy: +47.8%
- Mau giao dich van de:
```
entry_date  exit_date  holding_days  pnl_pct exit_reason  rule_overlap_pnl  avoidable_gap entry_trend  entry_ret_5d  entry_dist_sma20  breakout_entry
2021-01-13 2021-01-29            12    -5.31      signal             23.09          28.40    moderate          2.31              0.79           False
2022-05-27 2022-07-07             9    -0.22      signal             12.97          13.19        weak          0.47             -0.57           False
2020-07-09 2020-07-28             7    -6.22   stop_loss               NaN           6.22      strong          1.53              2.20           False
```

### AAV (gap -65.8%)
- V18 +252.7% vs Rule +318.5%
- Missed big-win proxy: +0.0%
- Undercapture gap proxy: +195.6%
- Avoidable loss proxy: +88.0%
- Mau giao dich van de:
```
entry_date  exit_date  holding_days  pnl_pct exit_reason  rule_overlap_pnl  avoidable_gap entry_trend  entry_ret_5d  entry_dist_sma20  breakout_entry
2023-06-23 2023-07-06             9   -11.11      signal             29.93          41.04      strong          1.61              5.35           False
2025-11-10 2025-12-12            23    -1.52      signal             12.83          14.35      strong         10.00              9.54           False
2023-02-20 2023-03-03             9   -13.33      signal               NaN          13.33        weak          7.14             -2.91           False
2023-07-18 2023-08-22            25    -4.92      signal              5.95          10.87    moderate          0.00              3.04           False
2022-08-05 2022-08-24            13    -4.17      signal              4.21           8.38      strong          7.99              7.30           False
```

### AAS (gap -51.8%)
- V18 +201.9% vs Rule +253.7%
- Missed big-win proxy: +0.0%
- Undercapture gap proxy: +144.2%
- Avoidable loss proxy: +82.9%
- Mau giao dich van de:
```
entry_date  exit_date  holding_days  pnl_pct exit_reason  rule_overlap_pnl  avoidable_gap entry_trend  entry_ret_5d  entry_dist_sma20  breakout_entry
2025-05-20 2025-06-16            19    -3.26      signal             15.50          18.76      strong         19.48             19.87           False
2022-08-03 2022-08-16             9    -4.27      signal             13.77          18.04    moderate          2.93              8.33           False
2023-04-07 2023-05-08            17    -2.55      signal             13.61          16.16      strong         15.60             15.84           False
2022-12-19 2022-12-30             9   -14.05      signal             -2.43          11.62    moderate          2.89              6.55           False
2021-04-15 2021-04-29             9   -14.73      signal             -3.54          11.19      strong         -2.86              5.82           False
2022-10-18 2022-10-31             9    -7.10      signal               NaN           7.10        weak         12.36             -6.74           False
```

### MBB (gap -37.2%)
- V18 +87.5% vs Rule +124.7%
- Missed big-win proxy: +0.0%
- Undercapture gap proxy: +52.6%
- Avoidable loss proxy: +30.4%
- Mau giao dich van de:
```
entry_date  exit_date  holding_days  pnl_pct exit_reason  rule_overlap_pnl  avoidable_gap entry_trend  entry_ret_5d  entry_dist_sma20  breakout_entry
2025-03-19 2025-04-04            12    -5.87      signal              9.32          15.19      strong         -3.03              0.88           False
2022-08-17 2022-09-09            15    -1.84      signal              7.94           9.78      strong          2.04              5.90            True
2020-01-22 2020-02-11             9    -5.38      signal               NaN           5.38      strong          4.33              4.05           False
```

### VND (gap -25.1%)
- V18 +277.0% vs Rule +302.1%
- Missed big-win proxy: +0.0%
- Undercapture gap proxy: +79.0%
- Avoidable loss proxy: +124.8%
- Mau giao dich van de:
```
entry_date  exit_date  holding_days  pnl_pct exit_reason  rule_overlap_pnl  avoidable_gap entry_trend  entry_ret_5d  entry_dist_sma20  breakout_entry
2022-12-15 2022-12-28             9   -11.31      signal             25.65          36.96      strong          3.89             19.04           False
2022-08-17 2022-09-05            11    -3.82      signal             17.77          21.59      strong          0.22             10.36           False
2022-11-02 2022-11-15             9   -18.17      signal               NaN          18.17        weak         13.57             -8.39           False
2022-06-27 2022-07-08             9    -0.26      signal             17.77          18.03        weak         16.81            -12.39           False
2022-01-06 2022-01-19             9   -17.87      signal               NaN          17.87        weak          0.00              0.00           False
2021-07-07 2021-07-09             2    -6.11   stop_loss               NaN           6.11      strong          0.28              1.69           False
```

### BID (gap -20.0%)
- V18 +22.0% vs Rule +42.0%
- Missed big-win proxy: +0.0%
- Undercapture gap proxy: +56.1%
- Avoidable loss proxy: +87.5%
- Mau giao dich van de:
```
entry_date  exit_date  holding_days  pnl_pct exit_reason  rule_overlap_pnl  avoidable_gap entry_trend  entry_ret_5d  entry_dist_sma20  breakout_entry
2025-12-31 2025-12-31             0     0.00         end             35.03          35.03      strong          1.30              2.25           False
2021-01-07 2021-01-19             8    -7.48   stop_loss             13.65          21.13        weak          0.00              0.00           False
2022-11-29 2022-12-20            15    -6.12      signal             14.96          21.08      strong         12.48             13.98           False
2021-05-31 2021-06-14            10    -7.80      signal              2.44          10.24      strong          5.87             12.33           False
```

## 4) Root cause (du lieu + logic)

1. V18 da cai thien entry quality (PF tang) nhung van thua Rule o nhom ma can momentum tiep dien (AAV, DGC, AAS, MBB, TCB, BID, VND).
2. Nhieu trade lo cua V18 tap trung o exit_reason `signal`; vao trend weak/moderate va hold ngan -> de bi whipsaw.
3. O cac nhom gap am lon, Rule thuong giu duoc 1 swing dai; V18 chia thanh nhieu lenh ngan hoac vao sau/ra som nen tong payoff kem.
4. Counter cho thay V18 co no luc mo khoa (`relaxed_ret5`, `relaxed_dp`) nhung chua du de bat cac pha tang manh keo dai o mot so ma.

## 5) De xuat cai tien de dot pha va on dinh hon

1. Two-stage entry policy: tach `entry_alpha` (cho phep vao) va `size_policy` (scale size theo overheat) thay vi block cung anti-chop/ret5.
2. Symbol-regime adapter: hoc threshold rieng theo nhom volatility/chop (bank, beta cao, phong thu) cho ret5/dp/exit confirm.
3. Exit quality model: bo sung classifier nho cho `signal_exit` de phan biet pullback vs trend-break (feature: MA slope, MACD hist delta, volume shock, ATR expansion).
4. Trade stitching: neu co re-entry trong <=7 ngay cung trend, danh gia theo campaign-level PnL de phat hien over-trading va toi uu cooldown.
5. Objective tuning theo payoff asymmetry: toi uu `total_pnl + PF - lambda*avoidable_gap` thay vi toi uu WR don le.
6. Offline counterfactual: replay lai cac window Rule-win-lon de tim nguong gay miss lon nhat (ret5 cap, dp cap, confirm bars).

## 6) File bang chung

- `results/v18_trades_20260420_detailed.csv`
- `results/rule_trades_20260420_detailed.csv`
- `results/v18_rule_symbol_metrics_20260420.csv`
- `results/v18_rule_gap_breakdown_20260420.csv`
- `results/v18_exit_reason_by_symbol_20260420.csv`