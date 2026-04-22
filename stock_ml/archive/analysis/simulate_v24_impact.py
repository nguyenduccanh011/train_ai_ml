"""V24 impact simulation on existing trade CSVs. Read-only analysis."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, os, numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def load(n):
    df = pd.read_csv(os.path.join(OUT, f"trades_{n}.csv"))
    return df

v191 = load("v19_1"); v22 = load("v22"); v23 = load("v23"); v193 = load("v19_3"); rule = load("rule")
if "symbol" in rule.columns and "entry_symbol" not in rule.columns:
    rule = rule.rename(columns={"symbol": "entry_symbol"})

print("=" * 100)
print("V24 IMPACT SIMULATION (derived from existing trade CSVs — no model re-run)")
print("=" * 100)

# ============================================================
# FIX 5.1: Smart hard_cap (confirm 1 bar in strong trend)
# Simulation: take V23 signal_hard_cap trades in strong trend.
# If V19.1 same (symbol,date±3d) trade avoided this exit (used signal/peak_protect),
# we assume V24 with confirm-1-bar would behave like V19.1.
# Replace V23 pnl with V19.1 pnl on these matched cases.
# ============================================================
print("\n[FIX 5.1] Smart hard_cap — confirm 1 bar in strong trend")
v23["dt"] = pd.to_datetime(v23["entry_date"])
v191["dt"] = pd.to_datetime(v191["entry_date"])

shc_strong = v23[(v23["exit_reason"] == "signal_hard_cap") & (v23["entry_trend"] == "strong")]
print(f"  V23 signal_hard_cap in strong trend: n={len(shc_strong)}, tot={shc_strong['pnl_pct'].sum():+.1f}%")

delta_51 = 0.0; improved = []
for _, t in shc_strong.iterrows():
    m = v191[(v191["entry_symbol"] == t["entry_symbol"]) &
             (abs((v191["dt"] - t["dt"]).dt.days) <= 3)]
    if len(m) == 0: continue
    v191_pnl = m.iloc[0]["pnl_pct"]
    if v191_pnl > t["pnl_pct"]:
        improved.append((t["entry_symbol"], str(t["dt"].date()), t["pnl_pct"], v191_pnl, v191_pnl - t["pnl_pct"]))
        delta_51 += v191_pnl - t["pnl_pct"]
print(f"  If V24 behaves like V19.1 on these (avoiding hard_cap via confirm bar): delta = {delta_51:+.1f}%")
print(f"  Cases improved: {len(improved)} of {len(shc_strong)}")
print(f"  Top cases:")
for sym, dt, v23p, v191p, d in sorted(improved, key=lambda x: -x[4])[:8]:
    print(f"    {sym} {dt}: V23 {v23p:+.1f}% → V24(≈V19.1) {v191p:+.1f}% (Δ+{d:.1f}%)")

# Also moderate hard_cap
shc_mod = v23[(v23["exit_reason"] == "signal_hard_cap") & (v23["entry_trend"] == "moderate")]
delta_51m = 0.0
for _, t in shc_mod.iterrows():
    m = v191[(v191["entry_symbol"] == t["entry_symbol"]) &
             (abs((v191["dt"] - t["dt"]).dt.days) <= 3)]
    if len(m) and m.iloc[0]["pnl_pct"] > t["pnl_pct"]:
        delta_51m += m.iloc[0]["pnl_pct"] - t["pnl_pct"]
print(f"  Moderate-trend hard_cap fix delta: {delta_51m:+.1f}%")
total_51 = delta_51 + delta_51m * 0.5  # moderate keeps some safety → 50% capture
print(f"  Estimated total delta (strong 100% + moderate 50%): {total_51:+.1f}%")

# ============================================================
# FIX 5.2: Restore peak_protect sensitivity V19.1 style
# Simulation: count V19.1 peak_protect trades NOT matched in V23 (V23 took signal instead).
# On matched pairs where V19.1 peak_protect > V23 signal, take the V19.1 PnL.
# ============================================================
print("\n[FIX 5.2] Peak_protect sensitivity restore (V19.1 style)")
pp_v191 = v191[v191["exit_reason"].str.startswith("peak_protect")]
print(f"  V19.1 peak_protect: n={len(pp_v191)}, tot={pp_v191['pnl_pct'].sum():+.1f}%")
pp_v23 = v23[v23["exit_reason"].str.startswith("peak_protect")]
print(f"  V23  peak_protect: n={len(pp_v23)}, tot={pp_v23['pnl_pct'].sum():+.1f}%")

gain_52 = 0; count_52 = 0
for _, t in pp_v191.iterrows():
    # find matching V23 trade
    m = v23[(v23["entry_symbol"] == t["entry_symbol"]) &
            (abs((v23["dt"] - t["dt"]).dt.days) <= 3)]
    if len(m) == 0: continue
    v23p = m.iloc[0]["pnl_pct"]
    if t["pnl_pct"] > v23p:
        gain_52 += t["pnl_pct"] - v23p
        count_52 += 1
print(f"  If V24 peak_protect matches V19.1 (where V19.1 wins): delta = +{gain_52:.1f}% over {count_52} trades")
# But we might lose some: V23 peak_protect wins
lose_52 = 0
for _, t in pp_v23.iterrows():
    m = v191[(v191["entry_symbol"] == t["entry_symbol"]) &
             (abs((v191["dt"] - t["dt"]).dt.days) <= 3)]
    if len(m) == 0: continue
    v191p = m.iloc[0]["pnl_pct"]
    if t["pnl_pct"] > v191p:
        lose_52 += t["pnl_pct"] - v191p
print(f"  V23 peak_protect cases where V23 > V19.1: would lose {lose_52:+.1f}%")
net_52 = gain_52 - lose_52
print(f"  Net estimated delta: {net_52:+.1f}%")

# ============================================================
# FIX 5.3: Long-horizon carry — capture Rule big wins
# Simulation: count rule trades > +20% that V23 missed or cut short.
# ============================================================
print("\n[FIX 5.3] Long-horizon carry (capture rule big wins)")
rule["dt"] = pd.to_datetime(rule["entry_date"])
big_rule = rule[rule["pnl_pct"] > 20].copy()
print(f"  Rule wins >+20%: n={len(big_rule)}, tot={big_rule['pnl_pct'].sum():+.1f}%")

captured = []; missed_entirely = []; captured_partial = []
for _, r in big_rule.iterrows():
    m = v23[(v23["entry_symbol"] == r["entry_symbol"]) &
            (abs((v23["dt"] - r["dt"]).dt.days) <= 7)]
    if len(m) == 0:
        missed_entirely.append((r["entry_symbol"], str(r["dt"].date()), r["pnl_pct"]))
        continue
    v23p = m.iloc[0]["pnl_pct"]
    if v23p >= r["pnl_pct"] * 0.7:
        captured.append((r["entry_symbol"], str(r["dt"].date()), r["pnl_pct"], v23p))
    else:
        captured_partial.append((r["entry_symbol"], str(r["dt"].date()), r["pnl_pct"], v23p, r["pnl_pct"] - v23p))

print(f"  Fully captured (V23 ≥70% of rule gain): {len(captured)}")
print(f"  Partially captured (V23 cut short): {len(captured_partial)}, sum lost = {sum(x[4] for x in captured_partial):+.1f}%")
print(f"  Entirely missed (no V23 entry): {len(missed_entirely)}, sum missed = {sum(x[2] for x in missed_entirely):+.1f}%")
print(f"\n  Top 10 partial-captures (V24 long-horizon carry target):")
for sym, dt, rp, v23p, d in sorted(captured_partial, key=lambda x: -x[4])[:10]:
    print(f"    {sym} {dt}: rule {rp:+.1f}% vs V23 {v23p:+.1f}% → recoverable +{d:.1f}%")
# Conservative: V24 could capture 40-50% of lost alpha from partial-captures
est_53 = sum(x[4] for x in captured_partial) * 0.45
print(f"  Conservative estimate (45% of partial-capture gap): +{est_53:.1f}%")

# ============================================================
# FIX 5.4: Symbol-specific tuning
# Simulation: quantify REE/MBB/AAS regression vs V19.1 and assume
# symbol tuning restores to V19.1 levels on these mães.
# ============================================================
print("\n[FIX 5.4] Symbol-specific tuning (REE/MBB/AAS restore)")
for sym in ["REE", "MBB", "AAS"]:
    v23_sym = v23[v23["entry_symbol"] == sym]["pnl_pct"].sum()
    v191_sym = v191[v191["entry_symbol"] == sym]["pnl_pct"].sum()
    print(f"  {sym}: V23 {v23_sym:+.1f}% vs V19.1 {v191_sym:+.1f}% → gap {v191_sym - v23_sym:+.1f}%")
# If we just restore V19.1 behavior selectively per symbol (REE especially):
gap_ree = v191[v191["entry_symbol"] == "REE"]["pnl_pct"].sum() - v23[v23["entry_symbol"] == "REE"]["pnl_pct"].sum()
gap_mbb = v191[v191["entry_symbol"] == "MBB"]["pnl_pct"].sum() - v23[v23["entry_symbol"] == "MBB"]["pnl_pct"].sum()
gap_aas = v191[v191["entry_symbol"] == "AAS"]["pnl_pct"].sum() - v23[v23["entry_symbol"] == "AAS"]["pnl_pct"].sum()
est_54 = gap_ree + gap_mbb + gap_aas
# But 5.1+5.2 already recover much of this, so apply 50% to avoid double counting
est_54_net = est_54 * 0.5
print(f"  Gross gap restore: {est_54:+.1f}%, after 50% double-count correction: +{est_54_net:.1f}%")

# ============================================================
# FIX 5.5: Rule + ML ensemble
# Simulation: Rule alone beats V23 by +122%. If V24 adds partial rule signal blend,
# could capture fraction of that.
# ============================================================
print("\n[FIX 5.5] Rule + ML ensemble (partial rule capture)")
gap_rule = rule["pnl_pct"].sum() - v23["pnl_pct"].sum()
print(f"  Rule total = {rule['pnl_pct'].sum():+.1f}%, V23 = {v23['pnl_pct'].sum():+.1f}%, gap = {gap_rule:+.1f}%")
# Ensemble can realistically capture 20-30% of gap (because overlap with 5.3 big wins)
est_55 = gap_rule * 0.10
print(f"  Conservative estimate (10% of gap, non-overlapping with 5.3): +{est_55:.1f}%")

# ============================================================
# TOTAL V24 PROJECTION
# ============================================================
print("\n" + "=" * 100)
print("V24 TOTAL PROJECTION (on top of V23-best +1860.3%)")
print("=" * 100)
v23_base = 1860.3
# Fix overlaps — don't double count symbol tuning vs 5.1/5.2
projected_delta = total_51 + net_52 + est_53 + est_54_net + est_55
# Apply realistic interaction discount (20%) because fixes overlap partially
projected_delta_realistic = projected_delta * 0.80
print(f"  5.1 Smart hard_cap:               +{total_51:.1f}%")
print(f"  5.2 Peak_protect restore:         {net_52:+.1f}%")
print(f"  5.3 Long-horizon carry:           +{est_53:.1f}%")
print(f"  5.4 Symbol tuning (post-overlap): +{est_54_net:.1f}%")
print(f"  5.5 Rule ensemble:                +{est_55:.1f}%")
print(f"  ─────────────────────────────────────────────")
print(f"  Gross delta:                      +{projected_delta:.1f}%")
print(f"  Realistic (20% interaction discount): +{projected_delta_realistic:.1f}%")
print(f"\n  V24 projection: +{v23_base:.1f}% + {projected_delta_realistic:.1f}% = +{v23_base + projected_delta_realistic:.1f}%")
print(f"  vs V19.1 baseline: +1866.8% → delta vs V19.1 = {v23_base + projected_delta_realistic - 1866.8:+.1f}%")
print(f"  vs Rule baseline:  +1982.6% → delta vs Rule  = {v23_base + projected_delta_realistic - 1982.6:+.1f}%")

# MaxLoss projection
# 5.1 reduces worst V23 losses (REE -18.58, AAS -18.18)
# New worst likely similar to V22/V19.1 range
print("\nMAX LOSS PROJECTION:")
v23_worst = v23.nsmallest(5, "pnl_pct")[["entry_symbol","entry_date","pnl_pct","exit_reason"]]
print(f"  V23 top-5 worst losses:\n{v23_worst.to_string(index=False)}")
# After 5.1, strong-trend hard_cap cases would behave like V19.1:
# REE 2022-05-11 V19.1 +11.88% (not -18.58), AAS 2021-04-08 V19.1 -12.46% (not -18.18)
# New worst would likely be AAV 2022-01-06 -16.31% (weak trend, kept) or similar
print(f"\n  Projected V24 max loss: ~ -16% to -17% (weak-trend hard_cap still fires, strong-trend softer)")

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
