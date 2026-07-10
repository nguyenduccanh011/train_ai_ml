"""un01: dry-run offline UNION 2 nguon entry (gb_x08 ML vs fc_rule2 price-gate).

Cau hoi: (1) rule entries khong trung gb theo nam (exact + fuzzy +/-5d);
(2) trong so do, bao nhieu roi vao luc slot gb (cung ma) DANG RANH (kha nang additive that);
(3) displacement: additive fill se chan gb entry nao trong khoang hold (proxy = hold cua rule);
(4) per-year gb entries/occupancy de doi chieu "gb doi tin hieu 2024-26".
"""
import pandas as pd
import numpy as np

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
gb = pd.read_csv(BASE + r"\exitmap\gbx08_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
ru = pd.read_csv(BASE + r"\pureml\fc_rule2_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
print(f"gb trades {len(gb)}  rule trades {len(ru)}")

gb["year"] = gb["entry_date"].dt.year
ru["year"] = ru["entry_date"].dt.year

print("\n=== entries per year ===")
py = pd.DataFrame({"gb": gb.groupby("year").size(), "rule": ru.groupby("year").size()}).fillna(0).astype(int)
print(py.to_string())

# --- overlap exact / fuzzy ---
gbk = set(zip(gb["symbol"], gb["entry_date"]))
ru["ov_exact"] = [ (s, d) in gbk for s, d in zip(ru["symbol"], ru["entry_date"]) ]
# fuzzy +/-5 calendar days
gb_sym = {s: g["entry_date"].values for s, g in gb.groupby("symbol")}
def fuzzy(s, d):
    a = gb_sym.get(s)
    if a is None: return False
    return bool((np.abs((a - np.datetime64(d)) / np.timedelta64(1, "D")) <= 5).any())
ru["ov_fuzzy"] = [fuzzy(s, d) for s, d in zip(ru["symbol"], ru["entry_date"])]

# --- slot-free: rule entry_date NOT inside any gb position [entry, exit] (+4d cooldown) for same symbol ---
gb_iv = {s: list(zip(g["entry_date"], g["exit_date"])) for s, g in gb.groupby("symbol")}
COOL = pd.Timedelta(days=6)  # ~4 phien
def slot_busy(s, d):
    for e, x in gb_iv.get(s, []):
        if e <= d <= x + COOL:
            return True
    return False
ru["gb_slot_busy"] = [slot_busy(s, d) for s, d in zip(ru["symbol"], ru["entry_date"])]

ru["additive"] = (~ru["ov_fuzzy"]) & (~ru["gb_slot_busy"])

print("\n=== rule vs gb per year ===")
agg = ru.groupby("year").agg(
    n_rule=("symbol", "size"),
    ov_exact=("ov_exact", "sum"),
    ov_fuzzy=("ov_fuzzy", "sum"),
    slot_busy=("gb_slot_busy", "sum"),
    additive=("additive", "sum"),
    additive_pnl=("pnl_pct", lambda s: s[ru.loc[s.index, "additive"]].sum()),
)
agg["pct_additive"] = (agg["additive"] / agg["n_rule"] * 100).round(1)
print(agg.to_string())
print(f"\nTOTAL rule {len(ru)}  ov_exact {ru.ov_exact.sum()} ({ru.ov_exact.mean()*100:.1f}%)  "
      f"ov_fuzzy {ru.ov_fuzzy.sum()} ({ru.ov_fuzzy.mean()*100:.1f}%)  "
      f"additive {ru.additive.sum()}  additive_pnl {ru.loc[ru.additive,'pnl_pct'].sum():.2f}")

# --- displacement risk: additive rule fill [entry, entry+hold] chua gb entry nao (se bi chan) ---
add = ru[ru["additive"]]
disp_rows = []
for _, r in add.iterrows():
    for e, x in gb_iv.get(r["symbol"], []):
        if r["entry_date"] < e <= r["exit_date"]:
            gpnl = gb[(gb["symbol"] == r["symbol"]) & (gb["entry_date"] == e)]["pnl_pct"].iloc[0]
            disp_rows.append(dict(symbol=r["symbol"], rule_entry=r["entry_date"], rule_pnl=r["pnl_pct"],
                                  gb_entry=e, gb_pnl=gpnl, year=r["year"]))
disp = pd.DataFrame(disp_rows)
print(f"\n=== displacement: additive fills chan gb entry (proxy hold=rule hold) ===")
if len(disp):
    d = disp.groupby("year").agg(n=("gb_pnl", "size"), gb_pnl_blocked=("gb_pnl", "sum"),
                                 rule_pnl_taken=("rule_pnl", "sum"))
    print(d.to_string())
    print(f"TOTAL blocked {len(disp)} gb entries, gb_pnl_blocked {disp.gb_pnl.sum():.2f} "
          f"vs rule_pnl cua cac fill chan {disp.drop_duplicates(['symbol','rule_entry']).rule_pnl.sum():.2f}")
else:
    print("none")

# --- additive theo nam, thong ke pnl ---
print("\n=== additive fills: pnl stats per year (pnl theo exit stack CU cua rule — proxy) ===")
st = add.groupby("year")["pnl_pct"].agg(["size", "sum", "mean", "median"])
print(st.round(4).to_string())

# --- gb occupancy per year (so ngay co >=1 position / so ngay giao dich, per-symbol slot view:
#     dung tong symbol-days held / (n_symbols_traded * trading days) khong can — chi can book-level:
#     concurrent positions trung binh & p50/p90 theo nam ---
days = pd.date_range(gb["entry_date"].min(), gb["exit_date"].max(), freq="B")
cnt = pd.Series(0, index=days)
for _, r in gb.iterrows():
    cnt[(cnt.index >= r["entry_date"]) & (cnt.index <= r["exit_date"])] += 1
occ = cnt.groupby(cnt.index.year).agg(["mean", "median", lambda s: (s >= 25).mean()])
occ.columns = ["mean_pos", "med_pos", "pct_days_ge25"]
print("\n=== gb concurrent positions per year (book khong cap; >=25 = K25 day) ===")
print(occ.round(2).to_string())

add_out = add[["symbol", "entry_date", "exit_date", "pnl_pct", "exit_reason", "year"]]
add_out.to_csv(BASE + r"\combo\un01_additive_fills.csv", index=False)
print(f"\nsaved {len(add_out)} additive fills -> combo\\un01_additive_fills.csv")
