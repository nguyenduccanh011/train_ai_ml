import json
import pandas as pd
R = json.load(open(r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\ohlcv\screen_results.json"))
rows = []
for cand, r in R["candidates"].items():
    for tgt in ("t_fwd21", "t_vexit"):
        raw, r10, r16 = r[f"{tgt}_raw"], r[f"{tgt}_resid_close10"], r[f"{tgt}_resid_strict16"]
        nb = r[f"{tgt}_null"]
        folds16 = [r16[f] for f in ("2022","2023","2024","2025","2026H1") if r16[f] is not None]
        same_sign = sum(1 for x in folds16 if x*(r16["pooled"] or 0) > 0)
        rows.append(dict(cand=cand, tgt=tgt, raw=raw["pooled"], raw_t=raw["t_cons"],
                         resid10=r10["pooled"], resid16=r16["pooled"], resid16_t=r16["t_cons"],
                         f22=r16["2022"], f23=r16["2023"], f24=r16["2024"], f25=r16["2025"], f26=r16["2026H1"],
                         nfold=len(folds16), sign_agree=same_sign,
                         null_hi=nb["null_hi"]))
df = pd.DataFrame(rows)
for tgt in ("t_fwd21", "t_vexit"):
    sub = df[df.tgt==tgt].copy()
    sub["absr16"] = sub.resid16.abs()
    sub = sub.sort_values("absr16", ascending=False)
    print("="*100); print(tgt)
    print(sub.drop(columns=["tgt","absr16"]).to_string(index=False))
nb = R["candidates"]["c_clv"]["t_fwd21_null"]
print("null validation (c_clv fwd21):", nb)
