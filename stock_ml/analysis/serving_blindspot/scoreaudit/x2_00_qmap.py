"""x2_00: quantile map cho kênh exit2 (reward_risk h10) — chọn mức z_threshold sweep.

Nguồn: bundles/bundle_n2_3h_brk05_z07_2025-01-01_wf/prediction_history.parquet
(exit_score = reward_risk_regression h10, cùng head audit sa04 đã demo).
Tính causal z 252/60 per-symbol (đúng _causal_zscore_by_symbol) rồi:
  - quantile z theo năm (q90/95/97/98/99)
  - % bar vượt các mức z ứng viên theo năm
Mức sweep chọn sao cho 2024-26 có mật độ sell hợp lý (~0.5-3% bar) và 2020-21 không bùng nổ.
"""
from __future__ import annotations
import pandas as pd

SRC = "f:/PROJECTS/train_ai_ml/bundles/bundle_n2_3h_brk05_z07_2025-01-01_wf/prediction_history.parquet"
W, MP = 252, 60


def causal_z(s: pd.Series, g: pd.Series) -> pd.Series:
    def _cz(x: pd.Series) -> pd.Series:
        m = x.rolling(W, min_periods=MP).mean()
        sd = x.rolling(W, min_periods=MP).std()
        return (x - m) / (sd + 1e-12)
    return s.groupby(g, sort=False).transform(_cz)


def main():
    df = pd.read_parquet(SRC, columns=["symbol", "date", "exit_score"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["z"] = causal_z(df["exit_score"], df["symbol"])
    df = df.dropna(subset=["z"])
    df["year"] = df["date"].dt.year

    qs = [0.90, 0.95, 0.97, 0.98, 0.99]
    thr = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    rows = []
    for y, gr in df.groupby("year"):
        r = {"year": y, "n": len(gr)}
        for q in qs:
            r[f"q{int(q*100)}"] = round(float(gr["z"].quantile(q)), 2)
        for t in thr:
            r[f"pct>{t}"] = round(100 * float((gr["z"] > t).mean()), 2)
        rows.append(r)
    out = pd.DataFrame(rows)
    # pooled
    r = {"year": "ALL", "n": len(df)}
    for q in qs:
        r[f"q{int(q*100)}"] = round(float(df["z"].quantile(q)), 2)
    for t in thr:
        r[f"pct>{t}"] = round(100 * float((df["z"] > t).mean()), 2)
    out = pd.concat([out, pd.DataFrame([r])], ignore_index=True)
    print(out.to_string(index=False))
    out.to_csv("f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/scoreaudit/x2_00_qmap.csv", index=False)


if __name__ == "__main__":
    main()
