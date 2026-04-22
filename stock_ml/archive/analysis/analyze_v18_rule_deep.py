import os
import numpy as np
import pandas as pd
from collections import defaultdict

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from compare_rule_vs_model import backtest_rule
from run_v18_compare import backtest_v18, calc_metrics

DEFAULT_SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"


def run_v18_collect(symbols_str):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    config = {
        "data": {"data_dir": data_dir},
        "split": {
            "method": "walk_forward",
            "train_years": 4,
            "test_years": 1,
            "gap_days": 0,
            "first_test_year": 2020,
            "last_test_year": 2025,
        },
        "target": {
            "type": "trend_regime",
            "trend_method": "dual_ma",
            "short_window": 5,
            "long_window": 20,
            "classes": 3,
        },
    }

    pick = [s.strip() for s in symbols_str.split(",") if s.strip()]
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    raw_df = loader.load_all(symbols=pick)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []
    counters_by_symbol = defaultdict(lambda: defaultdict(int))

    for _, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in pick:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            r = backtest_v18(
                y_pred,
                rets,
                sym_test,
                feature_cols,
                mod_a=True,
                mod_b=True,
                mod_c=False,
                mod_d=False,
                mod_e=True,
                mod_f=True,
                mod_g=True,
                mod_h=True,
                mod_i=True,
                mod_j=True,
            )

            for t in r["trades"]:
                t["symbol"] = sym
                all_trades.append(t)

            for k in (
                "n_vshape_entries",
                "n_peak_protect",
                "n_fast_loss_cut",
                "n_secondary_breakout",
                "n_bear_blocked",
                "n_chop_blocked",
                "n_confirmed_exit_blocked",
                "n_trend_carry_saved",
                "n_v18_relaxed_ret5_entries",
                "n_v18_relaxed_dp_entries",
                "n_v18_signal_quality_saves",
            ):
                counters_by_symbol[sym][k] += int(r.get(k, 0))

    return all_trades, counters_by_symbol


def run_rule_collect(symbols_str):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    pick = [s.strip() for s in symbols_str.split(",") if s.strip()]
    loader = DataLoader(data_dir)
    symbols = [s for s in pick if s in loader.symbols]

    raw_df = loader.load_all(symbols=symbols)
    all_trades = []
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        trades = backtest_rule(sym_test)
        for t in trades:
            t["symbol"] = sym
            all_trades.append(t)
    return all_trades


def to_df(trades):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades).copy()
    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    if "holding_days" not in df.columns:
        df["holding_days"] = (df["exit_date"] - df["entry_date"]).dt.days
    if "pnl_pct" not in df.columns:
        df["pnl_pct"] = 0.0
    return df


def overlap_windows(a_ent, a_ex, b_ent, b_ex):
    return not (a_ex < b_ent or b_ex < a_ent)


def analyze_symbol(sym, v18_sym, rule_sym):
    out = {}
    missed = []
    undercaptured = []
    for _, rr in rule_sym.iterrows():
        ov = v18_sym[(v18_sym["entry_date"] <= rr["exit_date"]) & (v18_sym["exit_date"] >= rr["entry_date"])]
        if ov.empty:
            missed.append(rr)
        else:
            cap = ov["pnl_pct"].sum()
            gap = float(rr["pnl_pct"] - cap)
            if rr["pnl_pct"] >= 8 and gap >= 8:
                row = rr.copy()
                row["v18_overlap_pnl"] = cap
                row["capture_gap"] = gap
                undercaptured.append(row)

    noisy_losses = v18_sym[(v18_sym["pnl_pct"] <= -5) & (v18_sym["holding_days"] <= 25)].copy()
    noisy_losses = noisy_losses.sort_values("pnl_pct")

    avoidable = []
    for _, vt in v18_sym.iterrows():
        ov = rule_sym[(rule_sym["entry_date"] <= vt["exit_date"]) & (rule_sym["exit_date"] >= vt["entry_date"])]
        if ov.empty:
            if vt["pnl_pct"] <= -5:
                r = vt.copy()
                r["rule_overlap_pnl"] = np.nan
                r["avoidable_gap"] = abs(float(vt["pnl_pct"]))
                avoidable.append(r)
        else:
            best_rule = float(ov["pnl_pct"].max())
            if vt["pnl_pct"] <= 0 and (best_rule - float(vt["pnl_pct"])) >= 8:
                r = vt.copy()
                r["rule_overlap_pnl"] = best_rule
                r["avoidable_gap"] = best_rule - float(vt["pnl_pct"])
                avoidable.append(r)

    out["missed_rule_big_wins"] = pd.DataFrame(missed)
    out["undercaptured_rule_wins"] = pd.DataFrame(undercaptured)
    out["v18_noisy_losses"] = noisy_losses
    out["v18_avoidable_losses"] = pd.DataFrame(avoidable)

    m_v18 = calc_metrics(v18_sym.to_dict("records")) if not v18_sym.empty else calc_metrics([])
    m_rule = calc_metrics(rule_sym.to_dict("records")) if not rule_sym.empty else calc_metrics([])
    out["metrics"] = {
        "symbol": sym,
        "v18_total": m_v18["total_pnl"],
        "rule_total": m_rule["total_pnl"],
        "gap_v18_minus_rule": m_v18["total_pnl"] - m_rule["total_pnl"],
        "v18_trades": m_v18["trades"],
        "rule_trades": m_rule["trades"],
        "v18_pf": m_v18["pf"],
        "rule_pf": m_rule["pf"],
    }
    out["missed_big_total"] = float(out["missed_rule_big_wins"].query("pnl_pct >= 8")["pnl_pct"].sum()) if not out["missed_rule_big_wins"].empty else 0.0
    out["undercapture_gap_total"] = float(out["undercaptured_rule_wins"]["capture_gap"].sum()) if not out["undercaptured_rule_wins"].empty else 0.0
    out["avoidable_gap_total"] = float(out["v18_avoidable_losses"]["avoidable_gap"].sum()) if not out["v18_avoidable_losses"].empty else 0.0

    return out


def top_rows(df, n=8, cols=None, sort_col=None, asc=True):
    if df is None or df.empty:
        return "(none)"
    d = df.copy()
    if sort_col and sort_col in d.columns:
        d = d.sort_values(sort_col, ascending=asc)
    if cols:
        keep = [c for c in cols if c in d.columns]
        d = d[keep]
    return d.head(n).to_string(index=False)


def main():
    symbols = DEFAULT_SYMBOLS
    out_dir_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    out_dir_docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(out_dir_results, exist_ok=True)
    os.makedirs(out_dir_docs, exist_ok=True)

    v18_trades, counters = run_v18_collect(symbols)
    rule_trades = run_rule_collect(symbols)

    v18_df = to_df(v18_trades)
    rule_df = to_df(rule_trades)

    v18_csv = os.path.join(out_dir_results, "v18_trades_20260420_detailed.csv")
    rule_csv = os.path.join(out_dir_results, "rule_trades_20260420_detailed.csv")
    v18_df.to_csv(v18_csv, index=False)
    rule_df.to_csv(rule_csv, index=False)

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

    analyses = {}
    metrics_rows = []
    for sym in symbol_list:
        v18_sym = v18_df[v18_df["symbol"] == sym].copy() if not v18_df.empty else pd.DataFrame()
        rule_sym = rule_df[rule_df["symbol"] == sym].copy() if not rule_df.empty else pd.DataFrame()
        a = analyze_symbol(sym, v18_sym, rule_sym)
        analyses[sym] = a
        metrics_rows.append(a["metrics"])

    metrics_df = pd.DataFrame(metrics_rows).sort_values("gap_v18_minus_rule")
    metrics_csv = os.path.join(out_dir_results, "v18_rule_symbol_metrics_20260420.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    gap_rows = []
    for sym in symbol_list:
        a = analyses[sym]
        gap_rows.append({
            "symbol": sym,
            "missed_rule_big_total": a["missed_big_total"],
            "undercapture_gap_total": a["undercapture_gap_total"],
            "avoidable_gap_total": a["avoidable_gap_total"],
            "combined_gap_proxy": a["missed_big_total"] + a["undercapture_gap_total"] + a["avoidable_gap_total"],
        })
    gap_df = pd.DataFrame(gap_rows).sort_values("combined_gap_proxy", ascending=False)
    gap_csv = os.path.join(out_dir_results, "v18_rule_gap_breakdown_20260420.csv")
    gap_df.to_csv(gap_csv, index=False)

    tcb = analyses["TCB"]
    weak_syms = metrics_df.head(6)["symbol"].tolist()

    reason_rows = []
    if not v18_df.empty and "exit_reason" in v18_df.columns:
        gb = v18_df.groupby(["symbol", "exit_reason"], as_index=False).agg(
            trades=("pnl_pct", "count"),
            total_pnl=("pnl_pct", "sum"),
            avg_pnl=("pnl_pct", "mean"),
            median_hold=("holding_days", "median"),
        )
        gb["total_pnl"] = gb["total_pnl"].round(2)
        gb["avg_pnl"] = gb["avg_pnl"].round(2)
        reason_rows = gb
        reason_rows.to_csv(os.path.join(out_dir_results, "v18_exit_reason_by_symbol_20260420.csv"), index=False)

    def sym_counter(sym, key):
        return int(counters.get(sym, {}).get(key, 0))

    report_path = os.path.join(out_dir_docs, "V18_RULE_DEEP_ANALYSIS_20260420.md")

    lines = []
    lines.append("# V18 vs Rule Deep Analysis (Run date: 2026-04-20)")
    lines.append("")

    m_v18 = calc_metrics(v18_df.to_dict("records"))
    m_rule = calc_metrics(rule_df.to_dict("records"))
    lines.append("## 1) Tong quan hieu nang (14 ma)")
    lines.append("")
    lines.append(f"- V18 tong: **{m_v18['total_pnl']:+.1f}%** ({m_v18['trades']} trades, PF {m_v18['pf']:.2f}, WR {m_v18['wr']:.1f}%)")
    lines.append(f"- Rule tong: **{m_rule['total_pnl']:+.1f}%** ({m_rule['trades']} trades, PF {m_rule['pf']:.2f}, WR {m_rule['wr']:.1f}%)")
    lines.append(f"- Chenh lech: **{m_v18['total_pnl'] - m_rule['total_pnl']:+.1f}%** (V18 {'thap hon' if m_v18['total_pnl'] < m_rule['total_pnl'] else 'cao hon'} Rule)")
    lines.append(f"- So ma V18 > Rule: **{int((metrics_df['gap_v18_minus_rule'] > 0).sum())}/{len(metrics_df)}**")
    lines.append(f"- So ma V18 < Rule: **{int((metrics_df['gap_v18_minus_rule'] < 0).sum())}/{len(metrics_df)}**")
    lines.append("")

    lines.append("## 2) TCB chi tiet (ma trong tam)")
    lines.append("")
    tcbm = tcb["metrics"]
    lines.append(f"- V18: **{tcbm['v18_total']:+.1f}%** ({tcbm['v18_trades']} trades, PF {tcbm['v18_pf']:.2f})")
    lines.append(f"- Rule: **{tcbm['rule_total']:+.1f}%** ({tcbm['rule_trades']} trades, PF {tcbm['rule_pf']:.2f})")
    lines.append(f"- Gap: **{tcbm['gap_v18_minus_rule']:+.1f}%**")
    lines.append("")

    lines.append("### 2.1 Nhip Rule kiem loi tot nhung V18 khong bat duoc / bat kem")
    lines.append("")
    tcb_missed_big = tcb["missed_rule_big_wins"]
    if not tcb_missed_big.empty:
        tcb_missed_big = tcb_missed_big[tcb_missed_big["pnl_pct"] >= 8].sort_values("pnl_pct", ascending=False)
    lines.append("```")
    lines.append(top_rows(
        tcb_missed_big,
        n=10,
        cols=["entry_date", "exit_date", "holding_days", "pnl_pct"],
        sort_col="pnl_pct",
        asc=False,
    ))
    lines.append("```")
    lines.append("")

    lines.append("### 2.2 Cac nhip Rule co overlap nhung V18 an qua it")
    lines.append("")
    lines.append("```")
    lines.append(top_rows(
        tcb["undercaptured_rule_wins"],
        n=10,
        cols=["entry_date", "exit_date", "holding_days", "pnl_pct", "v18_overlap_pnl", "capture_gap"],
        sort_col="capture_gap",
        asc=False,
    ))
    lines.append("```")
    lines.append("")

    lines.append("### 2.3 Giao dich V18 khong hieu qua gay thua lo / nhieu")
    lines.append("")
    lines.append("```")
    lines.append(top_rows(
        tcb["v18_avoidable_losses"],
        n=12,
        cols=["entry_date", "exit_date", "holding_days", "pnl_pct", "exit_reason", "rule_overlap_pnl", "avoidable_gap", "entry_trend", "entry_ret_5d", "entry_dist_sma20", "quick_reentry", "breakout_entry"],
        sort_col="avoidable_gap",
        asc=False,
    ))
    lines.append("```")
    lines.append("")

    lines.append("### 2.4 Counter module V18 tren TCB")
    lines.append("")
    lines.append(f"- relaxed_ret5_entries: **{sym_counter('TCB', 'n_v18_relaxed_ret5_entries')}**")
    lines.append(f"- relaxed_dp_entries: **{sym_counter('TCB', 'n_v18_relaxed_dp_entries')}**")
    lines.append(f"- signal_quality_saves: **{sym_counter('TCB', 'n_v18_signal_quality_saves')}**")
    lines.append(f"- chop_blocked: **{sym_counter('TCB', 'n_chop_blocked')}**")
    lines.append(f"- bear_blocked: **{sym_counter('TCB', 'n_bear_blocked')}**")
    lines.append("")

    lines.append("## 3) Cac ma khac co van de tuong tu")
    lines.append("")
    for sym in weak_syms:
        a = analyses[sym]
        m = a["metrics"]
        lines.append(f"### {sym} (gap {m['gap_v18_minus_rule']:+.1f}%)")
        lines.append(f"- V18 {m['v18_total']:+.1f}% vs Rule {m['rule_total']:+.1f}%")
        lines.append(f"- Missed big-win proxy: {a['missed_big_total']:+.1f}%")
        lines.append(f"- Undercapture gap proxy: {a['undercapture_gap_total']:+.1f}%")
        lines.append(f"- Avoidable loss proxy: {a['avoidable_gap_total']:+.1f}%")
        lines.append("- Mau giao dich van de:")
        lines.append("```")
        lines.append(top_rows(
            a["v18_avoidable_losses"],
            n=6,
            cols=["entry_date", "exit_date", "holding_days", "pnl_pct", "exit_reason", "rule_overlap_pnl", "avoidable_gap", "entry_trend", "entry_ret_5d", "entry_dist_sma20", "breakout_entry"],
            sort_col="avoidable_gap",
            asc=False,
        ))
        lines.append("```")
        lines.append("")

    lines.append("## 4) Root cause (du lieu + logic)")
    lines.append("")
    lines.append("1. V18 da cai thien entry quality (PF tang) nhung van thua Rule o nhom ma can momentum tiep dien (AAV, DGC, AAS, MBB, TCB, BID, VND).")
    lines.append("2. Nhieu trade lo cua V18 tap trung o exit_reason `signal`; vao trend weak/moderate va hold ngan -> de bi whipsaw.")
    lines.append("3. O cac nhom gap am lon, Rule thuong giu duoc 1 swing dai; V18 chia thanh nhieu lenh ngan hoac vao sau/ra som nen tong payoff kem.")
    lines.append("4. Counter cho thay V18 co no luc mo khoa (`relaxed_ret5`, `relaxed_dp`) nhung chua du de bat cac pha tang manh keo dai o mot so ma.")
    lines.append("")

    lines.append("## 5) De xuat cai tien de dot pha va on dinh hon")
    lines.append("")
    lines.append("1. Two-stage entry policy: tach `entry_alpha` (cho phep vao) va `size_policy` (scale size theo overheat) thay vi block cung anti-chop/ret5.")
    lines.append("2. Symbol-regime adapter: hoc threshold rieng theo nhom volatility/chop (bank, beta cao, phong thu) cho ret5/dp/exit confirm.")
    lines.append("3. Exit quality model: bo sung classifier nho cho `signal_exit` de phan biet pullback vs trend-break (feature: MA slope, MACD hist delta, volume shock, ATR expansion).")
    lines.append("4. Trade stitching: neu co re-entry trong <=7 ngay cung trend, danh gia theo campaign-level PnL de phat hien over-trading va toi uu cooldown.")
    lines.append("5. Objective tuning theo payoff asymmetry: toi uu `total_pnl + PF - lambda*avoidable_gap` thay vi toi uu WR don le.")
    lines.append("6. Offline counterfactual: replay lai cac window Rule-win-lon de tim nguong gay miss lon nhat (ret5 cap, dp cap, confirm bars).")
    lines.append("")

    lines.append("## 6) File bang chung")
    lines.append("")
    lines.append("- `results/v18_trades_20260420_detailed.csv`")
    lines.append("- `results/rule_trades_20260420_detailed.csv`")
    lines.append("- `results/v18_rule_symbol_metrics_20260420.csv`")
    lines.append("- `results/v18_rule_gap_breakdown_20260420.csv`")
    lines.append("- `results/v18_exit_reason_by_symbol_20260420.csv`")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote report: {report_path}")
    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote gap: {gap_csv}")


if __name__ == "__main__":
    main()
