"""Exit rewrite: red@s2 -> early-cut s3; green@s2 -> green-trail giveback."""
from __future__ import annotations

import pandas as pd


def rewrite(base_trades: pd.DataFrame, CLO, DIDX, INV, gt: float) -> pd.DataFrame:
    ned, nep, nreason = [], [], []
    for r in base_trades.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei:
            ned.append(r.exit_date); nep.append(r.exit_price); nreason.append(r.exit_reason); continue
        c = CLO[r.symbol]
        if xi > ei + 3 and c[ei + 2] / r.entry_price - 1.0 < 0.0:               # red@s2 -> early-cut s3
            ned.append(INV[r.symbol][ei + 3]); nep.append(float(c[ei + 3])); nreason.append("early_cut")
        else:                                                                  # green@s2 -> green-trail
            ek, ep_, rs, peak = xi, r.exit_price, r.exit_reason, c[ei]
            for b in range(ei + 2, xi + 1):
                peak = max(peak, c[b])
                if c[b] <= peak * (1 - gt):
                    ek, ep_ = (b + 1, float(c[b + 1])) if b + 1 <= xi else (b, float(c[b]))
                    rs = "green_trail"; break
            ned.append(INV[r.symbol][ek]); nep.append(ep_); nreason.append(rs)
    o = base_trades.copy(); o["exit_date"] = pd.to_datetime(ned); o["exit_price"] = nep; o["exit_reason"] = nreason
    return o
