"""resolved.json self-sufficient export record (ENGINE_UPGRADE §11.5.1 / R1, G1 write-only).

Guards that the export record actually captures the pieces ``config.json`` cannot: all 227 resolved
EngineConfig nodes (the 162-188 wheel-default ones), the causal z-window resolved from its literal
default, the DSL catalog fingerprint (feature semantics), the real wheel version, and a DATA
DECLARATION (symbols + as-of, never bytes).
"""

from __future__ import annotations

from types import SimpleNamespace

from stock_ml.src.backtest.engine import engine_config_from_dict
from stock_ml.src.serving.resolved import build_resolved_config


def _cfg(engine: dict) -> SimpleNamespace:
    return SimpleNamespace(
        engine=engine, feature_set="leading_v2", entry_features=None, exit_features=None
    )


def test_resolved_captures_all_227_engine_nodes() -> None:
    cfg = _cfg({"max_hold_bars": 20, "hard_stop_pct": -0.08})
    engine, _ = engine_config_from_dict(cfg.engine)
    r = build_resolved_config(
        cfg, engine, universe=["FPT", "SSI"], universe_slug="vn_x", as_of="2024-01-01"
    )

    assert r["schema"] == "resolved.json/1"
    assert len(r["engine"]) == 227, (
        "resolved.json must pin every EngineConfig node, not just the set ones"
    )
    # A node NOT set by the config (inherited default) must still be present — that is the whole point.
    assert "trailing_atr_mult" in r["engine"]
    assert r["catalog_fingerprint"] and len(r["catalog_fingerprint"]) >= 8
    assert r["wheel_version"] and r["wheel_version"] != "unknown"


def test_recombine_zwindow_resolved_from_default_and_override() -> None:
    # Default: absent in config -> the 252/60 literal recombine uses is recorded.
    r_def = build_resolved_config(*_engine(_cfg({"max_hold_bars": 20, "hard_stop_pct": None})))
    assert r_def["recombine"]["z_norm_window"] == 252
    assert r_def["recombine"]["z_norm_min_periods"] == 60
    # Override: a config that sets it is recorded resolved (not the literal).
    r_ovr = build_resolved_config(
        *_engine(_cfg({"max_hold_bars": 20, "hard_stop_pct": None, "z_norm_window": 300}))
    )
    assert r_ovr["recombine"]["z_norm_window"] == 300


def test_data_is_declared_not_embedded() -> None:
    cfg = _cfg({"max_hold_bars": 20, "hard_stop_pct": None})
    engine, _ = engine_config_from_dict(cfg.engine)
    r = build_resolved_config(
        cfg, engine, universe=["FPT", "SSI", "HPG"], universe_slug="s", as_of="2024-06-30"
    )
    dep = r["data_dependencies"]
    assert dep["universe"] == {"slug": "s", "as_of": "2024-06-30", "n_symbols": 3}
    assert "VNINDEX" in dep["market_context"]
    # No embedded bytes/paths — only declarations.
    assert "path" not in dep and "checksum" not in dep


def _engine(cfg: SimpleNamespace) -> tuple[SimpleNamespace, object]:
    engine, _ = engine_config_from_dict(cfg.engine)
    return cfg, engine
