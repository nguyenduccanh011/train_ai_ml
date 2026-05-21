from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tools.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import serve_train61_model as server  # noqa: E402
from model_registry import DEFAULT_MODEL, MODELS, get_model_cfg, model_availability  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train61 background worker commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-id", default=os.getenv("TRAIN61_DEFAULT_MODEL", DEFAULT_MODEL))
    common.add_argument("--symbols", default="", help="Comma-separated symbols or JSON file path")
    common.add_argument("--force", action="store_true", help="Regenerate even when cache is fresh")
    common.add_argument("--limit", type=int, default=0, help="Limit number of symbols processed")

    subparsers.add_parser("compute-symbol-signal", parents=[common])
    subparsers.add_parser("compute-universe-signals", parents=[common])
    subparsers.add_parser("run-once-after-ingest", parents=[common])
    subparsers.add_parser("list-models")

    return parser.parse_args()


def load_symbols(raw: str, model_cfg: dict[str, Any]) -> list[str]:
    raw = str(raw or "").strip()
    if raw:
        path = Path(raw)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("symbols", payload if isinstance(payload, list) else [])
            return sorted({str(sym).upper() for sym in values if str(sym).strip()})
        return sorted({part.strip().upper() for part in raw.split(",") if part.strip()})
    return server._load_symbols_for_model(model_cfg)


def cache_is_fresh(model_id: str, symbol: str, model_cfg: dict[str, Any]) -> bool:
    path = server._signal_cache_path(symbol, model_id)
    if not path.exists():
        return False
    try:
        payload = server._normalize_cached_payload_for_model(model_id, server._read_json(path))
    except Exception:
        return False
    return server._payload_is_fresh(payload, symbol, model_cfg=model_cfg)


def compute_symbol_signal(model_id: str, symbol: str, *, force: bool = False) -> dict[str, Any]:
    model_cfg = get_model_cfg(model_id)
    symbol = symbol.upper()
    if not force and cache_is_fresh(model_id, symbol, model_cfg):
        payload = server._normalize_cached_payload_for_model(
            model_id,
            server._read_json(server._signal_cache_path(symbol, model_id)),
        )
        return {
            "symbol": symbol,
            "status": "cached",
            "latest_bar_date": payload.get("latest_bar_date"),
        }

    server._load_ohlcv(symbol, model_cfg=model_cfg)
    started = time.time()
    payload = server._generate_signal_for_model(model_id, symbol)
    payload = server._sync_live_position_state(model_id, symbol, payload)
    server._write_json(server._signal_cache_path(symbol, model_id), payload)
    elapsed = round(time.time() - started, 2)
    return {
        "symbol": symbol,
        "status": "generated",
        "elapsed_seconds": elapsed,
        "latest_bar_date": payload.get("latest_bar_date"),
        "data_version_hash": payload.get("data_version_hash"),
    }


def compute_universe(model_id: str, symbols: list[str], *, force: bool = False) -> dict[str, Any]:
    summary = {
        "model_id": model_id,
        "symbol_count": len(symbols),
        "generated": 0,
        "cached": 0,
        "errors": 0,
        "results": [],
    }
    for idx, symbol in enumerate(symbols, start=1):
        try:
            result = compute_symbol_signal(model_id, symbol, force=force)
            summary["results"].append(result)
            if result["status"] == "generated":
                summary["generated"] += 1
            else:
                summary["cached"] += 1
            print(f"[{idx}/{len(symbols)}] {symbol} {result['status']}")
        except Exception as exc:
            summary["errors"] += 1
            error = {"symbol": symbol, "status": "error", "error": str(exc)}
            summary["results"].append(error)
            print(f"[{idx}/{len(symbols)}] {symbol} error: {exc}")
    return summary


def list_models() -> None:
    for model_id, cfg in MODELS.items():
        availability = model_availability(model_id)
        print(
            json.dumps(
                {
                    "id": model_id,
                    "type": cfg.get("type"),
                    "label": cfg.get("label", model_id),
                    "available": availability["available"],
                    "missing": availability["missing"],
                },
                ensure_ascii=False,
            )
        )


def main() -> int:
    args = parse_args()
    if args.command == "list-models":
        list_models()
        return 0

    model_cfg = get_model_cfg(args.model_id)
    symbols = load_symbols(args.symbols, model_cfg)
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if args.command == "compute-symbol-signal" and len(symbols) != 1:
        raise SystemExit("compute-symbol-signal requires exactly one symbol via --symbols")

    summary = compute_universe(args.model_id, symbols, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
