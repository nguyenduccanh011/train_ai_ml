#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve visualization pages with leaderboard access.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Serve from stock_ml root so "/results/..." resolves to stock_ml/results.
    os.chdir(PROJECT_ROOT)

    # Try to bind to port, fallback to next available if in use
    port = args.port
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
                print(f"Serving project root at http://localhost:{port}")
                print(f"Leaderboard: http://localhost:{port}/visualization/leaderboard.html")
                print(f"Dashboard:   http://localhost:{port}/visualization/dashboard.html")
                httpd.serve_forever()
                return
        except OSError as e:
            if attempt < max_attempts - 1:
                print(f"Port {port} in use, trying {port + 1}...")
                port += 1
            else:
                print(f"Error: Could not bind to any port. {e}")
                raise


if __name__ == "__main__":
    main()
