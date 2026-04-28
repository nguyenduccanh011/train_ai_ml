"""Entry point: python -m stock_ml <command> ..."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.cli import main

main()
