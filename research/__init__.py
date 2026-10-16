"""Research code (data generation, training, evaluation).

Makes the shared inference package `src/docforge` importable when scripts are run as
`python -m research.<module>` from the repository root (no install step needed).
"""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
