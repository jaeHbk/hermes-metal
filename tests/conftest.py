"""Shared pytest fixtures.

Adds the repo root to sys.path so `from src.X import Y` works whether tests
are run from the repo root or from inside `tests/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
