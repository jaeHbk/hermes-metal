"""hermes-metal server client package.

Re-exports the public surface of ``client`` so callers can simply do::

    from src.server import HermesClient, HermesError
"""
from __future__ import annotations

from .client import HermesClient, HermesError

__all__ = ["HermesClient", "HermesError"]
