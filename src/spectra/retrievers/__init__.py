"""Retrieval plugins turning scenes into column products.

Retrievers wrapping optional sensor packages are imported lazily on attribute
access so that importing this package never requires their extras.
"""

from __future__ import annotations

import importlib

from spectra.retrievers.toy import ToyRetriever

_LAZY_EXPORTS = {"Aviris4MatchedFilter": "spectra.retrievers.aviris4_mf"}


def __getattr__(name: str) -> object:
    """Import optional-dependency retrievers on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


__all__ = ["ToyRetriever"]
