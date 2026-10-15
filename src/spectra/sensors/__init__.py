"""Sensor adapter plugins.

Adapters wrapping optional sensor packages are imported lazily on attribute
access so that importing this package never requires their extras.
"""

from __future__ import annotations

import importlib

from spectra.sensors.toy import ToySensorAdapter

_LAZY_EXPORTS = {"Aviris4Adapter": "spectra.sensors.aviris4"}


def __getattr__(name: str) -> object:
    """Import optional-dependency adapters on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


__all__ = ["ToySensorAdapter"]
