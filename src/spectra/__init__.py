"""SPECTRA: Synthetic Plume Embedding in Cubes for Trace-gas Remote-sensing.

The package embeds simulated trace-gas plumes from large-eddy simulations into
real imaging-spectrometer observations and generates synthetic segmentation
datasets independently of the sensor.
"""

from importlib.metadata import PackageNotFoundError, version

from spectra.contracts.errors import (
    ContractError,
    MissingExtraError,
    PluginError,
    RegridderUnavailableError,
)

try:
    __version__ = version("spectra")
except PackageNotFoundError:  # pragma: no cover - editable installs without metadata
    __version__ = "0.0.0"

__all__ = [
    "ContractError",
    "MissingExtraError",
    "PluginError",
    "RegridderUnavailableError",
    "__version__",
]
