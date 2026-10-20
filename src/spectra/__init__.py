"""SPECTRA: Synthetic Plume Embedding in Cubes for Trace-gas Remote-sensing.

The package embeds simulated trace-gas plumes from large-eddy simulations into
real imaging-spectrometer observations (AVIRIS-3, AVIRIS-4, EMIT) and writes
the attenuated radiance back in the input's native file format, with the
embedding metadata attached. ``spectra.embed(run)`` runs one run file.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spectra")
except PackageNotFoundError:  # pragma: no cover - editable installs without metadata
    __version__ = "0.0.0"

# Imported after __version__, which the submodules read when they run.
from spectra.api import EmbedRun, embed
from spectra.contracts.errors import ContractError
from spectra.pipeline import embed_scene
from spectra.result import EmbedResult
from spectra.run import RunConfig, dump_run, load_run

__all__ = [
    "ContractError",
    "EmbedResult",
    "EmbedRun",
    "RunConfig",
    "__version__",
    "dump_run",
    "embed",
    "embed_scene",
    "load_run",
]
