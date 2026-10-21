"""Structural protocols implemented by sensor adapters and forward models.

The protocols are runtime-checkable and structural, so a third-party adapter
does not need to import SPECTRA to satisfy them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import xarray as xr

from spectra.types import (
    NativeLayout,
    Platform,
    SpectralResponse,
    ViewingGeometry,
)


@runtime_checkable
class SensorAdapter(Protocol):
    """Open one sensor's native files as a ``SensorScene`` and locate its radiance.

    Adapters never discover files: the user lists every file by role.
    :attr:`SensorAdapter.LAYOUT` tells the copy-and-patch exporter where the
    radiance lives in the native file, so import and export share one
    dimension mapping.
    """

    name: str
    platform: Platform
    ROLES: Mapping[str, bool]
    LAYOUT: NativeLayout

    def open_scene(
        self,
        files: Mapping[str, str | Path],
        *,
        chunks: Mapping[str, int] | None = None,
    ) -> xr.Dataset:
        """Open one scene lazily as a ``SensorScene`` dataset.

        Args:
            files: File path by role (:attr:`SensorAdapter.ROLES`).
            chunks: Dask chunks in common dimension names.
        """
        ...


@runtime_checkable
class ForwardModel(Protocol):
    """Compute absorption look-up tables for the Lambert-Beer embedding.

    Scene bands outside :attr:`spectral_range_nm`, or outside the run's
    ``wavelength_ranges_nm`` when given, pass the embedding unattenuated.
    """

    name: str

    @property
    def spectral_range_nm(self) -> tuple[float, float]:
        """Wavelength range the model covers, in nanometres."""
        ...

    @property
    def enhancements_ppm(self) -> tuple[float, ...]:
        """Absorption-table nodes in ppm."""
        ...

    @property
    def layer_height_m(self) -> float:
        """Nominal lowest-layer height in metres."""
        ...

    def check(self, gas: str) -> None:
        """Raise if the model cannot handle ``gas``.

        Raises:
            :exc:`~spectra.contracts.errors.ContractError`: If it cannot.
        """
        ...

    def lut(
        self, response: SpectralResponse, geometry: ViewingGeometry, *, gas: str
    ) -> xr.Dataset:
        """Return an absorption look-up table (``AbsorptionLut`` schema)."""
        ...

    def provenance(self) -> dict[str, Any]:
        """Return the settings recorded with every embedding."""
        ...


__all__ = ["ForwardModel", "SensorAdapter"]
