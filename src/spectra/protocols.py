"""Structural protocols implemented by sensor adapters and other plugins.

The protocols are runtime-checkable and structural, so a third-party adapter
does not need to import SPECTRA to satisfy them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import xarray as xr

from spectra.types import (
    Placement,
    Platform,
    RadianceLeg,
    SceneRecord,
    SpectralResponse,
    ViewingGeometry,
    Window,
)


@runtime_checkable
class SensorAdapter(Protocol):
    """Discover scenes of one sensor and open them as ``SensorScene`` datasets."""

    name: str
    platform: Platform

    def list_scenes(
        self,
        scenes: str | Path | Sequence[Mapping[str, object]],
        **filters: object,
    ) -> list[SceneRecord]:
        """Return the scenes of an explicit scene table (CSV/YAML path or rows).

        Adapters never discover files from directory layouts; every file of
        every scene is listed by the user (see :mod:`spectra.sensors.table`).
        """
        ...

    def open_scene(
        self, record: SceneRecord, *, chunks: Mapping[str, int] | None = None
    ) -> xr.Dataset:
        """Open one scene lazily as a ``SensorScene`` dataset."""
        ...

    def quality_masks(self, scene: xr.Dataset, record: SceneRecord) -> xr.Dataset:
        """Return boolean ``valid``/``cloud``/``shadow`` masks on the scene grid."""
        ...

    def background_group(self, record: SceneRecord) -> str:
        """Return the leakage group of a scene."""
        ...


@runtime_checkable
class UnitAbsorptionProvider(Protocol):
    """Compute unit absorption spectra and absorption tables for a sensor.

    A provider whose forward model covers a limited wavelength range may
    expose it as an optional ``spectral_range_nm = (min, max)`` attribute;
    scene bands outside it are then passed through the embedding unattenuated.
    """

    name: str

    def spectrum(
        self,
        response: SpectralResponse,
        geometry: ViewingGeometry,
        *,
        enhancement_ppm: float,
        gas: str = "CH4",
        layer_height_m: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        """Return a unit absorption spectrum (``UnitAbsorptionSpectrum`` schema)."""
        ...

    def lut(
        self,
        response: SpectralResponse,
        geometry: ViewingGeometry,
        *,
        enhancements_ppm: Sequence[float],
        gas: str = "CH4",
        layer_height_m: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        """Return an absorption look-up table (``AbsorptionLut`` schema)."""
        ...


@runtime_checkable
class ForwardModel(Protocol):
    """Compute absorption look-up tables for the Lambert-Beer embedding.

    Scene bands outside :attr:`spectral_range_nm` pass the embedding
    unattenuated.
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
        """Raise ``ContractError`` if the model cannot handle ``gas``."""
        ...

    def lut(
        self, response: SpectralResponse, geometry: ViewingGeometry, *, gas: str
    ) -> xr.Dataset:
        """Return an absorption look-up table (``AbsorptionLut`` schema)."""
        ...

    def provenance(self) -> dict[str, Any]:
        """Return the settings recorded with every embedding."""
        ...


@runtime_checkable
class Regridder(Protocol):
    """Conservatively place a plume column on a sensor grid."""

    name: str

    def to_scene(
        self,
        column: xr.DataArray,
        scene: xr.Dataset,
        placement: Placement,
        *,
        window: Window | None = None,
    ) -> xr.DataArray:
        """Return the column in kg m-2 on the sensor grid, cropped to ``window``."""
        ...


__all__ = [
    "ForwardModel",
    "Regridder",
    "SensorAdapter",
    "UnitAbsorptionProvider",
]
