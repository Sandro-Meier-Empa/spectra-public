"""Structural protocols implemented by sensor adapters and other plugins.

The protocols are runtime-checkable and structural, so a third-party adapter
does not need to import SPECTRA to satisfy them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import xarray as xr

from spectra.types import (
    MapGrid,
    Placement,
    Platform,
    RadianceLeg,
    RetrievalSettings,
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
class SupportsProduct(Protocol):
    """Optional adapter capability: open a retrieved column product."""

    def open_product(self, record: SceneRecord) -> xr.Dataset:
        """Open the retrieved column product of a scene (``ColumnProduct`` contract)."""
        ...


@runtime_checkable
class UnitAbsorptionProvider(Protocol):
    """Compute unit absorption spectra and absorption tables for a sensor."""

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
class Retriever(Protocol):
    """Retrieve a column product from a sensor scene."""

    name: str

    def retrieve(
        self,
        scene: xr.Dataset,
        target: xr.Dataset,
        *,
        settings: RetrievalSettings,
        background_stats: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Return a ``ColumnProduct`` in sensor geometry with uncertainties."""
        ...

    def background_statistics(
        self, scene: xr.Dataset, *, settings: RetrievalSettings
    ) -> xr.Dataset:
        """Return cacheable background statistics of a plume-free scene."""
        ...


@runtime_checkable
class Regridder(Protocol):
    """Conservatively place a plume column on a sensor or map grid."""

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

    def to_map(
        self, column: xr.DataArray, grid: MapGrid, placement: Placement
    ) -> xr.DataArray:
        """Return the column in kg m-2 on a regular map grid."""
        ...


__all__ = [
    "Regridder",
    "Retriever",
    "SensorAdapter",
    "SupportsProduct",
    "UnitAbsorptionProvider",
]
