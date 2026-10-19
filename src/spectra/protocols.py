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
class Retriever(Protocol):
    """Retrieve a column product from a sensor scene.

    The scene is any ``SensorScene`` (an embedded window or full scene of any
    sensor); the retriever selects the bands it needs with
    ``settings.band_windows_nm``. Wrappers of external matched filters can
    convert ppm-m outputs with
    :func:`spectra.retrievers.convert.matched_filter_to_column_product`.
    """

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


@runtime_checkable
class Exporter(Protocol):
    """Shape an embedded sample for a downstream consumer and write it.

    ``needs_retrieval`` tells the generator whether the exporter consumes a
    retrieved ``ColumnProduct`` (``True``) or the embedded ``SensorScene``
    (``False``).
    """

    name: str
    needs_retrieval: bool

    def prepare(
        self,
        sample: xr.Dataset,
        *,
        gas: str = "ch4",
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        """Return the dataset to write, with ``extra_attrs`` merged in."""
        ...

    def write(
        self, dataset: xr.Dataset, path: str | Path, *, overwrite: bool = False
    ) -> Path:
        """Write a prepared dataset and return its path."""
        ...


__all__ = [
    "Exporter",
    "Regridder",
    "Retriever",
    "SensorAdapter",
    "SupportsProduct",
    "UnitAbsorptionProvider",
]
