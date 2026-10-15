"""Shared dataclasses used by the contracts, protocols and plugins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pyproj
import xarray as xr

from spectra.contracts.errors import ContractError

Platform = Literal["airborne", "spaceborne"]
RadianceLeg = Literal["solar", "view", "both"]
AnchorCorner = Literal["lower left", "lower right", "upper left", "upper right"]
Flip = Literal["none", "x", "y"]
StatisticsScope = Literal["window", "full_scene", "cached_scene"]
GeometrySpace = Literal["sensor", "map"]

ANCHOR_CORNERS: tuple[AnchorCorner, ...] = (
    "lower left",
    "lower right",
    "upper left",
    "upper right",
)
FLIPS: tuple[Flip, ...] = ("none", "x", "y")


@dataclass(frozen=True)
class SceneRecord:
    """One scene known to a sensor adapter.

    Attributes:
        sensor: Registry name of the adapter that produced the record.
        scene_id: Identifier unique within the sensor.
        platform: Airborne or spaceborne.
        acquisition_time: UTC acquisition time.
        background_group: Leakage group shared by scenes that must not be split
            between training and evaluation.
        uris: Role to URI mapping, for example ``radiance``, ``metadata``,
            ``shadow``, ``cloud``, ``product``.
        center_lonlat: Scene centre in degrees.
        bbox_lonlat: West, south, east, north bounds in degrees, if known.
        has_radiance: Whether a radiance cube is available.
        has_product: Whether a retrieved column product is available.
        extra: Sensor-specific metadata such as flight line or orbit.
    """

    sensor: str
    scene_id: str
    platform: Platform
    acquisition_time: datetime
    background_group: str
    uris: Mapping[str, str]
    center_lonlat: tuple[float, float]
    bbox_lonlat: tuple[float, float, float, float] | None = None
    has_radiance: bool = True
    has_product: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewingGeometry:
    """Observation geometry used by unit-absorption providers.

    Attributes:
        sza: Solar zenith angle in degrees.
        saa: Solar azimuth angle in degrees, clockwise from north.
        vza: Viewing zenith angle in degrees.
        vaa: Viewing azimuth angle in degrees, clockwise from north.
        surface_elevation: Surface elevation in metres above sea level.
        sensor_altitude: Sensor altitude in metres above sea level.
        lat: Reference latitude in degrees.
        lon: Reference longitude in degrees.
        time_utc: Reference UTC time.
        platform: Airborne or spaceborne.
        h2o_column: Column water vapour in g cm-2, if known.
    """

    sza: float
    saa: float
    vza: float
    vaa: float
    surface_elevation: float
    sensor_altitude: float
    lat: float
    lon: float
    time_utc: np.datetime64
    platform: Platform
    h2o_column: float | None = None

    def geometric_amf(self, leg: RadianceLeg = "both") -> float:
        """Return the geometric air-mass factor of the selected photon path."""
        from spectra.geometry.amf import path_amf

        return path_amf(self.sza, self.vza, leg)

    @classmethod
    def from_scene(
        cls,
        scene: xr.Dataset,
        *,
        window: Window | None = None,
        reducer: Literal["median", "mean", "center"] = "median",
    ) -> ViewingGeometry:
        """Summarise the geometry variables of a sensor scene.

        Args:
            scene: Dataset following the ``SensorScene`` contract.
            window: Optional index window the geometry is reduced over.
            reducer: Statistic applied to two-dimensional geometry fields.

        Returns:
            Scalar geometry for the window.
        """
        subset = scene if window is None else scene.isel(window.indexers)

        def reduce(name: str) -> float:
            variable = subset[name]
            if variable.ndim == 0:
                return float(variable.values)
            values = np.asarray(variable.values, dtype=float)
            if reducer == "median":
                return float(np.nanmedian(values))
            if reducer == "mean":
                return float(np.nanmean(values))
            centre = tuple(size // 2 for size in values.shape)
            return float(values[centre])

        time = subset["time"]
        time_values = np.asarray(time.values).reshape(-1)
        finite = time_values[~np.isnat(time_values)]
        if finite.size == 0:
            raise ContractError("Scene time coordinate contains no valid value.")
        reference_time = finite[finite.size // 2]
        h2o = reduce("h2o_column") if "h2o_column" in subset else None
        return cls(
            sza=reduce("sza"),
            saa=reduce("saa"),
            vza=reduce("vza"),
            vaa=reduce("vaa"),
            surface_elevation=reduce("surface_elevation"),
            sensor_altitude=reduce("sensor_altitude"),
            lat=reduce("lat"),
            lon=reduce("lon"),
            time_utc=np.datetime64(reference_time),
            platform=str(scene.attrs["platform"]),  # type: ignore[arg-type]
            h2o_column=h2o,
        )


@dataclass(frozen=True, eq=False)
class SpectralResponse:
    """Band centres and widths of an imaging spectrometer.

    Attributes:
        wavelength: Band centre wavelengths in nanometres.
        fwhm: Full widths at half maximum in nanometres.
        shape: Slit function shape; only ``gaussian`` is supported.
    """

    wavelength: np.ndarray
    fwhm: np.ndarray
    shape: Literal["gaussian"] = "gaussian"

    def __post_init__(self) -> None:
        """Validate the band arrays."""
        wavelength = np.asarray(self.wavelength, dtype=float)
        fwhm = np.asarray(self.fwhm, dtype=float)
        if wavelength.ndim != 1 or wavelength.shape != fwhm.shape:
            raise ContractError(
                "wavelength and fwhm must be one-dimensional and equal."
            )
        if np.any(np.diff(wavelength) <= 0):
            raise ContractError("wavelength must be strictly increasing.")
        if np.any(fwhm <= 0):
            raise ContractError("fwhm must be positive.")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "fwhm", fwhm)

    @property
    def size(self) -> int:
        """Return the number of bands."""
        return int(self.wavelength.size)

    def select(self, windows: Sequence[tuple[float, float]]) -> np.ndarray:
        """Return indices of bands whose centres fall inside the given ranges.

        Args:
            windows: Inclusive wavelength ranges in nanometres.

        Returns:
            Sorted band indices.
        """
        mask = np.zeros(self.size, dtype=bool)
        for lower, upper in windows:
            mask |= (self.wavelength >= lower) & (self.wavelength <= upper)
        return np.flatnonzero(mask)

    @classmethod
    def from_scene(cls, scene: xr.Dataset) -> SpectralResponse:
        """Read the band coordinates of a sensor scene."""
        return cls(
            wavelength=np.asarray(scene["wavelength"].values, dtype=float),
            fwhm=np.asarray(scene["fwhm"].values, dtype=float),
        )


@dataclass(frozen=True)
class Window:
    """Positional index window on the sensor grid.

    Attributes:
        across: Across-track slice, stop exclusive.
        along: Along-track slice, stop exclusive.
    """

    across: slice
    along: slice

    def __post_init__(self) -> None:
        """Require explicit, non-negative, stop-exclusive slices."""
        for name in ("across", "along"):
            value = getattr(self, name)
            if value.start is None or value.stop is None or value.step not in (None, 1):
                raise ContractError(
                    f"Window.{name} must have start and stop and no step."
                )
            if value.start < 0 or value.stop <= value.start:
                raise ContractError(f"Window.{name} must satisfy 0 <= start < stop.")

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_across, n_along)``."""
        return (
            self.across.stop - self.across.start,
            self.along.stop - self.along.start,
        )

    @property
    def indexers(self) -> dict[str, slice]:
        """Return keyword arguments for :meth:`xarray.Dataset.isel`."""
        return {"across": self.across, "along": self.along}

    def attrs(self, prefix: str = "window_") -> dict[str, int]:
        """Return the window bounds as integer attributes."""
        return {
            f"{prefix}across_start": int(self.across.start),
            f"{prefix}across_stop": int(self.across.stop),
            f"{prefix}along_start": int(self.along.start),
            f"{prefix}along_stop": int(self.along.stop),
        }

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any], prefix: str = "window_") -> Window:
        """Rebuild a window from attributes written by :meth:`attrs`."""
        try:
            return cls(
                across=slice(
                    int(attrs[f"{prefix}across_start"]),
                    int(attrs[f"{prefix}across_stop"]),
                ),
                along=slice(
                    int(attrs[f"{prefix}along_start"]),
                    int(attrs[f"{prefix}along_stop"]),
                ),
            )
        except KeyError as error:
            raise ContractError(f"Missing window attribute {error}.") from error


@dataclass(frozen=True, eq=False)
class MapGrid:
    """Regular map grid of an orthorectified product.

    Attributes:
        x: One-dimensional cell-centre x coordinates.
        y: One-dimensional cell-centre y coordinates.
        crs: Coordinate reference system of the grid.
    """

    x: np.ndarray
    y: np.ndarray
    crs: pyproj.CRS

    def __post_init__(self) -> None:
        """Validate monotonic, uniform axes."""
        for name in ("x", "y"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or values.size < 2:
                raise ContractError(f"MapGrid.{name} must be a one-dimensional axis.")
            steps = np.diff(values)
            if np.any(steps == 0) or not np.allclose(steps, steps[0], rtol=1e-6):
                raise ContractError(f"MapGrid.{name} must be uniformly spaced.")
            object.__setattr__(self, name, values)

    @property
    def dx(self) -> float:
        """Return the signed x spacing."""
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        """Return the signed y spacing."""
        return float(self.y[1] - self.y[0])

    @classmethod
    def from_product(cls, product: xr.Dataset) -> MapGrid:
        """Read the map grid of a product in map geometry space."""
        from spectra.geometry.crs import crs_from_dataset

        if product.attrs.get("geometry_space") != "map":
            raise ContractError("MapGrid requires a product with geometry_space='map'.")
        return cls(
            x=np.asarray(product["x"].values, dtype=float),
            y=np.asarray(product["y"].values, dtype=float),
            crs=crs_from_dataset(product),
        )


@dataclass(frozen=True, eq=False)
class SceneGeometry:
    """Projected pixel-centre geometry of a sensor scene.

    This is the in-memory replacement of the AVIRIS-4 metadata file used by
    the placement and regridding code.

    Attributes:
        x: Projected pixel-centre x coordinates on the two spatial dimensions.
        y: Projected pixel-centre y coordinates on the same dimensions.
        crs: Projected coordinate reference system in metres.
        lon: Pixel-centre longitudes, if available.
        lat: Pixel-centre latitudes, if available.
        source: Name of the file or scene the geometry came from.
    """

    x: xr.DataArray
    y: xr.DataArray
    crs: pyproj.CRS
    lon: xr.DataArray | None = None
    lat: xr.DataArray | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate the dimension layout."""
        if (
            self.x.ndim != 2
            or self.x.dims != self.y.dims
            or self.x.shape != self.y.shape
        ):
            raise ContractError(
                "SceneGeometry x/y must share two dimensions and shape."
            )
        for name in ("lon", "lat"):
            value = getattr(self, name)
            if value is not None and (value.dims != self.x.dims):
                raise ContractError(f"SceneGeometry.{name} must share the x/y dims.")

    @property
    def spatial_dims(self) -> tuple[str, str]:
        """Return the two spatial dimension names, across-track first."""
        return (str(self.x.dims[0]), str(self.x.dims[1]))

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_across, n_along)``."""
        return (int(self.x.shape[0]), int(self.x.shape[1]))

    @classmethod
    def from_scene(cls, scene: xr.Dataset) -> SceneGeometry:
        """Build the geometry from a dataset carrying projected ``x``/``y``.

        Accepts a ``SensorScene`` (``spatial_ref`` coordinate) and an AVIRIS-4
        metadata dataset (``epsg`` attributes on ``x``/``y``).
        """
        from spectra.geometry.crs import crs_from_dataset

        if "x" not in scene or "y" not in scene:
            raise ContractError("Scene geometry requires projected 'x' and 'y'.")
        return cls(
            x=scene["x"],
            y=scene["y"],
            crs=crs_from_dataset(scene),
            lon=scene.get("lon", None),
            lat=scene.get("lat", None),
            source=str(scene.attrs.get("scene_id", scene.encoding.get("source", ""))),
        )


@dataclass(frozen=True)
class Placement:
    """How a plume column is placed on a sensor grid.

    Attributes:
        mode: ``anchor`` places the selected LES corner cell on an anchor pixel;
            ``georeferenced`` uses the georeference attributes of the column.
        anchor_along: Zero-based along-track index of the anchor pixel.
        anchor_across: Zero-based across-track index of the anchor pixel.
        anchor_corner: LES grid corner cell placed on the anchor pixel centre.
        rotation_deg: Whole-degree rotation around the anchor, counterclockwise
            positive, after the reflection.
        flip: LES axis reflected around the anchor before the rotation.
        emission_scale: Multiplicative factor applied to the column.
        time_index: LES time index of the embedded snapshot.
    """

    mode: Literal["anchor", "georeferenced"] = "anchor"
    anchor_along: int = 0
    anchor_across: int = 0
    anchor_corner: AnchorCorner = "lower left"
    rotation_deg: int = 0
    flip: Flip = "none"
    emission_scale: float = 1.0
    time_index: int = 0

    def __post_init__(self) -> None:
        """Validate the placement parameters."""
        if self.anchor_corner not in ANCHOR_CORNERS:
            raise ContractError(f"anchor_corner must be one of {ANCHOR_CORNERS}.")
        if self.flip not in FLIPS:
            raise ContractError(f"flip must be one of {FLIPS}.")
        if isinstance(self.rotation_deg, bool) or not isinstance(
            self.rotation_deg, (int, np.integer)
        ):
            raise ContractError("rotation_deg must be a whole number of degrees.")
        if not -359 <= int(self.rotation_deg) <= 359:
            raise ContractError("rotation_deg must lie between -359 and 359.")
        for name in ("anchor_along", "anchor_across", "time_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ContractError(f"{name} must be an integer.")
            if value < 0:
                raise ContractError(f"{name} must be non-negative.")
        if not np.isfinite(self.emission_scale) or self.emission_scale <= 0:
            raise ContractError("emission_scale must be a positive finite number.")

    @property
    def flip_axis(self) -> Literal["x", "y"] | None:
        """Return the reflected axis in the ``aviris4_ch4`` convention."""
        return None if self.flip == "none" else self.flip

    def attrs(self) -> dict[str, Any]:
        """Return the placement as serialisable attributes."""
        return {
            "placement_mode": self.mode,
            "anchor_along": int(self.anchor_along),
            "anchor_across": int(self.anchor_across),
            "anchor_corner": self.anchor_corner,
            "rotation_deg": int(self.rotation_deg),
            "flip": self.flip,
            "emission_scale": float(self.emission_scale),
            "time_index": int(self.time_index),
        }


@dataclass(frozen=True)
class RetrievalSettings:
    """Settings shared by all retrievers.

    Attributes:
        statistics_scope: Where background statistics come from: the embedded
            window, the full scene, or statistics cached per scene.
        band_windows_nm: Inclusive wavelength ranges used by the retrieval.
        target_enhancement_ppm: Enhancement at which the target spectrum is
            evaluated.
        min_n_samples: Minimum number of background spectra per statistic.
        exclude_injected_from_background: Mask the injected plume from the
            background statistics (idealised background).
        options: Retriever-specific options, validated by the retriever.
    """

    statistics_scope: StatisticsScope = "cached_scene"
    band_windows_nm: tuple[tuple[float, float], ...] = (
        (1480.0, 1800.0),
        (2080.0, 2500.0),
    )
    target_enhancement_ppm: float = 10.0
    min_n_samples: int = 7000
    exclude_injected_from_background: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.statistics_scope not in ("window", "full_scene", "cached_scene"):
            raise ContractError("statistics_scope is invalid.")
        for lower, upper in self.band_windows_nm:
            if not lower < upper:
                raise ContractError(
                    "band_windows_nm entries must satisfy lower < upper."
                )
        if self.target_enhancement_ppm <= 0:
            raise ContractError("target_enhancement_ppm must be positive.")
        if self.min_n_samples < 1:
            raise ContractError("min_n_samples must be at least 1.")


__all__ = [
    "ANCHOR_CORNERS",
    "FLIPS",
    "AnchorCorner",
    "Flip",
    "GeometrySpace",
    "MapGrid",
    "Placement",
    "Platform",
    "RadianceLeg",
    "RetrievalSettings",
    "SceneGeometry",
    "SceneRecord",
    "SpectralResponse",
    "StatisticsScope",
    "ViewingGeometry",
    "Window",
]
