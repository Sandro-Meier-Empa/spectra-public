"""Shared dataclasses used by the contracts, protocols and adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyproj
import xarray as xr

from spectra.contracts.errors import ContractError

Platform = Literal["airborne", "spaceborne"]
RadianceLeg = Literal["solar", "view", "both"]
AnchorCorner = Literal["lower left", "lower right", "upper left", "upper right"]
# Point of the LES grid pinned to the anchor: the emission source or a corner cell.
AnchorPoint = Literal[
    "source", "lower left", "lower right", "upper left", "upper right"
]
Flip = Literal["none", "x", "y"]

ANCHOR_CORNERS: tuple[AnchorCorner, ...] = (
    "lower left",
    "lower right",
    "upper left",
    "upper right",
)
ANCHOR_POINTS: tuple[AnchorPoint, ...] = ("source", *ANCHOR_CORNERS)
FLIPS: tuple[Flip, ...] = ("none", "x", "y")


@dataclass(frozen=True)
class NativeLayout:
    """Where an adapter's radiance lives in its native file.

    The copy-and-patch exporter writes the attenuated pixels back through this
    layout, so import and export share one dimension mapping.

    Attributes:
        role: File role holding the radiance, which is copied and patched.
        variable: ``/``-separated path of the radiance variable in that file.
        dims: Native dimension name to common dimension (``across``,
            ``along`` or ``band``) for every dimension of the variable.
    """

    role: str
    variable: str
    dims: Mapping[str, str]

    def __post_init__(self) -> None:
        """Require a bijection between the native and common dimensions."""
        values = sorted(self.dims.values())
        if values != sorted(("across", "along", "band")) or len(self.dims) != 3:
            raise ContractError(
                "NativeLayout.dims must map three native dimensions onto "
                f"'across', 'along' and 'band'; got {dict(self.dims)}."
            )
        if not self.role or not self.variable:
            raise ContractError("NativeLayout needs a role and a variable path.")
        object.__setattr__(self, "dims", dict(self.dims))

    @property
    def native_dims(self) -> dict[str, str]:
        """Return the common dimension to native dimension mapping."""
        return {common: native for native, common in self.dims.items()}


@dataclass(frozen=True)
class ViewingGeometry:
    """Observation geometry of an absorption table.

    Attributes:
        sza: Solar zenith angle in degrees.
        vza: Viewing zenith angle in degrees.
        surface_elevation: Surface elevation in metres above sea level.
        sensor_altitude: Sensor altitude in metres above sea level.
    """

    sza: float
    vza: float
    surface_elevation: float
    sensor_altitude: float

    def __post_init__(self) -> None:
        """Require finite values."""
        for name in ("sza", "vza", "surface_elevation", "sensor_altitude"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ContractError(f"ViewingGeometry.{name} is not finite.")
            object.__setattr__(self, name, value)

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

        return cls(
            sza=reduce("sza"),
            vza=reduce("vza"),
            surface_elevation=reduce("surface_elevation"),
            sensor_altitude=reduce("sensor_altitude"),
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
        anchor_along: Zero-based along-track index of the anchor pixel.
        anchor_across: Zero-based across-track index of the anchor pixel.
        anchor_corner: Point of the LES grid pinned to the anchor: the emission
            source (``source_x``/``source_y`` of the column) or a corner cell.
        rotation_deg: Whole-degree rotation around the anchor, counterclockwise
            positive, after the reflection.
        flip: LES axis reflected around the anchor before the rotation.
        emission_scale: Multiplicative factor applied to the column.
        time_index: LES time index of the embedded snapshot.
        anchor_xy: Exact projected anchor point; ``None`` uses the centre of
            the anchor pixel. The anchor indices then name the pixel nearest
            to the point.
    """

    anchor_along: int = 0
    anchor_across: int = 0
    anchor_corner: AnchorPoint = "lower left"
    rotation_deg: int = 0
    flip: Flip = "none"
    emission_scale: float = 1.0
    time_index: int = 0
    anchor_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate the placement parameters."""
        if self.anchor_corner not in ANCHOR_POINTS:
            raise ContractError(f"anchor_corner must be one of {ANCHOR_POINTS}.")
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
        if self.anchor_xy is not None:
            point = tuple(float(value) for value in self.anchor_xy)
            if len(point) != 2 or not all(np.isfinite(point)):
                raise ContractError("anchor_xy must be a finite (x, y) pair.")
            object.__setattr__(self, "anchor_xy", point)

    def attrs(self) -> dict[str, Any]:
        """Return the placement as serialisable attributes."""
        attrs: dict[str, Any] = {
            "anchor_along": int(self.anchor_along),
            "anchor_across": int(self.anchor_across),
            "anchor_corner": self.anchor_corner,
            "rotation_deg": int(self.rotation_deg),
            "flip": self.flip,
            "emission_scale": float(self.emission_scale),
            "time_index": int(self.time_index),
        }
        if self.anchor_xy is not None:
            attrs["anchor_x"], attrs["anchor_y"] = self.anchor_xy
        return attrs


__all__ = [
    "ANCHOR_CORNERS",
    "ANCHOR_POINTS",
    "FLIPS",
    "AnchorCorner",
    "AnchorPoint",
    "Flip",
    "NativeLayout",
    "Placement",
    "Platform",
    "RadianceLeg",
    "SceneGeometry",
    "SpectralResponse",
    "ViewingGeometry",
    "Window",
]
