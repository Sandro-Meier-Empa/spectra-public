"""Sensor scenes in sensor geometry, the common format adapters open.

The radiance keeps the native units of the sensor: the Lambert-Beer embedding
only multiplies it with a transmittance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyproj
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.geometry.crs import crs_from_dataset, require_projected_metre_crs

SENSOR_SCENE_CONTRACT = "spectra.SensorScene"
SENSOR_SCENE_VERSION = "1.0"
SENSOR_DIMS = ("across", "along", "band")
SPATIAL_DIMS = ("across", "along")
RADIANCE_UNITS = "W m-2 nm-1 sr-1"
PLATFORMS = ("airborne", "spaceborne")
REQUIRED_ATTRS = ("sensor", "platform", "scene_id", "acquisition_time")
GEOMETRY_VARIABLES = (
    "sza",
    "saa",
    "vza",
    "vaa",
    "surface_elevation",
    "sensor_altitude",
)


@dataclass(frozen=True)
class SceneInfo:
    """Summary of a validated sensor scene."""

    sensor: str
    scene_id: str
    platform: str
    n_across: int
    n_along: int
    n_band: int
    crs: pyproj.CRS
    is_lazy: bool


def _spatial(variable: xr.DataArray, name: str, allow_scalar: bool = False) -> None:
    dims = tuple(variable.dims)
    if allow_scalar and dims == ():
        return
    if dims != SPATIAL_DIMS:
        raise ContractError(
            f"SensorScene '{name}' must have dimensions {SPATIAL_DIMS}"
            f"{' or be scalar' if allow_scalar else ''}; got {dims}."
        )


def validate_sensor_scene(
    dataset: xr.Dataset, *, require_lazy: bool = False
) -> SceneInfo:
    """Validate a dataset against the ``SensorScene`` contract.

    Args:
        dataset: Candidate scene.
        require_lazy: Require the radiance to be dask-backed.

    Returns:
        Summary of the validated scene.

    Raises:
        ContractError: If the dataset violates the contract.
    """
    for dim in SENSOR_DIMS:
        if dim not in dataset.dims:
            raise ContractError(f"SensorScene is missing dimension '{dim}'.")
    for name in ("wavelength", "fwhm"):
        if name not in dataset.coords or tuple(dataset[name].dims) != ("band",):
            raise ContractError(f"SensorScene needs a '{name}(band)' coordinate.")
    wavelength = np.asarray(dataset["wavelength"].values, dtype=float)
    if np.any(np.diff(wavelength) <= 0):
        raise ContractError("SensorScene 'wavelength' must be strictly increasing.")
    if np.any(np.asarray(dataset["fwhm"].values, dtype=float) <= 0):
        raise ContractError("SensorScene 'fwhm' must be positive.")
    for name in ("lat", "lon", "x", "y"):
        if name not in dataset.coords and name not in dataset:
            raise ContractError(f"SensorScene needs a two-dimensional '{name}'.")
        _spatial(dataset[name], name)
    crs = crs_from_dataset(dataset)
    require_projected_metre_crs(crs, "SensorScene CRS")
    if "time" not in dataset.coords and "time" not in dataset:
        raise ContractError("SensorScene needs a 'time' coordinate.")
    time = dataset["time"]
    if tuple(time.dims) not in ((), ("along",)):
        raise ContractError("SensorScene 'time' must be scalar or along-track.")
    if not np.issubdtype(time.dtype, np.datetime64):
        raise ContractError("SensorScene 'time' must be datetime64.")

    if "radiance" not in dataset:
        raise ContractError("SensorScene must contain 'radiance'.")
    radiance = dataset["radiance"]
    if tuple(radiance.dims) != tuple(SENSOR_DIMS):
        raise ContractError(
            f"SensorScene 'radiance' must use dimensions {SENSOR_DIMS}; "
            f"got {radiance.dims}."
        )
    if "units" not in radiance.attrs:
        raise ContractError("SensorScene 'radiance' must carry a 'units' attribute.")
    is_lazy = radiance.chunks is not None
    if require_lazy and not is_lazy:
        raise ContractError("SensorScene 'radiance' must be dask-backed.")

    for name in GEOMETRY_VARIABLES:
        if name not in dataset:
            raise ContractError(f"SensorScene must contain '{name}'.")
        _spatial(dataset[name], name, allow_scalar=True)

    if "valid" not in dataset:
        raise ContractError("SensorScene must contain a boolean 'valid' mask.")
    _spatial(dataset["valid"], "valid")
    if dataset["valid"].dtype.kind != "b":
        raise ContractError("SensorScene 'valid' must be boolean.")
    if "pixel_area" not in dataset:
        raise ContractError("SensorScene must contain 'pixel_area'.")
    _spatial(dataset["pixel_area"], "pixel_area")

    missing = [name for name in REQUIRED_ATTRS if name not in dataset.attrs]
    if missing:
        raise ContractError(f"SensorScene is missing attributes: {missing}.")
    if dataset.attrs["platform"] not in PLATFORMS:
        raise ContractError(f"SensorScene platform must be one of {PLATFORMS}.")
    if dataset.attrs.get("geometry_space", "sensor") != "sensor":
        raise ContractError("SensorScene geometry_space must be 'sensor'.")

    return SceneInfo(
        sensor=str(dataset.attrs["sensor"]),
        scene_id=str(dataset.attrs["scene_id"]),
        platform=str(dataset.attrs["platform"]),
        n_across=int(dataset.sizes["across"]),
        n_along=int(dataset.sizes["along"]),
        n_band=int(dataset.sizes["band"]),
        crs=crs,
        is_lazy=is_lazy,
    )


def select_bands(
    dataset: xr.Dataset, windows: tuple[tuple[float, float], ...]
) -> xr.Dataset:
    """Return the bands whose centre wavelengths fall inside the given ranges.

    Args:
        dataset: Scene (or any dataset) with a ``wavelength(band)`` coordinate.
        windows: Inclusive wavelength ranges in nanometres.

    Returns:
        The dataset restricted along ``band``.

    Raises:
        ContractError: If no band falls inside the windows.
    """
    wavelength = np.asarray(dataset["wavelength"].values, dtype=float)
    mask = np.zeros(wavelength.size, dtype=bool)
    for lower, upper in windows:
        mask |= (wavelength >= lower) & (wavelength <= upper)
    if not mask.any():
        raise ContractError(f"No bands fall inside the wavelength windows {windows}.")
    return dataset.isel(band=np.flatnonzero(mask))


__all__ = [
    "GEOMETRY_VARIABLES",
    "PLATFORMS",
    "RADIANCE_UNITS",
    "REQUIRED_ATTRS",
    "SENSOR_DIMS",
    "SENSOR_SCENE_CONTRACT",
    "SENSOR_SCENE_VERSION",
    "SPATIAL_DIMS",
    "SceneInfo",
    "select_bands",
    "validate_sensor_scene",
]
