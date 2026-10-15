"""Coordinate reference system helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pyproj
import xarray as xr
from pyproj.exceptions import CRSError

from spectra.contracts.errors import ContractError

_METRE_UNITS = frozenset({"metre", "meter", "m", "metres", "meters"})


def require_projected_metre_crs(crs: pyproj.CRS, name: str = "CRS") -> None:
    """Raise unless ``crs`` is projected with metre axes."""
    if not crs.is_projected:
        raise ContractError(f"{name} must be a projected CRS; got {crs.to_string()}.")
    units = {axis.unit_name.lower() for axis in crs.axis_info}
    if not units or not units.issubset(_METRE_UNITS):
        raise ContractError(f"{name} must use metre axes; got units {sorted(units)}.")


def projected_metre_crs(value: str | int | pyproj.CRS) -> pyproj.CRS:
    """Return a validated projected coordinate reference system in metres."""
    try:
        crs = pyproj.CRS.from_user_input(value)
    except CRSError as error:
        raise ContractError(f"{value!r} is not a valid CRS.") from error
    require_projected_metre_crs(crs, name="target CRS")
    return crs


def projected_crs_from_xy(x: xr.DataArray, y: xr.DataArray) -> pyproj.CRS:
    """Return the projected CRS declared by ``epsg`` attributes on x and y."""
    epsg_x = x.attrs.get("epsg")
    epsg_y = y.attrs.get("epsg")
    if epsg_x is None or epsg_y is None:
        raise ContractError(
            "The projected x/y variables must both define an 'epsg' attribute."
        )
    try:
        code_x, code_y = int(epsg_x), int(epsg_y)
    except (TypeError, ValueError) as error:
        raise ContractError("The x/y 'epsg' attributes must be integers.") from error
    if code_x != code_y:
        raise ContractError(
            f"The x/y 'epsg' attributes do not match: {code_x} and {code_y}."
        )
    return projected_metre_crs(code_x)


def spatial_ref_coordinate(crs: pyproj.CRS) -> xr.DataArray:
    """Return a scalar CF grid-mapping coordinate describing ``crs``."""
    attrs: dict[str, Any] = dict(crs.to_cf())
    attrs.setdefault("crs_wkt", crs.to_wkt())
    attrs["spatial_ref"] = crs.to_wkt()
    epsg = crs.to_epsg()
    if epsg is not None:
        attrs["epsg"] = int(epsg)
    return xr.DataArray(np.int32(0), name="spatial_ref", attrs=attrs)


def crs_from_dataset(dataset: xr.Dataset) -> pyproj.CRS:
    """Return the CRS declared by a dataset.

    Looks for a ``spatial_ref`` grid-mapping coordinate (``crs_wkt`` or
    ``spatial_ref`` attribute), an ``epsg`` attribute on ``x``/``y`` as in the
    AVIRIS-4 metadata files, or ``crs``/``epsg`` dataset attributes.
    """
    if "spatial_ref" in dataset.coords or "spatial_ref" in dataset:
        attrs = dataset["spatial_ref"].attrs
        for key in ("crs_wkt", "spatial_ref"):
            if key in attrs:
                return pyproj.CRS.from_user_input(attrs[key])
        if "epsg" in attrs:
            return pyproj.CRS.from_epsg(int(attrs["epsg"]))
    if "x" in dataset and "y" in dataset and "epsg" in dataset["x"].attrs:
        return projected_crs_from_xy(dataset["x"], dataset["y"])
    for key in ("crs", "epsg", "embedding_crs", "target_crs"):
        if key in dataset.attrs:
            return pyproj.CRS.from_user_input(dataset.attrs[key])
    raise ContractError("Dataset declares no coordinate reference system.")


def to_lonlat(
    crs: pyproj.CRS, x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Transform projected coordinates to longitude and latitude."""
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def from_lonlat(
    crs: pyproj.CRS, lon: npt.ArrayLike, lat: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Transform longitude and latitude to projected coordinates."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(
        np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    )
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def utm_crs_for(lon: float, lat: float) -> pyproj.CRS:
    """Return the WGS 84 UTM zone CRS containing a point."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return pyproj.CRS.from_epsg(epsg)


__all__ = [
    "crs_from_dataset",
    "from_lonlat",
    "projected_crs_from_xy",
    "projected_metre_crs",
    "require_projected_metre_crs",
    "spatial_ref_coordinate",
    "to_lonlat",
    "utm_crs_for",
]
