"""Assembly of regridded data arrays and results shared by the regridders."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from spectra.geometry.placement import PlacedColumn
from spectra.regrid.base import RegridResult, relative_mass_error
from spectra.types import Window

SCENE_COORDS = ("x", "y", "lon", "lat")


def scene_data_array(
    values: np.ndarray,
    *,
    scene: xr.Dataset,
    window: Window,
    placed: PlacedColumn,
    name: Hashable | None = None,
) -> xr.DataArray:
    """Wrap regridded ``(across, along)`` values of a window as a data array.

    The ``across``/``along`` coordinates and any two-dimensional ``x``, ``y``,
    ``lon`` and ``lat`` coordinates of the scene are sliced to the window.
    """
    coords: dict[str, Any] = {}
    for dim, index in (("across", window.across), ("along", window.along)):
        if dim in scene.coords:
            coords[dim] = scene.coords[dim].isel({dim: index}).variable
        else:
            coords[dim] = np.arange(index.start, index.stop)
    for key in SCENE_COORDS:
        if key in scene and tuple(scene[key].dims) == ("across", "along"):
            sliced = scene[key].isel(window.indexers)
            coords[key] = (
                ("across", "along"),
                np.asarray(sliced.values),
                dict(scene[key].attrs),
            )
    if "spatial_ref" in scene.coords:
        coords["spatial_ref"] = scene.coords["spatial_ref"].variable
    attrs = {
        "units": placed.units,
        "long_name": "plume column on the sensor grid",
        "geometry_space": "sensor",
        **window.attrs("window_"),
        **placed.placement.attrs(),
    }
    return xr.DataArray(
        np.asarray(values, dtype=np.float64),
        dims=("across", "along"),
        coords=coords,
        attrs=attrs,
        name=name,
    )


def finalize_result(
    data: xr.DataArray,
    *,
    method: str,
    source_area: np.ndarray,
    target_area: np.ndarray,
    source_mass: float,
    source_mass_inside: float,
    target_mass: float,
    attrs: dict[str, Any] | None = None,
) -> RegridResult:
    """Build the result and write its provenance onto ``data.attrs``."""
    result = RegridResult(
        data=data,
        source_area=np.asarray(source_area, dtype=np.float64),
        target_area=np.asarray(target_area, dtype=np.float64),
        method=method,
        source_mass=float(source_mass),
        source_mass_inside=float(source_mass_inside),
        target_mass=float(target_mass),
        relative_error=relative_mass_error(source_mass_inside, target_mass),
        attrs=dict(attrs or {}),
    )
    data.attrs.update(result.provenance())
    return result


__all__ = [
    "SCENE_COORDS",
    "finalize_result",
    "scene_data_array",
]
