"""Pixel areas from projected pixel-centre coordinates."""

from __future__ import annotations

import xarray as xr


def pixel_area_from_centres(x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
    """Estimate pixel areas from neighbouring centre spacings (dask-aware).

    The area is the parallelogram spanned by the across- and along-track
    centre spacing vectors, so skew between the two directions is accounted
    for; this matches the polygon areas of the regridders to the lattice's
    local curvature.

    Args:
        x: Projected x coordinates of the pixel centres in metres, 2-D.
        y: Projected y coordinates on the same grid.

    Returns:
        Pixel areas in m2 on the grid of ``x``.
    """
    across, along = (str(dim) for dim in x.dims)
    # Work without index coordinates: padding would duplicate index values.
    plain_x = xr.DataArray(x.data, dims=x.dims)
    plain_y = xr.DataArray(y.data, dims=y.dims)
    dxa = plain_x.diff(across).pad({across: (0, 1)}, mode="edge")
    dya = plain_y.diff(across).pad({across: (0, 1)}, mode="edge")
    dxl = plain_x.diff(along).pad({along: (0, 1)}, mode="edge")
    dyl = plain_y.diff(along).pad({along: (0, 1)}, mode="edge")
    area = abs(dxa * dyl - dya * dxl).transpose(across, along)
    area = area.assign_coords({across: x[across], along: x[along]})
    area.attrs = {"units": "m2", "long_name": "pixel area from centre spacing"}
    return area


__all__ = ["pixel_area_from_centres"]
