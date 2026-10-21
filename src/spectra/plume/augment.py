"""Partial columns of ``PlumeColumn`` datasets below an altitude."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError

AltitudeMethod = Literal["floor", "linear"]
_ALTITUDE_RTOL = 1e-9


def column_below_altitude(
    dataset: xr.Dataset, altitude_m: float, *, method: AltitudeMethod = "linear"
) -> xr.DataArray:
    """Return the partial column below ``altitude_m``.

    Args:
        dataset: Plume column dataset with ``column_below_altitude``.
        altitude_m: Altitude above ground in metres.
        method: ``"floor"`` selects the largest tabulated altitude not above
            ``altitude_m``; ``"linear"`` interpolates between the tabulated
            altitudes. Both use the highest tabulated column above it.

    Returns:
        Partial column in the units of ``column_below_altitude`` with the
        ``altitude`` dimension removed and the attributes ``altitude`` and
        ``altitude_method`` recorded.

    Raises:
        :exc:`ContractError`: If partial columns are absent, ``altitude_m`` lies below
            the lowest tabulated altitude or ``method`` is unknown.
    """
    if "column_below_altitude" not in dataset:
        raise ContractError("Dataset has no 'column_below_altitude' variable.")
    if method not in ("floor", "linear"):
        raise ContractError(f"method must be 'floor' or 'linear', got {method!r}.")
    partial = dataset["column_below_altitude"]
    altitudes = np.asarray(partial["altitude"].values, dtype=float)
    altitude = float(altitude_m)
    if altitude < altitudes[0] * (1 - _ALTITUDE_RTOL):
        raise ContractError(
            f"altitude_m={altitude} lies below the lowest tabulated altitude "
            f"{altitudes[0]} m."
        )
    exact = np.flatnonzero(np.isclose(altitudes, altitude, rtol=_ALTITUDE_RTOL))
    if exact.size or method == "floor" or altitude >= altitudes[-1]:
        index = (
            int(exact[0])
            if exact.size
            else int(np.searchsorted(altitudes, altitude, "right") - 1)
        )
        result = partial.isel(altitude=index, drop=True)
    else:
        result = partial.interp(altitude=altitude).drop_vars("altitude")
    result = result.copy()
    result.attrs = {
        **partial.attrs,
        "altitude": altitude,
        "altitude_method": method,
    }
    result.name = "column_below_altitude"
    return result


def plume_fraction_below(
    dataset: xr.Dataset, altitude_m: float, *, method: AltitudeMethod = "linear"
) -> xr.DataArray:
    """Return the fraction of the plume mass below ``altitude_m`` per snapshot.

    Args:
        dataset: Plume column dataset with ``column_below_altitude``.
        altitude_m: Altitude above ground in metres.
        method: Altitude selection, see :func:`column_below_altitude`.

    Returns:
        Dimensionless fraction with the ``time`` dimension (scalar for a
        snapshot); ``NaN`` where the full column holds no mass.
    """
    partial = column_below_altitude(dataset, altitude_m, method=method)
    total = dataset["column"]
    spatial = [dim for dim in ("y", "x") if dim in total.dims]
    partial_mass = partial.sum(dim=spatial)
    total_mass = total.sum(dim=spatial)
    fraction = (partial_mass / total_mass.where(total_mass > 0)).astype(float)
    fraction.attrs = {
        "units": "1",
        "long_name": f"Fraction of the plume mass below {altitude_m:g} m",
        "altitude": float(altitude_m),
        "altitude_method": method,
    }
    fraction.name = "plume_fraction_below"
    return fraction


__all__ = [
    "column_below_altitude",
    "plume_fraction_below",
]
