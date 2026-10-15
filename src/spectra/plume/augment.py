"""Augmentation of ``PlumeColumn`` datasets: mirroring, scaling, partial columns."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.contracts.plume_column import scale_emission
from spectra.types import Placement

FlipAxis = Literal["x", "y"]
AltitudeMethod = Literal["floor", "linear"]
_WIND_COMPONENT = {"x": "wind_u", "y": "wind_v"}
_ALTITUDE_RTOL = 1e-9


def flip(dataset: xr.Dataset, axis: FlipAxis) -> xr.Dataset:
    """Mirror a plume column along ``x`` or ``y`` on its own grid.

    Data arrays are reversed along the axis while the coordinate stays
    increasing, the source position is mirrored to ``min + max - position``
    and the wind component along the axis changes sign. Flipping twice along
    the same axis restores the input, including the ``augment_flip``
    attribute that lists the axes mirrored an odd number of times.

    Args:
        dataset: Plume column dataset or snapshot.
        axis: Axis to mirror.

    Returns:
        Mirrored copy.
    """
    if axis not in ("x", "y"):
        raise ContractError(f"axis must be 'x' or 'y', got {axis!r}.")
    coordinate = dataset.coords[axis].values
    mirrored = dataset.isel({axis: slice(None, None, -1)})
    mirrored = mirrored.assign_coords({axis: coordinate})
    mirrored.coords[axis].attrs = dict(dataset.coords[axis].attrs)

    wind = _WIND_COMPONENT[axis]
    if wind in mirrored:
        attrs = dict(mirrored[wind].attrs)
        mirrored[wind] = -mirrored[wind]
        mirrored[wind].attrs = attrs

    attrs = dict(dataset.attrs)
    key = f"source_{axis}"
    if key in attrs:
        low = float(np.min(coordinate))
        high = float(np.max(coordinate))
        attrs[key] = low + high - float(attrs[key])
    flipped = {part for part in str(attrs.get("augment_flip", "")).split(",") if part}
    flipped.discard("none")
    flipped ^= {axis}
    attrs["augment_flip"] = ",".join(sorted(flipped)) if flipped else "none"
    mirrored.attrs = attrs
    return mirrored


def scale_to_emission_rate(
    dataset: xr.Dataset, target_kg_s: float
) -> tuple[xr.Dataset, float]:
    """Scale a plume column so that its emission rate equals ``target_kg_s``.

    Args:
        dataset: Plume column dataset with a constant ``emission_rate`` or a
            snapshot from :func:`~spectra.contracts.plume_column.select_time`.
        target_kg_s: Target emission rate in kg s-1.

    Returns:
        The scaled dataset and the applied factor.

    Raises:
        ContractError: If the emission rate varies in time, is not positive,
            or the target is not a positive finite number.
    """
    if not np.isfinite(target_kg_s) or target_kg_s <= 0:
        raise ContractError("target_kg_s must be a positive finite number.")
    rates = np.atleast_1d(np.asarray(dataset["emission_rate"].values, dtype=float))
    if rates.size > 1 and np.ptp(rates) != 0:
        raise ContractError(
            "emission_rate varies in time; select a snapshot with select_time first."
        )
    current = float(rates[0])
    if not np.isfinite(current) or current <= 0:
        raise ContractError(f"Cannot scale an emission rate of {current} kg s-1.")
    factor = float(target_kg_s) / current
    return scale_emission(dataset, factor), factor


def transform_wind(u: float, v: float, placement: Placement) -> tuple[float, float]:
    """Transform a model-frame wind vector like the plume column is placed.

    The reflection of ``placement.flip`` is applied first (``"x"`` negates
    ``u``, ``"y"`` negates ``v``), then the vector is rotated
    counterclockwise by ``placement.rotation_deg``.

    Args:
        u: Wind component along the model ``x`` axis in m s-1.
        v: Wind component along the model ``y`` axis in m s-1.
        placement: Placement whose ``flip`` and ``rotation_deg`` apply.

    Returns:
        Transformed ``(u, v)``.
    """
    u_out = float(u)
    v_out = float(v)
    if placement.flip == "x":
        u_out = -u_out
    elif placement.flip == "y":
        v_out = -v_out
    angle = np.deg2rad(float(placement.rotation_deg))
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return (u_out * cos - v_out * sin, u_out * sin + v_out * cos)


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
        ContractError: If partial columns are absent, ``altitude_m`` lies below
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
    "flip",
    "plume_fraction_below",
    "scale_to_emission_rate",
    "transform_wind",
]
