"""Contract B': retrieved or embedded column products.

A product is either in sensor geometry (``across``/``along``) or on a map grid
(``y``/``x``). ``to_prism_grid`` turns both into the ``(y, x)`` layout the
prism sample contract expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pyproj
import ucat
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.geometry.crs import crs_from_dataset

COLUMN_PRODUCT_CONTRACT = "spectra.ColumnProduct"
COLUMN_PRODUCT_VERSION = "1.0"
CANONICAL_COLUMN_UNITS = "kg m-2"
UCAT_COLUMN_UNITS = "kg/m2"
PPM_METRE_UNITS = "ppm-m"
SENSOR_SPACE_DIMS = ("across", "along")
MAP_SPACE_DIMS = ("y", "x")
_UNIT_ALIASES = {
    "kg/m2": CANONICAL_COLUMN_UNITS,
    "kg m-2": CANONICAL_COLUMN_UNITS,
    "kg m^-2": CANONICAL_COLUMN_UNITS,
    "kg/m^2": CANONICAL_COLUMN_UNITS,
    "ppm-m": PPM_METRE_UNITS,
    "ppm m": PPM_METRE_UNITS,
    "ppm*m": PPM_METRE_UNITS,
}


@dataclass(frozen=True)
class ProductInfo:
    """Summary of a validated column product."""

    sensor: str
    scene_id: str
    gas: str
    geometry_space: str
    spatial_dims: tuple[str, str]
    has_injected: bool
    crs: pyproj.CRS | None


def normalise_units(units: str) -> str:
    """Return the canonical spelling of a column unit."""
    key = " ".join(units.strip().split())
    return _UNIT_ALIASES.get(key, key)


def convert_column_units(
    column: xr.DataArray,
    to_unit: str,
    *,
    gas: str,
    from_unit: str | None = None,
) -> xr.DataArray:
    """Convert a column between units with ``ucat``.

    ``ucat.convert_columns`` validates the ``units`` attribute of labelled
    arrays against an internally derived unit and rejects correctly labelled
    ``kg m-2`` arrays, so the attribute is stripped before the call and the
    result is relabelled with the canonical spelling of ``to_unit``.

    Args:
        column: Column array.
        to_unit: Target unit, e.g. ``"kg m-2"`` or ``"ppm-m"``.
        gas: Gas name or molar mass accepted by ``ucat``.
        from_unit: Input unit; defaults to the ``units`` attribute.

    Returns:
        Converted array with the original attributes and canonical ``units``.
    """
    units = from_unit or column.attrs.get("units")
    if units is None:
        raise ContractError("Column units are unknown; set the 'units' attribute.")
    attrs = dict(column.attrs)
    source = normalise_units(str(units))
    target = normalise_units(to_unit)
    if source == target:
        converted = column.copy()
    else:
        bare = column.copy(deep=False)
        bare.attrs = {}
        converted = cast(
            xr.DataArray,
            ucat.convert_columns(
                bare,
                from_unit=UCAT_COLUMN_UNITS
                if source == CANONICAL_COLUMN_UNITS
                else source,
                to_unit=UCAT_COLUMN_UNITS
                if target == CANONICAL_COLUMN_UNITS
                else target,
                molar_mass=gas,
                warn=False,
            ),
        )
    converted.attrs = attrs
    converted.attrs["units"] = target
    converted.name = column.name
    return converted


def _spatial_dims(dataset: xr.Dataset) -> tuple[str, str]:
    space = dataset.attrs.get("geometry_space")
    if space == "sensor":
        return SENSOR_SPACE_DIMS
    if space == "map":
        return MAP_SPACE_DIMS
    raise ContractError(
        f"ColumnProduct must declare geometry_space 'sensor' or 'map'; got {space!r}."
    )


def validate_column_product(dataset: xr.Dataset, *, gas: str = "ch4") -> ProductInfo:
    """Validate a dataset against the ``ColumnProduct`` contract.

    Args:
        dataset: Candidate product.
        gas: Lower-case gas name used in the variable names.

    Returns:
        Summary of the validated product.

    Raises:
        ContractError: If the dataset violates the contract.
    """
    dims = _spatial_dims(dataset)
    for dim in dims:
        if dim not in dataset.dims:
            raise ContractError(f"ColumnProduct is missing dimension '{dim}'.")
    mass = f"{gas}_mass"
    uncertainty = f"{gas}_mass_uncertainty"
    for name in (mass, uncertainty):
        if name not in dataset:
            raise ContractError(f"ColumnProduct must contain '{name}'.")
        variable = dataset[name]
        if tuple(variable.dims) != dims:
            raise ContractError(
                f"ColumnProduct '{name}' must have dimensions {dims}; "
                f"got {variable.dims}."
            )
        if (
            normalise_units(str(variable.attrs.get("units", "")))
            != CANONICAL_COLUMN_UNITS
        ):
            raise ContractError(
                f"ColumnProduct '{name}' must use {CANONICAL_COLUMN_UNITS!r}; "
                f"got {variable.attrs.get('units')!r}."
            )
        if not np.issubdtype(variable.dtype, np.floating):
            raise ContractError(f"ColumnProduct '{name}' must be floating point.")
    if "retrieval_valid" not in dataset:
        raise ContractError("ColumnProduct must contain 'retrieval_valid'.")
    valid = dataset["retrieval_valid"]
    if tuple(valid.dims) != dims or valid.dtype.kind != "b":
        raise ContractError(
            "ColumnProduct 'retrieval_valid' must be boolean on the grid."
        )
    uncertainty_values = dataset[uncertainty].values
    valid_values = valid.values
    if np.any(~(uncertainty_values[valid_values] > 0)):
        raise ContractError(
            f"ColumnProduct '{uncertainty}' must be positive wherever retrieval_valid."
        )
    injected = f"injected_{gas}_mass"
    has_injected = injected in dataset
    if has_injected:
        variable = dataset[injected]
        if tuple(variable.dims) != dims:
            raise ContractError(
                f"ColumnProduct '{injected}' must have dimensions {dims}."
            )
        if (
            normalise_units(str(variable.attrs.get("units", "")))
            != CANONICAL_COLUMN_UNITS
        ):
            raise ContractError(f"ColumnProduct '{injected}' must use kg m-2.")
    for name in ("pixel_area", "sensitivity", "background_valid", "albedo"):
        if name in dataset and tuple(dataset[name].dims) != dims:
            raise ContractError(f"ColumnProduct '{name}' must have dimensions {dims}.")
    if dims == MAP_SPACE_DIMS:
        for name in dims:
            if name not in dataset.coords or tuple(dataset[name].dims) != (name,):
                raise ContractError(
                    f"Map-space ColumnProduct needs a 1-D '{name}' coordinate."
                )
    try:
        crs: pyproj.CRS | None = crs_from_dataset(dataset)
    except ContractError:
        crs = None
    for name in ("sensor", "scene_id"):
        if name not in dataset.attrs:
            raise ContractError(f"ColumnProduct is missing attribute '{name}'.")
    return ProductInfo(
        sensor=str(dataset.attrs["sensor"]),
        scene_id=str(dataset.attrs["scene_id"]),
        gas=gas,
        geometry_space=str(dataset.attrs["geometry_space"]),
        spatial_dims=dims,
        has_injected=has_injected,
        crs=crs,
    )


def to_prism_grid(dataset: xr.Dataset) -> xr.Dataset:
    """Return the product on the ``(y, x)`` grid of the prism sample contract.

    Sensor-space products are renamed positionally (``across`` becomes ``y``
    and ``along`` becomes ``x``); projected ``x``/``y`` coordinate variables
    are kept as ``projected_x``/``projected_y``. Map-space products are
    returned unchanged. Variables with a ``band`` dimension are dropped.

    Args:
        dataset: Column product in either geometry space.

    Returns:
        Product with spatial dimensions exactly ``("y", "x")``.
    """
    dims = _spatial_dims(dataset)
    output = dataset
    if "band" in output.dims:
        band_variables = [
            name
            for name, variable in output.variables.items()
            if "band" in variable.dims
        ]
        output = output.drop_vars(band_variables)
        if "band" in output.dims:
            output = output.drop_dims("band")
    if dims == SENSOR_SPACE_DIMS:
        renames: dict[str, str] = {}
        if "x" in output.variables:
            renames["x"] = "projected_x"
        if "y" in output.variables:
            renames["y"] = "projected_y"
        output = output.rename(renames)
        output = output.rename({"across": "y", "along": "x"})
        output = output.assign_coords(
            y=np.arange(output.sizes["y"]), x=np.arange(output.sizes["x"])
        )
        output.attrs["prism_grid_source"] = "sensor"
    else:
        output = output.transpose("y", "x", ...)
        output.attrs["prism_grid_source"] = "map"
    for name, variable in output.data_vars.items():
        if set(variable.dims) == {"y", "x"} and tuple(variable.dims) != ("y", "x"):
            output[name] = variable.transpose("y", "x")
    return output


__all__ = [
    "CANONICAL_COLUMN_UNITS",
    "COLUMN_PRODUCT_CONTRACT",
    "COLUMN_PRODUCT_VERSION",
    "MAP_SPACE_DIMS",
    "PPM_METRE_UNITS",
    "SENSOR_SPACE_DIMS",
    "ProductInfo",
    "convert_column_units",
    "normalise_units",
    "to_prism_grid",
    "validate_column_product",
]
