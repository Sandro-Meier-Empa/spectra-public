"""Column units: canonical spellings and ``ucat`` conversions.

Columns are stored in ``kg m-2``; plume columns and absorption tables may use
``ppm-m``.
"""

from __future__ import annotations

from typing import cast

import ucat
import xarray as xr

from spectra.contracts.errors import ContractError

CANONICAL_COLUMN_UNITS = "kg m-2"
UCAT_COLUMN_UNITS = "kg/m2"
PPM_METRE_UNITS = "ppm-m"
_UNIT_ALIASES = {
    "kg/m2": CANONICAL_COLUMN_UNITS,
    "kg m-2": CANONICAL_COLUMN_UNITS,
    "kg m^-2": CANONICAL_COLUMN_UNITS,
    "kg/m^2": CANONICAL_COLUMN_UNITS,
    "ppm-m": PPM_METRE_UNITS,
    "ppm m": PPM_METRE_UNITS,
    "ppm*m": PPM_METRE_UNITS,
}


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


__all__ = [
    "CANONICAL_COLUMN_UNITS",
    "PPM_METRE_UNITS",
    "convert_column_units",
    "normalise_units",
]
