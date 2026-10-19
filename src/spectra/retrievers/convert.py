"""Conversion of matched-filter outputs to the ``ColumnProduct`` contract.

Matched filters typically return a column enhancement and its uncertainty in
ppm m on the radiance grid. :func:`matched_filter_to_column_product` turns
such an output into a sensor-space ``ColumnProduct`` in kg m-2; retriever
wrappers (``aviris4_mf``, an emit-ghg wrapper, ...) and adapters opening
stored matched-filter products share it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from spectra.contracts.column_product import convert_column_units
from spectra.contracts.errors import ContractError
from spectra.geometry.area import pixel_area_from_centres
from spectra.geometry.crs import projected_crs_from_xy, spatial_ref_coordinate


def isfinite(array: xr.DataArray) -> xr.DataArray:
    """Return a dask-aware finite mask as a DataArray."""
    return xr.apply_ufunc(np.isfinite, array, dask="allowed")


def matched_filter_to_column_product(
    product: xr.Dataset,
    *,
    attrs: Mapping[str, Any],
    geometry: xr.Dataset | None = None,
    gas: str = "ch4",
    retriever: str | None = None,
    column_variable: str | None = None,
    uncertainty_variable: str = "uncertainty",
) -> xr.Dataset:
    """Convert a matched-filter output to a sensor-space ``ColumnProduct``.

    Args:
        product: Matched-filter output on ``(across, along)`` with the column
            and its uncertainty in units ``ucat`` converts (e.g. ppm m), and
            optional ``sensitivity``, ``retrieval_valid``, ``background_valid``.
        attrs: Scene attributes (``sensor``, ``scene_id``, ``background_group``).
        geometry: Dataset supplying ``x``, ``y``, ``lon``, ``lat`` on the same
            grid, if available.
        gas: Lower-case gas name of the output variables.
        retriever: Retriever name recorded when the output carries none.
        column_variable: Column variable of ``product``; the upper-case gas
            name by default.
        uncertainty_variable: Uncertainty variable of ``product``.

    Returns:
        Sensor-space column product in kg m-2.

    Raises:
        ContractError: If the column or the uncertainty is missing.
    """
    label = gas.upper()
    column = column_variable or label
    if column not in product or uncertainty_variable not in product:
        raise ContractError(
            f"Matched-filter product must contain '{column}' and "
            f"'{uncertainty_variable}'."
        )
    output = xr.Dataset(coords={"across": product["across"], "along": product["along"]})
    output[f"{gas}_mass"] = convert_column_units(product[column], "kg m-2", gas=label)
    output[f"{gas}_mass"].attrs.update(
        {"native_units": product[column].attrs.get("units"), "native_variable": column}
    )
    output[f"{gas}_mass_uncertainty"] = convert_column_units(
        product[uncertainty_variable], "kg m-2", gas=label
    )
    for name in ("sensitivity", "background_valid"):
        if name in product:
            output[name] = product[name]
    if "sensitivity" in output:
        # Matched filters label the dimensionless factor "-"; prism uses "1".
        output["sensitivity"].attrs["units"] = "1"
    # A retrieval without a finite, positive uncertainty is not usable (the
    # noise model is undefined for negative radiances), whatever the filter's
    # own validity flag says.
    finite = (
        isfinite(output[f"{gas}_mass"])
        & isfinite(output[f"{gas}_mass_uncertainty"])
        & (output[f"{gas}_mass_uncertainty"] > 0)
    )
    if "retrieval_valid" in product:
        valid = product["retrieval_valid"].astype(bool) & finite
    else:
        valid = finite
    output["retrieval_valid"] = valid.astype(bool)
    if geometry is not None:
        coords = {
            name: geometry[name]
            for name in ("lon", "lat", "x", "y")
            if name in geometry and geometry[name].dims == ("across", "along")
        }
        output = output.assign_coords(coords)
        if "x" in coords and "y" in coords:
            output = output.assign_coords(
                spatial_ref=spatial_ref_coordinate(
                    projected_crs_from_xy(coords["x"], coords["y"])
                )
            )
            output["pixel_area"] = pixel_area_from_centres(coords["x"], coords["y"])
    if "time" in product.coords:
        output = output.assign_coords(time=product["time"])
    output.attrs.update(product.attrs)
    output.attrs.update(dict(attrs))
    output.attrs.update(
        {
            "geometry_space": "sensor",
            "gas": label,
            "contract": "spectra.ColumnProduct",
        }
    )
    if retriever is not None:
        output.attrs.setdefault("retriever", retriever)
    return output


__all__ = ["isfinite", "matched_filter_to_column_product"]
