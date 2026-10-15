"""Lambert-Beer embedding of simulated columns into radiance and products.

Ported from ``aviris4_ch4.embedding`` without changing the numerics: product
embedding adds a vertical plume column to a retrieved column map, radiance
embedding attenuates a cube with a column already placed on the radiance
grid. The tabulated optical depths already contain the air-mass factor, so
the default input is a vertical column.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal, cast

import numpy as np
import ucat
import xarray as xr

import spectra

EMBEDDING_VERSION = "lambert-beer-v1"
COLUMN_UNITS = "kg/m2"
PPM_METRE_UNITS = "ppm-m"

ColumnConvention = Literal["vertical", "slant"]
EmbeddingMethod = Literal["product", "radiance"]


def _band_dimension(radiance: xr.DataArray) -> str:
    """Return the dimension associated with the wavelength coordinate.

    Raises:
        ValueError: If wavelength is invalid or unrelated to the radiance.
    """
    if "wavelength" not in radiance.coords or radiance["wavelength"].ndim != 1:
        raise ValueError(
            "Radiance must carry a one-dimensional 'wavelength' coordinate."
        )

    band_dim = str(radiance["wavelength"].dims[0])
    if band_dim in radiance.dims:
        return band_dim
    raise ValueError(
        f"The wavelength dimension '{band_dim}' is not a dimension of "
        f"the radiance {list(radiance.dims)}."
    )


def _validate_spatial_dimensions(
    column: xr.DataArray, spatial_dims: tuple[Hashable, ...], name: str
) -> None:
    """Require a column to match the target spatial dimensions.

    Raises:
        ValueError: If the dimension sets differ.
    """
    if set(column.dims) != set(spatial_dims):
        raise ValueError(
            f"{name} dimensions {list(column.dims)} must match the target spatial "
            f"dimensions {list(spatial_dims)} exactly."
        )


def _nonnegative_column(
    column: xr.DataArray, column_units: str, gas: str, to_unit: str
) -> xr.DataArray:
    """Return a column in ``to_unit`` with gaps zeroed and negatives clipped.

    The ``ucat`` call is kept exactly as in ``aviris4_ch4`` because the golden
    oracle was produced with it.
    """
    converted = cast(
        xr.DataArray,
        ucat.convert_columns(
            column,
            from_unit=column_units,
            to_unit=to_unit,
            molar_mass=gas,
            warn=False,
        ),
    )
    return converted.fillna(0.0).clip(min=0.0)


def _optical_depth_kernel(
    amount: np.ndarray, nodes: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Interpolate band optical depths for pixel path amounts."""
    output = np.empty((*amount.shape, table.shape[1]), dtype=np.float32)
    for index in range(table.shape[1]):
        output[..., index] = np.interp(amount, nodes, table[:, index])
    return output


def _select_optical_depth(lut: xr.Dataset, wavelength: xr.DataArray) -> xr.DataArray:
    """Select exact LUT wavelengths.

    Raises:
        ValueError: If any requested wavelength is absent.
    """
    requested = np.asarray(wavelength.values)
    available = np.asarray(lut["wavelength"].values)
    missing = requested[~np.isin(requested, available)]
    if missing.size:
        preview = ", ".join(f"{value:.3f}" for value in missing[:5])
        suffix = "..." if missing.size > 5 else ""
        raise ValueError(
            f"Cube wavelengths [{preview}{suffix}] nm are absent from the LUT; "
            "cube wavelengths must be an exact subset."
        )
    return lut["optical_depth"].sel(wavelength=wavelength)


def embed_column_in_radiance(
    radiance: xr.DataArray,
    simulated_column: xr.DataArray,
    lut: xr.Dataset,
    *,
    gas: str = "ch4",
    column_convention: ColumnConvention = "vertical",
) -> xr.DataArray:
    """Attenuate radiance with a Lambert-Beer plume.

    The LUT optical depths are the total attenuation caused by a vertical
    column, because the air mass factor cancels in
    ``sigma_slant * slant_amount``. The path amount of each pixel is therefore
    matched against ``vertical_amount`` for a vertical column and against
    ``slant_amount`` for a selected-path slant column. The matched filter
    retrieves vertical columns, so ``column_convention="vertical"`` keeps the
    injected truth and the retrieval on the same convention.

    The column must share the radiance spatial grid, and radiance wavelengths
    must be an exact LUT subset. Optical depth is negative by matched-filter
    convention, so transmittance is ``exp(optical_depth)``.

    Args:
        radiance: Radiance cube carrying a one-dimensional ``wavelength``
            coordinate.
        simulated_column: Column on the radiance spatial grid. Its ``units``
            attribute defines the input units.
        lut: Absorption table from
            :func:`spectra.spectral.lut.build_slant_absorption_lut`.
        gas: Gas name used for unit conversion.
        column_convention: Path convention of the embedded column.

    Returns:
        Attenuated radiance with the original name and metadata.

    Raises:
        ValueError: If units, grids, wavelengths, conventions, or LUT coverage
            are invalid.
    """
    if column_convention not in ("vertical", "slant"):
        raise ValueError(
            "Invalid column_convention "
            f"{column_convention!r}. Choose 'vertical' or 'slant'."
        )
    column_units = simulated_column.attrs.get("units")
    if column_units is None:
        raise ValueError(
            "Units of 'simulated_column' are unknown. Set its 'units' attribute."
        )

    band_dim = _band_dimension(radiance)
    spatial_dims = tuple(dim for dim in radiance.dims if dim != band_dim)
    _validate_spatial_dimensions(simulated_column, spatial_dims, "simulated_column")
    radiance, simulated_column = xr.align(radiance, simulated_column, join="exact")
    simulated_column = simulated_column.transpose(*spatial_dims)

    wavelength = radiance["wavelength"]
    optical_depth = _select_optical_depth(lut, wavelength)
    amount = _nonnegative_column(simulated_column, column_units, gas, PPM_METRE_UNITS)
    amount_coordinate = (
        "vertical_amount" if column_convention == "vertical" else "slant_amount"
    )
    nodes = np.asarray(lut[amount_coordinate].values, dtype=float)

    maximum = float(amount.max())
    if maximum > nodes[-1]:
        raise ValueError(
            f"Plume reaches {maximum:.1f} {PPM_METRE_UNITS} ({column_convention}), "
            f"beyond the tabulated maximum of {nodes[-1]:.1f} {PPM_METRE_UNITS}. "
            "Extend the enhancement nodes rather than letting the interpolation "
            "clamp and silently under-attenuate the plume."
        )

    table = np.concatenate(
        [
            np.zeros((1, wavelength.size), dtype=float),
            np.asarray(optical_depth.transpose("enhancement", band_dim).values),
        ],
        axis=0,
    )
    nodes = np.concatenate([[0.0], nodes])

    if radiance.chunks is not None:
        amount = amount.chunk({dim: radiance.chunksizes[dim] for dim in amount.dims})

    depth = xr.apply_ufunc(
        _optical_depth_kernel,
        amount,
        xr.DataArray(nodes, dims=("node",)),
        xr.DataArray(table, dims=("node", band_dim)),
        input_core_dims=[[], ["node"], ["node", band_dim]],
        output_core_dims=[[band_dim]],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {band_dim: wavelength.size}},
    )

    transmittance = np.exp(depth)
    embedded = (radiance * transmittance).transpose(*radiance.dims)
    embedded.name = radiance.name
    embedded.attrs = dict(radiance.attrs)
    return embedded


def embed_column_in_product(
    observed_column: xr.DataArray,
    simulated_column: xr.DataArray,
    *,
    output_units: str | None = None,
    gas: str = "ch4",
) -> xr.DataArray:
    """Add a simulated plume column to an observed column.

    Args:
        observed_column: Retrieved column on the product's spatial grid. Its
            ``units`` attribute defines the input units.
        simulated_column: Simulated column on the same spatial grid. Its
            ``units`` attribute defines the input units.
        output_units: Units of the combined column. The observed units are used
            when omitted.
        gas: Gas name used for unit conversion.

    Returns:
        Combined observed and simulated column in the requested units.

    Raises:
        ValueError: If units or the spatial grids are invalid.
    """
    observed_units = observed_column.attrs.get("units")
    if observed_units is None:
        raise ValueError(
            "Units of 'observed_column' are unknown. Set its 'units' attribute."
        )
    simulated_units = simulated_column.attrs.get("units")
    if simulated_units is None:
        raise ValueError(
            "Units of 'simulated_column' are unknown. Set its 'units' attribute."
        )
    units = output_units or observed_units

    spatial_dims = tuple(observed_column.dims)
    _validate_spatial_dimensions(simulated_column, spatial_dims, "simulated_column")

    observed_column, simulated_column = xr.align(
        observed_column, simulated_column, join="exact"
    )
    simulated_column = simulated_column.transpose(*spatial_dims)

    observed = cast(
        xr.DataArray,
        ucat.convert_columns(
            observed_column,
            from_unit=observed_units,
            to_unit=units,
            molar_mass=gas,
            warn=False,
        ),
    )
    simulated = _nonnegative_column(simulated_column, simulated_units, gas, units)

    embedded = (observed + simulated).transpose(*spatial_dims)
    embedded.name = observed_column.name
    embedded.attrs = dict(observed_column.attrs)
    embedded.attrs["units"] = units
    return embedded


def injected_column(
    column: xr.DataArray,
    *,
    column_units: str | None = None,
    gas: str = "ch4",
    convention: ColumnConvention | None = None,
) -> xr.DataArray:
    """Return the injected plume as a labelled, non-negative mass column.

    The result is the ground truth stored next to embedded data, so it is
    always in kg m-2, free of gaps and negatives, and single precision. It is
    named ``injected_{gas}_mass`` as required by the ``ColumnProduct``
    contract.

    Args:
        column: Simulated column on the embedding grid.
        column_units: Input units for a column without a ``units`` attribute;
            when both are present they must agree.
        gas: Gas name used for unit conversion and labelling.
        convention: Path convention recorded in the attributes, if any.

    Returns:
        Float32 column in kg m-2 with ``units``, ``long_name``, and, when given,
        ``column_convention`` attributes.

    Raises:
        ValueError: If the input units are unknown.
    """
    units = column_units or column.attrs.get("units")
    if units is None:
        raise ValueError("Units of 'column' are unknown. Set its 'units' attribute.")
    injected = _nonnegative_column(column, units, gas, COLUMN_UNITS).astype("float32")
    label = gas.upper()
    injected.name = f"injected_{gas}_mass"
    injected.attrs = {
        "units": "kg m-2",
        "long_name": (
            f"injected {label} column"
            if convention is None
            else f"injected {convention} {label} column"
        ),
    }
    if convention is not None:
        injected.attrs["column_convention"] = convention
    return injected


def embedding_provenance(
    method: EmbeddingMethod,
    geometric_amf: float | None = None,
    lut: xr.Dataset | None = None,
    column_convention: ColumnConvention | None = None,
    *,
    sensor: str | None = None,
    regridder: str | None = None,
) -> dict[str, object]:
    """Return embedding provenance attributes for an embedded dataset.

    Args:
        method: Embedding method, ``"product"`` or ``"radiance"``.
        geometric_amf: Geometric air mass factor of the radiance embedding.
        lut: Absorption table whose descriptive attributes are copied with an
            ``embedding_`` prefix.
        column_convention: Path convention of the embedded column.
        sensor: Registry name of the sensor adapter, if any.
        regridder: Registry name of the regridder that placed the column.

    Returns:
        Attributes prefixed with ``embedding_`` plus ``sensor`` when given.
    """
    attrs: dict[str, object] = {
        "embedding_method": method,
        "embedding_version": EMBEDDING_VERSION,
        "embedding_spectra_version": spectra.__version__,
    }
    if column_convention is not None:
        attrs["embedding_column_convention"] = column_convention
    if geometric_amf is not None:
        attrs["embedding_geometric_air_mass_factor"] = geometric_amf
    if lut is not None:
        for key in (
            "gas",
            "scenario",
            "layer_height",
            "effective_layer_height",
            "radiance_leg",
            "sza",
            "vza",
            "amf_convention",
        ):
            if key in lut.attrs:
                attrs[f"embedding_{key}"] = lut.attrs[key]
    if sensor is not None:
        attrs["sensor"] = sensor
    if regridder is not None:
        attrs["embedding_regridder"] = regridder
    return attrs


__all__ = [
    "COLUMN_UNITS",
    "EMBEDDING_VERSION",
    "PPM_METRE_UNITS",
    "ColumnConvention",
    "EmbeddingMethod",
    "embed_column_in_product",
    "embed_column_in_radiance",
    "embedding_provenance",
    "injected_column",
]
