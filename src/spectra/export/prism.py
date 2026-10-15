"""Export embedded column products to the prism sample contract.

prism owns the canonical machine-learning sample contract (one Zarr store per
sample on ``(y, x)`` in ``kg m-2``). SPECTRA writes products that satisfy it
and, when prism is installed, validates them with prism's own validator.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import spectra
from spectra.contracts.column_product import (
    CANONICAL_COLUMN_UNITS,
    normalise_units,
    to_prism_grid,
    validate_column_product,
)
from spectra.contracts.errors import ContractError

PRISM_EMBEDDING_METHODS = ("product", "radiance", "none")
SAMPLE_CHUNK = 512
_FLOAT_CHANNELS = (
    "ch4_mass",
    "ch4_mass_uncertainty",
    "injected_ch4_mass",
    "pixel_area",
    "albedo",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "sensitivity",
    "wind_u",
    "wind_v",
)


def _prism_validate(dataset: xr.Dataset) -> None:
    """Run prism's validator when prism is importable."""
    try:
        from prism.data.contracts import (  # pyright: ignore[reportMissingImports]
            validate_sample,
        )
    except ImportError:
        return
    validate_sample(dataset, require_target=False)


def to_prism_product(
    product: xr.Dataset,
    *,
    gas: str = "ch4",
    require_injected: bool = True,
    extra_attrs: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> xr.Dataset:
    """Return a column product as a prism-compatible product on ``(y, x)``.

    Args:
        product: Embedded ``ColumnProduct`` in sensor or map geometry space.
        gas: Lower-case gas name used in the variable names.
        require_injected: Require ``injected_{gas}_mass`` (the label source).
        extra_attrs: Attributes merged into the output.
        validate: Validate with prism's ``validate_sample`` when available.

    Returns:
        Dataset with every data variable on ``("y", "x")`` and canonical units.

    Raises:
        ContractError: If the product is invalid or lacks the injected column.
    """
    validate_column_product(product, gas=gas)
    injected = f"injected_{gas}_mass"
    if require_injected and injected not in product:
        raise ContractError(
            f"Product must contain '{injected}'; it is the label source for prism."
        )
    output = to_prism_grid(product)

    dropped = [
        name
        for name, variable in output.data_vars.items()
        if variable.dims != ("y", "x")
    ]
    if dropped:
        warnings.warn(
            f"Dropping non-spatial variables for prism: {sorted(dropped)}", stacklevel=2
        )
        output = output.drop_vars(dropped)

    for name in _FLOAT_CHANNELS:
        if name in output:
            attrs = dict(output[name].attrs)
            output[name] = output[name].astype("float32")
            output[name].attrs = attrs
    for name in (f"{gas}_mass", f"{gas}_mass_uncertainty", injected):
        if name in output:
            units = normalise_units(str(output[name].attrs.get("units", "")))
            if units != CANONICAL_COLUMN_UNITS:
                raise ContractError(f"'{name}' must be in {CANONICAL_COLUMN_UNITS}.")
            output[name].attrs["units"] = CANONICAL_COLUMN_UNITS
    for name in ("sensitivity", "albedo", "rgb_r", "rgb_g", "rgb_b"):
        if name in output and output[name].attrs.get("units") in (None, "", "-"):
            output[name].attrs["units"] = "1"
    for name in ("retrieval_valid", "background_valid"):
        if name in output:
            attrs = dict(output[name].attrs)
            output[name] = output[name].astype(bool)
            output[name].attrs = attrs

    method = output.attrs.get("embedding_method")
    if method not in PRISM_EMBEDDING_METHODS:
        raise ContractError(
            "Product attribute embedding_method must be one of "
            f"{PRISM_EMBEDDING_METHODS}; got {method!r}."
        )
    output.attrs.setdefault("sensor", str(product.attrs.get("sensor", "")))
    if "pixel_size_m" not in output.attrs and "pixel_area" in output:
        area = np.asarray(output["pixel_area"].values, dtype=float)
        output.attrs["pixel_size_m"] = float(np.sqrt(np.nanmedian(area)))
    output.attrs["spectra_version"] = spectra.__version__
    output.attrs["prism_product_version"] = "spectra-prism-v1"
    if extra_attrs:
        output.attrs.update(dict(extra_attrs))
    # NetCDF/Zarr attributes cannot hold None.
    output.attrs = {
        key: value for key, value in output.attrs.items() if value is not None
    }
    if validate:
        _prism_validate(output)
    return output


def write_product(
    dataset: xr.Dataset, uri: str | Path, *, overwrite: bool = False
) -> Path:
    """Write a prism product as consolidated Zarr (``.zarr``) or NetCDF.

    Args:
        dataset: Product from :func:`to_prism_product`.
        uri: Destination path; the suffix selects the format.
        overwrite: Replace an existing store or file.

    Returns:
        The written path.
    """
    path = Path(uri)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing product: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Samples are window sized: load them and give every spatial variable the
    # same chunk layout, otherwise readers that combine variables (prism's
    # sample derivation) end up with irregular dask chunks Zarr rejects.
    encoded = dataset.load().copy()
    for name, variable in encoded.variables.items():
        encoded[name].encoding = {}
        if np.issubdtype(variable.dtype, np.datetime64):
            encoded[name].encoding.update({"units": "seconds since 1970-01-01"})
    if path.suffix == ".zarr":
        if path.exists():
            import shutil

            shutil.rmtree(path)
        chunks = tuple(
            min(int(size), SAMPLE_CHUNK)
            for size in (encoded.sizes.get("y", 1), encoded.sizes.get("x", 1))
        )
        encoding = {
            name: {"chunks": chunks}
            for name, variable in encoded.data_vars.items()
            if tuple(variable.dims) == ("y", "x")
        }
        encoded.to_zarr(path, mode="w", consolidated=True, encoding=encoding)
    else:
        if path.exists():
            path.unlink()
        encoded.to_netcdf(path)
    return path


__all__ = ["PRISM_EMBEDDING_METHODS", "to_prism_product", "write_product"]
