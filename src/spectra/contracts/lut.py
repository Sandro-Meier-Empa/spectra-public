"""Schemas of unit absorption spectra and absorption look-up tables."""

from __future__ import annotations

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError

LUT_DIMS = ("enhancement", "wavelength")
LUT_VARIABLES = ("unit_abs_spect", "sigma_slant", "optical_depth")
LUT_REQUIRED_ATTRS = (
    "gas",
    "layer_height",
    "effective_layer_height",
    "radiance_leg",
    "sza",
    "vza",
    "geometric_air_mass_factor",
    "embedding_version",
)


def validate_lut(dataset: xr.Dataset) -> None:
    """Validate an absorption look-up table.

    Raises:
        :exc:`ContractError`: If variables, coordinates or attributes are missing or
            inconsistent.
    """
    for name in LUT_VARIABLES:
        if name not in dataset:
            raise ContractError(f"Absorption table must contain '{name}'.")
        if tuple(dataset[name].dims) != LUT_DIMS:
            raise ContractError(f"'{name}' must have dimensions {LUT_DIMS}.")
        if not np.all(np.isfinite(dataset[name].values)):
            raise ContractError(f"'{name}' must be finite.")
    enhancement = np.asarray(dataset["enhancement"].values, dtype=float)
    if (
        enhancement.ndim != 1
        or np.any(enhancement <= 0)
        or np.any(np.diff(enhancement) <= 0)
    ):
        raise ContractError("'enhancement' must be positive and strictly increasing.")
    wavelength = np.asarray(dataset["wavelength"].values, dtype=float)
    if np.any(np.diff(wavelength) <= 0):
        raise ContractError("'wavelength' must be strictly increasing.")
    for name in ("vertical_amount", "slant_amount"):
        if name not in dataset.coords or tuple(dataset[name].dims) != ("enhancement",):
            raise ContractError(
                f"Absorption table needs a '{name}(enhancement)' coordinate."
            )
        values = np.asarray(dataset[name].values, dtype=float)
        if np.any(np.diff(values) <= 0):
            raise ContractError(f"'{name}' must be strictly increasing.")
    missing = [name for name in LUT_REQUIRED_ATTRS if name not in dataset.attrs]
    if missing:
        raise ContractError(f"Absorption table is missing attributes: {missing}.")


__all__ = [
    "LUT_DIMS",
    "LUT_REQUIRED_ATTRS",
    "LUT_VARIABLES",
    "validate_lut",
]
