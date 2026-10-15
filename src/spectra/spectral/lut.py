"""Slant absorption look-up tables for one observation geometry.

Ported from ``aviris4_ch4.embedding.build_slant_absorption_lut``; the table
layout, attributes and arithmetic are unchanged so that tables built from the
same forward model are identical. The forward model is any object satisfying
:class:`UnitAbsorptionModel`, for example ``aviris4_ch4.mf.UnitAbsorption``
or the adapter returned by
:func:`spectra.spectral.providers.base.model_from_provider`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr

from spectra.contracts.lut import validate_lut
from spectra.geometry.amf import effective_layer_height, path_amf
from spectra.spectral.embedding import EMBEDDING_VERSION, PPM_METRE_UNITS
from spectra.types import RadianceLeg

# Attributes copied from the model only when it defines them, so forward models
# other than ``aviris4_ch4.mf.UnitAbsorption`` work without them.
_OPTIONAL_MODEL_ATTRS = ("scenario", "wmin", "wmax")


@runtime_checkable
class UnitAbsorptionModel(Protocol):
    """Forward model evaluating unit absorption spectra for one geometry.

    The members are declared read-only so both plain attributes and
    properties satisfy the protocol.
    """

    @property
    def gas(self) -> str:
        """Gas name."""
        ...

    @property
    def sza(self) -> float:
        """Solar zenith angle in degrees."""
        ...

    @property
    def vza(self) -> float:
        """Viewing zenith angle in degrees."""
        ...

    @property
    def elevation(self) -> float:
        """Surface elevation in metres."""
        ...

    @property
    def cw(self) -> np.ndarray:
        """Band centre wavelengths in nanometres."""
        ...

    def compute_unit_absorption_spectrum(
        self,
        enhancement: float,
        layer_height: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        """Return a dataset with ``unit_abs_spect(wavelength)`` in ppm-1 m-1."""
        ...


def build_slant_absorption_lut(
    unit_absorption: UnitAbsorptionModel,
    enhancements: Sequence[float] | np.ndarray,
    layer_height: float = 1000.0,
    radiance_leg: RadianceLeg = "both",
) -> xr.Dataset:
    """Build a nonlinear absorption table for one observation geometry.

    The table maps path amounts to band optical depth and accounts for
    saturation. The air mass factor cancels in
    ``optical_depth = sigma_slant * slant_amount``, so a tabulated row is the
    total optical depth of a vertical column of ``vertical_amount``; both
    abscissae are stored as coordinates. Retrievals should use an independent
    target spectrum.

    Args:
        unit_absorption: Configured forward model.
        enhancements: Increasing positive nodes in ppm.
        layer_height: Nominal lowest-layer height in metres.
        radiance_leg: Photon-path segment containing the plume.

    Returns:
        Absorption spectra and optical depths with path amounts in ppm m,
        satisfying :func:`spectra.contracts.lut.validate_lut`.

    Raises:
        ValueError: If the nodes or radiance leg are invalid.
        ContractError: If the model produced a table violating the contract.
    """
    nodes = np.atleast_1d(np.asarray(enhancements, dtype=float))
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("enhancements must be a non-empty one-dimensional sequence.")
    if np.any(nodes <= 0.0):
        raise ValueError(
            "Enhancement nodes must be positive; the unit absorption spectrum is "
            "undefined at zero enhancement."
        )
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("Enhancement nodes must be strictly increasing.")

    # Read once: ``aviris4_ch4`` evaluates the solar ephemeris on every access.
    sza = float(unit_absorption.sza)
    vza = float(unit_absorption.vza)
    geometric_amf = path_amf(sza, vza, radiance_leg)
    height = effective_layer_height(unit_absorption.elevation, layer_height)

    spectra = []
    for enhancement in nodes:
        spectrum = unit_absorption.compute_unit_absorption_spectrum(
            enhancement=float(enhancement),
            layer_height=layer_height,
            radiance_leg=radiance_leg,
        )
        spectra.append(np.asarray(spectrum["unit_abs_spect"].values, dtype=float))

    unit_abs_spect = np.atleast_2d(np.stack(spectra, axis=0))
    sigma_slant = unit_abs_spect / geometric_amf

    vertical_amount = nodes * height
    slant_amount = vertical_amount * geometric_amf

    optical_depth = sigma_slant * slant_amount[:, np.newaxis]

    wavelength = np.asarray(unit_absorption.cw, dtype=float)
    attrs: dict[str, object] = {
        "description": f"Slant absorption table for {unit_absorption.gas}",
        "gas": unit_absorption.gas,
    }
    scenario = getattr(unit_absorption, "scenario", None)
    if scenario is not None:
        attrs["scenario"] = scenario
    attrs.update(
        {
            "layer_height": layer_height,
            "effective_layer_height": height,
            "radiance_leg": radiance_leg,
            "sza": sza,
            "vza": vza,
            "geometric_air_mass_factor": geometric_amf,
            "amf_convention": "sigma_slant removes the selected-path AMF",
        }
    )
    for name in ("wmin", "wmax"):
        value = getattr(unit_absorption, name, None)
        if value is not None:
            attrs[name] = value
    attrs["embedding_version"] = EMBEDDING_VERSION

    lut = xr.Dataset(
        data_vars={
            "unit_abs_spect": (
                ["enhancement", "wavelength"],
                unit_abs_spect,
                {
                    "units": "ppm-1 m-1",
                    "long_name": "unit absorption spectrum per vertical path amount",
                },
            ),
            "sigma_slant": (
                ["enhancement", "wavelength"],
                sigma_slant,
                {
                    "units": "ppm-1 m-1",
                    "long_name": "absorption coefficient per slant path amount",
                },
            ),
            "optical_depth": (
                ["enhancement", "wavelength"],
                optical_depth,
                {
                    "units": "1",
                    "long_name": "added optical depth at this node",
                },
            ),
        },
        coords={
            "enhancement": (
                ["enhancement"],
                nodes,
                {"units": "ppm", "long_name": "lowest-layer enhancement"},
            ),
            "wavelength": (["wavelength"], wavelength, {"units": "nm"}),
            "slant_amount": (
                ["enhancement"],
                slant_amount,
                {
                    "units": PPM_METRE_UNITS,
                    "long_name": "selected slant path amount of this node",
                },
            ),
            "vertical_amount": (
                ["enhancement"],
                vertical_amount,
                {
                    "units": PPM_METRE_UNITS,
                    "long_name": "vertical path amount of this node",
                },
            ),
        },
        attrs=attrs,
    )
    validate_lut(lut)
    return lut


__all__ = ["UnitAbsorptionModel", "build_slant_absorption_lut"]
