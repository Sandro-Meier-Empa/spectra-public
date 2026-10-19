"""Analytic, resource-free forward model for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.geometry.amf import effective_layer_height, path_amf
from spectra.spectral.lut import build_slant_absorption_lut
from spectra.types import RadianceLeg, SpectralResponse, ViewingGeometry

# FWHM of a Gaussian in units of its standard deviation, 2 sqrt(2 ln 2).
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class ToyForwardModel:
    """Sum of Gaussian absorption lines with a simple saturation law.

    The unit absorption spectrum is

    ``unit_abs_spect(lambda) = -sum_i s_i exp(-(lambda - lambda_i)^2 / (2 sigma^2))
    / (1 + amount / saturation_ppm_m)``

    with ``sigma = width_nm / (2 sqrt(2 ln 2))`` and the slant path amount
    ``amount = enhancement_ppm * effective_layer_height * geometric AMF``. The
    spectrum is negative by convention and weakens with the path amount, so
    the absorption table is nonlinear like a real one. The sensor slit
    function is ignored; the lines are evaluated at the band centres.

    Satisfies :class:`spectra.protocols.ForwardModel`.

    Args:
        lines_nm: Line centre wavelengths in nanometres.
        strength: Peak unit absorption of each line in ppm-1 m-1.
        width_nm: Full width at half maximum of the lines in nanometres.
        saturation_ppm_m: Slant path amount at which the lines weaken to half.
        spectral_range_nm: Wavelength range the model covers.
        enhancements_ppm: Absorption-table nodes in ppm.
        layer_height_m: Nominal lowest-layer height in metres.
    """

    name = "toy"

    def __init__(
        self,
        *,
        lines_nm: Sequence[float] = (2200.0, 2300.0, 2370.0),
        strength: Sequence[float] = (2.0e-5, 3.0e-5, 2.5e-5),
        width_nm: float = 10.0,
        saturation_ppm_m: float = 1.0e6,
        spectral_range_nm: Sequence[float] = (1000.0, 2500.0),
        enhancements_ppm: Sequence[float] = tuple(
            np.logspace(np.log10(0.01), np.log10(200.0), 12)
        ),
        layer_height_m: float = 1000.0,
    ) -> None:
        """Validate and store the line parameters."""
        self.lines_nm = np.asarray(lines_nm, dtype=float)
        self.strength = np.asarray(strength, dtype=float)
        if self.lines_nm.ndim != 1 or self.lines_nm.shape != self.strength.shape:
            raise ContractError("lines_nm and strength must be equally long 1-D.")
        if np.any(self.strength <= 0):
            raise ContractError("strength must be positive.")
        if width_nm <= 0 or saturation_ppm_m <= 0:
            raise ContractError("width_nm and saturation_ppm_m must be positive.")
        lower, upper = (float(value) for value in spectral_range_nm)
        if not lower < upper:
            raise ContractError("spectral_range_nm must be an increasing pair.")
        nodes = np.asarray(enhancements_ppm, dtype=float)
        if nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0):
            raise ContractError("enhancements_ppm must be strictly increasing.")
        if nodes[0] <= 0 or layer_height_m <= 0:
            raise ContractError("Nodes and layer_height_m must be positive.")
        self.width_nm = float(width_nm)
        self.saturation_ppm_m = float(saturation_ppm_m)
        self._range = (lower, upper)
        self._nodes = tuple(float(node) for node in nodes)
        self._layer_height = float(layer_height_m)

    @property
    def spectral_range_nm(self) -> tuple[float, float]:
        """Wavelength range the model covers."""
        return self._range

    @property
    def enhancements_ppm(self) -> tuple[float, ...]:
        """Absorption-table nodes in ppm."""
        return self._nodes

    @property
    def layer_height_m(self) -> float:
        """Nominal lowest-layer height in metres."""
        return self._layer_height

    def check(self, gas: str) -> None:
        """Accept any gas; the toy model needs no resources."""

    def line_shape(self, wavelength: np.ndarray) -> np.ndarray:
        """Return the unsaturated, positive line profile sum on ``wavelength``."""
        sigma = self.width_nm / FWHM_TO_SIGMA
        wavelength = np.asarray(wavelength, dtype=float)
        shape = np.zeros_like(wavelength)
        for centre, strength in zip(self.lines_nm, self.strength, strict=True):
            shape += strength * np.exp(-((wavelength - centre) ** 2) / (2 * sigma**2))
        return shape

    def lut(
        self, response: SpectralResponse, geometry: ViewingGeometry, *, gas: str
    ) -> xr.Dataset:
        """Return the absorption table of one geometry."""
        model = _ToyModel(self, response, geometry, gas.upper())
        return build_slant_absorption_lut(
            model, self.enhancements_ppm, layer_height=self.layer_height_m
        )

    def provenance(self) -> dict[str, Any]:
        """Return the settings recorded as ``spectra_forward_model_*`` attributes."""
        return {
            "name": self.name,
            "lines_nm": self.lines_nm.tolist(),
            "strength": self.strength.tolist(),
            "width_nm": self.width_nm,
            "saturation_ppm_m": self.saturation_ppm_m,
            "wmin_nm": self._range[0],
            "wmax_nm": self._range[1],
            "layer_height_m": self._layer_height,
            "n_enhancements": len(self._nodes),
        }


@dataclass(frozen=True, eq=False)
class _ToyModel:
    """:class:`~spectra.spectral.lut.UnitAbsorptionModel` view of the toy model."""

    model: ToyForwardModel
    response: SpectralResponse
    geometry: ViewingGeometry
    gas: str

    @property
    def sza(self) -> float:
        return float(self.geometry.sza)

    @property
    def vza(self) -> float:
        return float(self.geometry.vza)

    @property
    def elevation(self) -> float:
        return float(self.geometry.surface_elevation)

    @property
    def cw(self) -> np.ndarray:
        return self.response.wavelength

    def compute_unit_absorption_spectrum(
        self,
        enhancement: float,
        layer_height: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        height = effective_layer_height(self.elevation, layer_height)
        amf = path_amf(self.sza, self.vza, radiance_leg)
        amount = float(enhancement) * height * amf
        values = -self.model.line_shape(self.response.wavelength) / (
            1.0 + amount / self.model.saturation_ppm_m
        )
        return xr.Dataset(
            {"unit_abs_spect": (("wavelength",), values, {"units": "ppm-1 m-1"})},
            coords={"wavelength": (("wavelength",), self.response.wavelength)},
        )


__all__ = ["FWHM_TO_SIGMA", "ToyForwardModel"]
