"""Analytic, resource-free unit-absorption provider for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.geometry.amf import effective_layer_height, path_amf
from spectra.spectral.providers.base import (
    UnitAbsorptionProviderBase,
    make_spectrum_dataset,
)
from spectra.types import RadianceLeg, SpectralResponse, ViewingGeometry

# FWHM of a Gaussian in units of its standard deviation, 2 sqrt(2 ln 2).
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class ToyUnitAbsorption(UnitAbsorptionProviderBase):
    """Sum of Gaussian absorption lines with a simple saturation law.

    The unit absorption spectrum is

    ``unit_abs_spect(lambda) = -sum_i s_i exp(-(lambda - lambda_i)^2 / (2 sigma^2))
    / (1 + amount / saturation_ppm_m)``

    with ``sigma = width_nm / (2 sqrt(2 ln 2))`` and the slant path amount
    ``amount = enhancement_ppm * effective_layer_height * geometric AMF`` of
    the selected radiance leg. The spectrum is negative by matched-filter
    convention and weakens with the path amount so the absorption table is
    nonlinear like a real one. The sensor slit function is ignored; the lines
    are evaluated at the band centres.

    Args:
        lines_nm: Line centre wavelengths in nanometres.
        strength: Peak unit absorption of each line in ppm-1 m-1.
        width_nm: Full width at half maximum of the lines in nanometres.
        saturation_ppm_m: Slant path amount at which the lines weaken to half.
        spectral_range_nm: Optional ``(min, max)`` wavelength range the
            provider covers, to mimic forward models with a limited range.
    """

    name = "toy"

    def __init__(
        self,
        *,
        lines_nm: Sequence[float] = (2200.0, 2300.0, 2370.0),
        strength: Sequence[float] = (2.0e-5, 3.0e-5, 2.5e-5),
        width_nm: float = 10.0,
        saturation_ppm_m: float = 1.0e6,
        spectral_range_nm: Sequence[float] | None = None,
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
        self.width_nm = float(width_nm)
        self.saturation_ppm_m = float(saturation_ppm_m)
        self.spectral_range_nm: tuple[float, float] | None = None
        if spectral_range_nm is not None:
            lower, upper = (float(value) for value in spectral_range_nm)
            if not lower < upper:
                raise ContractError("spectral_range_nm must be an increasing pair.")
            self.spectral_range_nm = (lower, upper)

    def line_shape(self, wavelength: np.ndarray) -> np.ndarray:
        """Return the unsaturated, positive line profile sum on ``wavelength``."""
        sigma = self.width_nm / FWHM_TO_SIGMA
        wavelength = np.asarray(wavelength, dtype=float)
        shape = np.zeros_like(wavelength)
        for centre, strength in zip(self.lines_nm, self.strength, strict=True):
            shape += strength * np.exp(-((wavelength - centre) ** 2) / (2 * sigma**2))
        return shape

    def spectrum(
        self,
        response: SpectralResponse,
        geometry: ViewingGeometry,
        *,
        enhancement_ppm: float,
        gas: str = "CH4",
        layer_height_m: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        """Evaluate the analytic spectrum for one enhancement."""
        height = effective_layer_height(geometry.surface_elevation, layer_height_m)
        amf = path_amf(geometry.sza, geometry.vza, radiance_leg)
        amount = float(enhancement_ppm) * height * amf
        values = -self.line_shape(response.wavelength) / (
            1.0 + amount / self.saturation_ppm_m
        )
        attrs = self.spectrum_attrs(
            geometry,
            gas=gas,
            enhancement_ppm=enhancement_ppm,
            layer_height_m=layer_height_m,
            radiance_leg=radiance_leg,
        )
        return make_spectrum_dataset(values, response, attrs)


__all__ = ["FWHM_TO_SIGMA", "ToyUnitAbsorption"]
