"""Unit-absorption provider wrapping ``aviris4_ch4.mf.UnitAbsorption``.

The forward model combines libRadtran molecular optical depths, ARTS
standard-atmosphere profiles and the TSIS-1 hybrid solar reference spectrum.
It is only available with the ``aviris4`` extra; the import happens lazily so
the registry can list the provider without it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from spectra.contracts.errors import MissingExtraError
from spectra.spectral.lut import build_slant_absorption_lut
from spectra.spectral.providers.base import UnitAbsorptionProviderBase
from spectra.types import RadianceLeg, SpectralResponse, ViewingGeometry

if TYPE_CHECKING:
    from aviris4_ch4.mf import UnitAbsorption

logger = logging.getLogger(__name__)

# Solar zenith angle difference between the scene geometry and the model's
# ephemeris above which the mismatch is reported.
_SZA_WARNING_DEG = 0.5


def _window_bound(value: float) -> int | float:
    """Return a spectral window bound in the spelling used by the file names.

    ``aviris4_ch4`` builds the libRadtran file name from ``wmin``/``wmax``, so
    ``1000.0`` has to become ``1000`` to match ``..._1000-2500nm-0.001nm.nc``.
    """
    number = float(value)
    return int(number) if number.is_integer() else number


class LibRadtranArtsProvider(UnitAbsorptionProviderBase):
    """Unit absorption spectra from libRadtran optical depths and ARTS profiles.

    The wrapped model computes the solar zenith angle itself from ``lat``,
    ``lon`` and ``time_utc`` of the geometry (astropy ephemeris); the
    ``sza`` of the :class:`~spectra.types.ViewingGeometry` is ignored and the
    model's value is recorded in the ``sza`` attributes. A warning is logged
    when the two differ by more than half a degree.

    Models are cached per gas, band arrays and geometry because the
    constructor loads the solar reference spectrum.

    Args:
        uvspec_path: Folder containing the libRadtran optical-depth files.
        sol_reference_path: Folder containing the solar reference spectrum.
        arts_path: Folder containing the ARTS standard-atmosphere profiles.
        scenario: Atmospheric scenario folder and file-name prefix.
        rho: Lambertian surface albedo of the simple radiance model.
        wmin: Minimum wavelength of the modelled window in nanometres.
        wmax: Maximum wavelength of the modelled window in nanometres.
        xch4_bg: Background column-averaged CH4 dry-air mole fraction in ppm.
        xco2_bg: Background column-averaged CO2 dry-air mole fraction in ppm.
        xh2o_bg: Background column-averaged H2O dry-air mole fraction in ppm.
    """

    name = "libradtran_arts"

    def __init__(
        self,
        uvspec_path: str | Path,
        sol_reference_path: str | Path,
        arts_path: str | Path,
        *,
        scenario: str = "midlatitude-summer",
        rho: float = 1.0,
        wmin: float = 1000.0,
        wmax: float = 2500.0,
        xch4_bg: float | None = None,
        xco2_bg: float | None = None,
        xh2o_bg: float | None = None,
    ) -> None:
        """Store the resource paths and model settings."""
        self.uvspec_path = Path(uvspec_path)
        self.sol_reference_path = Path(sol_reference_path)
        self.arts_path = Path(arts_path)
        self.scenario = scenario
        self.rho = float(rho)
        self.wmin = _window_bound(wmin)
        self.wmax = _window_bound(wmax)
        self.xch4_bg = xch4_bg
        self.xco2_bg = xco2_bg
        self.xh2o_bg = xh2o_bg
        self._models: dict[tuple[object, ...], UnitAbsorption] = {}

    @property
    def provider_version(self) -> str:
        """Version of ``aviris4_ch4`` providing the forward model."""
        try:
            return version("aviris4-ch4")
        except PackageNotFoundError:  # pragma: no cover - extra not installed
            return "unknown"

    @staticmethod
    def _cache_key(
        response: SpectralResponse, geometry: ViewingGeometry, gas: str
    ) -> tuple[object, ...]:
        return (
            gas,
            response.wavelength.tobytes(),
            response.fwhm.tobytes(),
            float(geometry.vza),
            float(geometry.surface_elevation),
            float(geometry.sensor_altitude),
            float(geometry.lat),
            float(geometry.lon),
            str(np.datetime64(geometry.time_utc)),
        )

    def model(
        self,
        response: SpectralResponse,
        geometry: ViewingGeometry,
        *,
        gas: str = "CH4",
    ) -> UnitAbsorption:
        """Return the cached ``aviris4_ch4`` forward model for one geometry.

        Args:
            response: Band centres and widths of the sensor.
            geometry: Observation geometry; ``sza`` is ignored, see the class
                docstring.
            gas: Gas name; upper-cased to match the resource file names.

        Returns:
            Configured ``aviris4_ch4.mf.UnitAbsorption``.

        Raises:
            MissingExtraError: If ``aviris4_ch4`` is not installed.
        """
        try:
            from aviris4_ch4.mf import UnitAbsorption
        except ImportError as error:
            raise MissingExtraError("aviris4", "aviris4_ch4", error) from error

        gas = gas.upper()
        key = self._cache_key(response, geometry, gas)
        model = self._models.get(key)
        if model is None:
            model = UnitAbsorption(
                gas=gas,
                scenario=self.scenario,
                lat=geometry.lat,
                lon=geometry.lon,
                time_utc=geometry.time_utc,
                wmin=self.wmin,
                wmax=self.wmax,
                vza=geometry.vza,
                cw=response.wavelength,
                fwhm=response.fwhm,
                rho=self.rho,
                elevation=geometry.surface_elevation,
                altitude=geometry.sensor_altitude,
                uvspec_path=self.uvspec_path,
                sol_reference_path=self.sol_reference_path,
                arts_path=self.arts_path,
                xch4_bg=self.xch4_bg,
                xco2_bg=self.xco2_bg,
                xh2o_bg=self.xh2o_bg,
            )
            model_sza = float(model.sza)
            if abs(model_sza - float(geometry.sza)) > _SZA_WARNING_DEG:
                logger.warning(
                    "ViewingGeometry.sza=%.3f deg differs from the %.3f deg the "
                    "libRadtran/ARTS model derives from lat/lon/time; the model "
                    "value is used.",
                    float(geometry.sza),
                    model_sza,
                )
            self._models[key] = model
        return model

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
        """Return the model's unit absorption spectrum with the schema attributes.

        The variables and attributes written by ``aviris4_ch4`` are kept; a
        ``fwhm`` coordinate and the ``UnitAbsorptionSpectrum`` attributes are
        added.
        """
        model = self.model(response, geometry, gas=gas)
        dataset = model.compute_unit_absorption_spectrum(
            enhancement=float(enhancement_ppm),
            layer_height=float(layer_height_m),
            radiance_leg=radiance_leg,
        )
        dataset = dataset.assign_coords(
            fwhm=(("wavelength",), response.fwhm, {"units": "nm"})
        )
        dataset.attrs.update(
            self.spectrum_attrs(
                geometry,
                gas=model.gas,
                enhancement_ppm=enhancement_ppm,
                layer_height_m=layer_height_m,
                radiance_leg=radiance_leg,
                sza=model.sza,
            )
        )
        return dataset

    def lut(
        self,
        response: SpectralResponse,
        geometry: ViewingGeometry,
        *,
        enhancements_ppm: Sequence[float],
        gas: str = "CH4",
        layer_height_m: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        """Build the table directly from the native model.

        The output equals ``aviris4_ch4.embedding.build_slant_absorption_lut``
        on the same model, plus ``provider`` and ``provider_version``.
        """
        model = self.model(response, geometry, gas=gas)
        table = build_slant_absorption_lut(
            model,
            enhancements_ppm,
            layer_height=layer_height_m,
            radiance_leg=radiance_leg,
        )
        return self.with_provider_attrs(table)


__all__ = ["LibRadtranArtsProvider"]
