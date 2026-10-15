"""Base class and helpers for unit-absorption providers.

A provider computes unit absorption spectra (``UnitAbsorptionSpectrum``
schema) for a sensor's spectral response and an observation geometry. The
base class derives the absorption look-up table from :meth:`spectrum` by
wrapping the provider in a :class:`~spectra.spectral.lut.UnitAbsorptionModel`
adapter; providers owning a native forward model override :meth:`lut`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

import spectra
from spectra.geometry.amf import effective_layer_height, path_amf
from spectra.spectral.lut import UnitAbsorptionModel, build_slant_absorption_lut
from spectra.types import RadianceLeg, SpectralResponse, ViewingGeometry

UNIT_ABS_SPECT_UNITS = "ppm-1 m-1"
SPECTRUM_ATTRS = (
    "provider",
    "provider_version",
    "gas",
    "enhancement_ppm",
    "layer_height_m",
    "effective_layer_height_m",
    "sza",
    "vza",
    "geometric_amf",
    "radiance_leg",
    "surface_elevation_m",
    "sensor_altitude_m",
    "latitude",
    "longitude",
    "time_utc",
)


class UnitAbsorptionProviderBase(ABC):
    """Abstract provider satisfying :class:`spectra.protocols.UnitAbsorptionProvider`.

    Subclasses set ``name`` and implement :meth:`spectrum`.
    """

    name: str

    @property
    def provider_version(self) -> str:
        """Version string recorded in the ``provider_version`` attribute."""
        return spectra.__version__

    @abstractmethod
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
        """Return a unit absorption spectrum (``UnitAbsorptionSpectrum`` schema).

        Args:
            response: Band centres and widths of the sensor.
            geometry: Observation geometry.
            enhancement_ppm: Lowest-layer enhancement the spectrum is
                evaluated at.
            gas: Gas name.
            layer_height_m: Nominal lowest-layer height in metres.
            radiance_leg: Photon-path segment containing the plume.

        Returns:
            Dataset with ``unit_abs_spect(wavelength)`` in ppm-1 m-1, negative
            by convention, coordinates ``wavelength`` and ``fwhm``, and the
            attributes listed in :data:`SPECTRUM_ATTRS`.
        """

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
        """Return an absorption look-up table (``AbsorptionLut`` schema).

        Evaluates :meth:`spectrum` at every node through a
        :class:`~spectra.spectral.lut.UnitAbsorptionModel` adapter.

        Args:
            response: Band centres and widths of the sensor.
            geometry: Observation geometry.
            enhancements_ppm: Increasing positive nodes in ppm.
            gas: Gas name.
            layer_height_m: Nominal lowest-layer height in metres.
            radiance_leg: Photon-path segment containing the plume.

        Returns:
            Table from :func:`~spectra.spectral.lut.build_slant_absorption_lut`
            with ``provider`` and ``provider_version`` attributes.
        """
        model = model_from_provider(self, response, geometry, gas=gas)
        table = build_slant_absorption_lut(
            model,
            enhancements_ppm,
            layer_height=layer_height_m,
            radiance_leg=radiance_leg,
        )
        return self.with_provider_attrs(table)

    def with_provider_attrs(self, dataset: xr.Dataset) -> xr.Dataset:
        """Record the provider name and version in the dataset attributes."""
        dataset.attrs["provider"] = self.name
        dataset.attrs["provider_version"] = self.provider_version
        return dataset

    def spectrum_attrs(
        self,
        geometry: ViewingGeometry,
        *,
        gas: str,
        enhancement_ppm: float,
        layer_height_m: float,
        radiance_leg: RadianceLeg,
        sza: float | None = None,
    ) -> dict[str, Any]:
        """Return the ``UnitAbsorptionSpectrum`` attributes for one evaluation.

        Args:
            geometry: Observation geometry.
            gas: Gas name.
            enhancement_ppm: Lowest-layer enhancement in ppm.
            layer_height_m: Nominal lowest-layer height in metres.
            radiance_leg: Photon-path segment containing the plume.
            sza: Solar zenith angle actually used by the model when it differs
                from ``geometry.sza``.

        Returns:
            Attributes in the order of :data:`SPECTRUM_ATTRS`; ``time_utc`` is
            an ISO string so the dataset can be written to NetCDF.
        """
        zenith = float(geometry.sza if sza is None else sza)
        return {
            "provider": self.name,
            "provider_version": self.provider_version,
            "gas": gas,
            "enhancement_ppm": float(enhancement_ppm),
            "layer_height_m": float(layer_height_m),
            "effective_layer_height_m": effective_layer_height(
                geometry.surface_elevation, layer_height_m
            ),
            "sza": zenith,
            "vza": float(geometry.vza),
            "geometric_amf": path_amf(zenith, geometry.vza, radiance_leg),
            "radiance_leg": radiance_leg,
            "surface_elevation_m": float(geometry.surface_elevation),
            "sensor_altitude_m": float(geometry.sensor_altitude),
            "latitude": float(geometry.lat),
            "longitude": float(geometry.lon),
            "time_utc": str(np.datetime64(geometry.time_utc)),
        }


def make_spectrum_dataset(
    values: np.ndarray, response: SpectralResponse, attrs: dict[str, Any]
) -> xr.Dataset:
    """Assemble a ``UnitAbsorptionSpectrum`` dataset from band values.

    Args:
        values: Unit absorption spectrum on the response bands in ppm-1 m-1.
        response: Band centres and widths.
        attrs: Dataset attributes, usually from
            :meth:`UnitAbsorptionProviderBase.spectrum_attrs`.

    Returns:
        Dataset with ``unit_abs_spect(wavelength)`` and a ``fwhm`` coordinate.
    """
    return xr.Dataset(
        data_vars={
            "unit_abs_spect": (
                ("wavelength",),
                np.asarray(values, dtype=float),
                {
                    "units": UNIT_ABS_SPECT_UNITS,
                    "long_name": "unit absorption spectrum per vertical path amount",
                },
            )
        },
        coords={
            "wavelength": (("wavelength",), response.wavelength, {"units": "nm"}),
            "fwhm": (("wavelength",), response.fwhm, {"units": "nm"}),
        },
        attrs=dict(attrs),
    )


@dataclass(frozen=True, eq=False)
class ProviderModel:
    """Adapter exposing a provider as a :class:`UnitAbsorptionModel`.

    Attributes:
        provider: Provider evaluating the spectra.
        response: Spectral response passed to the provider.
        geometry: Observation geometry passed to the provider.
        gas: Gas name.
        solar_zenith: Solar zenith angle in degrees used by the provider when
            it differs from ``geometry.sza``.
    """

    provider: UnitAbsorptionProviderBase
    response: SpectralResponse
    geometry: ViewingGeometry
    gas: str = "CH4"
    solar_zenith: float | None = None

    @property
    def sza(self) -> float:
        """Solar zenith angle in degrees."""
        if self.solar_zenith is None:
            return float(self.geometry.sza)
        return float(self.solar_zenith)

    @property
    def vza(self) -> float:
        """Viewing zenith angle in degrees."""
        return float(self.geometry.vza)

    @property
    def elevation(self) -> float:
        """Surface elevation in metres."""
        return float(self.geometry.surface_elevation)

    @property
    def cw(self) -> np.ndarray:
        """Band centre wavelengths in nanometres."""
        return self.response.wavelength

    def compute_unit_absorption_spectrum(
        self,
        enhancement: float,
        layer_height: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        """Evaluate the provider at one enhancement."""
        return self.provider.spectrum(
            self.response,
            self.geometry,
            enhancement_ppm=enhancement,
            gas=self.gas,
            layer_height_m=layer_height,
            radiance_leg=radiance_leg,
        )


def model_from_provider(
    provider: UnitAbsorptionProviderBase,
    response: SpectralResponse,
    geometry: ViewingGeometry,
    *,
    gas: str = "CH4",
    sza: float | None = None,
) -> UnitAbsorptionModel:
    """Wrap a provider so :func:`build_slant_absorption_lut` can evaluate it.

    Args:
        provider: Provider evaluating the spectra.
        response: Spectral response of the sensor.
        geometry: Observation geometry.
        gas: Gas name.
        sza: Solar zenith angle actually used by the provider, if it differs
            from ``geometry.sza``.

    Returns:
        Model whose ``compute_unit_absorption_spectrum`` calls
        ``provider.spectrum``.
    """
    return ProviderModel(
        provider=provider,
        response=response,
        geometry=geometry,
        gas=gas,
        solar_zenith=None if sza is None else float(sza),
    )


__all__ = [
    "SPECTRUM_ATTRS",
    "UNIT_ABS_SPECT_UNITS",
    "ProviderModel",
    "UnitAbsorptionProviderBase",
    "make_spectrum_dataset",
    "model_from_provider",
]
