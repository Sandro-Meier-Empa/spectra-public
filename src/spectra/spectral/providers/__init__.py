"""Unit-absorption spectrum providers."""

from spectra.spectral.providers.base import (
    SPECTRUM_ATTRS,
    ProviderModel,
    UnitAbsorptionProviderBase,
    make_spectrum_dataset,
    model_from_provider,
)
from spectra.spectral.providers.libradtran_arts import LibRadtranArtsProvider
from spectra.spectral.providers.toy import ToyUnitAbsorption

__all__ = [
    "SPECTRUM_ATTRS",
    "LibRadtranArtsProvider",
    "ProviderModel",
    "ToyUnitAbsorption",
    "UnitAbsorptionProviderBase",
    "make_spectrum_dataset",
    "model_from_provider",
]
