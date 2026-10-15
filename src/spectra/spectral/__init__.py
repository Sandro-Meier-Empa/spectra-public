"""Absorption tables, Lambert-Beer embedding and unit-absorption providers."""

from spectra.spectral.embedding import (
    COLUMN_UNITS,
    EMBEDDING_VERSION,
    PPM_METRE_UNITS,
    ColumnConvention,
    EmbeddingMethod,
    embed_column_in_product,
    embed_column_in_radiance,
    embedding_provenance,
    injected_column,
)
from spectra.spectral.lut import UnitAbsorptionModel, build_slant_absorption_lut
from spectra.spectral.providers import (
    LibRadtranArtsProvider,
    ToyUnitAbsorption,
    UnitAbsorptionProviderBase,
    make_spectrum_dataset,
    model_from_provider,
)

__all__ = [
    "COLUMN_UNITS",
    "EMBEDDING_VERSION",
    "PPM_METRE_UNITS",
    "ColumnConvention",
    "EmbeddingMethod",
    "LibRadtranArtsProvider",
    "ToyUnitAbsorption",
    "UnitAbsorptionModel",
    "UnitAbsorptionProviderBase",
    "build_slant_absorption_lut",
    "embed_column_in_product",
    "embed_column_in_radiance",
    "embedding_provenance",
    "injected_column",
    "make_spectrum_dataset",
    "model_from_provider",
]
