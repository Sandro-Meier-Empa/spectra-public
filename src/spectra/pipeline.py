"""Embed one plume snapshot into one scene: placement, regridding, embedding.

The functions here are the building blocks shared by the ``spectra embed``
command and the recipe-driven generator. Sensor-specific behaviour enters only
through the adapter, provider, retriever and regridder plugins.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

import spectra
from spectra.contracts.column_product import (
    CANONICAL_COLUMN_UNITS,
    convert_column_units,
)
from spectra.contracts.errors import ContractError
from spectra.contracts.sensor_scene import select_bands
from spectra.protocols import Regridder, Retriever, UnitAbsorptionProvider
from spectra.types import (
    Placement,
    RadianceLeg,
    RetrievalSettings,
    SpectralResponse,
    ViewingGeometry,
    Window,
)

EmbeddingMethod = Literal["radiance", "product"]


@dataclass(frozen=True)
class EmbeddingSpec:
    """Settings of the Lambert-Beer radiance embedding.

    Attributes:
        gas: Lower-case gas name used in variable names and unit conversion.
        column_convention: Path convention of the injected column.
        enhancements_ppm: Absorption-table nodes in ppm (positive, increasing).
        layer_height_m: Nominal lowest-layer height of the forward model.
        radiance_leg: Photon-path segment containing the plume.
    """

    gas: str = "ch4"
    column_convention: Literal["vertical", "slant"] = "vertical"
    enhancements_ppm: tuple[float, ...] = tuple(
        np.logspace(np.log10(0.01), np.log10(200.0), 12)
    )
    layer_height_m: float = 1000.0
    radiance_leg: RadianceLeg = "both"

    def __post_init__(self) -> None:
        """Validate the table nodes."""
        nodes = np.asarray(self.enhancements_ppm, dtype=float)
        if nodes.ndim != 1 or nodes.size == 0 or np.any(nodes <= 0):
            raise ContractError("enhancements_ppm must be positive values.")
        if np.any(np.diff(nodes) <= 0):
            raise ContractError("enhancements_ppm must be strictly increasing.")


def snapshot_column(
    dataset: xr.Dataset, *, time_index: int = 0, emission_scale: float = 1.0
) -> xr.DataArray:
    """Return one scaled ``(y, x)`` column snapshot of a plume-column dataset.

    Accepts a full ``PlumeColumn`` (with ``time``) or an already selected
    snapshot; the ``units`` attribute is required.
    """
    from spectra.contracts.plume_column import scale_emission, select_time

    data = dataset
    if "time" in data.dims:
        data = select_time(data, time_index)
    if emission_scale != 1.0:
        data = scale_emission(data, emission_scale)
    if "column" not in data:
        raise ContractError("Plume dataset has no 'column' variable.")
    column = data["column"]
    if "units" not in column.attrs:
        raise ContractError("Plume column lacks a 'units' attribute.")
    column = column.copy()
    column.attrs.update(dict(data.attrs))
    return column


def place_and_regrid(
    scene: xr.Dataset,
    column: xr.DataArray,
    placement: Placement,
    *,
    regridder: Regridder,
    window: Window | None = None,
) -> xr.DataArray:
    """Place a column snapshot on the scene grid and regrid it conservatively.

    Returns:
        The column on the sensor grid, cropped to the embedding window, in
        ``kg m-2``, with placement and regridding provenance in ``attrs``.
    """
    regridded = regridder.to_scene(column, scene, placement, window=window)
    injected = convert_column_units(
        regridded, CANONICAL_COLUMN_UNITS, gas=str(column.attrs.get("gas", "CH4"))
    )
    injected.name = f"injected_{column.attrs.get('gas', 'CH4').lower()}_mass"
    injected.attrs.update(placement.attrs())
    for key in (
        "case_id",
        "tracer",
        "source_id",
        "time_index",
        "simulation_time_s",
        "emission_scale",
    ):
        if key in column.attrs:
            injected.attrs[f"plume_{key}"] = column.attrs[key]
    return injected


def scene_window(
    scene: xr.Dataset, injected: xr.DataArray
) -> tuple[xr.Dataset, Window]:
    """Return the scene cropped to the window the injected column covers."""
    window = Window.from_attrs(injected.attrs, prefix="window_")
    return scene.isel(window.indexers), window


def build_lut(
    provider: UnitAbsorptionProvider,
    scene: xr.Dataset,
    spec: EmbeddingSpec,
    *,
    window: Window | None = None,
) -> xr.Dataset:
    """Build the absorption table of a scene window with a provider."""
    geometry = ViewingGeometry.from_scene(scene, window=window)
    response = SpectralResponse.from_scene(scene)
    return provider.lut(
        response,
        geometry,
        enhancements_ppm=spec.enhancements_ppm,
        gas=spec.gas.upper(),
        layer_height_m=spec.layer_height_m,
        radiance_leg=spec.radiance_leg,
    )


def embed_radiance(
    scene_window: xr.Dataset,
    injected: xr.DataArray,
    lut: xr.Dataset,
    *,
    spec: EmbeddingSpec,
) -> xr.Dataset:
    """Attenuate the window radiance with the injected column (Lambert-Beer)."""
    from spectra.spectral.embedding import (
        embed_column_in_radiance,
        embedding_provenance,
        injected_column,
    )

    column = injected.drop_vars(
        [name for name in injected.coords if name not in scene_window.coords],
        errors="ignore",
    )
    attenuated = embed_column_in_radiance(
        scene_window["radiance"],
        column,
        lut,
        gas=spec.gas,
        column_convention=spec.column_convention,
    )
    embedded = scene_window.copy()
    embedded["radiance"] = attenuated
    embedded[f"injected_{spec.gas}_mass"] = injected_column(
        column, gas=spec.gas, convention=spec.column_convention
    )
    embedded[f"injected_{spec.gas}_mass"].attrs.update(
        {key: value for key, value in injected.attrs.items() if key != "units"}
    )
    embedded.attrs.update(
        embedding_provenance(
            method="radiance",
            geometric_amf=lut.attrs.get("geometric_air_mass_factor"),
            lut=lut,
            column_convention=spec.column_convention,
            sensor=str(scene_window.attrs.get("sensor", "")),
            regridder=str(injected.attrs.get("regrid_method", "")),
        )
    )
    return embedded


_PROVENANCE_PREFIXES = ("window_", "regrid_", "plume_", "anchor_", "placement_")
_PROVENANCE_KEYS = ("rotation_deg", "flip", "emission_scale", "time_index")


def _provenance(attrs: Mapping[Any, Any]) -> dict[Any, Any]:
    """Return the window, regridding and placement attributes of an injected column."""
    return {
        key: value
        for key, value in attrs.items()
        if key.startswith(_PROVENANCE_PREFIXES) or key in _PROVENANCE_KEYS
    }


def retrieve_embedded(
    embedded: xr.Dataset,
    target: xr.Dataset,
    *,
    retriever: Retriever,
    settings: RetrievalSettings,
    gas: str = "ch4",
    background_stats: xr.Dataset | None = None,
) -> xr.Dataset:
    """Retrieve the embedded window and carry the injected truth and provenance."""
    product = retriever.retrieve(
        embedded, target, settings=settings, background_stats=background_stats
    )
    injected = f"injected_{gas}_mass"
    if injected in embedded:
        product[injected] = embedded[injected]
    product.attrs.update(
        {
            key: value
            for key, value in embedded.attrs.items()
            if key.startswith("embedding_")
        }
    )
    if injected in embedded:
        product.attrs.update(_provenance(embedded[injected].attrs))
    product.attrs["embedding_method"] = "radiance"
    return product


def embed_product(
    product_window: xr.Dataset,
    injected: xr.DataArray,
    *,
    gas: str = "ch4",
) -> xr.Dataset:
    """Add the injected column to a retrieved column product (additive method)."""
    from spectra.spectral.embedding import (
        embed_column_in_product,
        embedding_provenance,
        injected_column,
    )

    name = f"{gas}_mass"
    if name not in product_window:
        raise ContractError(f"Product lacks '{name}'.")
    column = injected.drop_vars(
        [c for c in injected.coords if c not in product_window.coords], errors="ignore"
    )
    embedded = product_window.copy()
    embedded[name] = embed_column_in_product(
        product_window[name], column, output_units=CANONICAL_COLUMN_UNITS, gas=gas
    )
    embedded[f"injected_{gas}_mass"] = injected_column(
        column, gas=gas, convention="vertical"
    )
    embedded[f"injected_{gas}_mass"].attrs.update(
        {key: value for key, value in injected.attrs.items() if key != "units"}
    )
    embedded.attrs.update(
        embedding_provenance(
            method="product",
            sensor=str(product_window.attrs.get("sensor", "")),
            regridder=str(injected.attrs.get("regrid_method", "")),
        )
    )
    embedded.attrs.update(_provenance(injected.attrs))
    embedded.attrs["embedding_method"] = "product"
    return embedded


def embed_sample(
    scene: xr.Dataset,
    column: xr.DataArray,
    placement: Placement,
    *,
    method: EmbeddingMethod,
    regridder: Regridder,
    spec: EmbeddingSpec | None = None,
    provider: UnitAbsorptionProvider | None = None,
    retriever: Retriever | None = None,
    settings: RetrievalSettings | None = None,
    target: xr.Dataset | None = None,
    product: xr.Dataset | None = None,
    window: Window | None = None,
) -> xr.Dataset:
    """Run the full embedding of one snapshot into one scene.

    Args:
        scene: ``SensorScene`` (radiance method) or any dataset carrying the
            geometry the regridder needs (product method).
        column: ``(y, x)`` plume snapshot with ``units``.
        placement: Anchor, rotation and reflection.
        method: ``"radiance"`` (Lambert-Beer + retrieval) or ``"product"``.
        regridder: Conservative regridder plugin.
        spec: Radiance-embedding settings.
        provider: Unit-absorption provider (radiance method).
        retriever: Retriever plugin (radiance method).
        settings: Retrieval settings (radiance method).
        target: Retrieval target spectrum; built with ``provider`` when omitted.
        product: Retrieved column product in sensor space (product method).
        window: Optional explicit embedding window.

    Returns:
        Column product on the embedding window carrying ``injected_*_mass``
        and the full provenance.
    """
    spec = spec or EmbeddingSpec()
    injected = place_and_regrid(
        scene, column, placement, regridder=regridder, window=window
    )
    if method == "product":
        if product is None:
            raise ContractError("The product method needs the retrieved product.")
        product_window, _ = scene_window(product, injected)
        return embed_product(product_window, injected, gas=spec.gas)
    if provider is None or retriever is None:
        raise ContractError("The radiance method needs a provider and a retriever.")
    settings = settings or RetrievalSettings(statistics_scope="window")
    window_scene, _ = scene_window(scene, injected)
    # Forward models cover the retrieval windows only; embedding other bands
    # would be wasted work (and some providers reject bands outside their range).
    window_scene = select_bands(window_scene, settings.band_windows_nm)
    lut = build_lut(provider, window_scene, spec)
    embedded = embed_radiance(window_scene, injected, lut, spec=spec)
    if target is None:
        geometry = ViewingGeometry.from_scene(window_scene)
        target = provider.spectrum(
            SpectralResponse.from_scene(window_scene),
            geometry,
            enhancement_ppm=settings.target_enhancement_ppm,
            gas=spec.gas.upper(),
            layer_height_m=spec.layer_height_m,
            radiance_leg=spec.radiance_leg,
        )
    retrieved = retrieve_embedded(
        embedded, target, retriever=retriever, settings=settings, gas=spec.gas
    )
    retrieved.attrs["spectra_version"] = spectra.__version__
    return retrieved


def log_nodes(minimum: float, maximum: float, count: int) -> tuple[float, ...]:
    """Return logarithmically spaced absorption-table nodes."""
    if minimum <= 0 or maximum <= minimum or count < 2:
        raise ContractError("Nodes need 0 < minimum < maximum and at least two nodes.")
    return tuple(
        float(value)
        for value in np.logspace(np.log10(minimum), np.log10(maximum), count)
    )


__all__ = [
    "EmbeddingMethod",
    "EmbeddingSpec",
    "build_lut",
    "embed_product",
    "embed_radiance",
    "embed_sample",
    "log_nodes",
    "place_and_regrid",
    "retrieve_embedded",
    "scene_window",
    "snapshot_column",
]
