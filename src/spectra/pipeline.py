"""Embed one plume snapshot into one scene: placement, regridding, embedding.

The functions here are the building blocks shared by the ``spectra embed``
command and the recipe-driven generator. Sensor-specific behaviour enters only
through the adapter, provider, retriever and regridder plugins.

The radiance method runs as three steps that can be called one after another:
:func:`place_and_regrid` (locate and regrid the column), :func:`embed_radiance_sample`
(Lambert-Beer embedding into the window or the full scene) and
:func:`retrieve_embedded` (retrieval of the embedded radiance with any
registered retriever). :func:`embed_sample` chains all of them. The embedded
radiance keeps every band it is given; bands the unit-absorption provider does
not cover pass through unattenuated.
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
EmbeddingExtent = Literal["window", "scene"]
EMBEDDING_EXTENTS: tuple[str, ...] = ("window", "scene")


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


def provider_bands(provider: UnitAbsorptionProvider, scene: xr.Dataset) -> xr.Dataset:
    """Return the scene bands inside the provider's optional ``spectral_range_nm``.

    Raises:
        ContractError: If no scene band lies inside the provider's range.
    """
    span = getattr(provider, "spectral_range_nm", None)
    if span is None:
        return scene
    lower, upper = (float(value) for value in span)
    return select_bands(scene, ((lower, upper),))


def target_spectrum(
    provider: UnitAbsorptionProvider,
    scene: xr.Dataset,
    settings: RetrievalSettings,
    spec: EmbeddingSpec,
    *,
    window: Window | None = None,
) -> xr.Dataset:
    """Return the retrieval target spectrum on the scene's retrieval bands.

    Only bands inside ``settings.band_windows_nm`` and the provider's spectral
    range are modelled, so full-spectrum cubes need no forward model outside
    the retrieval windows.
    """
    bands = select_bands(provider_bands(provider, scene), settings.band_windows_nm)
    return provider.spectrum(
        SpectralResponse.from_scene(bands),
        ViewingGeometry.from_scene(scene, window=window),
        enhancement_ppm=settings.target_enhancement_ppm,
        gas=spec.gas.upper(),
        layer_height_m=spec.layer_height_m,
        radiance_leg=spec.radiance_leg,
    )


def build_lut(
    provider: UnitAbsorptionProvider,
    scene: xr.Dataset,
    spec: EmbeddingSpec,
    *,
    window: Window | None = None,
) -> xr.Dataset:
    """Build the absorption table of a scene window with a provider.

    The table covers the scene bands inside the provider's
    ``spectral_range_nm`` (all bands when the provider declares none).
    """
    geometry = ViewingGeometry.from_scene(scene, window=window)
    response = SpectralResponse.from_scene(provider_bands(provider, scene))
    return provider.lut(
        response,
        geometry,
        enhancements_ppm=spec.enhancements_ppm,
        gas=spec.gas.upper(),
        layer_height_m=spec.layer_height_m,
        radiance_leg=spec.radiance_leg,
    )


def _attenuated_bands(radiance: xr.DataArray, lut: xr.Dataset) -> np.ndarray:
    """Return the band mask of the radiance wavelengths tabulated in ``lut``."""
    wavelength = np.asarray(radiance["wavelength"].values, dtype=float)
    covered = np.isin(wavelength, np.asarray(lut["wavelength"].values, dtype=float))
    if not covered.any():
        raise ContractError(
            "The absorption table covers none of the scene wavelengths; build it "
            "for this scene."
        )
    return covered


def _merge_bands(
    radiance: xr.DataArray, attenuated: xr.DataArray, covered: np.ndarray
) -> xr.DataArray:
    """Return ``radiance`` with the ``covered`` bands replaced by ``attenuated``."""
    if covered.all():
        return attenuated
    passthrough = radiance.isel(band=np.flatnonzero(~covered)).astype(attenuated.dtype)
    merged = xr.concat(
        [attenuated, passthrough], dim="band", coords="minimal", compat="override"
    )
    order = np.argsort(
        np.concatenate([np.flatnonzero(covered), np.flatnonzero(~covered)])
    )
    merged = merged.isel(band=order).transpose(*radiance.dims)
    merged.attrs = dict(attenuated.attrs)
    return merged


def embed_radiance(
    scene_window: xr.Dataset,
    injected: xr.DataArray,
    lut: xr.Dataset,
    *,
    spec: EmbeddingSpec,
) -> xr.Dataset:
    """Attenuate the window radiance with the injected column (Lambert-Beer).

    Bands tabulated in ``lut`` are attenuated; the other bands pass through
    unchanged (the forward model does not cover them).
    """
    from spectra.spectral.embedding import (
        embed_column_in_radiance,
        embedding_provenance,
        injected_column,
    )

    column = injected.drop_vars(
        [name for name in injected.coords if name not in scene_window.coords],
        errors="ignore",
    )
    radiance = scene_window["radiance"]
    covered = _attenuated_bands(radiance, lut)
    attenuated = embed_column_in_radiance(
        radiance.isel(band=np.flatnonzero(covered)) if not covered.all() else radiance,
        column,
        lut,
        gas=spec.gas,
        column_convention=spec.column_convention,
    )
    embedded = scene_window.copy()
    embedded["radiance"] = _merge_bands(radiance, attenuated, covered)
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
    wavelength = np.asarray(radiance["wavelength"].values, dtype=float)[covered]
    embedded.attrs.update(
        {
            "embedding_attenuated_band_count": int(covered.sum()),
            "embedding_attenuated_wavelength_min_nm": float(wavelength.min()),
            "embedding_attenuated_wavelength_max_nm": float(wavelength.max()),
        }
    )
    return embedded


def _paste_window(
    full: xr.DataArray, part: xr.DataArray, window: Window, *, fill: float
) -> xr.DataArray:
    """Return ``full`` with its positional ``window`` replaced by ``part`` (lazy)."""
    part = part.transpose(*full.dims)
    widths: dict[Any, tuple[int, int]] = {}
    for dim, piece in (("across", window.across), ("along", window.along)):
        widths[dim] = (int(piece.start), int(full.sizes[dim] - piece.stop))
    padded = xr.DataArray(part.data, dims=part.dims).pad(widths, constant_values=fill)
    inside = np.zeros((full.sizes["across"], full.sizes["along"]), dtype=bool)
    inside[window.across, window.along] = True
    values = xr.where(
        xr.DataArray(inside, dims=("across", "along")),
        padded,
        xr.DataArray(full.data, dims=full.dims),
    )
    return full.copy(data=values.transpose(*full.dims).data.astype(full.dtype))


def embedded_scene(
    scene: xr.Dataset, embedded_window: xr.Dataset, *, gas: str = "ch4"
) -> xr.Dataset:
    """Write an embedded window back into its full scene.

    Pixels outside the window keep their radiance; the injected column is
    zero outside the window. Dask-backed scenes stay lazy.

    Args:
        scene: Full ``SensorScene`` the window was cut from, with the same
            bands as ``embedded_window``.
        embedded_window: Window from :func:`embed_radiance`.
        gas: Lower-case gas name of the injected column.

    Returns:
        Full ``SensorScene`` carrying ``injected_*_mass`` and the embedding
        provenance.
    """
    name = f"injected_{gas}_mass"
    injected = embedded_window[name]
    window = Window.from_attrs(injected.attrs, prefix="window_")
    if not np.array_equal(
        np.asarray(scene["wavelength"].values),
        np.asarray(embedded_window["wavelength"].values),
    ):
        raise ContractError("The scene and the embedded window have different bands.")
    full = scene.copy()
    full["radiance"] = _paste_window(
        scene["radiance"], embedded_window["radiance"], window, fill=np.nan
    )
    template = xr.zeros_like(scene["valid"], dtype=injected.dtype)
    full[name] = _paste_window(template, injected, window, fill=0.0)
    full[name].attrs = dict(injected.attrs)
    full.attrs.update(
        {
            key: value
            for key, value in embedded_window.attrs.items()
            if key.startswith("embedding_") or key == "sensor"
        }
    )
    return full


def embed_radiance_sample(
    scene: xr.Dataset,
    injected: xr.DataArray,
    provider: UnitAbsorptionProvider | None = None,
    *,
    spec: EmbeddingSpec | None = None,
    band_windows_nm: tuple[tuple[float, float], ...] | None = None,
    lut: xr.Dataset | None = None,
    extent: EmbeddingExtent = "window",
) -> xr.Dataset:
    """Embed a regridded column into the radiance of a scene.

    The output follows the input: every band of ``scene`` is kept (bands the
    provider or ``lut`` does not cover pass through unattenuated), and
    ``extent`` selects the embedding window or the full scene.

    Args:
        scene: ``SensorScene`` the column was regridded onto.
        injected: Column from :func:`place_and_regrid`.
        provider: Unit-absorption provider building the absorption table.
        spec: Radiance-embedding settings.
        band_windows_nm: Inclusive wavelength ranges to keep; all bands of
            the scene when omitted.
        lut: Prebuilt absorption table of the window; skips the provider.
        extent: ``"window"`` returns the embedding window, ``"scene"`` the
            full scene with the window embedded.

    Returns:
        Embedded ``SensorScene`` carrying ``injected_*_mass`` and the
        embedding provenance.

    Raises:
        ContractError: If neither a provider nor an absorption table is
            given, or if ``extent`` is unknown.
    """
    if provider is None and lut is None:
        raise ContractError(
            "Radiance embedding needs a provider or an absorption table."
        )
    if extent not in EMBEDDING_EXTENTS:
        raise ContractError(f"extent must be one of {EMBEDDING_EXTENTS}.")
    spec = spec or EmbeddingSpec()
    if band_windows_nm is not None:
        scene = select_bands(scene, band_windows_nm)
    window_scene, _ = scene_window(scene, injected)
    if lut is None:
        assert provider is not None
        lut = build_lut(provider, window_scene, spec)
    embedded = embed_radiance(window_scene, injected, lut, spec=spec)
    if extent == "scene":
        embedded = embedded_scene(scene, embedded, gas=spec.gas)
    embedded.attrs["spectra_version"] = spectra.__version__
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
    target: xr.Dataset | None = None,
    *,
    retriever: Retriever,
    settings: RetrievalSettings,
    gas: str = "ch4",
    background_stats: xr.Dataset | None = None,
    provider: UnitAbsorptionProvider | None = None,
    spec: EmbeddingSpec | None = None,
) -> xr.Dataset:
    """Retrieve the embedded window and carry the injected truth and provenance.

    With ``settings.exclude_injected_from_background`` the retriever receives
    a ``plume_mask`` marking pixels whose injected column exceeds
    ``settings.plume_mask_threshold_kg_m2``, unless the scene already has one.

    Args:
        embedded: Embedded window or scene from :func:`embed_radiance_sample`.
        target: Retrieval target spectrum; built with ``provider`` when omitted.
        retriever: Retriever plugin.
        settings: Retrieval settings.
        gas: Lower-case gas name of the injected column.
        background_stats: Precomputed background statistics.
        provider: Unit-absorption provider building a missing target.
        spec: Radiance-embedding settings of the target spectrum.

    Raises:
        ContractError: If neither a target nor a provider is given, or if
            plume masking is requested but the scene carries no injected
            column.
    """
    if target is None:
        if provider is None:
            raise ContractError("Retrieval needs a target spectrum or a provider.")
        spec = spec or EmbeddingSpec(gas=gas)
        target = target_spectrum(provider, embedded, settings, spec)
    injected = f"injected_{gas}_mass"
    if settings.exclude_injected_from_background and "plume_mask" not in embedded:
        if injected not in embedded:
            raise ContractError(
                f"exclude_injected_from_background needs '{injected}' in the scene."
            )
        threshold = float(settings.plume_mask_threshold_kg_m2)
        plume_mask = (embedded[injected] > threshold).assign_attrs(
            long_name="injected plume excluded from the background statistics",
            threshold_kg_m2=threshold,
        )
        embedded = embedded.assign(plume_mask=plume_mask)
    product = retriever.retrieve(
        embedded, target, settings=settings, background_stats=background_stats
    )
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
    product.attrs["spectra_version"] = spectra.__version__
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
    extent: EmbeddingExtent = "window",
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
        extent: Retrieve the embedding window or the full embedded scene
            (radiance method).

    Returns:
        Column product on the embedding window (or the full scene) carrying
        ``injected_*_mass`` and the full provenance.
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
    embedded = embed_radiance_sample(
        scene,
        injected,
        provider,
        spec=spec,
        band_windows_nm=settings.band_windows_nm,
        extent=extent,
    )
    return retrieve_embedded(
        embedded,
        target,
        retriever=retriever,
        settings=settings,
        gas=spec.gas,
        provider=provider,
        spec=spec,
    )


def log_nodes(minimum: float, maximum: float, count: int) -> tuple[float, ...]:
    """Return logarithmically spaced absorption-table nodes."""
    if minimum <= 0 or maximum <= minimum or count < 2:
        raise ContractError("Nodes need 0 < minimum < maximum and at least two nodes.")
    return tuple(
        float(value)
        for value in np.logspace(np.log10(minimum), np.log10(maximum), count)
    )


__all__ = [
    "EMBEDDING_EXTENTS",
    "EmbeddingExtent",
    "EmbeddingMethod",
    "EmbeddingSpec",
    "build_lut",
    "embed_product",
    "embed_radiance",
    "embed_radiance_sample",
    "embed_sample",
    "embedded_scene",
    "log_nodes",
    "place_and_regrid",
    "provider_bands",
    "retrieve_embedded",
    "scene_window",
    "snapshot_column",
    "target_spectrum",
]
