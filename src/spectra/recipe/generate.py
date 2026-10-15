"""Dry run and per-sample generation of a recipe.

``dry_run`` resolves the context and the parameters of every sample without
touching radiance; ``generate`` embeds the requested samples and writes one
pairings shard. Plugins are built once per process (:class:`Plugins`) and
scenes, products, absorption tables and background statistics are cached
per scene within a run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import xarray as xr

import spectra
from spectra.contracts.column_product import (
    CANONICAL_COLUMN_UNITS,
    PPM_METRE_UNITS,
    convert_column_units,
)
from spectra.contracts.errors import ContractError
from spectra.export.pairings import pairings_row, scene_uri, write_pairings
from spectra.export.prism import to_prism_product, write_product
from spectra.pipeline import (
    EmbeddingSpec,
    build_lut,
    embed_radiance,
    embed_sample,
    log_nodes,
    place_and_regrid,
    retrieve_embedded,
    scene_window,
    snapshot_column,
)
from spectra.protocols import (
    Regridder,
    Retriever,
    SensorAdapter,
    UnitAbsorptionProvider,
)
from spectra.recipe.params import (
    CONTEXT_FILE,
    PARAMS_FILE,
    ContextResources,
    GenerationContext,
    SampleParams,
    build_context,
    build_parameter_table,
    params_from_table,
    read_parameter_table,
)
from spectra.recipe.schema import (
    Recipe,
    TimePolicy,
    dump_resolved,
    recipe_hash,
)
from spectra.registry import (
    get_regridder,
    get_retriever,
    get_sensor,
    get_unit_absorption_provider,
)
from spectra.types import (
    Placement,
    RetrievalSettings,
    SceneGeometry,
    SceneRecord,
    SpectralResponse,
    ViewingGeometry,
    Window,
)

RESOLVED_RECIPE_FILE = "recipe.resolved.yaml"
RECIPE_HASH_FILE = "recipe_hash.txt"
LUT_COVERAGE_FILE = "lut_coverage.json"
SHARDS_DIR = "shards"
SAMPLES_DIR = "samples"
_T = TypeVar("_T")


# -- plugins and caches ------------------------------------------------------
@dataclass
class Plugins:
    """Plugins of a recipe, built once per process.

    Attributes:
        adapter: Sensor adapter.
        regridder: Conservative regridder.
        spec: Radiance-embedding settings (nodes from the recipe LUT spec).
        provider: Unit-absorption provider (radiance method).
        retriever: Retriever (radiance method).
        settings: Retrieval settings (radiance method).
    """

    adapter: SensorAdapter
    regridder: Regridder
    spec: EmbeddingSpec
    provider: UnitAbsorptionProvider | None = None
    retriever: Retriever | None = None
    settings: RetrievalSettings | None = None


def embedding_spec(recipe: Recipe) -> EmbeddingSpec:
    """Return the ``EmbeddingSpec`` of a recipe."""
    lut = recipe.embedding.lut
    return EmbeddingSpec(
        column_convention=recipe.plumes.column_convention,  # type: ignore[arg-type]
        enhancements_ppm=log_nodes(
            lut.enhancement_min_ppm, lut.enhancement_max_ppm, lut.n_nodes
        ),
        layer_height_m=lut.layer_height_m,
        radiance_leg=lut.radiance_leg,  # type: ignore[arg-type]
    )


def retrieval_settings(recipe: Recipe) -> RetrievalSettings | None:
    """Return the ``RetrievalSettings`` of a radiance recipe."""
    retrieval = recipe.retrieval
    if retrieval is None:
        return None
    return RetrievalSettings(
        statistics_scope=retrieval.statistics_scope,  # type: ignore[arg-type]
        band_windows_nm=tuple(
            (float(low), float(high)) for low, high in retrieval.band_windows_nm
        ),
        target_enhancement_ppm=retrieval.target_enhancement_ppm,
        min_n_samples=retrieval.min_n_samples,
        exclude_injected_from_background=retrieval.exclude_injected_from_background,
        plume_mask_threshold_kg_m2=retrieval.plume_mask_threshold_kg_m2,
        options=dict(retrieval.options),
    )


def build_plugins(recipe: Recipe) -> Plugins:
    """Instantiate the adapter, regridder, provider and retriever of a recipe."""
    plugins = Plugins(
        adapter=get_sensor(recipe.sensor.name, **recipe.sensor.options),
        regridder=get_regridder(
            recipe.embedding.regridder, **recipe.embedding.regridder_options
        ),
        spec=embedding_spec(recipe),
    )
    if recipe.method == "radiance":
        provider = recipe.embedding.unit_absorption
        retrieval = recipe.retrieval
        assert provider is not None and retrieval is not None
        plugins.provider = get_unit_absorption_provider(
            provider.provider, **provider.options
        )
        plugins.retriever = get_retriever(retrieval.retriever, **retrieval.options)
        plugins.settings = retrieval_settings(recipe)
    return plugins


@dataclass
class RunCache:
    """Per-run caches keyed by scene (and window for the absorption table)."""

    scenes: dict[str, xr.Dataset] = field(default_factory=dict)
    products: dict[str, xr.Dataset] = field(default_factory=dict)
    geometries: dict[str, SceneGeometry] = field(default_factory=dict)
    background_stats: dict[str, xr.Dataset] = field(default_factory=dict)
    luts: dict[tuple[str, tuple[int, ...]], xr.Dataset] = field(default_factory=dict)
    targets: dict[tuple[str, tuple[int, ...]], xr.Dataset] = field(default_factory=dict)
    plumes: dict[str, xr.Dataset] = field(default_factory=dict)

    @staticmethod
    def _get(store: dict[Any, _T], key: Hashable, build: Callable[[], _T]) -> _T:
        if key not in store:
            store[key] = build()
        return store[key]

    def scene(
        self, record: SceneRecord, adapter: SensorAdapter, recipe: Recipe
    ) -> xr.Dataset:
        """Return the opened scene of a record."""
        chunks = recipe.compute.chunks or None
        return self._get(
            self.scenes,
            record.scene_id,
            lambda: adapter.open_scene(record, chunks=chunks),
        )

    def product(self, record: SceneRecord, adapter: SensorAdapter) -> xr.Dataset:
        """Return the retrieved product of a record."""
        open_product = getattr(adapter, "open_product", None)
        if not callable(open_product):
            raise ContractError(
                f"Adapter {adapter.name!r} cannot open products; the product "
                "method needs a SupportsProduct adapter."
            )
        return self._get(
            self.products,
            record.scene_id,
            lambda: cast(xr.Dataset, open_product(record)),
        )

    def geometry(self, scene: xr.Dataset) -> SceneGeometry:
        """Return the projected geometry of an opened scene."""
        return self._get(
            self.geometries,
            str(scene.attrs.get("scene_id", id(scene))),
            lambda: SceneGeometry.from_scene(scene),
        )

    def plume(self, uri: str) -> xr.Dataset:
        """Return an opened plume column."""
        from spectra.plume.io import open_plume_column

        return self._get(self.plumes, uri, lambda: open_plume_column(uri))

    def close(self) -> None:
        """Close the opened datasets."""
        for store in (self.scenes, self.products, self.plumes):
            for dataset in store.values():
                dataset.close()
            store.clear()


# -- dry run -------------------------------------------------------------------
def _policy_indices(policy: TimePolicy, plume_times: tuple[float, ...]) -> list[int]:
    if policy.mode == "uniform":
        return [
            i for i, t in enumerate(plume_times) if t >= policy.min_simulation_time_s
        ]
    assert policy.indices is not None
    return [i for i in policy.indices if i < len(plume_times)]


def lut_coverage(recipe: Recipe, context: GenerationContext) -> dict[str, Any]:
    """Compare the largest injectable column with the absorption-table range.

    The largest column of every plume file at the largest emission rate of
    the recipe is converted to ppm-m and compared with
    ``enhancement_max_ppm * layer_height_m``. The effective layer height of a
    scene is at most the nominal one, so the check uses the nominal height
    and reports the margin; the embedding itself refuses columns beyond the
    table at generation time.

    Returns:
        The report written as ``lut_coverage.json``.

    Raises:
        ContractError: If a radiance recipe cannot cover its plumes.
    """
    lut = recipe.embedding.lut
    lut_max_ppm_m = lut.enhancement_max_ppm * lut.layer_height_m
    max_kg_s = recipe.plumes.emission.maximum_kg_s
    entries: list[dict[str, Any]] = []
    exceeded: list[str] = []
    for plume in context.plumes:
        indices = _policy_indices(recipe.plumes.time_policy, plume.simulation_time_s)
        columns = [
            plume.column_max_kg_m2[i] * (max_kg_s / plume.emission_rate_kg_s[i])
            for i in indices
            if plume.emission_rate_kg_s[i] > 0
        ]
        max_kg_m2 = max(columns) if columns else math.nan
        max_ppm_m = float(
            convert_column_units(
                xr.DataArray(max_kg_m2, attrs={"units": CANONICAL_COLUMN_UNITS}),
                PPM_METRE_UNITS,
                gas=plume.gas,
            ).values
        )
        ratio = max_ppm_m / lut_max_ppm_m
        entries.append(
            {
                "uri": plume.uri,
                "case_id": plume.case_id,
                "max_emission_rate_kg_s": max_kg_s,
                "max_injected_kg_m2": max_kg_m2,
                "max_injected_ppm_m": max_ppm_m,
                "coverage_ratio": ratio,
                "covered": bool(ratio <= 1.0),
            }
        )
        if ratio > 1.0:
            exceeded.append(f"{plume.case_id} ({max_ppm_m:.0f} ppm-m)")
    report = {
        "applies": recipe.method == "radiance",
        "enhancement_max_ppm": lut.enhancement_max_ppm,
        "layer_height_m": lut.layer_height_m,
        "lut_max_ppm_m": lut_max_ppm_m,
        "note": (
            "The effective layer height of a scene is at most layer_height_m; the "
            "radiance embedding re-checks the coverage per sample."
        ),
        "plumes": entries,
    }
    if exceeded and recipe.method == "radiance":
        raise ContractError(
            "extend the LUT nodes: embedding.lut.enhancement_max_ppm="
            f"{lut.enhancement_max_ppm} ppm over {lut.layer_height_m} m covers "
            f"{lut_max_ppm_m:.0f} ppm-m but the plumes reach {', '.join(exceeded)}."
        )
    return report


def dry_run(
    recipe: Recipe, out_dir: str | Path, *, adapter: SensorAdapter | None = None
) -> Path:
    """Resolve the context and parameters of every sample without embedding.

    Writes ``context.json``, ``params.parquet``, ``recipe.resolved.yaml``,
    ``recipe_hash.txt`` and ``lut_coverage.json`` into ``out_dir``.

    Args:
        recipe: Validated recipe.
        out_dir: Run directory.
        adapter: Sensor adapter; built from the registry when omitted.

    Returns:
        The run directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context = build_context(recipe, adapter=adapter)
    report = lut_coverage(recipe, context)
    resources = ContextResources(recipe, context, adapter=adapter)
    try:
        table = build_parameter_table(recipe, context, resources=resources)
    finally:
        resources.close()
    context.write(out / CONTEXT_FILE)
    table.to_parquet(out / PARAMS_FILE, index=False)
    dump_resolved(recipe, out / RESOLVED_RECIPE_FILE)
    (out / RECIPE_HASH_FILE).write_text(recipe_hash(recipe) + "\n")
    (out / LUT_COVERAGE_FILE).write_text(json.dumps(report, indent=2))
    return out


# -- per-sample pipeline -----------------------------------------------------
@dataclass(frozen=True)
class GeneratedSample:
    """A written product and its pairings row."""

    path: Path
    row: dict[str, Any]


def _window_key(window: Window) -> tuple[int, ...]:
    return (
        window.across.start,
        window.across.stop,
        window.along.start,
        window.along.stop,
    )


def _zero_injected(template: xr.DataArray, gas: str) -> xr.DataArray:
    zeros = xr.zeros_like(template, dtype="float32")
    zeros.name = f"injected_{gas}_mass"
    zeros.attrs = {
        "units": CANONICAL_COLUMN_UNITS,
        "long_name": f"injected {gas.upper()} column (none: negative sample)",
    }
    return zeros


def _target(
    scene: xr.Dataset,
    window: Window,
    plugins: Plugins,
    cache: RunCache,
) -> xr.Dataset:
    """Return the retrieval target spectrum of a scene window (cached)."""
    provider, settings, spec = plugins.provider, plugins.settings, plugins.spec
    assert provider is not None and settings is not None
    key = (str(scene.attrs.get("scene_id", "")), _window_key(window))
    return cache._get(
        cache.targets,
        key,
        lambda: provider.spectrum(
            SpectralResponse.from_scene(scene),
            ViewingGeometry.from_scene(scene, window=window),
            enhancement_ppm=settings.target_enhancement_ppm,
            gas=spec.gas.upper(),
            layer_height_m=spec.layer_height_m,
            radiance_leg=spec.radiance_leg,
        ),
    )


def _background_stats(
    scene: xr.Dataset, plugins: Plugins, cache: RunCache
) -> xr.Dataset | None:
    """Return cached full-scene background statistics when the scope needs them."""
    retriever, settings = plugins.retriever, plugins.settings
    assert retriever is not None and settings is not None
    if settings.statistics_scope == "window":
        return None
    return cache._get(
        cache.background_stats,
        str(scene.attrs.get("scene_id", "")),
        lambda: retriever.background_statistics(scene, settings=settings),
    )


def _embed_radiance_sample(
    scene: xr.Dataset,
    column: xr.DataArray,
    placement: Placement,
    window: Window,
    plugins: Plugins,
    cache: RunCache,
) -> xr.Dataset:
    """Radiance method with cached absorption tables and background statistics."""
    provider, retriever, settings, spec = (
        plugins.provider,
        plugins.retriever,
        plugins.settings,
        plugins.spec,
    )
    assert provider is not None and retriever is not None and settings is not None
    injected = place_and_regrid(
        scene, column, placement, regridder=plugins.regridder, window=window
    )
    window_scene, window_obj = scene_window(scene, injected)
    key = (str(scene.attrs.get("scene_id", "")), _window_key(window_obj))
    lut = cache._get(
        cache.luts, key, lambda: build_lut(provider, scene, spec, window=window_obj)
    )
    embedded = embed_radiance(window_scene, injected, lut, spec=spec)
    product = retrieve_embedded(
        embedded,
        _target(scene, window_obj, plugins, cache),
        retriever=retriever,
        settings=settings,
        gas=spec.gas,
        background_stats=_background_stats(scene, plugins, cache),
    )
    product.attrs["spectra_version"] = spectra.__version__
    return product


def _negative_sample(
    scene: xr.Dataset,
    record: SceneRecord,
    placement: Placement,
    window: Window,
    recipe: Recipe,
    plugins: Plugins,
    cache: RunCache,
) -> xr.Dataset:
    """Plume-free window: the untouched product or the retrieval of the window."""
    from spectra.spectral.embedding import embedding_provenance

    gas = plugins.spec.gas
    if recipe.method == "product":
        product = cache.product(record, plugins.adapter).isel(window.indexers).copy()
    else:
        retriever, settings = plugins.retriever, plugins.settings
        assert retriever is not None and settings is not None
        product = retriever.retrieve(
            scene.isel(window.indexers),
            _target(scene, window, plugins, cache),
            settings=settings,
            background_stats=_background_stats(scene, plugins, cache),
        )
    product[f"injected_{gas}_mass"] = _zero_injected(product[f"{gas}_mass"], gas)
    product.attrs.update(
        embedding_provenance(
            recipe.method,  # type: ignore[arg-type]
            sensor=str(scene.attrs.get("sensor", "")),
            regridder=recipe.embedding.regridder,
        )
    )
    product.attrs.update(window.attrs("window_"))
    product.attrs.update(placement.attrs())
    product.attrs["embedding_method"] = recipe.method
    product.attrs["spectra_version"] = spectra.__version__
    return product


def injected_source_pixel(
    plume: xr.Dataset, placement: Placement, geometry: SceneGeometry, window: Window
) -> tuple[int, int]:
    """Return the window pixel ``(y, x)`` nearest to the placed LES source.

    The source position is translated and transformed exactly like the
    column grid (``place_column``) and mapped to the nearest pixel centre of
    the window; ``y`` is the across-track and ``x`` the along-track offset
    inside the window, matching ``to_prism_grid``.
    """
    from spectra.geometry.transform import anchor_cell_centre, transform_coordinates
    from spectra.regrid.windows import anchor_pixel_centre

    anchor_xy = anchor_pixel_centre(geometry, placement)
    anchor_source = anchor_cell_centre(
        plume["x"].values, plume["y"].values, placement.anchor_corner
    )
    source_x = float(plume.attrs["source_x"]) + anchor_xy[0] - anchor_source[0]
    source_y = float(plume.attrs["source_y"]) + anchor_xy[1] - anchor_source[1]
    tx, ty = transform_coordinates(
        source_x,
        source_y,
        pivot=anchor_xy,
        rotation_deg=placement.rotation_deg,
        flip=placement.flip,
    )
    dim0, dim1 = geometry.spatial_dims
    indexer = {dim0: window.across, dim1: window.along}
    grid_x = np.asarray(geometry.x.isel(indexer).values, dtype=float)
    grid_y = np.asarray(geometry.y.isel(indexer).values, dtype=float)
    distance = (grid_x - float(tx)) ** 2 + (grid_y - float(ty)) ** 2
    distance = np.where(np.isfinite(distance), distance, np.inf)
    y, x = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return int(y), int(x)


def _snapshot_wind(plume: xr.Dataset, time_index: int) -> tuple[float, float] | None:
    if "wind_u" not in plume or "wind_v" not in plume:
        return None
    components: list[float] = []
    for name in ("wind_u", "wind_v"):
        values = plume[name].isel(time=time_index).values
        components.append(float(np.nanmean(np.asarray(values, dtype=float))))
    if not all(math.isfinite(value) for value in components):
        return None
    return components[0], components[1]


def _pixel_size_m(product: xr.Dataset, scene: xr.Dataset, window: Window) -> float:
    """Return the square root of the median pixel area of the window."""
    if "pixel_area" in product:
        area = np.asarray(product["pixel_area"].values, dtype=float)
    elif "pixel_area" in scene:
        area = np.asarray(scene["pixel_area"].isel(window.indexers).values, dtype=float)
    else:
        return math.nan
    if not np.isfinite(area).any():
        return math.nan
    return float(np.sqrt(np.nanmedian(area)))


def _provenance(
    params: SampleParams,
    recipe: Recipe,
    context: GenerationContext,
    record: SceneRecord,
    plume: xr.Dataset,
) -> dict[str, Any]:
    placement = params.placement
    return {
        "sensor": params.sensor,
        "scene_id": params.scene_id,
        "background_group": params.background_group,
        "les_model": "none"
        if params.is_negative
        else str(plume.attrs.get("les_model", "")),
        "les_case": params.les_case,
        "tracer": params.tracer,
        "source_id": params.source_id,
        "time_index": int(params.time_index),
        "simulation_time_s": float(params.simulation_time_s),
        "emission_rate_kg_s": float(params.emission_rate_kg_s),
        "emission_scale": float(params.emission_scale),
        "rotation_deg": int(placement.rotation_deg),
        "flip": placement.flip,
        "anchor_along": int(placement.anchor_along),
        "anchor_across": int(placement.anchor_across),
        "anchor_corner": placement.anchor_corner,
        **params.window.attrs("window_"),
        "is_negative": int(params.is_negative),
        "split": params.split,
        "sample_id": params.sample_id,
        "pair_id": params.pair_id,
        "plume_uri": params.plume_uri,
        "scene_uri": scene_uri(record.uris),
        "recipe_name": recipe.name,
        "recipe_hash": context.recipe_hash,
        "context_hash": context.context_hash,
        "params_hash": params.params_hash,
        "spectra_version": spectra.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels_detection_snr": float(recipe.labels.detection_snr),
        "labels_ambiguous_snr": float(recipe.labels.ambiguous_snr),
        "column_convention": recipe.plumes.column_convention,
    }


def sample_path(recipe: Recipe, sample_id: str) -> Path:
    """Return the product path of a sample below ``output.root/samples``."""
    return Path(recipe.output.root) / SAMPLES_DIR / f"{sample_id}{recipe.output.suffix}"


def generate_sample(
    params: SampleParams,
    recipe: Recipe,
    context: GenerationContext,
    *,
    plugins: Plugins,
    cache: RunCache | None = None,
    overwrite: bool = False,
) -> GeneratedSample:
    """Embed one sample and write its prism product.

    Args:
        params: Sample parameters (one row of ``params.parquet``).
        recipe: Recipe.
        context: Generation context.
        plugins: Plugins from :func:`build_plugins`.
        cache: Per-run cache; a fresh one is used when omitted.
        overwrite: Replace an existing product.

    Returns:
        The written path and the pairings row.
    """
    cache = cache or RunCache()
    record = context.record(params.scene_id)
    scene = cache.scene(record, plugins.adapter, recipe)
    geometry = cache.geometry(scene)
    plume = cache.plume(params.plume_uri)
    placement, window = params.placement, params.window
    gas = plugins.spec.gas

    if params.is_negative:
        product = _negative_sample(
            scene, record, placement, window, recipe, plugins, cache
        )
    else:
        column = snapshot_column(
            plume, time_index=params.time_index, emission_scale=params.emission_scale
        )
        if recipe.method == "product":
            product = embed_sample(
                scene,
                column,
                placement,
                method="product",
                regridder=plugins.regridder,
                spec=plugins.spec,
                product=cache.product(record, plugins.adapter),
                window=window,
            )
        else:
            product = _embed_radiance_sample(
                scene, column, placement, window, plugins, cache
            )
        source_y, source_x = injected_source_pixel(plume, placement, geometry, window)
        product.attrs["injected_source_y"] = source_y
        product.attrs["injected_source_x"] = source_x

    wind = _snapshot_wind(plume, params.time_index)
    if wind is not None:
        from spectra.plume.augment import transform_wind

        u, v = transform_wind(wind[0], wind[1], placement)
        template = product[f"{gas}_mass"]
        for name, value, long_name in (
            ("wind_u", u, "eastward wind of the placed plume"),
            ("wind_v", v, "northward wind of the placed plume"),
        ):
            product[name] = xr.full_like(template, value, dtype="float32")
            product[name].attrs = {"units": "m s-1", "long_name": long_name}

    attrs = _provenance(params, recipe, context, record, plume)
    pixel_size = _pixel_size_m(product, scene, window)
    if math.isfinite(pixel_size) and pixel_size > 0:
        attrs["pixel_size_m"] = pixel_size
    prism_product = to_prism_product(product, gas=gas, extra_attrs=attrs)
    path = sample_path(recipe, params.sample_id)
    write_product(prism_product, path, overwrite=overwrite)
    return GeneratedSample(
        path=path, row=pairings_row(prism_product, params, recipe, path)
    )


# -- run ------------------------------------------------------------------------
def shard_path(out_dir: str | Path, shard: int | None) -> Path:
    """Return the pairings shard file of a run."""
    label = "all" if shard is None else str(shard)
    return Path(out_dir) / SHARDS_DIR / f"pairings_{label}.csv"


def run_files_exist(out_dir: str | Path) -> bool:
    """Return whether ``out_dir`` holds the dry-run files."""
    out = Path(out_dir)
    return (out / CONTEXT_FILE).exists() and (out / PARAMS_FILE).exists()


def generate(
    recipe: Recipe,
    out_dir: str | Path,
    *,
    index_range: tuple[int, int] | None = None,
    shard: int | None = None,
    overwrite: bool = False,
    plugins: Plugins | None = None,
    indices: Iterable[int] | None = None,
) -> Path:
    """Generate the samples of a recipe and write one pairings shard.

    Runs :func:`dry_run` first when ``out_dir`` holds no parameter table.

    Args:
        recipe: Recipe.
        out_dir: Run directory (dry-run files, shards).
        index_range: ``(start, stop)`` sample indices, stop exclusive.
        shard: Shard number of the pairings file (``all`` when omitted).
        overwrite: Replace existing products.
        plugins: Plugins; built once here when omitted.
        indices: Explicit indices (overrides ``index_range``).

    Returns:
        The written pairings shard.
    """
    out = Path(out_dir)
    if not run_files_exist(out):
        dry_run(recipe, out, adapter=plugins.adapter if plugins else None)
    context = GenerationContext.read(out / CONTEXT_FILE)
    if context.recipe_hash != recipe_hash(recipe):
        raise ContractError(
            f"{out / CONTEXT_FILE} was built for recipe hash {context.recipe_hash}; "
            f"the recipe now hashes to {recipe_hash(recipe)}. Re-run the dry run."
        )
    table = read_parameter_table(out / PARAMS_FILE)
    if indices is not None:
        selected = list(indices)
    elif index_range is not None:
        start, stop = index_range
        selected = list(range(max(0, start), min(stop, recipe.n_samples)))
    else:
        selected = list(range(recipe.n_samples))
    plugins = plugins or build_plugins(recipe)
    cache = RunCache()
    rows: list[dict[str, Any]] = []
    try:
        for index in selected:
            params = params_from_table(table, index)
            sample = generate_sample(
                params,
                recipe,
                context,
                plugins=plugins,
                cache=cache,
                overwrite=overwrite,
            )
            rows.append(sample.row)
    finally:
        cache.close()
    return write_pairings(rows, shard_path(out, shard))


__all__ = [
    "LUT_COVERAGE_FILE",
    "RECIPE_HASH_FILE",
    "RESOLVED_RECIPE_FILE",
    "SAMPLES_DIR",
    "SHARDS_DIR",
    "GeneratedSample",
    "Plugins",
    "RunCache",
    "build_plugins",
    "dry_run",
    "embedding_spec",
    "generate",
    "generate_sample",
    "injected_source_pixel",
    "lut_coverage",
    "retrieval_settings",
    "run_files_exist",
    "sample_path",
    "shard_path",
]
