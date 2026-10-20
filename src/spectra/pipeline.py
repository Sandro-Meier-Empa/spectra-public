"""Embed one plume snapshot into one scene: placement, regridding, embedding.

The radiance method runs as two steps: :func:`place_and_regrid` (locate and
regrid the column) and :func:`embed_radiance_sample` (Lambert-Beer embedding
into the window or the full scene). The embedded radiance keeps every band it
is given; bands the unit-absorption provider does not cover pass through
unattenuated.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyproj
import xarray as xr
from shapely.geometry import Polygon

import spectra
from spectra.contracts.errors import ContractError
from spectra.contracts.plume_column import scale_emission, select_time
from spectra.contracts.sensor_scene import select_bands
from spectra.contracts.units import CANONICAL_COLUMN_UNITS, convert_column_units
from spectra.geometry.amf import effective_layer_height
from spectra.geometry.crs import to_lonlat
from spectra.geometry.placement import anchor_source_point
from spectra.geometry.swath import ResolvedAnchor, resolve_anchor, swath_polygon
from spectra.geometry.transform import transform_coordinates
from spectra.plume.augment import plume_fraction_below
from spectra.plume.io import open_plume_column
from spectra.protocols import ForwardModel, Regridder, UnitAbsorptionProvider
from spectra.regrid.polygon import PolygonRegridder
from spectra.result import ClusterResult, EmbedResult, SourceResult, windows_overlap
from spectra.run import PlumeSpec
from spectra.types import (
    Placement,
    RadianceLeg,
    SceneGeometry,
    SpectralResponse,
    ViewingGeometry,
    Window,
)

logger = logging.getLogger(__name__)

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
    "EmbeddingSpec",
    "build_lut",
    "embed_radiance",
    "embed_radiance_sample",
    "embedded_scene",
    "log_nodes",
    "place_and_regrid",
    "provider_bands",
    "scene_window",
    "snapshot_column",
]


# -- multi-plume embedding ------------------------------------------------------


def open_plumes(plumes: Sequence[PlumeSpec]) -> dict[Path, xr.Dataset]:
    """Open every distinct plume file of a run once, chunked per snapshot."""
    datasets: dict[Path, xr.Dataset] = {}
    for plume in plumes:
        if plume.file not in datasets:
            datasets[plume.file] = open_plume_column(plume.file, chunks={"time": 1})
    return datasets


def plume_gas(datasets: Mapping[Path, xr.Dataset]) -> str:
    """Return the lower-case gas shared by every plume of a run.

    Raises:
        ContractError: If the plumes carry different gases.
    """
    gases = {str(dataset.attrs["gas"]).lower() for dataset in datasets.values()}
    if len(gases) != 1:
        raise ContractError(
            f"All plumes of a run must carry the same gas; found {sorted(gases)}."
        )
    return gases.pop()


def plume_snapshot(dataset: xr.Dataset, plume: PlumeSpec) -> xr.Dataset:
    """Return the scaled snapshot of a plume column selected by a plume spec."""
    snapshot = select_time(dataset, plume.time_index)
    if plume.emission_scale != 1.0:
        snapshot = scale_emission(snapshot, plume.emission_scale)
    return snapshot


def _snapshot_column(snapshot: xr.Dataset) -> xr.DataArray:
    column = snapshot["column"].copy()
    column.attrs.update(dict(snapshot.attrs))
    return column


def resolve_plumes(
    scene: xr.Dataset,
    plumes: Sequence[PlumeSpec],
    *,
    seed: int,
    datasets: Mapping[Path, xr.Dataset] | None = None,
    geometry: SceneGeometry | None = None,
    swath: Polygon | None = None,
) -> list[ResolvedAnchor]:
    """Resolve the anchor of every plume, drawing random anchors reproducibly.

    Plume ``i`` draws from the ``i``-th child of ``SeedSequence(seed)``, so
    adding or removing other plumes leaves its draw unchanged.
    """
    datasets = dict(datasets) if datasets is not None else open_plumes(plumes)
    geometry = geometry or SceneGeometry.from_scene(scene)
    swath = swath if swath is not None else swath_polygon(geometry)
    children = np.random.SeedSequence(int(seed)).spawn(len(plumes))
    anchors = []
    for plume, child in zip(plumes, children, strict=True):
        column = _snapshot_column(plume_snapshot(datasets[plume.file], plume))
        spec = plume.anchor
        anchors.append(
            resolve_anchor(
                column,
                geometry,
                mode=spec.mode,
                swath=swath,
                corner=spec.corner,
                rotation_deg=plume.rotation,
                flip=plume.flip,
                along=spec.along,
                across=spec.across,
                lonlat=None if spec.lon is None else (spec.lon, cast(float, spec.lat)),
                bounds=(spec.min_outside_fraction, spec.max_outside_fraction),
                rng=np.random.default_rng(child),
                max_attempts=spec.max_attempts,
                emission_scale=plume.emission_scale,
                time_index=plume.time_index,
            )
        )
    return anchors


def resolved_plume(plume: PlumeSpec, anchor: ResolvedAnchor) -> PlumeSpec:
    """Return the plume with a drawn random anchor recorded as its pixel."""
    if plume.anchor.mode != "random":
        return plume
    drawn = dataclasses.replace(
        plume.anchor,
        along=anchor.placement.anchor_along,
        across=anchor.placement.anchor_across,
    )
    return dataclasses.replace(plume, anchor=drawn)


def cluster_windows(windows: Sequence[Window]) -> list[tuple[Window, tuple[int, ...]]]:
    """Merge overlapping windows until the merged windows are pairwise disjoint.

    Returns:
        ``(window, members)`` per cluster, ordered by their first member.
    """
    boxes = list(windows)
    groups = [[index] for index in range(len(boxes))]
    merged = True
    while merged:
        merged = False
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                if windows_overlap(boxes[first], boxes[second]):
                    a, b = boxes[first], boxes[second]
                    boxes[first] = Window(
                        across=slice(
                            min(a.across.start, b.across.start),
                            max(a.across.stop, b.across.stop),
                        ),
                        along=slice(
                            min(a.along.start, b.along.start),
                            max(a.along.stop, b.along.stop),
                        ),
                    )
                    groups[first] += groups.pop(second)
                    boxes.pop(second)
                    merged = True
                    break
            if merged:
                break
    clusters = [
        (box, tuple(sorted(group))) for box, group in zip(boxes, groups, strict=True)
    ]
    return sorted(clusters, key=lambda item: item[1][0])


def attenuated_bands(
    scene: xr.Dataset, spectral_range_nm: tuple[float, float]
) -> slice:
    """Return the contiguous scene bands whose centres lie in the model range.

    Raises:
        ContractError: If no band lies in the range.
    """
    wavelength = np.asarray(scene["wavelength"].values, dtype=float)
    lower, upper = spectral_range_nm
    inside = np.flatnonzero((wavelength >= lower) & (wavelength <= upper))
    if inside.size == 0:
        raise ContractError(
            f"No scene band lies in the forward model's range {spectral_range_nm} nm."
        )
    return slice(int(inside[0]), int(inside[-1]) + 1)


def _source_lonlat(
    column: xr.DataArray, anchor: ResolvedAnchor, crs: pyproj.CRS
) -> tuple[float, float]:
    """Return the longitude and latitude of the emission source after placement."""
    placement = anchor.placement
    x = np.asarray(column["x"].values, dtype=float)
    y = np.asarray(column["y"].values, dtype=float)
    pivot = anchor_source_point(column, x, y, placement)
    source = (float(column.attrs["source_x"]), float(column.attrs["source_y"]))
    offset = (anchor.anchor_xy[0] - pivot[0], anchor.anchor_xy[1] - pivot[1])
    placed_x, placed_y = transform_coordinates(
        np.array([source[0] + offset[0]]),
        np.array([source[1] + offset[1]]),
        pivot=anchor.anchor_xy,
        rotation_deg=placement.rotation_deg,
        flip=placement.flip,
    )
    lon, lat = to_lonlat(crs, placed_x, placed_y)
    return float(lon[0]), float(lat[0])


def _warn_above_sensor(
    scene: xr.Dataset, snapshot: xr.Dataset, window: Window, index: int
) -> None:
    """Log the plume mass above an airborne sensor, ignored by the embedding."""
    if (
        scene.attrs.get("platform") != "airborne"
        or "column_below_altitude" not in snapshot
    ):
        return
    subset = scene.isel(window.indexers)
    height = subset["sensor_altitude"] - subset["surface_elevation"]
    above_ground = float(np.nanmedian(np.asarray(height.values, dtype=float)))
    try:
        below = float(plume_fraction_below(snapshot, above_ground))
    except ContractError:
        return
    if np.isfinite(below) and below < 0.99:
        logger.warning(
            "Plume %d has %.1f %% of its mass above the sensor (%.0f m above "
            "ground); the embedding attenuates both photon legs with the whole "
            "column.",
            index,
            100.0 * (1.0 - below),
            above_ground,
        )


def _check_table_range(
    column: np.ndarray,
    gas: str,
    forward_model: ForwardModel,
    geometry: ViewingGeometry,
    index: int,
) -> None:
    """Fail before the forward model runs if the table cannot hold the plume."""
    peak = float(np.max(column))
    amount = float(
        convert_column_units(
            xr.DataArray(peak, attrs={"units": CANONICAL_COLUMN_UNITS}),
            "ppm-m",
            gas=gas,
        )
    )
    height = effective_layer_height(
        max(float(geometry.surface_elevation), 0.0), forward_model.layer_height_m
    )
    capacity = float(forward_model.enhancements_ppm[-1]) * height
    if amount > capacity:
        raise ContractError(
            f"Cluster {index} reaches {amount:.0f} ppm-m, beyond the {capacity:.0f} "
            f"ppm-m the absorption table covers; raise enhancement_max_ppm to at "
            f"least {amount / height:.0f}."
        )


def embed_scene(
    scene: xr.Dataset,
    plumes: Sequence[PlumeSpec],
    forward_model: ForwardModel,
    *,
    seed: int,
    datasets: Mapping[Path, xr.Dataset] | None = None,
    regridder: PolygonRegridder | None = None,
) -> tuple[EmbedResult, tuple[PlumeSpec, ...]]:
    """Place, regrid and cluster plumes and build one absorption table per cluster.

    Args:
        scene: ``SensorScene`` dataset.
        plumes: Plumes to embed.
        forward_model: Model building the absorption tables.
        seed: Seed of the random anchors.
        datasets: Opened plume columns by file; opened when omitted.
        regridder: Conservative regridder; exact polygon overlap by default.

    Returns:
        The embedding result and the plumes with drawn anchors filled in.

    Raises:
        ContractError: If plumes, anchors or the forward model are invalid.
    """
    datasets = dict(datasets) if datasets is not None else open_plumes(plumes)
    gas = plume_gas(datasets)
    forward_model.check(gas.upper())
    regridder = regridder or PolygonRegridder()
    geometry = SceneGeometry.from_scene(scene)
    swath = swath_polygon(geometry)
    anchors = resolve_plumes(
        scene, plumes, seed=seed, datasets=datasets, geometry=geometry, swath=swath
    )

    placed = []
    for index, (plume, anchor) in enumerate(zip(plumes, anchors, strict=True)):
        snapshot = plume_snapshot(datasets[plume.file], plume)
        column = _snapshot_column(snapshot)
        regrid = regridder.regrid_to_scene(column, scene, anchor.placement)
        window = Window.from_attrs(regrid.data.attrs)
        injected = convert_column_units(regrid.data, CANONICAL_COLUMN_UNITS, gas=gas)
        lon, lat = _source_lonlat(column, anchor, geometry.crs)
        _warn_above_sensor(scene, snapshot, window, index)
        outside = 1.0 - regrid.source_mass_inside / regrid.source_mass
        placed.append(
            (
                window,
                np.asarray(injected.transpose("across", "along").values, dtype=float),
                {
                    "plume_file": str(plume.file),
                    "case_id": str(column.attrs.get("case_id", "")),
                    "time_index": plume.time_index,
                    "emission_rate_kg_s": float(snapshot["emission_rate"]),
                    "emission_scale": plume.emission_scale,
                    "lon": lon,
                    "lat": lat,
                    "anchor_along": anchor.placement.anchor_along,
                    "anchor_across": anchor.placement.anchor_across,
                    "anchor_mode": anchor.mode,
                    "rotation_deg": plume.rotation,
                    "flip": plume.flip,
                    "outside_fraction": float(min(1.0, max(0.0, outside))),
                    "injected_mass_kg": float(regrid.target_mass),
                },
            )
        )

    bands = attenuated_bands(scene, forward_model.spectral_range_nm)
    response = SpectralResponse(
        wavelength=np.asarray(scene["wavelength"].values)[bands],
        fwhm=np.asarray(scene["fwhm"].values)[bands],
    )
    clusters = []
    membership: dict[int, int] = {}
    for cluster_index, (window, members) in enumerate(
        cluster_windows([item[0] for item in placed])
    ):
        column = np.zeros(window.shape, dtype=float)
        for member in members:
            source_window, values, _ = placed[member]
            column[
                source_window.across.start
                - window.across.start : source_window.across.stop - window.across.start,
                source_window.along.start
                - window.along.start : source_window.along.stop - window.along.start,
            ] += values
            membership[member] = cluster_index
        cluster_geometry = ViewingGeometry.from_scene(scene, window=window)
        _check_table_range(column, gas, forward_model, cluster_geometry, cluster_index)
        lut = forward_model.lut(response, cluster_geometry, gas=gas.upper())
        clusters.append(
            ClusterResult(
                index=cluster_index,
                sources=members,
                window=window,
                bands=bands,
                column=column,
                geometry=cluster_geometry,
                lut=lut,
            )
        )
    sources = tuple(
        SourceResult(
            index=index,
            cluster=membership[index],
            window=window,
            injected=values.astype(np.float32),
            **fields,
        )
        for index, (window, values, fields) in enumerate(placed)
    )
    attrs: dict[str, Any] = {
        "spectra_regridder": regridder.name,
        "spectra_seed": int(seed),
        "spectra_sensor": str(scene.attrs.get("sensor", "")),
        "spectra_scene_id": str(scene.attrs.get("scene_id", "")),
    }
    for key, value in forward_model.provenance().items():
        attrs[f"spectra_forward_model_{key}"] = value
    result = EmbedResult(
        gas=gas, sources=sources, clusters=tuple(clusters), attrs=attrs
    )
    resolved = tuple(
        resolved_plume(plume, anchor)
        for plume, anchor in zip(plumes, anchors, strict=True)
    )
    return result, resolved
