"""Place, regrid and cluster plumes and build their absorption tables.

:func:`embed_scene` turns the plumes of a run into an
:class:`~spectra.result.EmbedResult`: every plume is anchored
(:func:`resolve_plumes`), regridded conservatively onto the sensor grid and
converted to kg m-2; plumes whose windows overlap form clusters that share one
summed column and one absorption table at the median geometry of their
window. The Lambert-Beer attenuation itself is applied when the result is
exported (:mod:`spectra.export.native`) or viewed in memory
(:meth:`~spectra.result.EmbedResult.to_dataset`).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyproj
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.contracts.plume_column import scale_emission, select_time
from spectra.contracts.units import CANONICAL_COLUMN_UNITS, convert_column_units
from spectra.geometry.amf import effective_layer_height
from spectra.geometry.corners import Outline
from spectra.geometry.crs import to_lonlat
from spectra.geometry.feasibility import AnchorTiles
from spectra.geometry.placement import anchor_source_point
from spectra.geometry.swath import ResolvedAnchor, resolve_anchor, swath_polygon
from spectra.geometry.transform import transform_coordinates
from spectra.plume.augment import plume_fraction_below
from spectra.plume.io import open_plume_column
from spectra.protocols import ForwardModel
from spectra.regrid.polygon import PolygonRegridder
from spectra.result import ClusterResult, EmbedResult, SourceResult, windows_overlap
from spectra.run import PlumeSpec, wavelength_ranges
from spectra.types import SceneGeometry, SpectralResponse, ViewingGeometry, Window

logger = logging.getLogger(__name__)


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
        :exc:`ContractError`: If the plumes carry different gases.
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
    swath: Outline | None = None,
) -> list[ResolvedAnchor]:
    """Resolve the anchor of every plume, drawing random anchors reproducibly.

    Plume ``i`` draws from the ``i``-th child of ``SeedSequence(seed)``, so
    adding or removing other plumes leaves its draw unchanged.
    """
    datasets = dict(datasets) if datasets is not None else open_plumes(plumes)
    geometry = geometry or SceneGeometry.from_scene(scene)
    swath = swath if swath is not None else swath_polygon(geometry)
    # Random anchors share one read of the pixel centres.
    tiles = (
        AnchorTiles.from_geometry(geometry)
        if any(plume.anchor.mode == "random" for plume in plumes)
        else None
    )
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
                tiles=tiles,
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
    scene: xr.Dataset,
    spectral_range_nm: tuple[float, float],
    ranges_nm: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return the scene bands whose centres lie in the attenuated ranges.

    Args:
        scene: ``SensorScene`` dataset.
        spectral_range_nm: Range the forward model covers.
        ranges_nm: Wavelength ranges to attenuate, each inside
            ``spectral_range_nm``; the whole model range when omitted.

    Returns:
        Strictly increasing band indices.

    Raises:
        :exc:`ContractError`: If a range leaves the model range or holds no band.
    """
    wavelength = np.asarray(scene["wavelength"].values, dtype=float)
    model_lower, model_upper = spectral_range_nm
    if ranges_nm is None:
        ranges, where = (spectral_range_nm,), "the forward model's range"
    else:
        ranges, where = wavelength_ranges(ranges_nm), "the wavelength range"
    runs = []
    for lower, upper in ranges:
        if lower < model_lower or upper > model_upper:
            raise ContractError(
                f"The wavelength range ({lower}, {upper}) nm leaves the forward "
                f"model's range {spectral_range_nm} nm."
            )
        inside = np.flatnonzero((wavelength >= lower) & (wavelength <= upper))
        if inside.size == 0:
            raise ContractError(f"No scene band lies in {where} {(lower, upper)} nm.")
        runs.append(np.arange(inside[0], inside[-1] + 1))
    return np.concatenate(runs)


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
    wavelength_ranges_nm: Sequence[tuple[float, float]] | None = None,
) -> tuple[EmbedResult, tuple[PlumeSpec, ...]]:
    """Place, regrid and cluster plumes and build one absorption table per cluster.

    Args:
        scene: ``SensorScene`` dataset.
        plumes: Plumes to embed.
        forward_model: Model building the absorption tables.
        seed: Seed of the random anchors.
        datasets: Opened plume columns by file; opened when omitted.
        regridder: Conservative regridder; exact polygon overlap by default.
        wavelength_ranges_nm: Wavelength ranges whose bands are attenuated;
            every band in the forward model's range when omitted.

    Returns:
        The embedding result and the plumes with drawn anchors filled in.

    Raises:
        :exc:`ContractError`: If plumes, anchors, wavelength ranges or the
            forward model are invalid.
    """
    datasets = dict(datasets) if datasets is not None else open_plumes(plumes)
    gas = plume_gas(datasets)
    forward_model.check(gas.upper())
    bands = attenuated_bands(
        scene, forward_model.spectral_range_nm, wavelength_ranges_nm
    )
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


__all__ = [
    "attenuated_bands",
    "cluster_windows",
    "embed_scene",
    "open_plumes",
    "plume_gas",
    "plume_snapshot",
    "resolve_plumes",
    "resolved_plume",
]
