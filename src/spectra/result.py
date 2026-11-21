"""In-memory result of one embedding run, consumed by the native exporter.

The pipeline places every plume, groups overlapping plume windows into
clusters and builds one absorption table per cluster; the result carries the
summed column and the table of each cluster and the injected column and
provenance of each plume. The Lambert-Beer attenuation itself is applied by
the exporter to the radiance it reads from the native file, so the result
never holds radiance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.types import ViewingGeometry, Window


@dataclass(frozen=True, eq=False)
class SourceResult:
    """One embedded plume.

    Attributes:
        index: Position of the plume in the run file.
        plume_file: Path of the plume-column file.
        case_id: LES case identifier of the plume column.
        time_index: LES time index of the embedded snapshot.
        emission_rate_kg_s: Emission rate of the snapshot after scaling.
        emission_scale: Emission scale applied to the column.
        lon: Longitude of the emission source after placement, in degrees.
        lat: Latitude of the emission source after placement, in degrees.
        source_model_x: Emission source in the LES model frame, the plume
            column's ``source_x`` [m].
        source_model_y: Emission source in the LES model frame, the plume
            column's ``source_y`` [m].
        source_along: Along-track index of the pixel nearest the placed
            emission source, ``-1`` when it lies outside the swath.
        source_across: Across-track index of that pixel, ``-1`` when outside.
        anchor_along: Along-track index of the anchor pixel.
        anchor_across: Across-track index of the anchor pixel.
        anchor_point: Point of the LES grid pinned to the anchor: ``source``
            or a corner such as ``lower left``.
        anchor_mode: ``lonlat``, ``pixel`` or ``random``.
        rotation_deg: Rotation of the plume, counterclockwise positive.
        flip: Reflected LES axis: ``none``, ``x`` or ``y``.
        outside_fraction: Plume mass fraction outside the swath.
        injected_mass_kg: Plume mass on the sensor grid.
        cluster: Index of the cluster the plume belongs to.
        window: Index window of the injected column.
        injected: ``(n_across, n_along)`` injected column in kg m-2 on
            ``window``, float32.
    """

    index: int
    plume_file: str
    case_id: str
    time_index: int
    emission_rate_kg_s: float
    emission_scale: float
    lon: float
    lat: float
    source_model_x: float
    source_model_y: float
    source_along: int
    source_across: int
    anchor_along: int
    anchor_across: int
    anchor_point: str
    anchor_mode: str
    rotation_deg: int
    flip: str
    outside_fraction: float
    injected_mass_kg: float
    cluster: int
    window: Window
    injected: np.ndarray

    def __post_init__(self) -> None:
        """Require the injected column to fill its window."""
        if self.injected.shape != self.window.shape:
            raise ContractError(
                f"Injected column of source {self.index} has shape "
                f"{self.injected.shape}, its window {self.window.shape}."
            )


@dataclass(frozen=True, eq=False)
class ClusterResult:
    """Plumes with overlapping windows, attenuated together.

    Attributes:
        index: Cluster index.
        sources: Indices of the member plumes.
        window: Index window covering every member.
        bands: Strictly increasing scene band indices attenuated (those in the
            run's wavelength ranges); the table's wavelengths are these bands'.
        column: ``(n_across, n_along)`` summed injected column in kg m-2 on
            ``window``, float64.
        geometry: Observation geometry the table was built for (median over
            the window).
        lut: Absorption table of the cluster geometry.
    """

    index: int
    sources: tuple[int, ...]
    window: Window
    bands: np.ndarray
    column: np.ndarray
    geometry: ViewingGeometry
    lut: xr.Dataset

    def __post_init__(self) -> None:
        """Validate the band indices against the table and the column shape."""
        bands = np.asarray(self.bands)
        if (
            bands.ndim != 1
            or bands.size == 0
            or bands.dtype.kind not in "iu"
            or bands[0] < 0
            or np.any(np.diff(bands) <= 0)
        ):
            raise ContractError(
                "ClusterResult.bands must be strictly increasing band indices."
            )
        object.__setattr__(self, "bands", bands)
        if bands.size != self.lut.sizes.get("wavelength"):
            raise ContractError(
                f"Cluster {self.index} attenuates {bands.size} bands but its table "
                f"has {self.lut.sizes.get('wavelength')} wavelengths."
            )
        if self.column.shape != self.window.shape:
            raise ContractError(
                f"Column of cluster {self.index} has shape {self.column.shape}, "
                f"its window {self.window.shape}."
            )

    @property
    def band_runs(self) -> tuple[tuple[slice, slice], ...]:
        """Contiguous runs of :attr:`bands` as (scene band, table row) slices."""
        breaks = np.flatnonzero(np.diff(self.bands) > 1) + 1
        starts = np.concatenate([[0], breaks])
        stops = np.concatenate([breaks, [self.bands.size]])
        return tuple(
            (
                slice(int(self.bands[start]), int(self.bands[stop - 1]) + 1),
                slice(int(start), int(stop)),
            )
            for start, stop in zip(starts, stops, strict=True)
        )

    @property
    def amf(self) -> float:
        """Geometric air-mass factor of the cluster table."""
        return float(self.lut.attrs["geometric_air_mass_factor"])

    @property
    def model_surface_elevation(self) -> float:
        """Surface elevation the forward model used (negative values clamp to 0)."""
        if "surface_elevation_clamped_from" in self.lut.attrs:
            return 0.0
        return float(self.geometry.surface_elevation)


@dataclass(frozen=True, eq=False)
class EmbedResult:
    """Everything the exporter writes for one run.

    Attributes:
        gas: Lower-case gas name.
        sources: Embedded plumes in run-file order.
        clusters: Pairwise disjoint clusters.
        attrs: Run-level ``spectra_*`` attributes.
    """

    gas: str
    sources: tuple[SourceResult, ...]
    clusters: tuple[ClusterResult, ...]
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require disjoint cluster windows and consistent memberships."""
        windows = [cluster.window for cluster in self.clusters]
        for i, first in enumerate(windows):
            for second in windows[i + 1 :]:
                if windows_overlap(first, second):
                    raise ContractError("Cluster windows must be pairwise disjoint.")
        for source in self.sources:
            if not 0 <= source.cluster < len(self.clusters):
                raise ContractError(f"Source {source.index} names an unknown cluster.")

    @property
    def injected_name(self) -> str:
        """Name of the injected-column variable."""
        return f"injected_{self.gas.lower()}_mass"

    def to_dataset(self, scene: xr.Dataset) -> xr.Dataset:
        """Return the embedded scene in memory, for notebooks and small scenes.

        The radiance is loaded and attenuated exactly as the native exporter
        does it; the per-plume injected columns are added on ``source``.

        Args:
            scene: The ``SensorScene`` the result was computed for.

        Returns:
            Copy of ``scene`` with attenuated radiance and
            ``injected_<gas>_mass(source, across, along)``.
        """
        from spectra.spectral.embedding import embed_column_in_radiance

        radiance = scene["radiance"].transpose("across", "along", "band")
        values = np.array(radiance.values)
        for cluster in self.clusters:
            window = cluster.window
            wavelength = cluster.lut["wavelength"].values
            for bands, rows in cluster.band_runs:
                block = values[window.across, window.along, bands]
                attenuated = embed_column_in_radiance(
                    xr.DataArray(
                        block,
                        dims=("across", "along", "band"),
                        coords={"wavelength": ("band", wavelength[rows])},
                    ),
                    xr.DataArray(
                        cluster.column,
                        dims=("across", "along"),
                        attrs={"units": "kg m-2"},
                    ),
                    cluster.lut,
                    gas=self.gas.lower(),
                ).values
                patch = (cluster.column > 0)[..., None] & np.isfinite(block)
                block[patch] = attenuated[patch]
        field = np.zeros(
            (len(self.sources), scene.sizes["across"], scene.sizes["along"]),
            dtype=np.float32,
        )
        for position, source in enumerate(self.sources):
            field[position, source.window.across, source.window.along] = source.injected
        embedded = scene.copy()
        embedded["radiance"] = radiance.copy(data=values)
        embedded[self.injected_name] = xr.DataArray(
            field,
            dims=("source", "across", "along"),
            attrs={"units": "kg m-2", "column_convention": "vertical"},
        )
        embedded.attrs.update(self.attrs)
        return embedded


def windows_overlap(first: Window, second: Window) -> bool:
    """Return whether two index windows share at least one pixel."""
    return (
        first.across.start < second.across.stop
        and second.across.start < first.across.stop
        and first.along.start < second.along.stop
        and second.along.start < first.along.stop
    )


__all__ = ["ClusterResult", "EmbedResult", "SourceResult", "windows_overlap"]
