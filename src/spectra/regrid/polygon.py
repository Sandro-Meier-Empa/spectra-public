"""Exact polygon-overlap conservative regridding of placed plume columns.

Every LES cell is intersected with every sensor pixel (or map cell) it
overlaps and the planar intersection areas weight the source values, as the
first-order conservative scheme of SCRIP/CDO ``remapcon`` does. Areas are
planar in the projected CRS; CDO integrates on the sphere in lon/lat, so the
two agree to the variation of the projection scale factor across a cell
(about ``1e-6`` relative over a few hundred metres in UTM), far below the
``float32`` precision of CDO output.

Overlap areas are computed by a vectorised Sutherland-Hodgman clipping of
the target polygons against the (convex) LES cells, which is exact for
convex clip polygons and an order of magnitude faster than the GEOS overlay;
``shapely.intersection`` remains available as the reference engine and is
used automatically when a source polygon is not convex.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import shapely
import xarray as xr
from shapely.errors import GEOSException

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import cell_areas, cell_polygons
from spectra.geometry.placement import PlacedColumn, place_column
from spectra.regrid.base import RegridResult, check_mass, target_window_cells
from spectra.regrid.results import finalize_result, map_data_array, scene_data_array
from spectra.regrid.windows import (
    anchor_pixel_centre,
    map_anchor_centre,
    map_window,
    map_window_cells,
    support_window,
)
from spectra.types import MapGrid, Placement, SceneGeometry, Window

Norm = Literal["destarea", "fracarea"]
NORMS: tuple[Norm, ...] = ("destarea", "fracarea")
Engine = Literal["auto", "clip", "shapely"]
ENGINES: tuple[Engine, ...] = ("auto", "clip", "shapely")

# Pairs per chunk of the vectorised clipping; bounds the temporary arrays.
_CLIP_CHUNK = 1 << 17
# Relative tolerance of the convexity test selecting the clipping engine.
_CONVEX_RTOL = 1e-9

# Target cells whose source overlap fraction is below this threshold are
# treated as uncovered under ``fracarea``; it removes degenerate slivers
# produced by rounding on shared edges, which would otherwise be filled with
# the neighbouring source value.
_FRACAREA_MIN_FRACTION = 1e-9


def _candidate_pairs(
    sources: np.ndarray, targets: np.ndarray, *, exact: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(source_index, target_index)`` of candidate overlapping pairs.

    The spatial index is built on the larger set so the smaller set is
    queried. With ``exact`` the ``intersects`` predicate filters the bounding
    box candidates; otherwise the caller discards zero-area pairs itself.
    """
    predicate = "intersects" if exact else None
    if targets.size >= sources.size:
        tree = shapely.STRtree(targets)
        source_index, target_index = tree.query(sources, predicate=predicate)
    else:
        tree = shapely.STRtree(sources)
        target_index, source_index = tree.query(targets, predicate=predicate)
    return np.asarray(source_index, dtype=np.intp), np.asarray(
        target_index, dtype=np.intp
    )


def _ring_vertices(polygons: np.ndarray) -> np.ndarray | None:
    """Return ``(n, m, 2)`` exterior-ring vertices padded with the last vertex.

    Returns ``None`` if a polygon is empty, degenerate or has holes, in which
    case the clipping engine cannot be used.
    """
    if polygons.size == 0:
        return np.zeros((0, 1, 2), dtype=np.float64)
    if np.any(shapely.get_num_interior_rings(polygons) > 0):
        return None
    rings = shapely.get_exterior_ring(polygons)
    counts = np.asarray(shapely.get_num_coordinates(rings), dtype=np.intp) - 1
    if np.any(counts < 3):
        return None
    coordinates = shapely.get_coordinates(rings)
    offsets = np.concatenate(([0], np.cumsum(counts + 1)[:-1]))
    positions = np.arange(int(counts.max()))
    index = offsets[:, None] + np.minimum(positions[None, :], (counts - 1)[:, None])
    return np.asarray(coordinates[index], dtype=np.float64)


def _is_convex(polygons: np.ndarray) -> bool:
    """Return whether every polygon equals its convex hull (within rounding)."""
    if polygons.size == 0:
        return True
    area = shapely.area(polygons)
    hull_area = shapely.area(shapely.convex_hull(polygons))
    return bool(np.all(hull_area <= area * (1.0 + _CONVEX_RTOL)))


def _signed_area(vertices: np.ndarray) -> np.ndarray:
    """Return signed shoelace areas of ``(n, m, 2)`` rings (positive if CCW).

    Coordinates are taken relative to the first vertex: projected coordinates
    of several million metres would otherwise lose the area of a square metre
    to cancellation.
    """
    x = vertices[..., 0] - vertices[..., :1, 0]
    y = vertices[..., 1] - vertices[..., :1, 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)


def _compact(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Keep the valid points of every ring in order, padded by the last one."""
    counts = valid.sum(axis=1)
    width = int(counts.max()) if counts.size else 0
    if width == 0:
        return np.zeros((points.shape[0], 1, 2), dtype=np.float64)
    position = np.cumsum(valid, axis=1) - 1
    rows = np.nonzero(valid)[0]
    compacted = np.zeros((points.shape[0], width, 2), dtype=np.float64)
    compacted[rows, position[valid]] = points[valid]
    index = np.minimum(np.arange(width)[None, :], np.maximum(counts - 1, 0)[:, None])
    return compacted[np.arange(points.shape[0])[:, None], index]


def clip_areas(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Return planar areas of ``subject[i] & clip[i]`` by Sutherland-Hodgman.

    Args:
        subject: ``(n, ms, 2)`` simple polygons (any orientation), padded by
            repeating their last vertex.
        clip: ``(n, mc, 2)`` convex polygons (any orientation), padded the
            same way. Repeated vertices form zero-length edges that clip
            nothing.

    Returns:
        ``(n,)`` intersection areas; exact for convex clip polygons.
    """
    count = subject.shape[0]
    areas = np.empty(count, dtype=np.float64)
    for start in range(0, count, _CLIP_CHUNK):
        stop = min(count, start + _CLIP_CHUNK)
        rings = subject[start:stop]
        clips = clip[start:stop]
        # Interior lies left of the edges of a CCW ring; flip for CW rings.
        sign = np.where(_signed_area(clips) >= 0, 1.0, -1.0)[:, None]
        for edge in range(clips.shape[1]):
            a = clips[:, edge]
            b = clips[:, (edge + 1) % clips.shape[1]]
            ex = (b[:, 0] - a[:, 0])[:, None]
            ey = (b[:, 1] - a[:, 1])[:, None]
            distance = sign * (
                ex * (rings[..., 1] - a[:, None, 1])
                - ey * (rings[..., 0] - a[:, None, 0])
            )
            inside = distance >= 0
            next_rings = np.roll(rings, -1, axis=1)
            next_distance = np.roll(distance, -1, axis=1)
            next_inside = np.roll(inside, -1, axis=1)
            crossing = inside != next_inside
            denominator = np.where(crossing, distance - next_distance, 1.0)
            fraction = np.where(crossing, distance / denominator, 0.0)
            cut = rings + fraction[..., None] * (next_rings - rings)
            points = np.empty((rings.shape[0], 2 * rings.shape[1], 2))
            points[:, 0::2] = cut
            points[:, 1::2] = next_rings
            valid = np.empty((rings.shape[0], 2 * rings.shape[1]), dtype=bool)
            valid[:, 0::2] = crossing
            valid[:, 1::2] = next_inside
            rings = _compact(points, valid)
        areas[start:stop] = np.abs(_signed_area(rings))
    return areas


def _overlap_areas(
    sources: np.ndarray, targets: np.ndarray, engine: Engine
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(source_index, target_index, overlap_area)`` of overlapping pairs."""
    source_rings = target_rings = None
    if engine != "shapely":
        source_rings = _ring_vertices(sources)
        target_rings = _ring_vertices(targets)
        if source_rings is None or target_rings is None or not _is_convex(sources):
            if engine == "clip":
                raise ContractError(
                    "The clipping engine needs convex, hole-free source polygons."
                )
            source_rings = target_rings = None
    if source_rings is None or target_rings is None:
        source_index, target_index = _candidate_pairs(sources, targets, exact=True)
        if source_index.size == 0:
            return source_index, target_index, np.zeros(0, dtype=np.float64)
        try:
            overlap = shapely.area(
                shapely.intersection(sources[source_index], targets[target_index])
            )
        except GEOSException as error:
            raise ContractError(
                f"Cell polygons are invalid for intersection: {error}"
            ) from error
        return source_index, target_index, np.asarray(overlap, dtype=np.float64)
    source_index, target_index = _candidate_pairs(sources, targets, exact=False)
    overlap = clip_areas(target_rings[target_index], source_rings[source_index])
    keep = overlap > 0
    return source_index[keep], target_index[keep], overlap[keep]


def regrid_polygons(
    source_polygons: npt.ArrayLike,
    source_values: npt.ArrayLike,
    source_areas: npt.ArrayLike,
    target_vx: npt.ArrayLike,
    target_vy: npt.ArrayLike,
    *,
    norm: Norm = "destarea",
    engine: Engine = "auto",
) -> tuple[np.ndarray, float, float, float]:
    """Conservatively remap cell values by exact planar polygon overlap.

    Args:
        source_polygons: Flat, row-major object array of ``ny * nx`` shapely
            polygons of the source cells.
        source_values: ``(ny, nx)`` source values (zeros are skipped).
        source_areas: ``(ny, nx)`` planar source cell areas in m2.
        target_vx: ``(nt0, nt1, 4)`` perimeter-ordered target vertex x.
        target_vy: ``(nt0, nt1, 4)`` perimeter-ordered target vertex y.
        norm: ``destarea`` divides the overlap-weighted sum by the full
            target cell area and conserves mass; ``fracarea`` (CDO's default
            ``REMAP_NORM``) divides by the overlapped target area and
            extrapolates at the edge of the source domain.
        engine: ``clip`` uses the vectorised clipping (convex sources only),
            ``shapely`` the GEOS overlay, ``auto`` the clipping when possible.

    Returns:
        ``(values, source_mass, source_mass_inside, target_mass)`` with
        ``values`` of shape ``(nt0, nt1)``, ``source_mass`` the integral of
        the source field, ``source_mass_inside`` the source mass overlapping
        the target cells and ``target_mass`` the integral of ``values``.

    Raises:
        ContractError: If the inputs are inconsistent or a polygon is invalid.
    """
    if norm not in NORMS:
        raise ContractError(f"norm must be one of {NORMS}; got {norm!r}.")
    if engine not in ENGINES:
        raise ContractError(f"engine must be one of {ENGINES}; got {engine!r}.")
    values = np.asarray(source_values, dtype=np.float64)
    areas = np.asarray(source_areas, dtype=np.float64)
    polygons = np.asarray(source_polygons, dtype=object).reshape(-1)
    if values.ndim != 2 or areas.shape != values.shape:
        raise ContractError("source_values and source_areas must be (ny, nx).")
    if polygons.size != values.size:
        raise ContractError("source_polygons must hold one polygon per source cell.")
    vx = np.asarray(target_vx, dtype=np.float64)
    vy = np.asarray(target_vy, dtype=np.float64)
    if vx.ndim != 3 or vx.shape[-1] != 4 or vx.shape != vy.shape:
        raise ContractError("target vertices must be (nt0, nt1, 4) arrays.")
    target_shape = (int(vx.shape[0]), int(vx.shape[1]))
    target_polygons = cell_polygons(vx, vy)
    target_areas = cell_areas(vx, vy).reshape(-1)
    n_target = target_areas.size

    flat_values = values.reshape(-1)
    flat_areas = areas.reshape(-1)
    source_mass = float(np.sum(flat_values * flat_areas))
    # Zero cells carry no mass, so ``destarea`` skips them; ``fracarea``
    # needs them in the overlap denominator, or targets at the edge of the
    # nonzero support would be extrapolated instead of diluted.
    if norm == "destarea":
        active = np.flatnonzero(flat_values != 0)
    else:
        active = np.arange(flat_values.size)
    active_values = flat_values[active]

    overlap_sum = np.zeros(n_target, dtype=np.float64)
    weighted_sum = np.zeros(n_target, dtype=np.float64)
    source_overlap = np.zeros(active.size, dtype=np.float64)
    if active.size and n_target:
        source_index, target_index, overlap = _overlap_areas(
            polygons[active], target_polygons, engine
        )
        if source_index.size:
            overlap_sum = np.bincount(target_index, weights=overlap, minlength=n_target)
            weighted_sum = np.bincount(
                target_index,
                weights=overlap * active_values[source_index],
                minlength=n_target,
            )
            source_overlap = np.bincount(
                source_index, weights=overlap, minlength=active.size
            )
    source_mass_inside = float(np.sum(active_values * source_overlap))

    positive_area = target_areas > 0
    safe_area = np.where(positive_area, target_areas, 1.0)
    if norm == "destarea":
        result = np.where(positive_area, weighted_sum / safe_area, 0.0)
    else:
        covered = positive_area & (overlap_sum >= _FRACAREA_MIN_FRACTION * safe_area)
        safe_overlap = np.where(covered, overlap_sum, 1.0)
        result = np.where(covered, weighted_sum / safe_overlap, 0.0)
    target_mass = float(np.sum(result * target_areas))
    return result.reshape(target_shape), source_mass, source_mass_inside, target_mass


class PolygonRegridder:
    """Conservative regridder based on exact planar polygon overlaps.

    Args:
        norm: ``destarea`` (mass conserving) or ``fracarea`` (CDO default,
            extrapolating at the source domain edge).
        rtol: Maximum relative mass error accepted for ``destarea`` results.
        engine: Overlap engine passed to :func:`regrid_polygons`.
    """

    name: str = "polygon"

    def __init__(
        self, norm: Norm = "destarea", rtol: float = 1e-6, engine: Engine = "auto"
    ) -> None:
        if norm not in NORMS:
            raise ContractError(f"norm must be one of {NORMS}; got {norm!r}.")
        if engine not in ENGINES:
            raise ContractError(f"engine must be one of {ENGINES}; got {engine!r}.")
        if not np.isfinite(rtol) or rtol < 0:
            raise ContractError("rtol must be a finite nonnegative number.")
        self.norm: Norm = norm
        self.rtol = float(rtol)
        self.engine: Engine = engine

    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"{type(self).__name__}(norm={self.norm!r}, rtol={self.rtol!r}, "
            f"engine={self.engine!r})"
        )

    def _regrid(
        self,
        placed: PlacedColumn,
        target_vx: np.ndarray,
        target_vy: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        return regrid_polygons(
            placed.cell_polygons(),
            placed.values,
            placed.cell_areas(),
            target_vx,
            target_vy,
            norm=self.norm,
            engine=self.engine,
        )

    def _finalize(
        self,
        data: xr.DataArray,
        placed: PlacedColumn,
        target_area: np.ndarray,
        source_mass: float,
        source_mass_inside: float,
        target_mass: float,
    ) -> RegridResult:
        result = finalize_result(
            data,
            method=self.name,
            source_area=placed.cell_areas(),
            target_area=target_area,
            source_mass=source_mass,
            source_mass_inside=source_mass_inside,
            target_mass=target_mass,
            attrs={"regrid_norm": self.norm, "regrid_engine": self.engine},
        )
        if self.norm == "destarea":
            check_mass(result, self.rtol)
        return result

    def regrid_to_scene(
        self,
        column: xr.DataArray,
        scene: xr.Dataset,
        placement: Placement,
        *,
        window: Window | None = None,
    ) -> RegridResult:
        """Place ``column`` on the sensor grid and return the full result.

        Args:
            column: Plume column on ``(y, x)`` in metres with ``units``.
            scene: Dataset carrying the projected pixel-centre geometry.
            placement: Anchor pixel, corner, reflection and rotation.
            window: Target index window; derived from the support if omitted.

        Returns:
            Result with the regridded ``(across, along)`` data of the window.
        """
        geometry = SceneGeometry.from_scene(scene)
        placed = place_column(
            column,
            crs=geometry.crs,
            anchor_target_xy=anchor_pixel_centre(geometry, placement),
            placement=placement,
        )
        if window is None:
            window = support_window(placed, geometry)
        elif window.across.stop > geometry.shape[0] or (
            window.along.stop > geometry.shape[1]
        ):
            raise ContractError("window lies outside the scene geometry.")
        vx, vy, target_area = target_window_cells(geometry, window)
        values, source_mass, inside, target_mass = self._regrid(placed, vx, vy)
        data = scene_data_array(
            values, scene=scene, window=window, placed=placed, name=column.name
        )
        return self._finalize(
            data, placed, target_area, source_mass, inside, target_mass
        )

    def to_scene(
        self,
        column: xr.DataArray,
        scene: xr.Dataset,
        placement: Placement,
        *,
        window: Window | None = None,
    ) -> xr.DataArray:
        """Return the column on the sensor grid, cropped to ``window``."""
        return self.regrid_to_scene(column, scene, placement, window=window).data

    def regrid_to_map(
        self, column: xr.DataArray, grid: MapGrid, placement: Placement
    ) -> RegridResult:
        """Place ``column`` on a regular map grid and return the full result.

        The anchor cell is ``(grid.y[anchor_across], grid.x[anchor_along])``.
        """
        placed = place_column(
            column,
            crs=grid.crs,
            anchor_target_xy=map_anchor_centre(grid, placement),
            placement=placement,
        )
        y_slice, x_slice = map_window(placed, grid)
        vx, vy, target_area = map_window_cells(grid, y_slice, x_slice)
        values, source_mass, inside, target_mass = self._regrid(placed, vx, vy)
        data = map_data_array(
            values,
            grid=grid,
            y_slice=y_slice,
            x_slice=x_slice,
            placed=placed,
            name=column.name,
        )
        return self._finalize(
            data, placed, target_area, source_mass, inside, target_mass
        )

    def to_map(
        self, column: xr.DataArray, grid: MapGrid, placement: Placement
    ) -> xr.DataArray:
        """Return the column on a regular map grid on dims ``(y, x)``."""
        return self.regrid_to_map(column, grid, placement).data


__all__ = [
    "ENGINES",
    "NORMS",
    "Engine",
    "Norm",
    "PolygonRegridder",
    "clip_areas",
    "regrid_polygons",
]
