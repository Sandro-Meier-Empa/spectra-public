"""Plume placement inside a sensor swath.

The swath outline, the anchor-pixel draws and the outside-mass test are
ported from ``aviris4_ch4.regridding`` and generalised to any sensor through
:class:`~spectra.types.SceneGeometry`; geographic anchors and the emission
source as anchor point are SPECTRA additions. Only boundary rows and columns of the
pixel-centre arrays are read for the swath outline, because real swaths hold
tens of millions of pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import shapely
import xarray as xr
from shapely.geometry import Point, Polygon

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import (
    Outline,
    axis_nodes,
    outline_polygon,
    ring_outline,
    window_corner_nodes,
)
from spectra.geometry.crs import from_lonlat
from spectra.geometry.placement import place_column
from spectra.geometry.transform import (
    support_vertices,
    transform_coordinates,
    validate_flip,
    validate_rotation,
)
from spectra.types import AnchorPoint, Flip, Placement, SceneGeometry, Window

DEFAULT_MAX_ATTEMPTS = 10_000
_ACROSS_CHUNK = 8
AnchorMode = Literal["lonlat", "pixel", "random"]
# Number of pixel centres loaded by coarse searches over the scene geometry.
_COARSE_SAMPLES = 20_000


@dataclass(frozen=True)
class AnchorDraw:
    """One accepted anchor pixel with its centre and outside mass fraction.

    Attributes:
        along: Zero-based along-track index of the anchor pixel.
        across: Zero-based across-track index of the anchor pixel.
        anchor_x: Projected x coordinate of the anchor pixel centre.
        anchor_y: Projected y coordinate of the anchor pixel centre.
        outside_fraction: Plume mass fraction outside the swath.
        attempts: Random draws needed; ``None`` for an explicit anchor.
    """

    along: int
    across: int
    anchor_x: float
    anchor_y: float
    outside_fraction: float
    attempts: int | None


@dataclass(frozen=True, eq=False)
class MassPoints:
    """Transformed nonzero plume cell centres relative to the LES anchor.

    The centres are reflected and rotated once around the anchor corner cell
    centre of the LES grid; the outside-mass test for any candidate pixel is
    then a translation, as in ``aviris4_ch4``.

    Attributes:
        dx: Transformed centre x offsets from the anchor.
        dy: Transformed centre y offsets from the anchor.
        weights: Column value of every nonzero cell.
        total: Sum of ``weights``.
    """

    dx: np.ndarray
    dy: np.ndarray
    weights: np.ndarray
    total: float

    @classmethod
    def from_values(
        cls,
        values: npt.ArrayLike,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        anchor_source: tuple[float, float],
        rotation_deg: int,
        flip: Flip,
    ) -> MassPoints:
        """Build the mass points of a ``(ny, nx)`` field on native coordinates.

        Args:
            values: Non-negative column values, gaps zeroed.
            x: Native cell-centre x coordinates.
            y: Native cell-centre y coordinates.
            anchor_source: LES corner cell centre used as pivot.
            rotation_deg: Counterclockwise rotation in degrees.
            flip: Axis mirrored before the rotation.

        Returns:
            Transformed nonzero cell centres relative to the anchor.
        """
        field = np.asarray(values, dtype=np.float64)
        nonzero = field > 0
        if not nonzero.any():
            raise ContractError("Column has empty nonzero support.")
        grid_x, grid_y = np.meshgrid(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            indexing="xy",
        )
        tx, ty = transform_coordinates(
            grid_x[nonzero],
            grid_y[nonzero],
            pivot=anchor_source,
            rotation_deg=rotation_deg,
            flip=flip,
        )
        weights = field[nonzero]
        return cls(
            dx=tx - anchor_source[0],
            dy=ty - anchor_source[1],
            weights=weights,
            total=float(weights.sum()),
        )


def _boundary_nodes(
    values: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the outer grid-node rows and columns of a centre array.

    Only the first/last two rows and columns are read. The arithmetic matches
    the chunked node construction of ``aviris4_ch4`` exactly.

    Returns:
        ``(top, right, bottom, left)`` node coordinates in index order.
    """
    dim0, dim1 = (str(dim) for dim in values.dims)
    n0, n1 = values.shape
    if n0 < 2 or n1 < 2:
        raise ContractError("Scene geometry needs at least two pixels per dimension.")
    first_rows = np.asarray(values.isel({dim0: slice(0, 2)}).values, dtype=np.float64)
    last_rows = np.asarray(
        values.isel({dim0: slice(n0 - 2, n0)}).values, dtype=np.float64
    )
    first_columns = np.asarray(
        values.isel({dim1: slice(0, 2)}).values, dtype=np.float64
    )
    last_columns = np.asarray(
        values.isel({dim1: slice(n1 - 2, n1)}).values, dtype=np.float64
    )
    nodes_first = axis_nodes(first_rows, axis=1)
    nodes_last = axis_nodes(last_rows, axis=1)
    top = 1.5 * nodes_first[0] - 0.5 * nodes_first[1]
    bottom = 1.5 * nodes_last[1] - 0.5 * nodes_last[0]
    left = axis_nodes(1.5 * first_columns[:, 0] - 0.5 * first_columns[:, 1])
    right = axis_nodes(1.5 * last_columns[:, 1] - 0.5 * last_columns[:, 0])
    return top, right, bottom, left


def _ring(
    top: np.ndarray, right: np.ndarray, bottom: np.ndarray, left: np.ndarray
) -> np.ndarray:
    """Concatenate boundary nodes into a closed ring in grid order."""
    return np.concatenate((top, right[1:], bottom[-2::-1], left[-2:0:-1]))


def swath_polygon(geometry: SceneGeometry) -> Outline:
    """Return the region enclosed by the outer pixel boundaries of a scene."""
    return ring_outline(
        _ring(*_boundary_nodes(geometry.x)), _ring(*_boundary_nodes(geometry.y))
    )


def window_nodes(
    geometry: SceneGeometry, window: Window
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(n_across + 1, n_along + 1)`` node coordinates of a window.

    Neighbouring pixel centres outside the window give the boundary nodes
    where they exist, exactly as the CDO target grids define them.
    """
    n_across, n_along = geometry.shape
    dim0, dim1 = geometry.spatial_dims
    start = (max(0, window.across.start - 1), max(0, window.along.start - 1))
    indexers = {
        dim0: slice(start[0], min(n_across, window.across.stop + 1)),
        dim1: slice(start[1], min(n_along, window.along.stop + 1)),
    }
    x = np.asarray(geometry.x.isel(indexers).values, dtype=np.float64)
    y = np.asarray(geometry.y.isel(indexers).values, dtype=np.float64)
    return (
        window_corner_nodes(x, window, start=start),
        window_corner_nodes(y, window, start=start),
    )


def window_polygon(geometry: SceneGeometry, window: Window) -> Outline:
    """Return the outer pixel-boundary polygon of an index window."""
    node_x, node_y = window_nodes(geometry, window)
    return outline_polygon(node_x, node_y)


def pixel_centre(
    geometry: SceneGeometry, along: int, across: int
) -> tuple[float, float]:
    """Return the projected centre of one pixel."""
    dim0, dim1 = geometry.spatial_dims
    indexer = {dim0: across, dim1: along}
    centre_x = float(np.asarray(geometry.x.isel(indexer).values, dtype=float))
    centre_y = float(np.asarray(geometry.y.isel(indexer).values, dtype=float))
    if not (np.isfinite(centre_x) and np.isfinite(centre_y)):
        raise ContractError(
            f"Pixel (along={along}, across={across}) has no finite geolocation."
        )
    return centre_x, centre_y


def anchor_pixel_spacing(geometry: SceneGeometry, along: int, across: int) -> float:
    """Return the largest centre spacing in the anchor pixel's 2x2 neighbourhood."""
    n_across, n_along = geometry.shape
    dim0, dim1 = geometry.spatial_dims
    across_start = across if across + 1 < n_across else across - 1
    along_start = along if along + 1 < n_along else along - 1
    indexers = {
        dim0: slice(across_start, across_start + 2),
        dim1: slice(along_start, along_start + 2),
    }
    x = np.asarray(geometry.x.isel(indexers).values, dtype=float)
    y = np.asarray(geometry.y.isel(indexers).values, dtype=float)
    spacing = max(
        float(np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).max()),
        float(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)).max()),
    )
    if not np.isfinite(spacing) or spacing <= 0:
        raise ContractError("Pixel spacing at the anchor is not positive.")
    return spacing


def block(
    geometry: SceneGeometry, across: slice, along: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Load a block of pixel-centre coordinates as float64 arrays."""
    dim_across, dim_along = geometry.spatial_dims
    indexer = {dim_across: across, dim_along: along}
    x = np.asarray(geometry.x.isel(indexer).values, dtype=np.float64)
    y = np.asarray(geometry.y.isel(indexer).values, dtype=np.float64)
    return x, y


def coarse_strides(n_across: int, n_along: int) -> tuple[int, int]:
    stride = max(1, math.ceil(math.sqrt(n_across * n_along / _COARSE_SAMPLES)))
    return min(stride, max(1, n_across // 2)), min(stride, max(1, n_along // 2))


def candidate_region(
    geometry: SceneGeometry, box: tuple[float, float, float, float]
) -> Window:
    """Return an index window that contains every pixel centre inside ``box``.

    A strided subsample of the centres is tested against the box expanded by
    the largest distance between a pixel and its coarse representative, so no
    fine pixel inside the box can be missed; the fine test runs on the
    returned block only.
    """
    n_across, n_along = geometry.shape
    stride_across, stride_along = coarse_strides(n_across, n_along)
    x, y = block(
        geometry,
        slice(0, n_across, stride_across),
        slice(0, n_along, stride_along),
    )
    step_across = float(np.nanmax(np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))))
    step_along = float(np.nanmax(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))))
    if not math.isfinite(step_across) or not math.isfinite(step_along):
        raise ContractError("Scene pixel centres contain no finite spacing.")
    # A fine pixel lies within one stride of its coarse representative along
    # each axis; the factor two absorbs local spacing variations.
    margin = 2.0 * (step_across + step_along)
    found = (
        (x >= box[0] - margin)
        & (x <= box[2] + margin)
        & (y >= box[1] - margin)
        & (y <= box[3] + margin)
    )
    rows, columns = np.nonzero(found)
    if rows.size == 0:
        raise ContractError("No pixel centre lies inside the placed support.")
    return Window(
        across=slice(
            max(0, int(rows.min()) * stride_across - stride_across),
            min(n_across, int(rows.max()) * stride_across + stride_across + 1),
        ),
        along=slice(
            max(0, int(columns.min()) * stride_along - stride_along),
            min(n_along, int(columns.max()) * stride_along + stride_along + 1),
        ),
    )


def outside_mass_fraction(
    region: Outline,
    support_rel: npt.NDArray[np.floating[Any]],
    centre: tuple[float, float],
    points: MassPoints,
) -> float:
    """Return the plume mass fraction outside ``region`` for one anchor centre.

    A support rectangle covered by the region contains the whole plume and
    one disjoint from it none, so only straddling rectangles need the
    cell-centre test. Cells count by their centre, so the fraction is resolved
    at the LES cell size.
    """
    support_polygon = Polygon(np.add(support_rel, np.asarray(centre)))
    if region.covers(support_polygon):
        return 0.0
    if not region.intersects(support_polygon):
        return 1.0
    inside = np.asarray(
        shapely.contains_xy(region, points.dx + centre[0], points.dy + centre[1]),
        dtype=bool,
    )
    inside_mass = float(points.weights[inside].sum())
    return float(min(1.0, max(0.0, 1.0 - inside_mass / points.total)))


def _validate_index(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _validate_indices(
    along: int, across: int, geometry: SceneGeometry
) -> tuple[int, int]:
    """Validate one ``(along, across)`` anchor index pair."""
    n_across, n_along = geometry.shape
    along = _validate_index(along, "along")
    across = _validate_index(across, "across")
    if not 0 <= along < n_along:
        raise ContractError(
            f"along index {along} is outside the along range [0, {n_along})."
        )
    if not 0 <= across < n_across:
        raise ContractError(
            f"across index {across} is outside the across range [0, {n_across})."
        )
    return along, across


def outside_fraction_bounds(
    min_outside_fraction: float, max_outside_fraction: float
) -> tuple[float, float]:
    """Return validated minimum and maximum outside plume mass fractions."""
    values: list[float] = []
    for name, value in (
        ("min_outside_fraction", min_outside_fraction),
        ("max_outside_fraction", max_outside_fraction),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a number, got {value!r}.")
        number = float(value)
        if not np.isfinite(number) or not 0 <= number < 1:
            raise ContractError(
                f"{name} must be in the interval [0, 1); got {value!r}."
            )
        values.append(number)
    minimum, maximum = values
    if minimum > maximum:
        raise ContractError(
            f"min_outside_fraction ({minimum}) must not exceed "
            f"max_outside_fraction ({maximum})."
        )
    return minimum, maximum


def _validate_max_attempts(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"max_attempts must be an integer, got {value!r}.")
    attempts = int(value)
    if attempts < 1:
        raise ContractError(f"max_attempts must be at least 1; got {value!r}.")
    return attempts


def explicit_anchor(
    geometry: SceneGeometry,
    swath: Outline,
    points: MassPoints,
    support_rel: npt.NDArray[np.floating[Any]],
    *,
    along: int,
    across: int,
) -> AnchorDraw:
    """Return the given anchor pixel after checking that plume mass is inside."""
    along, across = _validate_indices(along, across, geometry)
    anchor_x, anchor_y = pixel_centre(geometry, along, across)
    fraction = outside_mass_fraction(swath, support_rel, (anchor_x, anchor_y), points)
    if fraction >= 1.0:
        raise ContractError(
            f"Embedding location (along={along}, across={across}) places the "
            "whole plume outside the swath."
        )
    return AnchorDraw(
        along=along,
        across=across,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        outside_fraction=fraction,
        attempts=None,
    )


def random_anchor(
    geometry: SceneGeometry,
    swath: Outline,
    points: MassPoints,
    support_rel: npt.NDArray[np.floating[Any]],
    *,
    bounds: tuple[float, float],
    rng: np.random.Generator | None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> AnchorDraw:
    """Draw uniform anchor pixels until the outside mass fraction is in bounds.

    Every pixel is drawn with equal probability and the first eligible one is
    returned, so accepted anchors are uniformly distributed over the eligible
    pixels without evaluating the whole swath.
    """
    n_across, n_along = geometry.shape
    generator = np.random.default_rng() if rng is None else rng
    minimum, maximum = outside_fraction_bounds(*bounds)
    max_attempts = _validate_max_attempts(max_attempts)
    for attempt in range(1, max_attempts + 1):
        along = int(generator.integers(0, n_along))
        across = int(generator.integers(0, n_across))
        anchor_x, anchor_y = pixel_centre(geometry, along, across)
        fraction = outside_mass_fraction(
            swath, support_rel, (anchor_x, anchor_y), points
        )
        if minimum <= fraction <= maximum:
            return AnchorDraw(
                along=along,
                across=across,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                outside_fraction=fraction,
                attempts=attempt,
            )
    raise ContractError(
        "No random anchor pixel with a plume mass fraction outside the swath "
        f"between {minimum:.3f} and {maximum:.3f} was found in {max_attempts} "
        "draws."
    )


def locate_pixel(geometry: SceneGeometry, xy: tuple[float, float]) -> tuple[int, int]:
    """Return ``(along, across)`` of the pixel whose centre is nearest to ``xy``.

    A strided subsample locates the neighbourhood of the point; only that block
    of the (possibly lazy) pixel-centre geometry is loaded.

    Raises:
        :exc:`ContractError`: If no finite pixel centre lies near the point.
    """
    x0, y0 = (float(value) for value in xy)
    region = candidate_region(geometry, (x0, y0, x0, y0))
    x, y = block(geometry, region.across, region.along)
    distance = np.hypot(x - x0, y - y0)
    if not np.isfinite(distance).any():
        raise ContractError(f"No finite pixel centre lies near {xy}.")
    row, column = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
    return region.along.start + int(column), region.across.start + int(row)


@dataclass(frozen=True)
class ResolvedAnchor:
    """An anchor resolved on a scene, ready for regridding.

    Attributes:
        placement: Placement with the anchor pixel; for geographic anchors it
            carries the exact projected point in :attr:`Placement.anchor_xy`
            and the nearest pixel as anchor indices.
        mode: How the anchor was given: ``lonlat``, ``pixel`` or ``random``.
        anchor_xy: Projected point the LES anchor point is placed on.
        outside_fraction: Plume mass fraction outside the swath.
        attempts: Random draws needed; ``None`` for given anchors.
    """

    placement: Placement
    mode: AnchorMode
    anchor_xy: tuple[float, float]
    outside_fraction: float
    attempts: int | None


def resolve_anchor(
    column: xr.DataArray,
    geometry: SceneGeometry,
    *,
    mode: AnchorMode,
    swath: Outline | None = None,
    corner: AnchorPoint = "source",
    rotation_deg: int = 0,
    flip: Flip = "none",
    along: int | None = None,
    across: int | None = None,
    lonlat: tuple[float, float] | None = None,
    bounds: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    emission_scale: float = 1.0,
    time_index: int = 0,
) -> ResolvedAnchor:
    """Resolve where a plume column is anchored in a sensor swath.

    The LES anchor point (``corner``: the emission source or a corner cell
    centre) is pinned to a geographic position (``lonlat``), a pixel centre
    (``along``/``across``) or a random pixel centre. Random pixels are drawn
    uniformly until the plume mass fraction outside the swath lies within
    ``bounds``; the default requires a fully contained plume. Given anchors
    are accepted whatever part of the plume lies beyond the swath, unless it
    is all of it.

    Args:
        column: Non-negative ``(y, x)`` column on uniform coordinates with a
            ``units`` attribute.
        geometry: Scene geometry the column is placed in.
        mode: ``lonlat``, ``pixel`` or ``random``.
        swath: Swath outline; computed from ``geometry`` when omitted.
        corner: LES point pinned to the anchor.
        rotation_deg: Whole-degree rotation around the anchor.
        flip: LES axis reflected before the rotation.
        along: Anchor pixel along-track index (``pixel`` mode).
        across: Anchor pixel across-track index (``pixel`` mode).
        lonlat: Anchor longitude and latitude in degrees (``lonlat`` mode).
        bounds: Accepted outside mass fractions (``random`` mode).
        rng: Random generator (``random`` mode).
        max_attempts: Maximum number of random draws.
        emission_scale: Emission scale carried by the placement.
        time_index: LES time index carried by the placement.

    Returns:
        The resolved anchor with its placement and outside mass fraction.

    Raises:
        :exc:`ContractError`: If the inputs are invalid, the anchor leaves no plume
            mass inside the swath, or no random draw satisfies ``bounds``.
    """
    rotation_deg = validate_rotation(rotation_deg)
    flip = validate_flip(flip)
    base = Placement(
        anchor_corner=corner,
        rotation_deg=rotation_deg,
        flip=flip,
        emission_scale=emission_scale,
        time_index=time_index,
    )
    # A zero-offset placement validates the column and yields its anchor point
    # and support bounds before the anchor position is known.
    probe = place_column(
        column, crs=geometry.crs, anchor_target_xy=(0.0, 0.0), placement=base
    )
    anchor_source = probe.anchor_source_xy
    support_rel = support_vertices(
        probe.support_bounds, anchor_source, rotation_deg, flip
    )
    points = MassPoints.from_values(
        probe.values,
        column["x"].values,
        column["y"].values,
        anchor_source=anchor_source,
        rotation_deg=rotation_deg,
        flip=flip,
    )
    if swath is None:
        swath = swath_polygon(geometry)
    shapely.prepare(swath)
    anchor_xy: tuple[float, float] | None = None
    if mode == "pixel":
        if along is None or across is None:
            raise ContractError("A pixel anchor needs along and across indices.")
        draw = explicit_anchor(
            geometry, swath, points, support_rel, along=along, across=across
        )
    elif mode == "random":
        draw = random_anchor(
            geometry,
            swath,
            points,
            support_rel,
            bounds=bounds,
            rng=rng,
            max_attempts=max_attempts,
        )
    elif mode == "lonlat":
        if lonlat is None:
            raise ContractError("A geographic anchor needs lon and lat.")
        lon, lat = (float(value) for value in lonlat)
        x, y = from_lonlat(geometry.crs, lon, lat)
        anchor_xy = (float(x), float(y))
        if not all(np.isfinite(anchor_xy)) or not swath.covers(Point(anchor_xy)):
            raise ContractError(
                f"Anchor (lon={lon}, lat={lat}) lies outside the swath."
            )
        fraction = outside_mass_fraction(swath, support_rel, anchor_xy, points)
        if fraction >= 1.0:
            raise ContractError(
                f"Anchor (lon={lon}, lat={lat}) places the whole plume outside "
                "the swath."
            )
        pixel_along, pixel_across = locate_pixel(geometry, anchor_xy)
        draw = AnchorDraw(
            along=pixel_along,
            across=pixel_across,
            anchor_x=anchor_xy[0],
            anchor_y=anchor_xy[1],
            outside_fraction=fraction,
            attempts=None,
        )
    else:
        raise ContractError(f"Unknown anchor mode {mode!r}.")
    placement = Placement(
        anchor_along=draw.along,
        anchor_across=draw.across,
        anchor_corner=corner,
        rotation_deg=rotation_deg,
        flip=flip,
        emission_scale=emission_scale,
        time_index=time_index,
        anchor_xy=anchor_xy,
    )
    return ResolvedAnchor(
        placement=placement,
        mode=mode,
        anchor_xy=(draw.anchor_x, draw.anchor_y),
        outside_fraction=draw.outside_fraction,
        attempts=draw.attempts,
    )


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "AnchorDraw",
    "AnchorMode",
    "MassPoints",
    "ResolvedAnchor",
    "anchor_pixel_spacing",
    "block",
    "candidate_region",
    "coarse_strides",
    "explicit_anchor",
    "locate_pixel",
    "outside_fraction_bounds",
    "outside_mass_fraction",
    "pixel_centre",
    "random_anchor",
    "resolve_anchor",
    "swath_polygon",
    "window_nodes",
    "window_polygon",
]
