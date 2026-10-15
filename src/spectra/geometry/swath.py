"""Plume placement inside a sensor swath.

The swath outline, the anchor-pixel draws, the outside-mass test and the
index window covering the placed plume are ported from
``aviris4_ch4.regridding`` and generalised to any sensor through
:class:`~spectra.types.SceneGeometry`. Only boundary rows and columns of the
pixel-centre arrays are read for the swath outline, because real swaths hold
tens of millions of pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import shapely
import xarray as xr
from shapely.geometry import Polygon

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import (
    axis_nodes,
    outline_polygon,
    require_finite,
    window_corner_nodes,
)
from spectra.geometry.placement import PlacedColumn, place_column
from spectra.geometry.transform import (
    support_vertices,
    transform_coordinates,
    validate_flip,
    validate_rotation,
    vertices_bounds,
)
from spectra.types import AnchorCorner, Flip, Placement, SceneGeometry, Window

DEFAULT_MAX_ATTEMPTS = 10_000
_ACROSS_CHUNK = 8
PlacementMode = Literal["explicit", "random"]


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


def swath_polygon(geometry: SceneGeometry) -> Polygon:
    """Return the polygon following the outer pixel boundaries of a scene."""
    outline_x = _ring(*_boundary_nodes(geometry.x))
    outline_y = _ring(*_boundary_nodes(geometry.y))
    require_finite(outline_x, outline_y)
    polygon = Polygon(zip(outline_x, outline_y, strict=True))
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ContractError("Pixel boundary forms an invalid swath polygon.")
    return polygon


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


def window_polygon(geometry: SceneGeometry, window: Window) -> Polygon:
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


def outside_mass_fraction(
    region: Polygon,
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
    swath: Polygon,
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
    swath: Polygon,
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


def window_covering_support(
    geometry: SceneGeometry,
    support_vertices: npt.NDArray[np.floating[Any]],
    *,
    swath: Polygon,
    covered: bool,
    anchor_index: tuple[int, int],
) -> Window:
    """Return a padded index window covering the in-swath plume support.

    A support covered by the swath is bounded directly. A partially embedded
    support is clipped to the swath first, because the window cannot extend
    past the swath edge, and the window must then capture the whole in-swath
    area.

    Args:
        geometry: Scene geometry.
        support_vertices: ``(4, 2)`` absolute support rectangle vertices.
        swath: Swath outline polygon.
        covered: Whether the swath covers the support rectangle.
        anchor_index: ``(along, across)`` anchor pixel indices.

    Returns:
        The covering window.
    """
    n_across, n_along = geometry.shape
    dim0 = geometry.spatial_dims[0]
    support_polygon = Polygon(support_vertices)
    partial = not covered
    if partial:
        inside = swath.intersection(support_polygon)
        if inside.is_empty or inside.area <= 0:
            raise ContractError("Translated plume support lies outside the swath.")
        inside_area = float(inside.area)
        # Pad by one pixel so a sub-pixel sliver still contains a pixel centre.
        pad = anchor_pixel_spacing(geometry, anchor_index[0], anchor_index[1])
        xmin, ymin, xmax, ymax = inside.bounds
        xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad
    else:
        inside_area = float(support_polygon.area)
        xmin, ymin, xmax, ymax = vertices_bounds(support_vertices)
    across_min = n_across
    across_max = -1
    along_min = n_along
    along_max = -1
    for start in range(0, n_across, _ACROSS_CHUNK):
        stop = min(n_across, start + _ACROSS_CHUNK)
        rows = {dim0: slice(start, stop)}
        x = np.asarray(geometry.x.isel(rows).values, dtype=float)
        y = np.asarray(geometry.y.isel(rows).values, dtype=float)
        inside_box = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        row_index, column_index = np.nonzero(inside_box)
        if row_index.size:
            across_min = min(across_min, start + int(row_index.min()))
            across_max = max(across_max, start + int(row_index.max()))
            along_min = min(along_min, int(column_index.min()))
            along_max = max(along_max, int(column_index.max()))
    if across_max < 0:
        raise ContractError("Translated plume support contains no pixel centres.")

    window = Window(
        across=slice(max(0, across_min - 1), min(n_across, across_max + 2)),
        along=slice(max(0, along_min - 1), min(n_along, along_max + 2)),
    )
    while True:
        outline = window_polygon(geometry, window)
        if partial:
            # Intersection vertices lie on the shared swath/window edge, so an
            # area comparison is robust where ``covers`` is not.
            captured = outline.intersection(support_polygon).area
            if captured >= inside_area * (1.0 - 1e-9):
                break
        elif outline.covers(support_polygon):
            break
        expanded = Window(
            across=slice(
                max(0, window.across.start - 1),
                min(n_across, window.across.stop + 1),
            ),
            along=slice(
                max(0, window.along.start - 1),
                min(n_along, window.along.stop + 1),
            ),
        )
        if expanded == window:
            raise ContractError("Could not construct a window covering the plume.")
        window = expanded
    return window


@dataclass(frozen=True, eq=False)
class ScenePlacement:
    """A plume column placed in a sensor swath.

    Attributes:
        placed: Transformed column in the scene CRS.
        window: Index window covering the in-swath plume support.
        draw: Anchor pixel and outside mass fraction.
        placement: Placement with the resolved anchor indices.
        swath: Swath outline polygon in the scene CRS.
        mode: Whether the anchor was given explicitly or drawn.
        bounds: Accepted outside-fraction bounds of a random draw.
        column: The input column on its native ``(y, x)`` grid.
    """

    placed: PlacedColumn
    window: Window
    draw: AnchorDraw
    placement: Placement
    swath: Polygon
    mode: PlacementMode
    bounds: tuple[float, float]
    column: xr.DataArray

    def placement_attrs(self) -> dict[str, Any]:
        """Return attributes describing how the anchor pixel was chosen."""
        attrs: dict[str, Any] = {
            "embedding_placement": self.mode,
            "embedding_outside_mass_fraction": float(self.draw.outside_fraction),
        }
        if self.mode == "random":
            attrs["embedding_min_outside_fraction"] = float(self.bounds[0])
            attrs["embedding_max_outside_fraction"] = float(self.bounds[1])
            attrs["embedding_attempts"] = int(self.draw.attempts or 0)
        return attrs

    def attrs(self) -> dict[str, Any]:
        """Return the placement attributes and the ``embedding_*`` window bounds."""
        return {**self.placement_attrs(), **self.window.attrs("embedding_")}


def place_in_scene(
    column: xr.DataArray,
    geometry: SceneGeometry,
    *,
    placement: Placement | None = None,
    anchor_corner: AnchorCorner = "lower left",
    rotation_deg: int = 0,
    flip: Flip = "none",
    min_outside_fraction: float = 0.0,
    max_outside_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> ScenePlacement:
    """Place a plume column in a sensor swath.

    The selected LES grid-corner cell centre is anchored to a pixel centre.
    With ``placement`` the plume is embedded at its anchor pixel, whatever
    part of it lies beyond the swath. Without it, anchor pixels are drawn
    uniformly from the whole swath until the plume mass fraction outside the
    swath lies between ``min_outside_fraction`` and ``max_outside_fraction``;
    the defaults require a fully contained plume.

    Args:
        column: Non-negative ``(y, x)`` column on uniform coordinates with a
            ``units`` attribute.
        geometry: Scene geometry the column is placed in.
        placement: Explicit anchor indices, corner, rotation and reflection.
        anchor_corner: LES corner cell placed on the anchor pixel centre
            (random placement).
        rotation_deg: Whole-degree rotation around the anchor (random
            placement).
        flip: LES axis reflected before the rotation (random placement).
        min_outside_fraction: Smallest accepted outside mass fraction.
        max_outside_fraction: Largest accepted outside mass fraction.
        rng: Random generator for the draw.
        max_attempts: Maximum number of random draws.

    Returns:
        The placed column, its covering window and the draw.

    Raises:
        ContractError: If the column, bounds, indices or placement are
            invalid, if an explicit placement leaves no plume mass inside the
            swath, or if no draw satisfies the bounds.
    """
    if placement is not None:
        if (
            min_outside_fraction != 0.0
            or max_outside_fraction != 0.0
            or rng is not None
            or max_attempts != DEFAULT_MAX_ATTEMPTS
        ):
            raise ContractError(
                "min_outside_fraction, max_outside_fraction, rng and max_attempts "
                "only apply to random placement; omit placement to draw an "
                "anchor pixel."
            )
        anchor_corner = placement.anchor_corner
        rotation_deg = placement.rotation_deg
        flip = placement.flip
    bounds = outside_fraction_bounds(min_outside_fraction, max_outside_fraction)
    max_attempts = _validate_max_attempts(max_attempts)
    rotation_deg = validate_rotation(rotation_deg)
    flip = validate_flip(flip)
    base = Placement(
        anchor_corner=anchor_corner,
        rotation_deg=rotation_deg,
        flip=flip,
        emission_scale=placement.emission_scale if placement else 1.0,
        time_index=placement.time_index if placement else 0,
    )
    # A zero-offset placement validates the column and yields its anchor cell
    # centre and support bounds before the anchor pixel is known.
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
    swath = swath_polygon(geometry)
    shapely.prepare(swath)
    if placement is not None:
        draw = explicit_anchor(
            geometry,
            swath,
            points,
            support_rel,
            along=placement.anchor_along,
            across=placement.anchor_across,
        )
        mode: PlacementMode = "explicit"
    else:
        draw = random_anchor(
            geometry,
            swath,
            points,
            support_rel,
            bounds=bounds,
            rng=rng,
            max_attempts=max_attempts,
        )
        mode = "random"
    resolved = Placement(
        mode="anchor",
        anchor_along=draw.along,
        anchor_across=draw.across,
        anchor_corner=anchor_corner,
        rotation_deg=rotation_deg,
        flip=flip,
        emission_scale=base.emission_scale,
        time_index=base.time_index,
    )
    placed = place_column(
        column,
        crs=geometry.crs,
        anchor_target_xy=(draw.anchor_x, draw.anchor_y),
        placement=resolved,
    )
    absolute_support = placed.support_vertices
    window = window_covering_support(
        geometry,
        absolute_support,
        swath=swath,
        covered=bool(swath.covers(Polygon(absolute_support))),
        anchor_index=(draw.along, draw.across),
    )
    return ScenePlacement(
        placed=placed,
        window=window,
        draw=draw,
        placement=resolved,
        swath=swath,
        mode=mode,
        bounds=bounds,
        column=column,
    )


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "AnchorDraw",
    "MassPoints",
    "PlacementMode",
    "ScenePlacement",
    "anchor_pixel_spacing",
    "explicit_anchor",
    "outside_fraction_bounds",
    "outside_mass_fraction",
    "pixel_centre",
    "place_in_scene",
    "random_anchor",
    "swath_polygon",
    "window_covering_support",
    "window_nodes",
    "window_polygon",
]
