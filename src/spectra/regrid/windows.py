"""Index windows of sensor and map grids covering a placed plume column.

The sensor-grid helpers only ever load small blocks of the (possibly lazy)
pixel-centre geometry: the anchor neighbourhood, the four border strips that
form the swath outline, a coarse subsample used to locate the candidate
region, and finally the candidate block itself. Real swaths are of the order
of ``1241 x 25000`` pixels and must never be loaded as a whole.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import (
    axis_nodes,
    cell_areas,
    nodes_to_cell_vertices,
    outline_polygon,
    require_finite,
)
from spectra.geometry.placement import PlacedColumn
from spectra.regrid.base import target_window_nodes
from spectra.types import MapGrid, Placement, SceneGeometry, Window

# Number of pixel centres loaded by the coarse pass of ``support_window``.
_COARSE_SAMPLES = 20_000
# Relative tolerance of the "window captures the support" area test. The
# intersection vertices lie on shared edges, so ``covers`` is not robust.
_COVER_RTOL = 1e-9


def _block(
    geometry: SceneGeometry, across: slice, along: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Load a block of pixel-centre coordinates as float64 arrays."""
    dim_across, dim_along = geometry.spatial_dims
    indexer = {dim_across: across, dim_along: along}
    x = np.asarray(geometry.x.isel(indexer).values, dtype=np.float64)
    y = np.asarray(geometry.y.isel(indexer).values, dtype=np.float64)
    return x, y


def _require_pixel_pairs(geometry: SceneGeometry) -> tuple[int, int]:
    n_across, n_along = geometry.shape
    if n_across < 2 or n_along < 2:
        raise ContractError(
            "Scene geometry needs at least two pixels along each dimension."
        )
    return n_across, n_along


def anchor_pixel_centre(
    geometry: SceneGeometry, placement: Placement
) -> tuple[float, float]:
    """Return the projected centre of the anchor pixel of a placement.

    Raises:
        ContractError: If the anchor index lies outside the scene or the pixel
            centre is not finite.
    """
    n_across, n_along = geometry.shape
    across = int(placement.anchor_across)
    along = int(placement.anchor_along)
    if across >= n_across or along >= n_along:
        raise ContractError(
            f"Anchor pixel (across={across}, along={along}) lies outside the "
            f"scene of shape {geometry.shape}."
        )
    x, y = _block(geometry, slice(across, across + 1), slice(along, along + 1))
    centre = (float(x[0, 0]), float(y[0, 0]))
    if not all(math.isfinite(value) for value in centre):
        raise ContractError(
            f"Anchor pixel (across={across}, along={along}) has no finite centre."
        )
    return centre


def anchor_pixel_spacing(geometry: SceneGeometry, across: int, along: int) -> float:
    """Return the largest centre spacing in the 2x2 neighbourhood of a pixel.

    Mirrors ``aviris4_ch4.regridding._anchor_pixel_spacing``.
    """
    n_across, n_along = _require_pixel_pairs(geometry)
    if not 0 <= across < n_across or not 0 <= along < n_along:
        raise ContractError("Pixel index lies outside the scene.")
    across0 = across if across + 1 < n_across else across - 1
    along0 = along if along + 1 < n_along else along - 1
    x, y = _block(geometry, slice(across0, across0 + 2), slice(along0, along0 + 2))
    spacing = max(
        float(np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).max()),
        float(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)).max()),
    )
    if not math.isfinite(spacing) or spacing <= 0:
        raise ContractError("Pixel spacing at the anchor is not positive.")
    return spacing


def swath_outline(geometry: SceneGeometry) -> Polygon:
    """Return the outer pixel-boundary polygon of the whole scene.

    Only the two outermost rows and columns of pixel centres are loaded; the
    edge nodes are extrapolated from them exactly as
    :func:`spectra.geometry.corners.cell_corner_nodes` does for the full grid.
    """
    n_across, n_along = _require_pixel_pairs(geometry)
    rings: list[np.ndarray] = []
    for axis_index in (0, 1):
        top_strip = _block(geometry, slice(0, 2), slice(0, n_along))[axis_index]
        bottom_strip = _block(
            geometry, slice(n_across - 2, n_across), slice(0, n_along)
        )[axis_index]
        left_strip = _block(geometry, slice(0, n_across), slice(0, 2))[axis_index]
        right_strip = _block(geometry, slice(0, n_across), slice(n_along - 2, n_along))[
            axis_index
        ]
        top = axis_nodes(axis_nodes(top_strip, axis=1), axis=0)[0]
        bottom = axis_nodes(axis_nodes(bottom_strip, axis=1), axis=0)[-1]
        left = axis_nodes(axis_nodes(left_strip, axis=0), axis=1)[:, 0]
        right = axis_nodes(axis_nodes(right_strip, axis=0), axis=1)[:, -1]
        rings.append(np.concatenate((top, right[1:], bottom[-2::-1], left[-2:0:-1])))
    require_finite(*rings)
    polygon = Polygon(zip(rings[0], rings[1], strict=True))
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ContractError("Scene pixel boundaries form an invalid swath polygon.")
    return polygon


def window_outline(geometry: SceneGeometry, window: Window) -> Polygon:
    """Return the outer pixel-boundary polygon of an index window."""
    node_x, node_y = target_window_nodes(geometry, window)
    return outline_polygon(node_x, node_y)


def _coarse_strides(n_across: int, n_along: int) -> tuple[int, int]:
    stride = max(1, math.ceil(math.sqrt(n_across * n_along / _COARSE_SAMPLES)))
    return min(stride, max(1, n_across // 2)), min(stride, max(1, n_along // 2))


def _candidate_region(
    geometry: SceneGeometry, box: tuple[float, float, float, float]
) -> Window:
    """Return an index window that contains every pixel centre inside ``box``.

    A strided subsample of the centres is tested against the box expanded by
    the largest distance between a pixel and its coarse representative, so no
    fine pixel inside the box can be missed; the fine test runs on the
    returned block only.
    """
    n_across, n_along = geometry.shape
    stride_across, stride_along = _coarse_strides(n_across, n_along)
    x, y = _block(
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


def _grow(window: Window, shape: tuple[int, int], amount: int) -> Window:
    return Window(
        across=slice(
            max(0, window.across.start - amount),
            min(shape[0], window.across.stop + amount),
        ),
        along=slice(
            max(0, window.along.start - amount),
            min(shape[1], window.along.stop + amount),
        ),
    )


def support_window(
    placed: PlacedColumn, geometry: SceneGeometry, *, margin_px: int = 1
) -> Window:
    """Return the smallest padded index window covering the placed support.

    Pixels whose centres fall inside the bounding box of the support (clipped
    to the swath) expanded by one pixel spacing are selected, padded by
    ``margin_px`` and clipped to the scene. The window is then grown until
    its pixel boundaries capture the in-swath support, as in ``aviris4_ch4``.

    Args:
        placed: Placed plume column in the CRS of ``geometry``.
        geometry: Pixel-centre geometry of the scene.
        margin_px: Padding in pixels around the selected pixels.

    Returns:
        Index window on the sensor grid.

    Raises:
        ContractError: If the support does not overlap the scene or no pixel
            centre qualifies.
    """
    if margin_px < 0:
        raise ContractError("margin_px must be non-negative.")
    n_across, n_along = _require_pixel_pairs(geometry)
    swath = swath_outline(geometry)
    support = placed.support_polygon
    region = support if swath.covers(support) else swath.intersection(support)
    if region.is_empty or region.area <= 0:
        raise ContractError("The placed support does not overlap the scene.")
    spacing = anchor_pixel_spacing(
        geometry, placed.placement.anchor_across, placed.placement.anchor_along
    )
    xmin, ymin, xmax, ymax = region.bounds
    box = (xmin - spacing, ymin - spacing, xmax + spacing, ymax + spacing)
    candidate = _candidate_region(geometry, box)
    x, y = _block(geometry, candidate.across, candidate.along)
    inside = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
    rows, columns = np.nonzero(inside)
    if rows.size == 0:
        raise ContractError("No pixel centre lies inside the placed support.")
    window = Window(
        across=slice(
            max(0, candidate.across.start + int(rows.min()) - margin_px),
            min(n_across, candidate.across.start + int(rows.max()) + 1 + margin_px),
        ),
        along=slice(
            max(0, candidate.along.start + int(columns.min()) - margin_px),
            min(n_along, candidate.along.start + int(columns.max()) + 1 + margin_px),
        ),
    )
    target_area = region.area * (1.0 - _COVER_RTOL)
    while window_outline(geometry, window).intersection(region).area < target_area:
        grown = _grow(window, (n_across, n_along), 1)
        if grown == window:
            break
        window = grown
    return window


def _axis_window(
    centres: np.ndarray, low: float, high: float, margin: int
) -> slice | None:
    """Return the index slice of cells overlapping ``[low, high]`` on one axis."""
    nodes = axis_nodes(centres)
    size = centres.size
    ascending = nodes[-1] > nodes[0]
    ordered = nodes if ascending else nodes[::-1]
    first = int(np.searchsorted(ordered, low, side="right")) - 1
    last = int(np.searchsorted(ordered, high, side="left")) - 1
    first = max(first, 0)
    last = min(last, size - 1)
    if first > last:
        return None
    if not ascending:
        first, last = size - 1 - last, size - 1 - first
    return slice(max(0, first - margin), min(size, last + 1 + margin))


def map_window(
    placed: PlacedColumn, grid: MapGrid, margin: int = 1
) -> tuple[slice, slice]:
    """Return the ``(y, x)`` index slices of map cells covering the support.

    Cells intersecting the bounding box of the placed support are selected,
    padded by ``margin`` cells and clipped to the grid.

    Raises:
        ContractError: If the support does not overlap the grid.
    """
    if margin < 0:
        raise ContractError("margin must be non-negative.")
    xmin, ymin, xmax, ymax = placed.support_polygon.bounds
    x_slice = _axis_window(grid.x, xmin, xmax, margin)
    y_slice = _axis_window(grid.y, ymin, ymax, margin)
    if x_slice is None or y_slice is None:
        raise ContractError("The placed support does not overlap the map grid.")
    return y_slice, x_slice


def map_anchor_centre(grid: MapGrid, placement: Placement) -> tuple[float, float]:
    """Return the centre of the anchor cell ``(y[anchor_across], x[anchor_along])``."""
    along = int(placement.anchor_along)
    across = int(placement.anchor_across)
    if along >= grid.x.size or across >= grid.y.size:
        raise ContractError(
            f"Anchor cell (across={across}, along={along}) lies outside the map "
            f"grid of shape {(grid.y.size, grid.x.size)}."
        )
    return float(grid.x[along]), float(grid.y[across])


def map_window_nodes(
    grid: MapGrid, y_slice: slice, x_slice: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ny + 1, nx + 1)`` node coordinates of a map window."""
    x_nodes = axis_nodes(grid.x)[x_slice.start : x_slice.stop + 1]
    y_nodes = axis_nodes(grid.y)[y_slice.start : y_slice.stop + 1]
    node_x, node_y = np.meshgrid(x_nodes, y_nodes, indexing="xy")
    return node_x, node_y


def map_window_cells(
    grid: MapGrid, y_slice: slice, x_slice: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vertices_x, vertices_y, areas)`` of the cells in a map window."""
    node_x, node_y = map_window_nodes(grid, y_slice, x_slice)
    vx = nodes_to_cell_vertices(node_x)
    vy = nodes_to_cell_vertices(node_y)
    return vx, vy, cell_areas(vx, vy)


__all__ = [
    "anchor_pixel_centre",
    "anchor_pixel_spacing",
    "map_anchor_centre",
    "map_window",
    "map_window_cells",
    "map_window_nodes",
    "support_window",
    "swath_outline",
    "window_outline",
]
