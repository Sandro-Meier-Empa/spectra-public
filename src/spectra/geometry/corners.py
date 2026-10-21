"""Pixel corner geometry derived from pixel-centre coordinates.

Grid nodes are the corners shared by neighbouring cells. Inside the grid a
node is the midpoint of the adjacent centres; at the outer edge it is
linearly extrapolated, exactly as the CDO target grids written by
``aviris4_ch4.regridding`` define them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import MultiPolygon, Polygon

from spectra.contracts.errors import ContractError
from spectra.types import Window

# Outline of a grid; a MultiPolygon when folded edge pixels make the boundary
# ring cross itself.
Outline = Polygon | MultiPolygon


def require_finite(*arrays: npt.ArrayLike) -> None:
    """Reject arrays containing missing or infinite values."""
    for array in arrays:
        if not np.all(np.isfinite(np.asarray(array, dtype=float))):
            raise ContractError("Geolocation contains missing or infinite values.")


def axis_nodes(centres: npt.ArrayLike, axis: int = -1) -> np.ndarray:
    """Return node coordinates along one axis of a centre array.

    Interior nodes are midpoints of neighbouring centres; the two outer nodes
    are extrapolated by half a cell.
    """
    values = np.moveaxis(np.asarray(centres, dtype=np.float64), axis, -1)
    if values.shape[-1] < 2:
        raise ContractError("At least two cell centres are needed to build nodes.")
    nodes = np.empty((*values.shape[:-1], values.shape[-1] + 1), dtype=np.float64)
    nodes[..., 1:-1] = 0.5 * (values[..., :-1] + values[..., 1:])
    nodes[..., 0] = 1.5 * values[..., 0] - 0.5 * values[..., 1]
    nodes[..., -1] = 1.5 * values[..., -1] - 0.5 * values[..., -2]
    return np.moveaxis(nodes, -1, axis)


def cell_corner_nodes(centres: npt.ArrayLike) -> np.ndarray:
    """Return the ``(n0 + 1, n1 + 1)`` grid nodes of a two-dimensional centre array."""
    values = np.asarray(centres, dtype=np.float64)
    if values.ndim != 2:
        raise ContractError("Cell centres must be two-dimensional.")
    return axis_nodes(axis_nodes(values, axis=1), axis=0)


def window_corner_nodes(
    centres: npt.ArrayLike, window: Window, *, start: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Return grid nodes of an index window using neighbouring centres.

    Nodes on the window boundary that lie inside the grid are midpoints with
    the centres just outside the window, so ``centres`` must contain at least
    one extra row and column around the window where available. ``start``
    gives the grid index of ``centres[0, 0]``.

    Args:
        centres: Pixel-centre coordinates covering the window and its halo.
        window: Index window in grid coordinates.
        start: Grid indices of the first centre row and column.

    Returns:
        ``(n_across + 1, n_along + 1)`` node coordinates of the window.
    """
    nodes = cell_corner_nodes(centres)
    row0 = window.across.start - start[0]
    col0 = window.along.start - start[1]
    rows = window.across.stop - window.across.start
    cols = window.along.stop - window.along.start
    if row0 < 0 or col0 < 0:
        raise ContractError("Centres do not cover the window.")
    if row0 + rows + 1 > nodes.shape[0] or col0 + cols + 1 > nodes.shape[1]:
        raise ContractError("Centres do not cover the window.")
    return nodes[row0 : row0 + rows + 1, col0 : col0 + cols + 1]


def nodes_to_cell_vertices(nodes: npt.ArrayLike) -> np.ndarray:
    """Convert grid nodes to ``(n0, n1, 4)`` perimeter-ordered cell vertices."""
    values = np.asarray(nodes, dtype=np.float64)
    return np.stack(
        (
            values[:-1, :-1],
            values[:-1, 1:],
            values[1:, 1:],
            values[1:, :-1],
        ),
        axis=-1,
    )


def cell_vertices_to_nodes(vertices: npt.ArrayLike) -> np.ndarray:
    """Rebuild the grid-node array from perimeter-ordered cell vertices."""
    corners = np.asarray(vertices, dtype=np.float64)
    rows, columns = corners.shape[:2]
    nodes = np.empty((rows + 1, columns + 1), dtype=np.float64)
    nodes[:-1, :-1] = corners[:, :, 0]
    nodes[:-1, -1] = corners[:, -1, 1]
    nodes[-1, :-1] = corners[-1, :, 3]
    nodes[-1, -1] = corners[-1, -1, 2]
    return nodes


def outline_from_nodes(nodes: npt.ArrayLike) -> np.ndarray:
    """Return node values along the outer grid boundary in ring order."""
    values = np.asarray(nodes, dtype=np.float64)
    top = values[0, :]
    right = values[:, -1]
    bottom = values[-1, :]
    left = values[:, 0]
    return np.concatenate((top, right[1:], bottom[-2::-1], left[-2:0:-1]))


def ring_outline(outline_x: npt.ArrayLike, outline_y: npt.ArrayLike) -> Outline:
    """Return the region enclosed by a closed boundary ring of grid nodes.

    Georeferenced swaths can fold at their edges (pixels displaced backwards
    along track by aircraft roll), so the boundary ring may cross itself; the
    enclosed region is then repaired into its polygonal parts instead of
    rejected.

    Raises:
        :exc:`ContractError`: If the ring is not finite or encloses no area.
    """
    require_finite(outline_x, outline_y)
    polygon = Polygon(zip(np.asarray(outline_x), np.asarray(outline_y), strict=True))
    if not polygon.is_valid:
        repaired = shapely.make_valid(polygon)
        parts = [
            part
            for part in getattr(repaired, "geoms", [repaired])
            if isinstance(part, (Polygon, MultiPolygon))
        ]
        merged = shapely.union_all(parts) if parts else Polygon()
        polygon = merged if isinstance(merged, (Polygon, MultiPolygon)) else Polygon()
    if polygon.is_empty or polygon.area <= 0:
        raise ContractError("Grid nodes enclose no area.")
    return polygon


def outline_polygon(nodes_x: npt.ArrayLike, nodes_y: npt.ArrayLike) -> Outline:
    """Return the region enclosed by the outer boundary of a grid's nodes."""
    return ring_outline(outline_from_nodes(nodes_x), outline_from_nodes(nodes_y))


def cell_areas(vertices_x: npt.ArrayLike, vertices_y: npt.ArrayLike) -> np.ndarray:
    """Return planar cell areas from ``(..., 4)`` vertex arrays (shoelace formula)."""
    x = np.asarray(vertices_x, dtype=np.float64)
    y = np.asarray(vertices_y, dtype=np.float64)
    x = x - x[..., :1]
    y = y - y[..., :1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1), axis=-1)
    )


def cell_polygons(vertices_x: npt.ArrayLike, vertices_y: npt.ArrayLike) -> np.ndarray:
    """Return a flat object array of shapely polygons for ``(..., 4)`` vertices."""
    x = np.asarray(vertices_x, dtype=np.float64)
    y = np.asarray(vertices_y, dtype=np.float64)
    coordinates = np.stack((x, y), axis=-1).reshape(-1, 4, 2)
    return np.asarray(shapely.polygons(coordinates))


__all__ = [
    "Outline",
    "axis_nodes",
    "cell_areas",
    "cell_corner_nodes",
    "cell_polygons",
    "cell_vertices_to_nodes",
    "nodes_to_cell_vertices",
    "outline_from_nodes",
    "outline_polygon",
    "require_finite",
    "ring_outline",
    "window_corner_nodes",
]
