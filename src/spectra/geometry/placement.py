"""Pure placement of a plume snapshot in a projected coordinate system.

A placement translates the selected corner cell of the LES grid onto a target
point (usually a pixel centre), reflects the grid around that point and then
rotates it, exactly as ``aviris4_ch4.regridding`` does. The result keeps the
transformed cell nodes, so conservative regridders can intersect LES cells
with sensor pixels without any further assumption about the sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyproj
import shapely
import xarray as xr
from shapely.geometry import Polygon

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import (
    axis_nodes,
    cell_areas,
    cell_polygons,
    nodes_to_cell_vertices,
    outline_polygon,
)
from spectra.geometry.transform import (
    anchor_cell_centre,
    support_vertices,
    transform_coordinates,
)
from spectra.types import Placement

_SPACING_RTOL = 1e-9


@dataclass(frozen=True, eq=False)
class PlacedColumn:
    """One plume snapshot placed in a projected CRS.

    Attributes:
        values: ``(ny, nx)`` column values, gaps zeroed, in ``units``.
        units: Units of ``values``.
        node_x: ``(ny + 1, nx + 1)`` transformed cell-node x coordinates.
        node_y: ``(ny + 1, nx + 1)`` transformed cell-node y coordinates.
        centre_x: ``(ny, nx)`` transformed cell-centre x coordinates.
        centre_y: ``(ny, nx)`` transformed cell-centre y coordinates.
        crs: Projected CRS of the transformed coordinates.
        placement: Placement parameters that produced the transform.
        anchor_source_xy: LES corner cell centre in the model frame.
        anchor_target_xy: Target point the corner cell centre was placed on.
        dx: LES cell size along x in metres.
        dy: LES cell size along y in metres.
        support_bounds: Cell-edge bounds of the nonzero cells, model frame.
        attrs: Attributes of the input column, carried for provenance.
    """

    values: np.ndarray
    units: str
    node_x: np.ndarray
    node_y: np.ndarray
    centre_x: np.ndarray
    centre_y: np.ndarray
    crs: pyproj.CRS
    placement: Placement
    anchor_source_xy: tuple[float, float]
    anchor_target_xy: tuple[float, float]
    dx: float
    dy: float
    support_bounds: tuple[float, float, float, float]
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(ny, nx)``."""
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    @property
    def offset_xy(self) -> tuple[float, float]:
        """Return the translation from the model frame to the target CRS."""
        return (
            self.anchor_target_xy[0] - self.anchor_source_xy[0],
            self.anchor_target_xy[1] - self.anchor_source_xy[1],
        )

    @property
    def cell_area(self) -> float:
        """Return the planar area of one LES cell in m2."""
        return abs(self.dx * self.dy)

    @property
    def total_mass(self) -> float:
        """Return the integral of ``values`` over the grid."""
        return float(np.sum(self.values) * self.cell_area)

    @property
    def support_vertices(self) -> np.ndarray:
        """Return the transformed support rectangle, ``(4, 2)`` absolute."""
        relative = support_vertices(
            self.support_bounds,
            self.anchor_source_xy,
            self.placement.rotation_deg,
            self.placement.flip,
        )
        return relative + np.asarray(self.anchor_target_xy)

    @property
    def support_polygon(self) -> Polygon:
        """Return the transformed support rectangle as a polygon."""
        return Polygon(self.support_vertices)

    def cell_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ny, nx, 4)`` perimeter-ordered cell vertex coordinates."""
        return nodes_to_cell_vertices(self.node_x), nodes_to_cell_vertices(self.node_y)

    def cell_polygons(self) -> np.ndarray:
        """Return a flat, row-major object array of cell polygons."""
        vx, vy = self.cell_vertices()
        return cell_polygons(vx, vy)

    def cell_areas(self) -> np.ndarray:
        """Return ``(ny, nx)`` planar cell areas from the transformed nodes."""
        vx, vy = self.cell_vertices()
        return cell_areas(vx, vy)

    def outline(self) -> Polygon:
        """Return the outer boundary of the placed grid."""
        return outline_polygon(self.node_x, self.node_y)

    def mass_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return centre coordinates and weights of the nonzero cells."""
        nonzero = self.values > 0
        return (
            self.centre_x[nonzero],
            self.centre_y[nonzero],
            self.values[nonzero] * self.cell_area,
        )

    def outside_fraction(self, region: Polygon) -> float:
        """Return the mass fraction of nonzero cells centred outside ``region``.

        A support rectangle covered by the region contains the whole plume and
        one disjoint from it none, so only straddling rectangles need the
        cell-centre test. Cells count by their centre, so the fraction is
        resolved at the LES cell size, as in ``aviris4_ch4``.
        """
        support = self.support_polygon
        if region.covers(support):
            return 0.0
        if not region.intersects(support):
            return 1.0
        x, y, weights = self.mass_points()
        total = float(weights.sum())
        if total <= 0:
            return 1.0
        inside = np.asarray(shapely.contains_xy(region, x, y), dtype=bool)
        return float(min(1.0, max(0.0, 1.0 - float(weights[inside].sum()) / total)))


def _validate_axis(coordinate: xr.DataArray, name: str) -> tuple[np.ndarray, float]:
    if coordinate.ndim != 1 or coordinate.size < 1:
        raise ContractError(f"Column '{name}' must be a one-dimensional coordinate.")
    values = np.asarray(coordinate.values, dtype=np.float64)
    if values.size == 1:
        raise ContractError(f"Column '{name}' needs at least two cells.")
    steps = np.diff(values)
    if not np.all(np.isfinite(values)) or np.any(steps <= 0):
        raise ContractError(f"Column '{name}' must be finite and increasing.")
    if not np.allclose(steps, steps[0], rtol=_SPACING_RTOL, atol=1e-9):
        raise ContractError(f"Column '{name}' must be uniformly spaced.")
    return values, float(steps[0])


def column_support_bounds(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float
) -> tuple[float, float, float, float]:
    """Return cell-edge bounds of the nonzero cells of a ``(ny, nx)`` field."""
    indices = np.argwhere(values != 0)
    if indices.size == 0:
        raise ContractError("Column has empty nonzero support.")
    return (
        float(x[indices[:, 1].min()] - dx / 2),
        float(y[indices[:, 0].min()] - dy / 2),
        float(x[indices[:, 1].max()] + dx / 2),
        float(y[indices[:, 0].max()] + dy / 2),
    )


def place_column(
    column: xr.DataArray,
    *,
    crs: pyproj.CRS,
    anchor_target_xy: tuple[float, float],
    placement: Placement,
) -> PlacedColumn:
    """Place a ``(y, x)`` plume snapshot in a projected CRS.

    The selected corner cell centre of the LES grid is translated onto
    ``anchor_target_xy``; the grid is then reflected around that point and
    rotated around it, in this order.

    Args:
        column: Non-negative column on uniform ``x``/``y`` cell-centre
            coordinates in metres, with a ``units`` attribute.
        crs: Projected CRS of ``anchor_target_xy``.
        anchor_target_xy: Point the corner cell centre is placed on.
        placement: Corner, reflection and rotation of the placement.

    Returns:
        The placed snapshot with transformed nodes and centres.

    Raises:
        ContractError: If the column grid, units or values are invalid.
    """
    if set(column.dims) != {"x", "y"}:
        raise ContractError(
            f"Column must have dimensions ('y', 'x'); got {column.dims}."
        )
    units = column.attrs.get("units")
    if units is None:
        raise ContractError("Column units are unknown; set the 'units' attribute.")
    x, dx = _validate_axis(column["x"], "x")
    y, dy = _validate_axis(column["y"], "y")
    values = np.asarray(column.transpose("y", "x").values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    if np.any(values < 0):
        raise ContractError("Column must not contain negative values.")
    support = column_support_bounds(values, x, y, dx, dy)
    anchor_source = anchor_cell_centre(x, y, placement.anchor_corner)
    offset = (
        anchor_target_xy[0] - anchor_source[0],
        anchor_target_xy[1] - anchor_source[1],
    )
    node_x_1d = axis_nodes(x) + offset[0]
    node_y_1d = axis_nodes(y) + offset[1]
    grid_node_x, grid_node_y = np.meshgrid(node_x_1d, node_y_1d, indexing="xy")
    grid_centre_x, grid_centre_y = np.meshgrid(
        x + offset[0], y + offset[1], indexing="xy"
    )
    node_x, node_y = transform_coordinates(
        grid_node_x,
        grid_node_y,
        pivot=anchor_target_xy,
        rotation_deg=placement.rotation_deg,
        flip=placement.flip,
    )
    centre_x, centre_y = transform_coordinates(
        grid_centre_x,
        grid_centre_y,
        pivot=anchor_target_xy,
        rotation_deg=placement.rotation_deg,
        flip=placement.flip,
    )
    return PlacedColumn(
        values=values,
        units=str(units),
        node_x=node_x,
        node_y=node_y,
        centre_x=centre_x,
        centre_y=centre_y,
        crs=crs,
        placement=placement,
        anchor_source_xy=anchor_source,
        anchor_target_xy=(float(anchor_target_xy[0]), float(anchor_target_xy[1])),
        dx=dx,
        dy=dy,
        support_bounds=support,
        attrs=dict(column.attrs),
    )


__all__ = ["PlacedColumn", "column_support_bounds", "place_column"]
