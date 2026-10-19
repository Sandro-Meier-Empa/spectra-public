"""Supersampled nearest-centre ("bucket") conservative regridding.

Every nonzero LES cell is split into ``k x k`` sub-cells whose centres are
mapped through the placement transform; each sub-cell drops its mass
``v_s A_s / k**2`` into the target pixel with the nearest centre. The target
field is the accumulated mass divided by the pixel area, so mass is conserved
exactly up to the classification of sub-cells as inside or outside the window
outline. The method is approximate at pixel edges (resolution ``dx / k``) but
needs no polygon intersections.
"""

from __future__ import annotations

import numpy as np
import shapely
import xarray as xr
from scipy.spatial import KDTree

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import (
    cell_areas,
    nodes_to_cell_vertices,
    outline_polygon,
)
from spectra.geometry.placement import PlacedColumn, place_column
from spectra.geometry.swath import window_nodes
from spectra.geometry.transform import transform_coordinates
from spectra.regrid.base import RegridResult, check_mass
from spectra.regrid.results import finalize_result, scene_data_array
from spectra.regrid.windows import anchor_pixel_centre, support_window
from spectra.types import Placement, SceneGeometry, Window


def subsample_points(
    placed: PlacedColumn, supersample: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sub-cell centres and masses of the nonzero cells in the target CRS.

    The placement is affine, so sub-cell offsets are rotated and reflected
    with the linear part of the transform and added to the transformed cell
    centres; this reproduces the exact transform of the sub-cell centres.

    Args:
        placed: Placed column.
        supersample: Number of sub-cells per cell edge ``k``.

    Returns:
        ``(x, y, mass)`` flat arrays of the ``k * k`` sub-cells of every
        nonzero cell, with masses ``v_s A_s / k**2`` from the placed areas.
    """
    if isinstance(supersample, bool) or int(supersample) != supersample:
        raise ContractError("supersample must be an integer.")
    k = int(supersample)
    if k < 1:
        raise ContractError("supersample must be at least 1.")
    offsets = (np.arange(k, dtype=np.float64) + 0.5) / k - 0.5
    du, dv = np.meshgrid(placed.dx * offsets, placed.dy * offsets, indexing="xy")
    lx, ly = transform_coordinates(
        du,
        dv,
        pivot=(0.0, 0.0),
        rotation_deg=placed.placement.rotation_deg,
        flip=placed.placement.flip,
    )
    nonzero = placed.values != 0
    areas = placed.cell_areas()[nonzero]
    centre_x = placed.centre_x[nonzero]
    centre_y = placed.centre_y[nonzero]
    mass = placed.values[nonzero] * areas / float(k * k)
    x = (centre_x[:, None] + lx.reshape(1, -1)).reshape(-1)
    y = (centre_y[:, None] + ly.reshape(1, -1)).reshape(-1)
    return x, y, np.repeat(mass, k * k)


def regrid_buckets(
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_mass: np.ndarray,
    target_centre_x: np.ndarray,
    target_centre_y: np.ndarray,
    target_node_x: np.ndarray,
    target_node_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Drop point masses into the target cells with the nearest centres.

    Args:
        point_x: Point x coordinates.
        point_y: Point y coordinates.
        point_mass: Mass carried by every point.
        target_centre_x: ``(nt0, nt1)`` target cell-centre x.
        target_centre_y: ``(nt0, nt1)`` target cell-centre y.
        target_node_x: ``(nt0 + 1, nt1 + 1)`` target node x.
        target_node_y: ``(nt0 + 1, nt1 + 1)`` target node y.

    Returns:
        ``(values, target_areas, mass_inside, target_mass)`` where
        ``mass_inside`` is the mass of the points inside the target outline.
    """
    centre_x = np.asarray(target_centre_x, dtype=np.float64)
    centre_y = np.asarray(target_centre_y, dtype=np.float64)
    node_x = np.asarray(target_node_x, dtype=np.float64)
    node_y = np.asarray(target_node_y, dtype=np.float64)
    if centre_x.ndim != 2 or centre_x.shape != centre_y.shape:
        raise ContractError("target centres must be (nt0, nt1) arrays.")
    if node_x.shape != (centre_x.shape[0] + 1, centre_x.shape[1] + 1) or (
        node_x.shape != node_y.shape
    ):
        raise ContractError("target nodes must be (nt0 + 1, nt1 + 1) arrays.")
    vx = nodes_to_cell_vertices(node_x)
    vy = nodes_to_cell_vertices(node_y)
    areas = cell_areas(vx, vy)
    outline = outline_polygon(node_x, node_y)
    x = np.asarray(point_x, dtype=np.float64).reshape(-1)
    y = np.asarray(point_y, dtype=np.float64).reshape(-1)
    mass = np.asarray(point_mass, dtype=np.float64).reshape(-1)
    inside = np.asarray(shapely.contains_xy(outline, x, y), dtype=bool)
    accumulated = np.zeros(centre_x.size, dtype=np.float64)
    if inside.any():
        tree = KDTree(np.column_stack((centre_x.reshape(-1), centre_y.reshape(-1))))
        _, index = tree.query(np.column_stack((x[inside], y[inside])), k=1)
        accumulated = np.bincount(
            np.asarray(index, dtype=np.intp),
            weights=mass[inside],
            minlength=centre_x.size,
        )
    accumulated = accumulated.reshape(centre_x.shape)
    positive = areas > 0
    values = np.where(positive, accumulated / np.where(positive, areas, 1.0), 0.0)
    mass_inside = float(np.sum(mass[inside]))
    target_mass = float(np.sum(values * areas))
    return values, areas, mass_inside, target_mass


class BucketRegridder:
    """Approximate conservative regridder by supersampled nearest-centre binning.

    Args:
        supersample: Sub-cells per LES cell edge; the edge resolution is
            ``dx / supersample``.
        rtol: Maximum relative mass error accepted.
    """

    name: str = "bucket"

    def __init__(self, supersample: int = 4, rtol: float = 1e-4) -> None:
        if isinstance(supersample, bool) or int(supersample) != supersample:
            raise ContractError("supersample must be an integer.")
        if supersample < 1:
            raise ContractError("supersample must be at least 1.")
        if not np.isfinite(rtol) or rtol < 0:
            raise ContractError("rtol must be a finite nonnegative number.")
        self.supersample = int(supersample)
        self.rtol = float(rtol)

    def __repr__(self) -> str:
        """Return the constructor call."""
        return (
            f"{type(self).__name__}(supersample={self.supersample!r}, "
            f"rtol={self.rtol!r})"
        )

    def _regrid(
        self,
        placed: PlacedColumn,
        centre_x: np.ndarray,
        centre_y: np.ndarray,
        node_x: np.ndarray,
        node_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        x, y, mass = subsample_points(placed, self.supersample)
        return regrid_buckets(x, y, mass, centre_x, centre_y, node_x, node_y)

    def _finalize(
        self,
        data: xr.DataArray,
        placed: PlacedColumn,
        target_area: np.ndarray,
        source_mass_inside: float,
        target_mass: float,
    ) -> RegridResult:
        source_area = placed.cell_areas()
        result = finalize_result(
            data,
            method=self.name,
            source_area=source_area,
            target_area=target_area,
            source_mass=float(np.sum(placed.values * source_area)),
            source_mass_inside=source_mass_inside,
            target_mass=target_mass,
            attrs={"regrid_supersample": self.supersample},
        )
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
        dim_across, dim_along = geometry.spatial_dims
        indexer = {dim_across: window.across, dim_along: window.along}
        centre_x = np.asarray(geometry.x.isel(indexer).values, dtype=np.float64)
        centre_y = np.asarray(geometry.y.isel(indexer).values, dtype=np.float64)
        node_x, node_y = window_nodes(geometry, window)
        values, target_area, inside, target_mass = self._regrid(
            placed, centre_x, centre_y, node_x, node_y
        )
        data = scene_data_array(
            values, scene=scene, window=window, placed=placed, name=column.name
        )
        return self._finalize(data, placed, target_area, inside, target_mass)

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


__all__ = ["BucketRegridder", "regrid_buckets", "subsample_points"]
