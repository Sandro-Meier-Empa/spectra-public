"""Shared regridding results and planar mass checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import cell_areas, nodes_to_cell_vertices
from spectra.geometry.swath import window_nodes
from spectra.types import SceneGeometry, Window


@dataclass(frozen=True, eq=False)
class RegridResult:
    """Output of a conservative regridder.

    Attributes:
        data: Regridded column on the target grid in the source units.
        source_area: Planar area of every source cell in m2.
        target_area: Planar area of every target cell in m2.
        method: Regridder name recorded in provenance.
        source_mass: Integral of the source field over the source cells.
        source_mass_inside: Source mass whose cells overlap the target window.
        target_mass: Integral of the regridded field over the target cells.
        relative_error: ``|target_mass - source_mass_inside| / source_mass_inside``.
        attrs: Additional provenance written by the regridder.
    """

    data: xr.DataArray
    source_area: np.ndarray
    target_area: np.ndarray
    method: str
    source_mass: float
    source_mass_inside: float
    target_mass: float
    relative_error: float
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def inside_mass_fraction(self) -> float:
        """Return the fraction of the source mass inside the target window."""
        return self.source_mass_inside / self.source_mass if self.source_mass else 0.0

    def provenance(self) -> dict[str, Any]:
        """Return attributes describing the regridding for a product."""
        return {
            "regrid_method": self.method,
            "regrid_source_mass": float(self.source_mass),
            "regrid_source_mass_inside": float(self.source_mass_inside),
            "regrid_target_mass": float(self.target_mass),
            "regrid_inside_mass_fraction": float(self.inside_mass_fraction),
            "regrid_max_relative_mass_error": float(self.relative_error),
            **self.attrs,
        }


def relative_mass_error(source_mass_inside: float, target_mass: float) -> float:
    """Return the relative mass error with a zero-safe denominator."""
    denominator = abs(source_mass_inside)
    difference = abs(target_mass - source_mass_inside)
    return float(difference / denominator) if denominator > 0 else float(difference)


def check_mass(result: RegridResult, rtol: float) -> None:
    """Raise if the regridding error exceeds ``rtol``."""
    if not np.isfinite(rtol) or rtol < 0:
        raise ContractError("rtol must be a finite nonnegative number.")
    if result.relative_error > rtol:
        raise ContractError(
            f"Conservative regridding error {result.relative_error:.3g} exceeds "
            f"rtol={rtol:.3g} ({result.method})."
        )


def target_window_cells(
    geometry: SceneGeometry, window: Window
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vertices_x, vertices_y, areas)`` of the pixels in a window."""
    node_x, node_y = window_nodes(geometry, window)
    vx = nodes_to_cell_vertices(node_x)
    vy = nodes_to_cell_vertices(node_y)
    return vx, vy, cell_areas(vx, vy)


__all__ = [
    "RegridResult",
    "check_mass",
    "relative_mass_error",
    "target_window_cells",
]
