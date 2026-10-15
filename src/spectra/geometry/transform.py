"""Reflection, rotation and translation of LES grid coordinates.

The transform mirrors ``aviris4_ch4.regridding``: coordinates are reflected
around a pivot first and rotated afterwards, so provenance written by either
package describes the same placement.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from spectra.contracts.errors import ContractError
from spectra.types import ANCHOR_CORNERS, AnchorCorner, Flip


def validate_rotation(value: int) -> int:
    """Return a whole-degree rotation between -359 and 359."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"rotation must be an integer number of degrees; got {value!r}."
        )
    rotation = int(value)
    if not -359 <= rotation <= 359:
        raise ContractError(f"rotation must lie between -359 and 359; got {rotation}.")
    return rotation


def validate_flip(value: Flip | None) -> Flip:
    """Return a normalised reflection axis (``none``, ``x`` or ``y``)."""
    if value is None:
        return "none"
    if value not in ("none", "x", "y"):
        raise ContractError(f"flip must be 'none', 'x' or 'y'; got {value!r}.")
    return value


def transform_coordinates(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    pivot: tuple[float, float],
    rotation_deg: float,
    flip: Flip = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect coordinates around ``pivot`` and then rotate them around it.

    Args:
        x: X coordinates.
        y: Y coordinates.
        pivot: Point the reflection and rotation are applied around.
        rotation_deg: Counterclockwise rotation in degrees.
        flip: Axis whose coordinates are mirrored before the rotation.

    Returns:
        Transformed x and y arrays.
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    relative_x = x_values - pivot[0]
    relative_y = y_values - pivot[1]
    if flip == "x":
        relative_x = -relative_x
    elif flip == "y":
        relative_y = -relative_y
    if rotation_deg == 0:
        return pivot[0] + relative_x, pivot[1] + relative_y
    angle = math.radians(rotation_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        pivot[0] + cosine * relative_x - sine * relative_y,
        pivot[1] + sine * relative_x + cosine * relative_y,
    )


def translate(
    x: npt.ArrayLike, y: npt.ArrayLike, offset: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Shift coordinates by ``offset``."""
    return np.asarray(x, dtype=float) + offset[0], np.asarray(y, dtype=float) + offset[
        1
    ]


def anchor_cell_centre(
    x: npt.ArrayLike, y: npt.ArrayLike, corner: AnchorCorner
) -> tuple[float, float]:
    """Return the centre of the selected corner cell of a regular grid.

    Args:
        x: Increasing one-dimensional cell-centre x coordinates.
        y: Increasing one-dimensional cell-centre y coordinates.
        corner: Grid corner, e.g. ``"upper left"`` (minimum x, maximum y).

    Returns:
        Corner cell centre.
    """
    if corner not in ANCHOR_CORNERS:
        choices = ", ".join(repr(choice) for choice in ANCHOR_CORNERS)
        raise ContractError(f"anchor corner must be one of {choices}; got {corner!r}.")
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    x_index = -1 if corner.endswith("right") else 0
    y_index = -1 if corner.startswith("upper") else 0
    return float(x_values[x_index]), float(y_values[y_index])


def support_vertices(
    bounds: tuple[float, float, float, float],
    anchor: tuple[float, float],
    rotation_deg: float,
    flip: Flip = "none",
) -> np.ndarray:
    """Return the transformed corners of a bounding box relative to the anchor.

    Args:
        bounds: ``(xmin, ymin, xmax, ymax)`` in the untransformed frame.
        anchor: Pivot of the reflection and rotation.
        rotation_deg: Counterclockwise rotation in degrees.
        flip: Axis mirrored before the rotation.

    Returns:
        ``(4, 2)`` array of vertices relative to the anchor, perimeter order.
    """
    xmin, ymin, xmax, ymax = bounds
    vertices = np.asarray(
        ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), dtype=float
    )
    x, y = transform_coordinates(
        vertices[:, 0],
        vertices[:, 1],
        pivot=anchor,
        rotation_deg=rotation_deg,
        flip=flip,
    )
    return np.column_stack((x - anchor[0], y - anchor[1]))


def vertices_bounds(
    vertices: npt.NDArray[np.floating],
) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounds of ``(n, 2)`` polygon vertices."""
    return (
        float(np.min(vertices[:, 0])),
        float(np.min(vertices[:, 1])),
        float(np.max(vertices[:, 0])),
        float(np.max(vertices[:, 1])),
    )


__all__ = [
    "anchor_cell_centre",
    "support_vertices",
    "transform_coordinates",
    "translate",
    "validate_flip",
    "validate_rotation",
    "vertices_bounds",
]
