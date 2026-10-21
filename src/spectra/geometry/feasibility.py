"""Conservative map of the anchor pixels that can hold a plume.

Random anchors are drawn from the pixels whose outside mass fraction can lie
within the requested bounds rather than from the whole scene. The fraction is
bounded, not computed, for every anchor at once: the swath is rasterised into
cells certainly inside and certainly outside it, the plume mass is binned on
the same cell size, and one FFT correlation per side gives the mass certainly
outside and certainly inside the swath for an anchor centre anywhere in a
cell. Pixels are grouped into tiles whose bounds hold for all their pixels,
so the candidate tiles contain every eligible pixel; testing each draw
exactly then keeps the accepted anchors uniform over the eligible pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import shapely
from scipy import ndimage, signal

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import Outline
from spectra.types import SceneGeometry

if TYPE_CHECKING:
    from spectra.geometry.swath import MassPoints

# Upper bound on the number of anchor tiles of a scene.
_MAX_TILES = 1_000_000
# Upper bound on the number of cells of the swath raster.
_MAX_CELLS = 20_000_000
# Cells across the plume extent at the finest raster resolution.
_KERNEL_CELLS = 128
# Largest tile span, in cells per axis, resolved by the rectangle gather.
_GATHER = 6
# Margin of the raster around the reachable plume positions, in cells.
_MARGIN = 3
# Across-track rows of pixel centres read at once when tiles are built.
_STRIP_ROWS = 256
# Absorbs the FFT round-off of the bounds.
_TOLERANCE = 1e-9


def _tile_reduce(
    values: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the NaN-ignoring min, max and finite count of square blocks."""
    rows, columns = values.shape
    padded = np.full((-(-rows // size) * size, -(-columns // size) * size), np.nan)
    padded[:rows, :columns] = values
    blocks = padded.reshape(
        padded.shape[0] // size, size, padded.shape[1] // size, size
    )
    return (
        np.fmin.reduce(blocks, axis=(1, 3)),
        np.fmax.reduce(blocks, axis=(1, 3)),
        np.isfinite(blocks).sum(axis=(1, 3)),
    )


def _centres(
    geometry: SceneGeometry, across: slice, along: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Load pixel centres, with NaN where either coordinate is not finite."""
    dim_across, dim_along = geometry.spatial_dims
    indexer = {dim_across: across, dim_along: along}
    x = np.asarray(geometry.x.isel(indexer).values, dtype=np.float64)
    y = np.asarray(geometry.y.isel(indexer).values, dtype=np.float64)
    missing = ~(np.isfinite(x) & np.isfinite(y))
    x[missing] = np.nan
    y[missing] = np.nan
    return x, y


@dataclass(frozen=True, eq=False)
class AnchorTiles:
    """Extent of the pixel centres of square pixel tiles of a scene.

    Attributes:
        size: Tile edge in pixels.
        x_min: Smallest finite pixel-centre x per tile, NaN for empty tiles,
            on ``(n_tiles_across, n_tiles_along)``.
        x_max: Largest finite pixel-centre x per tile.
        y_min: Smallest finite pixel-centre y per tile.
        y_max: Largest finite pixel-centre y per tile.
        count: Number of pixels with a finite centre per tile.
        spacing: Median distance between neighbouring pixel centres along
            the axis with the finer sampling, in metres; 0 for a single pixel.
    """

    size: int
    x_min: np.ndarray
    x_max: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    count: np.ndarray
    spacing: float

    @classmethod
    def from_geometry(cls, geometry: SceneGeometry) -> AnchorTiles:
        """Read the pixel centres once, in across-track strips, and tile them."""
        n_across, n_along = geometry.shape
        size = max(1, math.ceil(math.sqrt(n_across * n_along / _MAX_TILES)))
        strip = size * max(1, _STRIP_ROWS // size)
        parts: list[tuple[np.ndarray, ...]] = []
        steps: tuple[list[float], list[float]] = ([], [])
        for start in range(0, n_across, strip):
            x, y = _centres(
                geometry, slice(start, min(start + strip, n_across)), slice(0, n_along)
            )
            x_min, x_max, count = _tile_reduce(x, size)
            y_min, y_max, _ = _tile_reduce(y, size)
            parts.append((x_min, x_max, y_min, y_max, count))
            for axis, medians in enumerate(steps):
                step = np.hypot(np.diff(x, axis=axis), np.diff(y, axis=axis))
                step = step[np.isfinite(step)]
                if step.size:
                    medians.append(float(np.median(step)))
        x_min, x_max, y_min, y_max, count = (
            np.concatenate(arrays) for arrays in zip(*parts, strict=True)
        )
        return cls(
            size=size,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            count=count,
            spacing=min(
                (float(np.median(medians)) for medians in steps if medians),
                default=0.0,
            ),
        )


def _cell_span(
    low: np.ndarray,
    high: np.ndarray,
    filled: np.ndarray,
    origin: float,
    cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first and last raster cell index of per-tile extents."""
    first = np.floor((np.where(filled, low, origin) - origin) / cell)
    last = np.floor((np.where(filled, high, origin) - origin) / cell)
    return first.astype(np.intp), last.astype(np.intp)


def _erode(cells: np.ndarray) -> np.ndarray:
    """Keep the cells whose 2x2 block starting at them lies within ``cells``."""
    eroded = np.zeros_like(cells)
    eroded[:-1, :-1] = cells[:-1, :-1] & cells[1:, :-1] & cells[:-1, 1:] & cells[1:, 1:]
    return eroded


def _swath_cells(
    swath: Outline, origin: tuple[float, float], cell: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cells certainly inside and certainly outside the swath.

    Every cell the outline passes through is excluded: the rings are sampled
    at half a cell, so a crossed cell lies within one cell of a sample and the
    3x3 dilation covers it. The remaining cells form components that do not
    touch the outline, each wholly inside or outside, which one centre test
    decides. Both results are eroded for the two cells per axis a binned
    point can reach (see :meth:`FeasibilityMap.build`).
    """
    boundary = np.zeros(shape, dtype=bool)
    for ring in shapely.get_rings(shapely.get_parts(swath)):
        coordinates = shapely.get_coordinates(ring)
        start, delta = coordinates[:-1], np.diff(coordinates, axis=0)
        steps = np.maximum(
            1, np.ceil(np.hypot(delta[:, 0], delta[:, 1]) / (0.5 * cell))
        ).astype(np.intp)
        segment = np.repeat(np.arange(steps.size), steps + 1)
        first = np.repeat(np.cumsum(steps + 1) - (steps + 1), steps + 1)
        fraction = (np.arange(segment.size) - first) / steps[segment]
        samples = start[segment] + fraction[:, None] * delta[segment]
        index = np.floor((samples - np.asarray(origin)) / cell).astype(np.intp)
        boundary[index[:, 0], index[:, 1]] = True
    boundary = np.asarray(
        ndimage.binary_dilation(boundary, structure=np.ones((3, 3), dtype=bool)),
        dtype=bool,
    )
    labels, count = cast(tuple[np.ndarray, int], ndimage.label(~boundary))
    ids, first_cell = (
        np.asarray(values, dtype=np.intp)
        for values in np.unique(labels.ravel(), return_index=True)
    )
    ids, first_cell = ids[ids > 0], first_cell[ids > 0]
    rows, columns = np.unravel_index(first_cell, shape)
    inside_label = np.zeros(count + 1, dtype=bool)
    inside_label[ids] = shapely.contains_xy(
        swath, origin[0] + (rows + 0.5) * cell, origin[1] + (columns + 0.5) * cell
    )
    labelled = labels > 0
    inside = labelled & inside_label[labels]
    outside = labelled & ~inside_label[labels]
    return _erode(inside), _erode(outside)


@dataclass(frozen=True, eq=False)
class FeasibilityMap:
    """Bounds of the outside mass fraction of the anchor pixels of every tile.

    Attributes:
        tiles: Tiles the bounds belong to.
        lower: Lower bound of the outside mass fraction of any pixel of a tile.
        upper: Upper bound of the outside mass fraction of any pixel of a tile.
        cell: Raster cell size in metres.
    """

    tiles: AnchorTiles
    lower: np.ndarray
    upper: np.ndarray
    cell: float

    @classmethod
    def build(
        cls, tiles: AnchorTiles, swath: Outline, points: MassPoints
    ) -> FeasibilityMap:
        """Bound the outside mass fraction of every tile for one placed plume.

        An anchor centre in raster cell ``k`` and a point binned into
        ``j = floor(offset / cell)`` put that point in one of the cells
        ``k + j + {0, 1}`` per axis, so the eroded rasters count it as surely
        inside or outside for every anchor in ``k``. Points on the outline
        are never counted, which keeps the bounds valid for the exact test
        of :func:`~spectra.geometry.swath.outside_mass_fraction`.

        Args:
            tiles: Anchor tiles of the scene.
            swath: Swath outline.
            points: Transformed plume mass points relative to the anchor.

        Returns:
            Per-tile bounds of the outside mass fraction.

        Raises:
            ContractError: If no pixel of the scene has a finite centre.
        """
        filled = tiles.count > 0
        if not filled.any():
            raise ContractError("No pixel of the scene has a finite geolocation.")
        x0, y0, x1, y1 = (float(value) for value in swath.bounds)
        # Offsets widened to include zero so the raster also holds the outline.
        dx = (min(float(points.dx.min()), 0.0), max(float(points.dx.max()), 0.0))
        dy = (min(float(points.dy.min()), 0.0), max(float(points.dy.max()), 0.0))
        footprint = np.maximum(tiles.x_max - tiles.x_min, tiles.y_max - tiles.y_min)
        width = x1 - x0 + dx[1] - dx[0]
        height = y1 - y0 + dy[1] - dy[0]
        # Finer than the plume, the tiles or a quarter pixel adds nothing.
        cell = max(
            max(np.ptp(points.dx), np.ptp(points.dy)) / _KERNEL_CELLS,
            float(np.percentile(footprint[filled], 99)) / (_GATHER - 2),
            tiles.spacing / 4,
            math.sqrt(width * height / _MAX_CELLS),
        )
        origin = (x0 + dx[0] - _MARGIN * cell, y0 + dy[0] - _MARGIN * cell)
        shape = (
            math.ceil((x1 + dx[1] + _MARGIN * cell - origin[0]) / cell),
            math.ceil((y1 + dy[1] + _MARGIN * cell - origin[1]) / cell),
        )
        inside, outside = _swath_cells(swath, origin, cell, shape)

        bins_x = np.floor(points.dx / cell).astype(np.intp)
        bins_y = np.floor(points.dy / cell).astype(np.intp)
        low = (int(bins_x.min()), int(bins_y.min()))
        kernel_shape = (int(bins_x.max()) - low[0] + 1, int(bins_y.max()) - low[1] + 1)
        kernel = np.bincount(
            np.ravel_multi_index((bins_x - low[0], bins_y - low[1]), kernel_shape),
            weights=points.weights,
            minlength=kernel_shape[0] * kernel_shape[1],
        ).reshape(kernel_shape)
        # Convolving with the flipped kernel correlates; the full output at
        # ``k + shift`` sums the kernel over the cells ``k + j``.
        flipped = kernel[::-1, ::-1]
        shift = (low[0] + kernel_shape[0] - 1, low[1] + kernel_shape[1] - 1)
        surely_outside = signal.fftconvolve(outside.astype(np.float64), flipped)
        surely_inside = signal.fftconvolve(inside.astype(np.float64), flipped)
        cell_lower = surely_outside / points.total
        cell_upper = 1.0 - surely_inside / points.total

        rows = _cell_span(tiles.x_min, tiles.x_max, filled, origin[0], cell)
        columns = _cell_span(tiles.y_min, tiles.y_max, filled, origin[1], cell)
        lower = np.full(filled.shape, np.inf)
        upper = np.full(filled.shape, -np.inf)
        # Clipping each offset to the tile's last cell keeps the gather inside
        # the cells its centres span, so min and max are over exactly those.
        last = (cell_lower.shape[0] - 1, cell_lower.shape[1] - 1)
        for a in range(_GATHER):
            row = np.clip(np.minimum(rows[0] + a, rows[1]) + shift[0], 0, last[0])
            for b in range(_GATHER):
                column = np.minimum(columns[0] + b, columns[1]) + shift[1]
                column = np.clip(column, 0, last[1])
                np.minimum(lower, cell_lower[row, column], out=lower)
                np.maximum(upper, cell_upper[row, column], out=upper)
        # Tiles spanning more cells than gathered keep the trivial bounds.
        oversized = (rows[1] - rows[0] >= _GATHER) | (
            columns[1] - columns[0] >= _GATHER
        )
        lower[oversized] = 0.0
        upper[oversized] = 1.0
        lower[~filled] = np.nan
        upper[~filled] = np.nan
        return cls(tiles=tiles, lower=lower, upper=upper, cell=cell)

    def cumulative_counts(self, bounds: tuple[float, float]) -> np.ndarray:
        """Return the running candidate-pixel count over the flattened tiles.

        A tile is a candidate when its bounds overlap ``bounds``; its pixels
        then include every eligible pixel of the tile.
        """
        minimum, maximum = bounds
        candidate = (self.upper >= minimum - _TOLERANCE) & (
            self.lower <= maximum + _TOLERANCE
        )
        return np.cumsum(np.where(candidate, self.tiles.count, 0).ravel())

    def attainable(self) -> tuple[float, float]:
        """Return the range of outside mass fractions any pixel can reach."""
        filled = self.tiles.count > 0
        return (
            float(np.clip(np.min(self.lower[filled]), 0.0, 1.0)),
            float(np.clip(np.max(self.upper[filled]), 0.0, 1.0)),
        )

    def pixel(
        self, geometry: SceneGeometry, cumulative: np.ndarray, rank: int
    ) -> tuple[int, int]:
        """Return ``(along, across)`` of the candidate pixel of a given rank.

        Args:
            geometry: Scene geometry the tiles were built from.
            cumulative: Result of :meth:`cumulative_counts`.
            rank: Zero-based rank among the candidate pixels.
        """
        tile = int(np.searchsorted(cumulative, rank, side="right"))
        before = int(cumulative[tile - 1]) if tile else 0
        tile_across, tile_along = np.unravel_index(tile, self.tiles.count.shape)
        size = self.tiles.size
        across0, along0 = int(tile_across) * size, int(tile_along) * size
        x, _ = _centres(
            geometry, slice(across0, across0 + size), slice(along0, along0 + size)
        )
        offset = int(np.flatnonzero(np.isfinite(x))[rank - before])
        row, column = divmod(offset, x.shape[1])
        return along0 + column, across0 + row


__all__ = ["AnchorTiles", "FeasibilityMap"]
