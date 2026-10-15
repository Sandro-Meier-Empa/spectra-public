"""Conservative remapping of placed plume columns with CDO ``remapcon``.

The module writes the two files CDO needs, a georeferenced CF source grid
with cell corners and a cropped curvilinear target window, runs the remap and
checks mass conservation. File layouts, attributes and chunking follow
``aviris4_ch4.regridding`` and ``microhh_processing_chain.postprocessing``
so files written by either package are interchangeable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyproj
import shapely
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import Variable as NetCDFVariable
from shapely.geometry import Polygon

from spectra.contracts.errors import ContractError, RegridderUnavailableError
from spectra.geometry.corners import (
    cell_vertices_to_nodes,
    nodes_to_cell_vertices,
    outline_from_nodes,
    require_finite,
    window_corner_nodes,
)
from spectra.geometry.crs import require_projected_metre_crs
from spectra.geometry.swath import ScenePlacement, place_in_scene
from spectra.geometry.transform import transform_coordinates
from spectra.scene.window import expand_window, normalise_remapped_dimensions
from spectra.types import MapGrid, Placement, SceneGeometry, Window

logger = logging.getLogger(__name__)

HORIZONTAL_DIMS = ("across", "along")
SOURCE_DIMS = ("nrows", "nobs")
_NETCDF_CHUNK_BYTES = 4 * 1024**2
_GEOMETRY_ACROSS_CHUNK = 8
_SWATH_SIMPLIFY_TOLERANCE = 1e-3
_SAFE_ENCODING_KEYS = frozenset(
    {
        "_FillValue",
        "add_offset",
        "chunksizes",
        "complevel",
        "contiguous",
        "dtype",
        "endian",
        "fletcher32",
        "least_significant_digit",
        "quantize_mode",
        "scale_factor",
        "shuffle",
        "significant_digits",
        "zlib",
    }
)
_AUXILIARY_VARIABLES = ("x", "y", "lon", "lat", "lonc", "latc", "dummy")


@dataclass(frozen=True)
class CdoPreparation:
    """Files and placement metadata for a column-to-scene CDO remap.

    Attributes:
        source_path: Georeferenced CDO source file of the placed column.
        target_grid_path: Cropped curvilinear CDO target grid.
        along_index: Zero-based along-track anchor pixel.
        across_index: Zero-based across-track anchor pixel.
        along_slice: Along-track index window of the target grid.
        across_slice: Across-track index window of the target grid.
        longitude: Anchor pixel centre longitude in degrees.
        latitude: Anchor pixel centre latitude in degrees.
        x_offset: Projected x translation from the LES frame to the scene CRS.
        y_offset: Projected y translation from the LES frame to the scene CRS.
        rotation: Rotation around the anchor in whole degrees.
        flip_axis: LES axis reflected before the rotation, or ``None``.
        outside_fraction: Plume mass fraction outside the swath.
        attempts: Random draws needed; ``None`` for an explicit anchor.
    """

    source_path: Path
    target_grid_path: Path
    along_index: int
    across_index: int
    along_slice: slice
    across_slice: slice
    longitude: float
    latitude: float
    x_offset: float
    y_offset: float
    rotation: int = 0
    flip_axis: str | None = None
    outside_fraction: float = 0.0
    attempts: int | None = None

    @property
    def window(self) -> Window:
        """Return the target window."""
        return Window(across=self.across_slice, along=self.along_slice)

    @property
    def command(self) -> str:
        """Return the conservative CDO command for the prepared files."""
        return f"cdo remapcon,{self.target_grid_path} {self.source_path} <output.nc>"


def temporary_output_path(path: Path) -> Path:
    """Return a unique temporary path next to a final output file."""
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _validate_output_paths(output_paths: tuple[Path, ...], overwrite: bool) -> None:
    """Validate output path relationships and the overwrite policy."""
    resolved = [path.resolve() for path in output_paths]
    if len(set(resolved)) != len(resolved):
        raise ContractError("Source and target-grid output paths must be different.")
    if not overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError("Output file already exists: " + ", ".join(existing))


def _to_lonlat_scalar(crs: pyproj.CRS, x: float, y: float) -> tuple[float, float]:
    """Transform one projected point to longitude and latitude."""
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x, y)
    return float(longitude), float(latitude)


def _coordinate_crs_attrs(crs: pyproj.CRS) -> dict[str, Any]:
    """Return compact CRS attributes for projected coordinate variables."""
    authority = crs.to_authority()
    if authority is not None and authority[0].upper() == "EPSG":
        return {"epsg": int(authority[1])}
    return {"crs": crs.to_string()}


def _grid_mapping_attrs(crs: pyproj.CRS) -> dict[str, Any]:
    """Return NetCDF-safe CF grid-mapping attributes."""
    attrs = {
        key: value
        for key, value in crs.to_cf().items()
        if value is not None and not isinstance(value, dict)
    }
    attrs["spatial_ref"] = crs.to_wkt()
    return attrs


def embedding_attrs(
    scene_placement: ScenePlacement,
    geometry: SceneGeometry,
    *,
    variable: str,
    metadata_source: str,
) -> dict[str, Any]:
    """Return the ``embedding_*`` provenance written to both CDO files.

    The attributes, including the ``embedding_*`` window bounds, document the
    files; SPECTRA never reads the window back from them but passes the
    :attr:`CdoPreparation.window` explicitly.
    """
    placed = scene_placement.placed
    draw = scene_placement.draw
    placement = scene_placement.placement
    support = placed.support_vertices
    xmin, ymin, xmax, ymax = (
        float(np.min(support[:, 0])),
        float(np.min(support[:, 1])),
        float(np.max(support[:, 0])),
        float(np.max(support[:, 1])),
    )
    return {
        **scene_placement.placement_attrs(),
        "embedding_grid_method": "cdo-remapcon",
        "embedding_anchor": (
            f"{placement.anchor_corner.replace(' ', '-')} MicroHH cell center"
        ),
        "embedding_plume_variable": variable,
        "embedding_metadata_source": metadata_source,
        "embedding_crs": geometry.crs.to_string(),
        "embedding_along_index": draw.along,
        "embedding_across_index": draw.across,
        "embedding_anchor_x": draw.anchor_x,
        "embedding_anchor_y": draw.anchor_y,
        "embedding_x_offset": placed.offset_xy[0],
        "embedding_y_offset": placed.offset_xy[1],
        "embedding_rotation": int(placement.rotation_deg),
        "embedding_flip_axis": placement.flip_axis or "none",
        "embedding_support_bounds": np.asarray(
            (xmin, ymin, xmax, ymax), dtype=np.float64
        ),
        "embedding_support_vertices": np.asarray(support, dtype=np.float64).ravel(),
        **scene_placement.window.attrs("embedding_"),
    }


def _safe_encoding(
    encoding: dict[str, Any],
    original_dims: tuple[Hashable, ...],
    output_dims: tuple[Hashable, ...],
) -> dict[str, Any]:
    """Keep portable NetCDF encodings and reorder dimension-based chunks."""
    cleaned = {k: v for k, v in encoding.items() if k in _SAFE_ENCODING_KEYS}
    chunksizes = cleaned.get("chunksizes")
    if chunksizes is not None and original_dims != output_dims:
        if isinstance(chunksizes, (tuple, list)) and len(chunksizes) == len(
            original_dims
        ):
            by_dim = dict(zip(original_dims, tuple(chunksizes), strict=True))
            cleaned["chunksizes"] = tuple(by_dim[dim] for dim in output_dims)
        else:
            cleaned.pop("chunksizes")
    if cleaned.get("contiguous") and "chunksizes" in cleaned:
        cleaned.pop("contiguous")
    return cleaned


def georeferenced_source(
    scene_placement: ScenePlacement,
    geometry: SceneGeometry,
    *,
    variable: str,
) -> xr.Dataset:
    """Return the placed column as a CF dataset with cell corners for CDO.

    The layout reproduces ``convert_microhh_grid_to_coords`` followed by the
    ``aviris4_ch4`` reflection/rotation and ``format_data_for_cdo``: the
    native grid is translated by the projected anchor pixel centre obtained
    through a longitude/latitude round trip, transformed around the original
    anchor pixel centre and written on ``(nrows, nobs)`` with ``lonc``/``latc``
    vertices.
    """
    column = scene_placement.column
    placed = scene_placement.placed
    draw = scene_placement.draw
    placement = scene_placement.placement
    crs = geometry.crs
    longitude, latitude = _to_lonlat_scalar(crs, draw.anchor_x, draw.anchor_y)
    to_projected = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected_x, projected_y = to_projected.transform(longitude, latitude)
    if not np.isfinite(projected_x) or not np.isfinite(projected_y):
        raise ContractError("The anchor cannot be represented in the scene CRS.")
    anchor_source = placed.anchor_source_xy
    x_values = np.asarray(column["x"].values, dtype=float) + (
        projected_x - anchor_source[0]
    )
    y_values = np.asarray(column["y"].values, dtype=float) + (
        projected_y - anchor_source[1]
    )
    centre_x, centre_y = np.meshgrid(x_values, y_values, indexing="xy")
    x_centres = centre_x[0, :]
    y_centres = centre_y[:, 0]
    dx = float(x_centres[1] - x_centres[0])
    dy = float(y_centres[1] - y_centres[0])
    x_nodes = np.concatenate(
        (
            [x_centres[0] - dx / 2],
            0.5 * (x_centres[:-1] + x_centres[1:]),
            [x_centres[-1] + dx / 2],
        )
    )
    y_nodes = np.concatenate(
        (
            [y_centres[0] - dy / 2],
            0.5 * (y_centres[:-1] + y_centres[1:]),
            [y_centres[-1] + dy / 2],
        )
    )
    node_x, node_y = np.meshgrid(x_nodes, y_nodes, indexing="xy")
    pivot = (draw.anchor_x, draw.anchor_y)
    centre_x, centre_y = transform_coordinates(
        centre_x,
        centre_y,
        pivot=pivot,
        rotation_deg=placement.rotation_deg,
        flip=placement.flip,
    )
    node_x, node_y = transform_coordinates(
        node_x,
        node_y,
        pivot=pivot,
        rotation_deg=placement.rotation_deg,
        flip=placement.flip,
    )
    to_geo = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    centre_lon, centre_lat = to_geo.transform(centre_x, centre_y)
    node_lon, node_lat = to_geo.transform(node_x, node_y)
    cell_lon = nodes_to_cell_vertices(node_lon)
    cell_lat = nodes_to_cell_vertices(node_lat)
    if placement.flip != "none":
        # Reflection reverses winding; retain the original CDO corner winding.
        cell_lon = np.flip(cell_lon, axis=-1)
        cell_lat = np.flip(cell_lat, axis=-1)

    values = placed.values
    if np.issubdtype(column.dtype, np.floating):
        values = values.astype(column.dtype, copy=False)
    data_attrs = {
        key: value for key, value in column.attrs.items() if key != "coordinates"
    }
    data_attrs["coordinates"] = "lat lon"
    data = xr.DataArray(values, dims=SOURCE_DIMS, attrs=data_attrs)
    dataset = xr.Dataset({variable: data})
    if "time" in column.coords and column["time"].ndim == 0:
        time = column["time"].copy(deep=False)
        time = time.drop_vars([name for name in time.coords if name != "time"])
        time.attrs = dict(time.attrs)
        time.encoding = _safe_encoding(time.encoding, time.dims, time.dims)
        if np.issubdtype(time.dtype, np.datetime64):
            time.attrs.pop("units", None)
            time.attrs.pop("calendar", None)
            time.encoding.update(
                {
                    "units": "seconds since 1970-01-01 00:00:00",
                    "calendar": "proleptic_gregorian",
                }
            )
        dataset = dataset.assign_coords(time=time)
    dataset = dataset.assign_coords(
        lon=(
            SOURCE_DIMS,
            centre_lon,
            {
                "units": "degrees_east",
                "standard_name": "longitude",
                "epsg": 4326,
                "bounds": "lonc",
            },
        ),
        lat=(
            SOURCE_DIMS,
            centre_lat,
            {
                "units": "degrees_north",
                "standard_name": "latitude",
                "epsg": 4326,
                "bounds": "latc",
            },
        ),
    )
    dataset["lonc"] = ((*SOURCE_DIMS, "vertices"), cell_lon, {"units": "degrees_east"})
    dataset["latc"] = (
        (*SOURCE_DIMS, "vertices"),
        cell_lat,
        {"units": "degrees_north"},
    )
    return dataset


def _geometry_chunks(nacross: int, nalong: int, vertices: int) -> tuple[int, ...]:
    """Return bounded NetCDF chunks for grid centres or vertices."""
    across_chunk = min(_GEOMETRY_ACROSS_CHUNK, nacross)
    along_chunk = max(
        1,
        _NETCDF_CHUNK_BYTES
        // (np.dtype(np.float64).itemsize * across_chunk * vertices),
    )
    base = (across_chunk, min(nalong, along_chunk))
    return base if vertices == 1 else (*base, vertices)


def _dimension_values(array: xr.DataArray, dim: str, selection: slice) -> np.ndarray:
    """Return stored dimension coordinates or their integer indices."""
    if dim in array.coords and array.coords[dim].dims == (dim,):
        return np.asarray(array.coords[dim].values)[selection]
    return np.arange(array.sizes[dim], dtype=np.int32)[selection]


def _write_grid_descriptor(
    output: NetCDFDataset,
    crs: pyproj.CRS,
    horizontal_dims: tuple[str, str],
    chunksizes: tuple[int, ...],
) -> NetCDFVariable:
    """Write a CF grid mapping and return CDO's spatial dummy variable."""
    crs_variable = output.createVariable("crs", "i4")
    crs_variable.assignValue(0)
    crs_variable.setncatts(_grid_mapping_attrs(crs))
    dummy = output.createVariable("dummy", "f4", horizontal_dims, chunksizes=chunksizes)
    dummy.setncatts({"coordinates": "lat lon", "grid_mapping": "crs"})
    return dummy


def _create_lonlat_variables(
    output: NetCDFDataset,
    horizontal_dims: tuple[str, str],
    centre_chunks: tuple[int, ...],
    corner_chunks: tuple[int, ...],
) -> tuple[NetCDFVariable, NetCDFVariable, NetCDFVariable, NetCDFVariable]:
    """Create chunked geographic centre and bounds variables."""
    lon = output.createVariable(
        "lon", "f8", horizontal_dims, fill_value=np.nan, chunksizes=centre_chunks
    )
    lat = output.createVariable(
        "lat", "f8", horizontal_dims, fill_value=np.nan, chunksizes=centre_chunks
    )
    lonc = output.createVariable(
        "lonc",
        "f8",
        (*horizontal_dims, "vertices"),
        fill_value=np.nan,
        chunksizes=corner_chunks,
    )
    latc = output.createVariable(
        "latc",
        "f8",
        (*horizontal_dims, "vertices"),
        fill_value=np.nan,
        chunksizes=corner_chunks,
    )
    lon.setncatts(
        {"units": "degrees_east", "standard_name": "longitude", "bounds": "lonc"}
    )
    lat.setncatts(
        {"units": "degrees_north", "standard_name": "latitude", "bounds": "latc"}
    )
    lonc.setncattr("units", "degrees_east")
    latc.setncattr("units", "degrees_north")
    return lon, lat, lonc, latc


def _window_row_blocks(
    geometry: SceneGeometry, window: Window
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(start, stop, x, y, node_x, node_y)`` of across-track row blocks.

    Centres cover the block rows and the window columns; nodes have one more
    row and column, built from neighbouring centres where they exist.
    """
    n_across, n_along = geometry.shape
    dim0, dim1 = geometry.spatial_dims
    column_start = max(0, window.along.start - 1)
    columns = slice(column_start, min(n_along, window.along.stop + 1))
    for start in range(window.across.start, window.across.stop, _GEOMETRY_ACROSS_CHUNK):
        stop = min(window.across.stop, start + _GEOMETRY_ACROSS_CHUNK)
        row_start = max(0, start - 1)
        rows = slice(row_start, min(n_across, stop + 1))
        block_x = np.asarray(
            geometry.x.isel({dim0: rows, dim1: columns}).values, dtype=np.float64
        )
        block_y = np.asarray(
            geometry.y.isel({dim0: rows, dim1: columns}).values, dtype=np.float64
        )
        block_window = Window(across=slice(start, stop), along=window.along)
        offset = (row_start, column_start)
        node_x = window_corner_nodes(block_x, block_window, start=offset)
        node_y = window_corner_nodes(block_y, block_window, start=offset)
        centre_rows = slice(start - row_start, stop - row_start)
        centre_columns = slice(
            window.along.start - column_start, window.along.stop - column_start
        )
        yield (
            start,
            stop,
            block_x[centre_rows, centre_columns],
            block_y[centre_rows, centre_columns],
            node_x,
            node_y,
        )


def _write_window_geometry(
    output: NetCDFDataset,
    geometry: SceneGeometry,
    window: Window,
    *,
    extent_crs: pyproj.CRS | None = None,
) -> tuple[float, float, float, float] | None:
    """Write chunked geographic centres and cell bounds of a window."""
    n_across, n_along = window.shape
    centre_chunks = _geometry_chunks(n_across, n_along, vertices=1)
    corner_chunks = _geometry_chunks(n_across, n_along, vertices=4)
    lon, lat, lonc, latc = _create_lonlat_variables(
        output, HORIZONTAL_DIMS, centre_chunks, corner_chunks
    )
    to_geo = pyproj.Transformer.from_crs(geometry.crs, "EPSG:4326", always_xy=True)
    to_extent = (
        None
        if extent_crs is None
        else pyproj.Transformer.from_crs(geometry.crs, extent_crs, always_xy=True)
    )
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for start, stop, x, y, node_x, node_y in _window_row_blocks(geometry, window):
        out = slice(start - window.across.start, stop - window.across.start)
        require_finite(x, y)
        centre_lon, centre_lat = to_geo.transform(x, y)
        require_finite(centre_lon, centre_lat)
        lon[out] = centre_lon
        lat[out] = centre_lat
        require_finite(node_x, node_y)
        corner_lon, corner_lat = to_geo.transform(node_x, node_y)
        require_finite(corner_lon, corner_lat)
        lonc[out] = nodes_to_cell_vertices(corner_lon)
        latc[out] = nodes_to_cell_vertices(corner_lat)
        if to_extent is not None:
            extent_x, extent_y = to_extent.transform(node_x, node_y)
            require_finite(extent_x, extent_y)
            xmin = min(xmin, float(np.min(extent_x)))
            xmax = max(xmax, float(np.max(extent_x)))
            ymin = min(ymin, float(np.min(extent_y)))
            ymax = max(ymax, float(np.max(extent_y)))
    if to_extent is None:
        return None
    return xmin, ymin, xmax, ymax


def write_target_window(
    path: Path,
    geometry: SceneGeometry,
    window: Window,
    *,
    attrs: dict[str, Any],
) -> None:
    """Write one cropped curvilinear target grid for CDO.

    Args:
        path: Output NetCDF path.
        geometry: Scene geometry.
        window: Index window of the target grid.
        attrs: Global attributes, typically :func:`embedding_attrs`.
    """
    dim0, dim1 = geometry.spatial_dims
    n_across, n_along = window.shape
    with NetCDFDataset(str(path), mode="w", format="NETCDF4") as output:
        output.createDimension("across", n_across)
        output.createDimension("along", n_along)
        output.createDimension("vertices", 4)
        across = output.createVariable("across", "i4", ("across",))
        along = output.createVariable("along", "i4", ("along",))
        across[:] = _dimension_values(geometry.x, dim0, window.across)
        along[:] = _dimension_values(geometry.x, dim1, window.along)
        across.setncattr("long_name", "original AVIRIS across-track index")
        along.setncattr("long_name", "original AVIRIS along-track index")
        centre_chunks = _geometry_chunks(n_across, n_along, vertices=1)
        dummy = _write_grid_descriptor(
            output,
            crs=geometry.crs,
            horizontal_dims=HORIZONTAL_DIMS,
            chunksizes=centre_chunks,
        )
        _write_window_geometry(output, geometry, window)
        dummy[:] = 0.0
        output.setncatts(
            {**attrs, "target_grid_kind": "cropped AVIRIS curvilinear grid"}
        )


def prepare_cdo_files(
    scene_placement: ScenePlacement,
    geometry: SceneGeometry,
    source_path: str | Path,
    target_grid_path: str | Path,
    *,
    variable: str,
    metadata_source: str = "",
    overwrite: bool = False,
) -> CdoPreparation:
    """Write the georeferenced source and cropped target grid for CDO.

    CDO is not executed; see :func:`run_cdo_remapcon` and
    :attr:`CdoPreparation.command`.

    Args:
        scene_placement: Placement from :func:`~spectra.geometry.swath.place_in_scene`.
        geometry: Scene geometry used for the placement.
        source_path: Destination of the georeferenced source file.
        target_grid_path: Destination of the cropped target grid.
        variable: Name the column is written under.
        metadata_source: Name recorded as ``embedding_metadata_source``;
            defaults to the geometry source file name.
        overwrite: Replace existing output files.

    Returns:
        Prepared paths and placement metadata.

    Raises:
        FileExistsError: If an output exists and ``overwrite`` is false.
        ContractError: If the output paths coincide.
    """
    source_path = Path(source_path)
    target_grid_path = Path(target_grid_path)
    _validate_output_paths((source_path, target_grid_path), overwrite=overwrite)
    if not metadata_source and geometry.source:
        metadata_source = Path(geometry.source).name
    attrs = embedding_attrs(
        scene_placement, geometry, variable=variable, metadata_source=metadata_source
    )
    for output_path in (source_path, target_grid_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    source_tmp = temporary_output_path(source_path)
    target_tmp = temporary_output_path(target_grid_path)
    try:
        write_target_window(target_tmp, geometry, scene_placement.window, attrs=attrs)
        formatted = georeferenced_source(scene_placement, geometry, variable=variable)
        formatted.attrs.update(attrs)
        formatted.to_netcdf(source_tmp)
        source_tmp.replace(source_path)
        target_tmp.replace(target_grid_path)
    except Exception:
        source_tmp.unlink(missing_ok=True)
        target_tmp.unlink(missing_ok=True)
        raise
    draw = scene_placement.draw
    placed = scene_placement.placed
    window = scene_placement.window
    longitude, latitude = _to_lonlat_scalar(geometry.crs, draw.anchor_x, draw.anchor_y)
    result = CdoPreparation(
        source_path=source_path,
        target_grid_path=target_grid_path,
        along_index=draw.along,
        across_index=draw.across,
        along_slice=window.along,
        across_slice=window.across,
        longitude=longitude,
        latitude=latitude,
        x_offset=placed.offset_xy[0],
        y_offset=placed.offset_xy[1],
        rotation=int(scene_placement.placement.rotation_deg),
        flip_axis=scene_placement.placement.flip_axis,
        outside_fraction=draw.outside_fraction,
        attempts=draw.attempts,
    )
    logger.info("For remapping, run \n%s", result.command)
    return result


def prepare_regular_target_grid(
    geometry: SceneGeometry,
    resolution: float,
    target_crs: str | int | pyproj.CRS | None,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a regular projected CDO target grid covering a scene.

    Args:
        geometry: Scene geometry whose pixel nodes define the extent.
        resolution: Grid spacing in metres.
        target_crs: Projected target CRS; defaults to the scene CRS.
        path: Destination of the target-grid file.
        overwrite: Replace an existing file.

    Returns:
        The written path.
    """
    path = Path(path)
    _validate_output_paths((path,), overwrite=overwrite)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ContractError("'resolution' must be a positive finite number.")
    crs = geometry.crs if target_crs is None else pyproj.CRS.from_user_input(target_crs)
    require_projected_metre_crs(crs, name="target CRS")
    n_across, n_along = geometry.shape
    full = Window(across=slice(0, n_across), along=slice(0, n_along))
    to_extent = pyproj.Transformer.from_crs(geometry.crs, crs, always_xy=True)
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for _, _, _, _, node_x, node_y in _window_row_blocks(geometry, full):
        require_finite(node_x, node_y)
        extent_x, extent_y = to_extent.transform(node_x, node_y)
        require_finite(extent_x, extent_y)
        xmin = min(xmin, float(np.min(extent_x)))
        xmax = max(xmax, float(np.max(extent_x)))
        ymin = min(ymin, float(np.min(extent_y)))
        ymax = max(ymax, float(np.max(extent_y)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temporary_output_path(path)
    try:
        _write_regular_grid(tmp, (xmin, ymin, xmax, ymax), float(resolution), crs)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _target_geometry_chunks(nrows: int, ncols: int, vertices: int) -> tuple[int, ...]:
    """Return bounded chunks for regular target-grid variables."""
    row_chunk = min(8, nrows)
    col_chunk = max(
        1,
        _NETCDF_CHUNK_BYTES // (np.dtype(np.float64).itemsize * row_chunk * vertices),
    )
    base = (row_chunk, min(ncols, col_chunk))
    return base if vertices == 1 else (*base, vertices)


def _write_regular_grid(
    path: Path,
    bounds: tuple[float, float, float, float],
    resolution: float,
    target_crs: pyproj.CRS,
) -> None:
    """Write a regular projected grid with geographic centres and bounds."""
    xmin, ymin, xmax, ymax = bounds
    snapped_xmin = np.floor(xmin / resolution) * resolution
    snapped_ymin = np.floor(ymin / resolution) * resolution
    snapped_xmax = np.ceil(xmax / resolution) * resolution
    snapped_ymax = np.ceil(ymax / resolution) * resolution
    nx = round((snapped_xmax - snapped_xmin) / resolution)
    ny = round((snapped_ymax - snapped_ymin) / resolution)
    if nx < 1 or ny < 1:
        raise ContractError("The calculated target grid is empty.")
    x_centres = snapped_xmin + (np.arange(nx, dtype=np.float64) + 0.5) * resolution
    y_centres = snapped_ymax - (np.arange(ny, dtype=np.float64) + 0.5) * resolution
    x_nodes = snapped_xmin + np.arange(nx + 1, dtype=np.float64) * resolution
    to_geo = pyproj.Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

    with NetCDFDataset(str(path), mode="w", format="NETCDF4") as output:
        output.createDimension("y", ny)
        output.createDimension("x", nx)
        output.createDimension("vertices", 4)
        x_variable = output.createVariable("x", "f8", ("x",))
        y_variable = output.createVariable("y", "f8", ("y",))
        x_variable[:] = x_centres
        y_variable[:] = y_centres
        crs_attrs = _coordinate_crs_attrs(target_crs)
        x_variable.setncatts(
            {
                "units": "m",
                "standard_name": "projection_x_coordinate",
                "axis": "X",
                **crs_attrs,
            }
        )
        y_variable.setncatts(
            {
                "units": "m",
                "standard_name": "projection_y_coordinate",
                "axis": "Y",
                **crs_attrs,
            }
        )
        centre_chunks = _target_geometry_chunks(ny, nx, vertices=1)
        corner_chunks = _target_geometry_chunks(ny, nx, vertices=4)
        lon, lat, lonc, latc = _create_lonlat_variables(
            output, ("y", "x"), centre_chunks, corner_chunks
        )
        dummy = _write_grid_descriptor(
            output, crs=target_crs, horizontal_dims=("y", "x"), chunksizes=centre_chunks
        )
        row_chunk = min(64, ny)
        for start in range(0, ny, row_chunk):
            stop = min(start + row_chunk, ny)
            centre_x, centre_y = np.meshgrid(x_centres, y_centres[start:stop])
            centre_lon, centre_lat = to_geo.transform(centre_x, centre_y)
            lon[start:stop, :] = centre_lon
            lat[start:stop, :] = centre_lat
            y_nodes = snapped_ymax - np.arange(start, stop + 1) * resolution
            node_x, node_y = np.meshgrid(x_nodes, y_nodes)
            node_lon, node_lat = to_geo.transform(node_x, node_y)
            lonc[start:stop, :, :] = nodes_to_cell_vertices(node_lon)
            latc[start:stop, :, :] = nodes_to_cell_vertices(node_lat)
            dummy[start:stop, :] = np.zeros((stop - start, nx), dtype=np.float32)
        output.setncatts(
            {
                "target_crs": target_crs.to_string(),
                "resolution_m": resolution,
                "projected_bounds": np.array(
                    [snapped_xmin, snapped_ymin, snapped_xmax, snapped_ymax],
                    dtype=np.float64,
                ),
                "GeoTransform": (
                    f"{snapped_xmin} {resolution} 0 {snapped_ymax} 0 {-resolution}"
                ),
            }
        )


def _open_dataset_input(
    value: xr.Dataset | str | Path, *, decode_times: bool = True
) -> tuple[xr.Dataset, bool]:
    """Open a dataset path lazily and report whether it must be closed."""
    if isinstance(value, xr.Dataset):
        return value, False
    return xr.open_dataset(value, chunks="auto", decode_times=decode_times), True


def _projected_cell_corners(data: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return CDO cell corners transformed to the dataset's projected CRS."""
    if "lonc" not in data or "latc" not in data:
        raise ContractError("Dataset must contain 'lonc' and 'latc' cell bounds.")
    lonc = data["lonc"]
    latc = data["latc"]
    if lonc.dims != latc.dims or lonc.ndim != 3 or lonc.shape[-1] != 4:
        raise ContractError(
            "lonc/latc must share two spatial dimensions and 4 vertices."
        )
    crs_value = data.attrs.get("embedding_crs") or data.attrs.get("target_crs")
    if crs_value is None:
        raise ContractError("Dataset lacks the projected CRS needed for pixel areas.")
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", pyproj.CRS.from_user_input(crs_value), always_xy=True
    )
    x, y = transformer.transform(lonc.values, latc.values)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _cell_areas_from_lonlat(data: xr.Dataset) -> xr.DataArray:
    """Calculate projected cell areas from CDO longitude/latitude bounds."""
    x, y = _projected_cell_corners(data)
    x = x - x[..., :1]
    y = y - y[..., :1]
    area_values = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1), axis=-1)
    )
    spatial_dims = data["lonc"].dims[:-1]
    coordinates = {dim: data.coords[dim] for dim in spatial_dims if dim in data.coords}
    return xr.DataArray(
        area_values,
        dims=spatial_dims,
        coords=coordinates,
        attrs={"units": "m2", "long_name": "projected pixel area"},
    )


def _area_integral(
    data: xr.Dataset, variable: str, weights: xr.DataArray | None = None
) -> xr.DataArray:
    """Return the area integral of one field using its grid-cell corners.

    ``weights`` scales every cell area, for example by the fraction of the
    cell inside a target window.
    """
    if variable not in data:
        raise ContractError(f"Dataset has no '{variable}' variable.")
    area = _cell_areas_from_lonlat(data)
    if weights is not None:
        area = area * weights
    field = data[variable]
    if not set(area.dims).issubset(field.dims):
        raise ContractError(
            f"'{variable}' does not use the spatial dimensions {list(area.dims)}."
        )
    return (field * area).sum(dim=area.dims, skipna=True)


def _target_window_outline(target: xr.Dataset) -> Polygon:
    """Return the outer boundary of a CDO target window in its projected CRS."""
    x, y = _projected_cell_corners(target)
    outline_x = outline_from_nodes(cell_vertices_to_nodes(x))
    outline_y = outline_from_nodes(cell_vertices_to_nodes(y))
    require_finite(outline_x, outline_y)
    polygon = Polygon(zip(outline_x, outline_y, strict=True))
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ContractError("Target grid corners form an invalid window polygon.")
    return polygon


def _source_window_overlap(source: xr.Dataset, window: Polygon) -> xr.DataArray:
    """Return the fraction of every source cell's area inside the target window."""
    x, y = _projected_cell_corners(source)
    cells = np.asarray(shapely.polygons(np.stack((x, y), axis=-1)))
    areas = _cell_areas_from_lonlat(source)
    outline = shapely.simplify(
        window, _SWATH_SIMPLIFY_TOLERANCE, preserve_topology=True
    )
    if outline.is_empty or not outline.is_valid:
        outline = window
    shapely.prepare(outline)
    overlap = np.asarray(shapely.covers(outline, cells), dtype=np.float64)
    boundary = np.asarray(shapely.intersects(outline, cells), dtype=bool) & (
        overlap == 0
    )
    if boundary.any():
        cell_area = np.asarray(areas.values, dtype=np.float64)[boundary]
        parts = shapely.area(shapely.intersection(outline, cells[boundary]))
        overlap[boundary] = np.clip(
            parts / np.where(cell_area > 0, cell_area, np.inf), 0.0, 1.0
        )
    return xr.DataArray(
        overlap,
        dims=areas.dims,
        coords=areas.coords,
        attrs={"units": "1", "long_name": "cell area fraction inside target window"},
    )


def _spatial_variable(data: xr.Dataset, variable: str) -> xr.DataArray:
    """Return a field whose spatial dimensions match its lon coordinate."""
    if variable not in data:
        raise ContractError(f"Dataset has no '{variable}' variable.")
    if "lon" not in data or data["lon"].ndim != 2:
        raise ContractError("Remapped dataset must contain a two-dimensional 'lon'.")
    spatial_dims = data["lon"].dims
    field = data[variable]
    if not set(spatial_dims).issubset(field.dims):
        raise ContractError(
            f"'{variable}' does not use remapped spatial dimensions {spatial_dims}."
        )
    return field


def validate_conservative_mass(
    source: xr.Dataset | str | Path,
    remapped: xr.Dataset | str | Path,
    target_grid_path: str | Path,
    variable: str,
    rtol: float = 1e-3,
) -> xr.Dataset:
    """Compare the remapped mass with the source mass inside the target window.

    A conservative remap can only conserve the source mass overlapping the
    target window. Source cells beyond the window, which occur with partial
    embedding, are excluded through the area fraction of every cell inside
    the window outline, so the check holds for full and partial embeds alike.

    Args:
        source: Formatted CDO source dataset or its path.
        remapped: CDO remapping result or its path.
        target_grid_path: Target grid used for remapping.
        variable: Areal-density variable to integrate.
        rtol: Maximum accepted relative mass error.

    Returns:
        Dataset with the total source mass, the source mass inside the target
        window, the inside mass fraction, the remapped mass and the relative
        error for every retained leading-dimension coordinate.

    Raises:
        ContractError: If inputs are incompatible or the error exceeds ``rtol``.
    """
    if not np.isfinite(rtol) or rtol < 0:
        raise ContractError("rtol must be a finite nonnegative number.")
    source_ds, close_source = _open_dataset_input(source, decode_times=False)
    remapped_ds, close_remapped = _open_dataset_input(remapped, decode_times=False)
    try:
        with xr.open_dataset(target_grid_path, decode_times=False) as target:
            target_loaded = target[["lonc", "latc"]].load()
        source_mass = _area_integral(source_ds, variable)
        window = _target_window_outline(target_loaded)
        overlap = _source_window_overlap(source_ds, window)
        source_mass_inside = _area_integral(source_ds, variable, weights=overlap)
        target_area = _cell_areas_from_lonlat(target_loaded)
        remapped_variable = _spatial_variable(remapped_ds, variable)
        remapped_mass = (remapped_variable * target_area).sum(
            dim=target_area.dims, skipna=True
        )
        source_mass, source_mass_inside, remapped_mass = xr.align(
            source_mass, source_mass_inside, remapped_mass, join="exact"
        )
        denominator = abs(source_mass_inside)
        relative_error = abs(remapped_mass - source_mass_inside) / denominator.where(
            denominator != 0, 1.0
        )
        relative_error = relative_error.where(
            denominator != 0, abs(remapped_mass - source_mass_inside)
        )
        inside_mass_fraction = source_mass_inside / source_mass.where(source_mass != 0)
        result = xr.Dataset(
            {
                "source_mass": source_mass,
                "source_mass_inside": source_mass_inside,
                "inside_mass_fraction": inside_mass_fraction,
                "remapped_mass": remapped_mass,
                "relative_error": relative_error,
            }
        )
        result["relative_error"].attrs["units"] = "1"
        result["inside_mass_fraction"].attrs["units"] = "1"
        result = result.compute()
        maximum = float(result["relative_error"].max())
        result.attrs.update(
            {
                "variable": variable,
                "relative_tolerance": rtol,
                "maximum_relative_error": maximum,
            }
        )
        if maximum > rtol:
            raise ContractError(
                f"Conservative remapping error for '{variable}' is {maximum:.3g}, "
                f"which exceeds rtol={rtol:.3g}."
            )
        return result
    finally:
        if close_source:
            source_ds.close()
        if close_remapped:
            remapped_ds.close()


def run_cdo_remapcon(
    preparation: CdoPreparation,
    output_path: str | Path,
    *,
    executable: str = "cdo",
    remap_norm: str = "fracarea",
) -> Path:
    """Run ``cdo remapcon`` on prepared files.

    Args:
        preparation: Prepared source and target-grid files.
        output_path: Destination of the remapped file.
        executable: CDO executable name or path.
        remap_norm: Value of CDO's ``REMAP_NORM`` environment variable.

    Returns:
        The remapped file path.

    Raises:
        RegridderUnavailableError: If the executable is not found.
        RuntimeError: If CDO exits with a non-zero status.
    """
    resolved = shutil.which(executable)
    if resolved is None:
        raise RegridderUnavailableError(
            f"CDO executable {executable!r} was not found on PATH."
        )
    output_path = Path(output_path)
    command = [
        resolved,
        f"remapcon,{preparation.target_grid_path}",
        str(preparation.source_path),
        str(output_path),
    ]
    env = {**os.environ, "REMAP_NORM": remap_norm}
    completed = subprocess.run(
        command, env=env, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cdo remapcon failed with status {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return output_path


class CdoRegridder:
    """Conservative regridder delegating to the CDO ``remapcon`` operator.

    Attributes:
        name: Registry name.
        executable: CDO executable name or path.
        workdir: Directory for the intermediate files; a temporary directory
            when omitted.
        keep_files: Keep the intermediate files after the remap.
        remap_norm: CDO ``REMAP_NORM`` normalisation.
        rtol: Accepted relative mass error of the remap.
    """

    name = "cdo"

    def __init__(
        self,
        executable: str = "cdo",
        workdir: str | Path | None = None,
        keep_files: bool = False,
        remap_norm: str = "fracarea",
        rtol: float = 1e-3,
    ) -> None:
        """Configure the regridder."""
        self.executable = executable
        self.workdir = None if workdir is None else Path(workdir)
        self.keep_files = keep_files
        self.remap_norm = remap_norm
        self.rtol = rtol

    @contextmanager
    def _working_directory(self) -> Iterator[Path]:
        """Yield a fresh directory for one remap and clean it up unless kept."""
        if self.workdir is None:
            directory = Path(tempfile.mkdtemp(prefix="spectra-cdo-"))
        else:
            self.workdir.mkdir(parents=True, exist_ok=True)
            directory = self.workdir / f"cdo-{uuid4().hex}"
            directory.mkdir()
        try:
            yield directory
        finally:
            if self.keep_files:
                logger.info("CDO files kept in %s", directory)
            else:
                shutil.rmtree(directory, ignore_errors=True)

    def to_scene(
        self,
        column: xr.DataArray,
        scene: xr.Dataset,
        placement: Placement,
        *,
        window: Window | None = None,
    ) -> xr.DataArray:
        """Place and conservatively remap a column onto the scene grid.

        Args:
            column: ``(y, x)`` plume column with a ``units`` attribute.
            scene: Scene carrying projected ``x``/``y`` pixel centres.
            placement: Explicit anchor placement.
            window: Window the result is cropped or expanded to; the placement
                window when omitted.

        Returns:
            The column on ``(across, along)`` with ``window_*`` and
            ``embedding_*`` provenance and the mass-check results. The CDO
            output is expanded from the preparation window, not from the
            file attributes.
        """
        if shutil.which(self.executable) is None:
            raise RegridderUnavailableError(
                f"CDO executable {self.executable!r} was not found on PATH."
            )
        geometry = SceneGeometry.from_scene(scene)
        scene_placement = place_in_scene(column, geometry, placement=placement)
        variable = str(column.name) if column.name is not None else "column"
        with self._working_directory() as directory:
            preparation = prepare_cdo_files(
                scene_placement,
                geometry,
                directory / "cdo_source.nc",
                directory / "cdo_target_grid.nc",
                variable=variable,
                overwrite=True,
            )
            remapped_path = run_cdo_remapcon(
                preparation,
                directory / "cdo_remapped.nc",
                executable=self.executable,
                remap_norm=self.remap_norm,
            )
            with xr.open_dataset(remapped_path, decode_times=False) as remapped:
                remapped = remapped.load()
            mass = validate_conservative_mass(
                preparation.source_path,
                remapped,
                preparation.target_grid_path,
                variable,
                rtol=self.rtol,
            )
            with xr.open_dataset(preparation.source_path, decode_times=False) as src:
                provenance = {
                    key: value
                    for key, value in src.attrs.items()
                    if key.startswith("embedding_")
                }
        placement_window = preparation.window
        output_window = placement_window if window is None else window
        expanded = expand_window(
            normalise_remapped_dimensions(remapped)[[variable]],
            placement_window,
            geometry,
        )
        result = expanded[variable].isel(output_window.indexers)
        coords: dict[str, Any] = {}
        for dim, selection in output_window.indexers.items():
            if dim in scene.coords:
                coords[dim] = scene[dim].isel({dim: selection})
            else:
                coords[dim] = np.arange(selection.start, selection.stop)
        result = result.assign_coords(coords)
        result.name = variable
        result.attrs = {
            "units": str(column.attrs["units"]),
            "regrid_method": "cdo-remapcon",
            **output_window.attrs("window_"),
            **provenance,
            "regrid_source_mass": float(mass["source_mass"]),
            "regrid_source_mass_inside": float(mass["source_mass_inside"]),
            "regrid_target_mass": float(mass["remapped_mass"]),
            "regrid_inside_mass_fraction": float(mass["inside_mass_fraction"]),
            "regrid_max_relative_mass_error": float(
                mass.attrs["maximum_relative_error"]
            ),
        }
        return result.transpose(*HORIZONTAL_DIMS)

    def to_map(
        self, column: xr.DataArray, grid: MapGrid, placement: Placement
    ) -> xr.DataArray:
        """Not implemented: CDO remapping onto regular map grids is a follow-up."""
        raise NotImplementedError(
            "CdoRegridder.to_map is not implemented; CDO remapping onto regular "
            "map grids is a follow-up. Use the polygon or bucket regridder."
        )


__all__ = [
    "CdoPreparation",
    "CdoRegridder",
    "embedding_attrs",
    "georeferenced_source",
    "prepare_cdo_files",
    "prepare_regular_target_grid",
    "run_cdo_remapcon",
    "temporary_output_path",
    "validate_conservative_mass",
    "write_target_window",
]
