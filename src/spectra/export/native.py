"""Copy-and-patch export of an embedding into the native radiance file.

The output is a byte copy of the input radiance file in which only the plume
pixels of the attenuated bands are rewritten; every other byte, variable,
group, attribute and storage property stays as it was. The embedding metadata
(:mod:`spectra.contracts.embedded`) is added to the root group. The
Lambert-Beer attenuation is applied here, block by block, to the raw values
read from the copy, with the same function the golden fixtures check, so the
radiance is read once and memory stays bounded for any file size.

The copy is written to a hidden staging file next to the output and published
atomically, so a failed run never leaves a partial output behind.
"""

from __future__ import annotations

import datetime
import errno
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import xarray as xr

import spectra
from spectra.contracts.embedded import (
    ANCHOR_MODE_CODES,
    FLIP_CODES,
    RUN_CONFIG_ATTR,
    SOURCE_DIM,
    SOURCE_VARIABLES,
    STRING_DIM,
    anchor_variable,
    check_not_embedded,
)
from spectra.contracts.errors import ContractError
from spectra.result import ClusterResult, EmbedResult, SourceResult
from spectra.spectral.embedding import EMBEDDING_VERSION, embed_column_in_radiance
from spectra.types import NativeLayout

COMMON_ORDER = ("across", "along", "band")
# Size of the radiance blocks read and written at once.
_BLOCK_BYTES = 64 * 2**20
# Chunk edge of the injected-column variable.
_FIELD_CHUNK = 512


def check_output_target(
    files: Mapping[str, str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Validate the output path against the inputs and return it resolved.

    Raises:
        ContractError: If the output is one of the input files (also through
            symbolic or hard links) or a directory.
        FileExistsError: If the output exists and ``overwrite`` is false.
    """
    target = Path(output).expanduser().resolve()
    for role, path in files.items():
        source = Path(path).expanduser().resolve()
        if target == source or (
            target.exists() and source.exists() and target.samefile(source)
        ):
            raise ContractError(
                f"The output {target} is the input {role!r} file; SPECTRA never "
                "modifies its inputs."
            )
    if target.is_dir():
        raise ContractError(f"The output {target} is a directory.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; set overwrite to replace it.")
    return target


def write_native(
    layout: NativeLayout,
    files: Mapping[str, str | os.PathLike[str]],
    output: str | os.PathLike[str],
    result: EmbedResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the embedded radiance file in the input's native format.

    Args:
        layout: Location and dimension mapping of the radiance variable.
        files: Input files by role; ``files[layout.role]`` is copied.
        output: Output path.
        result: Embedding to apply.
        overwrite: Replace an existing output.

    Returns:
        The resolved output path.

    Raises:
        ContractError: If the output is an input, the input is already
            embedded or not NetCDF4, or the result does not fit the file.
        FileExistsError: If the output exists and ``overwrite`` is false.
        OSError: If the output file system lacks space for the copy.
    """
    if layout.role not in files:
        raise ContractError(f"No {layout.role!r} file to export into.")
    source = Path(files[layout.role]).expanduser().resolve()
    target = check_output_target(files, output, overwrite=overwrite)
    check_not_embedded(source)
    with netCDF4.Dataset(source) as dataset:
        if dataset.data_model != "NETCDF4":
            raise ContractError(
                f"{source} is {dataset.data_model}; only NETCDF4 files can carry "
                "the embedding metadata."
            )
        variable = _variable(dataset, layout.variable)
        _validate(variable, layout, result)
    target.parent.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    free = shutil.disk_usage(target.parent).free
    if free < stat.st_size:
        raise OSError(
            errno.ENOSPC,
            f"{target.parent} has {free} bytes free; the copy needs {stat.st_size}.",
        )
    staging = target.parent / f".{target.name}.spectra-{uuid.uuid4().hex}.partial"
    try:
        _copy(source, staging, stat)
        with netCDF4.Dataset(staging, "r+") as dataset:
            variable = _variable(dataset, layout.variable)
            variable.set_auto_maskandscale(False)
            for cluster in result.clusters:
                _patch_cluster(variable, layout, cluster, result.gas)
            _write_metadata(dataset, variable, layout, result, source, stat)
        with staging.open("rb+") as handle:
            os.fsync(handle.fileno())
        _publish(staging, target, overwrite=overwrite)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return target


def _copy(source: Path, staging: Path, stat: os.stat_result) -> None:
    """Copy the input byte for byte and verify it did not change meanwhile."""
    shutil.copyfile(source, staging)
    copied = source.stat()
    if (copied.st_size, copied.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ContractError(f"{source} changed while it was copied.")


def _variable(dataset: netCDF4.Dataset, path: str) -> netCDF4.Variable:
    """Return the variable at a ``/``-separated path."""
    *groups, name = path.strip("/").split("/")
    node = dataset
    for group in groups:
        if group not in node.groups:
            raise ContractError(f"Group {group!r} of {path!r} not found.")
        node = node.groups[group]
    if name not in node.variables:
        raise ContractError(f"Variable {path!r} not found.")
    return node.variables[name]


def _common_dims(variable: netCDF4.Variable, layout: NativeLayout) -> list[str]:
    if set(variable.dimensions) != set(layout.dims):
        raise ContractError(
            f"{variable.name} has dimensions {variable.dimensions}; the layout "
            f"expects {sorted(layout.dims)}."
        )
    return [layout.dims[dim] for dim in variable.dimensions]


def _validate(
    variable: netCDF4.Variable, layout: NativeLayout, result: EmbedResult
) -> None:
    """Require the result to fit the radiance variable."""
    common = _common_dims(variable, layout)
    sizes = dict(zip(common, variable.shape, strict=True))
    if not result.sources:
        raise ContractError("The result embeds no plume.")
    if variable.dtype.kind != "f":
        raise ContractError(
            f"{variable.name} is {variable.dtype}; only floating-point radiance "
            "is supported."
        )
    windows = [cluster.window for cluster in result.clusters] + [
        source.window for source in result.sources
    ]
    for window in windows:
        if window.across.stop > sizes["across"] or window.along.stop > sizes["along"]:
            raise ContractError(
                f"Window {window} lies outside the scene of across={sizes['across']}, "
                f"along={sizes['along']}."
            )
    for cluster in result.clusters:
        if cluster.bands.stop > sizes["band"]:
            raise ContractError(f"Cluster {cluster.index} names bands beyond the cube.")


def _fill_values(variable: netCDF4.Variable) -> list[Any]:
    """Return the raw values marking missing radiance."""
    values = []
    for name in ("_FillValue", "missing_value"):
        if name in variable.ncattrs():
            values.extend(np.atleast_1d(variable.getncattr(name)).tolist())
    values.append(netCDF4.default_fillvals[variable.dtype.str[1:]])
    return [value for value in values if not np.isnan(value)]


def _patch_cluster(
    variable: netCDF4.Variable, layout: NativeLayout, cluster: ClusterResult, gas: str
) -> None:
    """Attenuate the plume pixels of one cluster in place, block by block.

    Only pixels with a positive column and valid radiance, in the cluster's
    bands, are rewritten; every other element is written back from the raw
    values just read, which keeps its bytes.
    """
    common = _common_dims(variable, layout)
    order = [common.index(dim) for dim in COMMON_ORDER]
    inverse = [order.index(axis) for axis in range(3)]
    window = cluster.window
    slices = {"across": window.across, "along": window.along, "band": cluster.bands}
    outer = common[0]
    inner_elements = 1
    for dim in common[1:]:
        span = slices[dim]
        inner_elements *= span.stop - span.start
    step = max(1, _BLOCK_BYTES // max(1, inner_elements * variable.dtype.itemsize))
    fill_values = _fill_values(variable)
    wavelength = np.asarray(cluster.lut["wavelength"].values)
    outer_span = slices[outer]
    for start in range(outer_span.start, outer_span.stop, step):
        block = dict(slices)
        block[outer] = slice(start, min(start + step, outer_span.stop))
        across = slice(
            block["across"].start - window.across.start,
            block["across"].stop - window.across.start,
        )
        along = slice(
            block["along"].start - window.along.start,
            block["along"].stop - window.along.start,
        )
        column = cluster.column[across, along]
        plume = column > 0
        if not plume.any():
            continue
        selection = tuple(block[dim] for dim in common)
        raw = np.asarray(variable[selection])
        view = raw.transpose(order)
        bad = ~np.isfinite(view)
        for value in fill_values:
            bad |= view == value
        radiance = xr.DataArray(
            view,
            dims=COMMON_ORDER,
            coords={"wavelength": ("band", wavelength)},
        )
        attenuated = embed_column_in_radiance(
            radiance,
            xr.DataArray(column, dims=("across", "along"), attrs={"units": "kg m-2"}),
            cluster.lut,
            gas=gas.lower(),
        ).values
        patch = plume[..., None] & ~bad
        patched = view.copy()
        patched[patch] = attenuated[patch].astype(view.dtype)
        variable[selection] = np.ascontiguousarray(patched.transpose(inverse))


def _write_metadata(
    dataset: netCDF4.Dataset,
    variable: netCDF4.Variable,
    layout: NativeLayout,
    result: EmbedResult,
    source: Path,
    stat: os.stat_result,
) -> None:
    """Add the ``source`` dimension, per-plume variables and ``spectra_*`` attrs."""
    spatial = [dim for dim in variable.dimensions if layout.dims[dim] != "band"]
    y_dim, x_dim = spatial
    dataset.createDimension(SOURCE_DIM, len(result.sources))
    ny = len(dataset.dimensions[y_dim])
    nx = len(dataset.dimensions[x_dim])
    field = dataset.createVariable(
        result.injected_name,
        "f4",
        (SOURCE_DIM, y_dim, x_dim),
        zlib=True,
        complevel=4,
        shuffle=True,
        chunksizes=(1, min(ny, _FIELD_CHUNK), min(nx, _FIELD_CHUNK)),
        fill_value=False,
    )
    field.units = "kg m-2"
    field.long_name = f"injected vertical {result.gas.upper()} column"
    field.column_convention = "vertical"
    for position, plume in enumerate(result.sources):
        _write_field(field, position, plume, layout, y_dim, nx)

    clusters = {cluster.index: cluster for cluster in result.clusters}
    values: dict[str, Sequence[Any]] = {
        "source_plume_file": [plume.plume_file for plume in result.sources],
        "source_case_id": [plume.case_id for plume in result.sources],
        "source_anchor_mode": [
            ANCHOR_MODE_CODES[plume.anchor_mode] for plume in result.sources
        ],
        "source_time_index": [plume.time_index for plume in result.sources],
        "source_emission_rate": [plume.emission_rate_kg_s for plume in result.sources],
        "source_emission_scale": [plume.emission_scale for plume in result.sources],
        "source_lon": [plume.lon for plume in result.sources],
        "source_lat": [plume.lat for plume in result.sources],
        "source_rotation": [plume.rotation_deg for plume in result.sources],
        "source_flip": [FLIP_CODES[plume.flip] for plume in result.sources],
        "source_injected_mass": [plume.injected_mass_kg for plume in result.sources],
        "source_outside_fraction": [plume.outside_fraction for plume in result.sources],
        "source_cluster": [plume.cluster for plume in result.sources],
        "source_lut_sza": [
            float(clusters[plume.cluster].geometry.sza) for plume in result.sources
        ],
        "source_lut_vza": [
            float(clusters[plume.cluster].geometry.vza) for plume in result.sources
        ],
        "source_lut_amf": [clusters[plume.cluster].amf for plume in result.sources],
        "source_lut_surface_elevation": [
            clusters[plume.cluster].model_surface_elevation for plume in result.sources
        ],
    }
    strings = [
        str(value).encode()
        for name, (dtype, _, _) in SOURCE_VARIABLES.items()
        if dtype == "S1"
        for value in values[name]
    ]
    dataset.createDimension(STRING_DIM, max([1, *map(len, strings)]))
    for name, (dtype, units, long_name) in SOURCE_VARIABLES.items():
        _write_source_variable(dataset, name, dtype, values[name], units, long_name)
    for name, codes in (
        ("source_flip", FLIP_CODES),
        ("source_anchor_mode", ANCHOR_MODE_CODES),
    ):
        dataset.variables[name].flag_values = np.array(
            sorted(codes.values()), dtype="i1"
        )
        dataset.variables[name].flag_meanings = " ".join(
            sorted(codes, key=codes.__getitem__)
        )
    for dim in spatial:
        common = layout.dims[dim]
        _write_source_variable(
            dataset,
            anchor_variable(dim),
            "i4",
            [getattr(plume, f"anchor_{common}") for plume in result.sources],
            None,
            f"{dim} index of the anchor pixel ({common})",
        )

    wavelengths = [
        np.asarray(cluster.lut["wavelength"].values, dtype=float)
        for cluster in result.clusters
    ]
    attrs: dict[str, Any] = {
        "spectra_version": spectra.__version__,
        "spectra_embedding_version": EMBEDDING_VERSION,
        "spectra_created": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        "spectra_gas": result.gas.lower(),
        "spectra_column_convention": "vertical",
        "spectra_source_radiance_path": str(source),
        "spectra_source_radiance_size": int(stat.st_size),
        "spectra_source_radiance_mtime": datetime.datetime.fromtimestamp(
            stat.st_mtime, tz=datetime.UTC
        ).isoformat(timespec="seconds"),
        "spectra_n_sources": len(result.sources),
        "spectra_n_clusters": len(result.clusters),
    }
    if wavelengths:
        attrs["spectra_attenuated_wavelength_min_nm"] = min(
            w.min() for w in wavelengths
        )
        attrs["spectra_attenuated_wavelength_max_nm"] = max(
            w.max() for w in wavelengths
        )
    attrs.update(result.attrs)
    for name, value in attrs.items():
        _set_attribute(dataset, name, value)


def _write_field(
    field: netCDF4.Variable,
    position: int,
    plume: SourceResult,
    layout: NativeLayout,
    y_dim: str,
    nx: int,
) -> None:
    """Write one plume's injected column over the whole scene, in row blocks."""
    rows_common = layout.dims[y_dim]
    window = plume.window
    injected = plume.injected if rows_common == "across" else plume.injected.T
    rows = window.across if rows_common == "across" else window.along
    columns = window.along if rows_common == "across" else window.across
    ny = field.shape[1]
    for start in range(0, ny, _FIELD_CHUNK):
        stop = min(start + _FIELD_CHUNK, ny)
        block = np.zeros((stop - start, nx), dtype=np.float32)
        low, high = max(start, rows.start), min(stop, rows.stop)
        if low < high:
            block[low - start : high - start, columns] = injected[
                low - rows.start : high - rows.start
            ]
        field[position, start:stop, :] = block


def _write_source_variable(
    dataset: netCDF4.Dataset,
    name: str,
    dtype: str,
    values: Sequence[Any],
    units: str | None,
    long_name: str,
) -> None:
    if dtype == "S1":
        width = len(dataset.dimensions[STRING_DIM])
        variable = dataset.createVariable(name, "S1", (SOURCE_DIM, STRING_DIM))
        variable._Encoding = "utf-8"
        variable[:] = np.array([str(value) for value in values], dtype=f"U{width}")
    else:
        variable = dataset.createVariable(name, dtype, (SOURCE_DIM,))
        variable[:] = np.asarray(values, dtype=dtype)
    variable.long_name = long_name
    if units is not None:
        variable.units = units


def _set_attribute(dataset: netCDF4.Dataset, name: str, value: object) -> None:
    """Write one global attribute with an explicit netCDF type."""
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        dataset.setncattr(name, np.int8(value))
    elif isinstance(value, (int, np.integer)):
        dataset.setncattr(name, np.int64(value))
    elif isinstance(value, (float, np.floating)):
        dataset.setncattr(name, np.float64(value))
    elif isinstance(value, str):
        dataset.setncattr_string(name, value)
    elif isinstance(value, Mapping):
        dataset.setncattr_string(name, json.dumps(value, sort_keys=True, default=str))
    elif isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind in "iuf" and array.ndim == 1:
            dataset.setncattr(name, array.astype(np.float64))
        else:
            dataset.setncattr_string(name, json.dumps(np.asarray(value).tolist()))
    else:
        dataset.setncattr_string(name, str(value))


def _publish(staging: Path, target: Path, *, overwrite: bool) -> None:
    """Move the finished staging file into place atomically."""
    if overwrite:
        staging.replace(target)
        return
    try:
        os.link(staging, target)
    except FileExistsError:
        raise FileExistsError(f"{target} appeared while it was written.") from None
    except OSError:
        # File systems without hard links: replace unless the target appeared.
        if target.exists():
            raise FileExistsError(f"{target} appeared while it was written.") from None
        staging.replace(target)
        return
    staging.unlink()


__all__ = ["RUN_CONFIG_ATTR", "check_output_target", "write_native"]
