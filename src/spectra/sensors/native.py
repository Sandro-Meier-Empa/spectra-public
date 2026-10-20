"""Helpers shared by the sensor adapters.

:func:`open_native_radiance` maps the radiance variable onto the common
``(across, along, band)`` dimensions through the adapter's
:class:`~spectra.types.NativeLayout`, the same mapping the exporter uses to
write the attenuated pixels back. :func:`resolve_files` checks the files a
user lists by role; adapters never look for files themselves.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.contracts.sensor_scene import SENSOR_DIMS
from spectra.types import NativeLayout

_COVERAGE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def resolve_files(
    roles: Mapping[str, bool],
    files: Mapping[str, str | os.PathLike[str]],
    *,
    base: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    """Return the files of one scene as resolved paths, checked against roles.

    Args:
        roles: Role to required flag of the adapter.
        files: File path by role, as given by the user.
        base: Directory relative paths are resolved against; the working
            directory when omitted.

    Returns:
        Resolved path by role.

    Raises:
        ContractError: If a role is unknown or missing, a path is not an
            existing regular file, or two roles name the same file.
    """
    unknown = sorted(set(files) - set(roles))
    if unknown:
        raise ContractError(f"Unknown file roles {unknown}; expected {sorted(roles)}.")
    missing = sorted(
        role for role, required in roles.items() if required and role not in files
    )
    if missing:
        raise ContractError(f"Missing required file roles {missing}.")
    root = Path(base) if base is not None else Path.cwd()
    resolved: dict[str, Path] = {}
    for role, value in files.items():
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise ContractError(f"The {role!r} file {path} does not exist.")
        resolved[role] = path
    seen: dict[Path, str] = {}
    for role, path in resolved.items():
        if path in seen:
            raise ContractError(
                f"Roles {seen[path]!r} and {role!r} name the same file {path}."
            )
        seen[path] = role
    return resolved


def split_variable(path: str) -> tuple[str | None, str]:
    """Return ``(group, name)`` of a ``/``-separated variable path."""
    *groups, name = path.strip("/").split("/")
    return ("/".join(groups) or None), name


def aligned_chunks(
    path: Path,
    variable: str,
    requested: Mapping[str, int],
    *,
    group: str | None = None,
) -> dict[str, int]:
    """Return dask chunks rounded up to multiples of the on-disk chunks.

    Args:
        path: NetCDF file.
        variable: Variable name in ``group``.
        requested: Chunk size per native dimension; -1 means the whole
            dimension.
        group: Group of the variable.

    Returns:
        Chunks per native dimension, aligned to the storage chunks of a
        chunked variable, unchanged for contiguous ones.
    """
    with xr.open_dataset(path, group=group, decode_cf=False) as dataset:
        encoding = dataset[variable].encoding
        dims = dataset[variable].dims
        sizes = dataset[variable].shape
    storage = encoding.get("chunksizes")
    chunks: dict[str, int] = {}
    for index, (dim, size) in enumerate(zip(dims, sizes, strict=True)):
        wanted = int(requested.get(str(dim), -1))
        if wanted <= 0 or wanted >= size:
            chunks[str(dim)] = -1
            continue
        if storage:
            disk = int(storage[index])
            wanted = int(np.ceil(wanted / disk)) * disk
        chunks[str(dim)] = -1 if wanted >= size else wanted
    return chunks


def open_native_radiance(
    path: Path, layout: NativeLayout, *, chunks: Mapping[str, int]
) -> xr.DataArray:
    """Open the radiance lazily on the common ``(across, along, band)`` dims.

    Args:
        path: Radiance file.
        layout: Location and dimension mapping of the radiance variable.
        chunks: Dask chunks in common dimension names.

    Returns:
        Dask-backed radiance with the fill value masked to NaN.
    """
    group, name = split_variable(layout.variable)
    native = {layout.native_dims[dim]: size for dim, size in chunks.items()}
    dataset = xr.open_dataset(
        path,
        group=group,
        chunks=aligned_chunks(path, name, native, group=group),
    )
    if name not in dataset:
        raise ContractError(f"{path} has no {layout.variable!r} variable.")
    radiance = dataset[name]
    if set(radiance.dims) != set(layout.dims):
        raise ContractError(
            f"{layout.variable!r} in {path} has dimensions {radiance.dims}; "
            f"expected {sorted(layout.dims)}."
        )
    radiance = radiance.rename({dim: layout.dims[str(dim)] for dim in radiance.dims})
    radiance = radiance.drop_vars(
        [name for name in radiance.coords if name not in SENSOR_DIMS]
    )
    return radiance.transpose(*SENSOR_DIMS)


def to_scene_dims(variable: xr.DataArray, layout: NativeLayout) -> xr.DataArray:
    """Rename native spatial dimensions through ``layout`` and order them.

    Returns:
        The variable on ``(across, along, ...)``.
    """
    renames = {
        dim: layout.dims[str(dim)] for dim in variable.dims if dim in layout.dims
    }
    return variable.rename(renames).transpose("across", "along", ...)


def global_attrs(path: Path, keys: Iterable[str]) -> dict[str, str]:
    """Return root-group attributes of a file as strings, empty when missing."""
    with xr.open_dataset(path) as dataset:
        return {key: str(dataset.attrs.get(key, "")) for key in keys}


def coverage_start(value: str, path: Path) -> np.datetime64:
    """Return a ``time_coverage_start`` attribute as a naive UTC datetime64.

    Both the ``Z`` and the ``+0000`` suffix of NASA L1B granules are accepted.

    Raises:
        ContractError: If the value does not parse.
    """
    try:
        stamp = datetime.datetime.strptime(value, _COVERAGE_TIME_FORMAT)
    except ValueError as error:
        raise ContractError(
            f"{path} has no parsable time_coverage_start ({value!r})."
        ) from error
    naive = stamp.astimezone(datetime.UTC).replace(tzinfo=None)
    return np.datetime64(naive, "ns")


__all__ = [
    "aligned_chunks",
    "coverage_start",
    "global_attrs",
    "open_native_radiance",
    "resolve_files",
    "split_variable",
    "to_scene_dims",
]
