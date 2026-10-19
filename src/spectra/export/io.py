"""Writing generated samples as consolidated Zarr stores or NetCDF files."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import xarray as xr

SAMPLE_CHUNK = 512
# Dimensions chunked in tiles of SAMPLE_CHUNK; other dimensions stay whole.
_SPATIAL_DIMS = ("y", "x", "across", "along")


def _chunk_sizes(dataset: xr.Dataset) -> dict[str, int]:
    """Return one uniform chunk size per dimension of a dataset."""
    return {
        str(dim): min(int(size), SAMPLE_CHUNK) if dim in _SPATIAL_DIMS else int(size)
        for dim, size in dataset.sizes.items()
    }


def write_dataset(
    dataset: xr.Dataset,
    uri: str | Path,
    *,
    overwrite: bool = False,
    load: bool = True,
) -> Path:
    """Write a sample as consolidated Zarr (``.zarr``) or NetCDF.

    Args:
        dataset: Sample to write (any geometry space).
        uri: Destination path; the suffix selects the format.
        overwrite: Replace an existing store or file.
        load: Load the sample into memory first (window-sized samples). With
            ``False`` the data are rechunked uniformly and streamed, which
            suits full-scene radiance cubes.

    Returns:
        The written path.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
    """
    path = Path(uri)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing product: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Every spatial variable gets the same chunk layout, otherwise readers that
    # combine variables (prism's sample derivation) end up with irregular dask
    # chunks Zarr rejects.
    sizes = _chunk_sizes(dataset)
    encoded = dataset.load().copy() if load else dataset.copy().chunk(sizes)
    for name, variable in encoded.variables.items():
        encoded[name].encoding = {}
        if np.issubdtype(variable.dtype, np.datetime64):
            encoded[name].encoding.update({"units": "seconds since 1970-01-01"})
    if path.suffix == ".zarr":
        if path.exists():
            shutil.rmtree(path)
        encoding = {
            name: {"chunks": tuple(sizes[str(dim)] for dim in variable.dims)}
            for name, variable in encoded.data_vars.items()
            if any(dim in _SPATIAL_DIMS for dim in variable.dims)
        }
        encoded.to_zarr(path, mode="w", consolidated=True, encoding=encoding)
    else:
        if path.exists():
            path.unlink()
        encoded.to_netcdf(path)
    return path


__all__ = ["SAMPLE_CHUNK", "write_dataset"]
