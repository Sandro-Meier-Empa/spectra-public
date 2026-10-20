"""Read and write ``PlumeColumn`` datasets as NetCDF."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import xarray as xr

from spectra.contracts.plume_column import validate_plume_column

_TIME_ENCODING = {
    "units": "seconds since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "float64",
}
_COLUMN_VARIABLES = ("column", "column_below_altitude", "concentration")


def open_plume_column(
    uri: str | Path, *, chunks: Mapping[str, int] | str | None = None
) -> xr.Dataset:
    """Open and validate a plume column stored as NetCDF.

    Args:
        uri: Path of the NetCDF file.
        chunks: Dask chunking passed to xarray; ``None`` loads lazily without
            Dask.

    Returns:
        Validated dataset following the ``PlumeColumn`` contract.

    Raises:
        FileNotFoundError: If ``uri`` does not exist.
        ContractError: If the stored dataset violates the contract.
    """
    path = Path(uri)
    if not path.exists():
        raise FileNotFoundError(f"Plume column {path} does not exist.")
    dataset = xr.open_dataset(path, engine="netcdf4", chunks=chunks)
    validate_plume_column(dataset)
    return dataset


def _prepare_for_writing(dataset: xr.Dataset) -> tuple[xr.Dataset, dict[str, dict]]:
    """Return a copy with float32 columns and the matching encoding."""
    prepared = dataset.copy()
    # NetCDF attributes have no boolean type; store flags as integers.
    prepared.attrs = {
        k: int(v) if isinstance(v, bool) else v for k, v in prepared.attrs.items()
    }
    for name in prepared.variables:
        prepared[name].encoding = {}
        prepared[name].attrs = {
            k: int(v) if isinstance(v, bool) else v
            for k, v in prepared[name].attrs.items()
        }
    encoding: dict[str, dict] = {}
    for name in _COLUMN_VARIABLES:
        if name in prepared:
            attrs = dict(prepared[name].attrs)
            prepared[name] = prepared[name].astype("float32")
            prepared[name].attrs = attrs
            encoding[name] = {"dtype": "float32"}
    # CF time units must come from the encoding, not from the attributes.
    time_attrs = dict(prepared["time"].attrs)
    for key in ("units", "calendar"):
        time_attrs.pop(key, None)
    prepared["time"].attrs = time_attrs
    encoding["time"] = dict(_TIME_ENCODING)
    return prepared, encoding


def write_plume_column(
    dataset: xr.Dataset, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Validate a plume column and write it as compressed NetCDF.

    The file is written next to the target and moved into place, so a dataset
    opened from ``path`` can be rewritten with ``overwrite=True``.

    Args:
        dataset: Dataset following the ``PlumeColumn`` contract.
        path: Target file.
        overwrite: Whether an existing target may be replaced.

    Returns:
        Path of the written file.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is false.
        ContractError: If the dataset violates the contract.
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists. Use overwrite=True to replace it."
        )
    validate_plume_column(dataset)
    prepared, encoding = _prepare_for_writing(dataset)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    temporary = staging / target.name
    try:
        ny, nx = prepared.sizes["y"], prepared.sizes["x"]
        for name in _COLUMN_VARIABLES:
            if name in encoding:
                leading = (1,) * (prepared[name].ndim - 2)
                encoding[name].update(
                    {"zlib": True, "complevel": 4, "chunksizes": (*leading, ny, nx)}
                )
        prepared.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        temporary.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


__all__ = ["open_plume_column", "write_plume_column"]
