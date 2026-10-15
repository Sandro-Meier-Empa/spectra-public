"""Scene windows: expanding remapped columns and merging them into scenes.

Ported from ``aviris4_ch4.regridding.expand_aviris_window`` and
``merge_aviris_with_microhh``; windows are positional index ranges on the
``(across, along)`` sensor grid, given explicitly as a
:class:`~spectra.types.Window` or carried as the ``window_*`` attributes the
regridders write.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.types import SceneGeometry, Window

HORIZONTAL_DIMS = ("across", "along")
WINDOW_ATTR_PREFIX = "window_"
_AUXILIARY_VARIABLES = (
    "x",
    "y",
    "lon",
    "lat",
    "lonc",
    "latc",
    "lon_bnds",
    "lat_bnds",
    "dummy",
)


def _dimension_coordinate(dataset: xr.Dataset, dimension: str) -> np.ndarray:
    """Return a dimension coordinate or zero-based indices."""
    if dimension in dataset.coords and dataset.coords[dimension].dims == (dimension,):
        return np.asarray(dataset.coords[dimension].values)
    return np.arange(dataset.sizes[dimension], dtype=np.int32)


def _require_horizontal_dimensions(data: xr.Dataset, label: str) -> None:
    missing = [dim for dim in HORIZONTAL_DIMS if dim not in data.dims]
    if missing:
        raise ContractError(f"{label} dataset is missing dimensions: {missing}.")


def normalise_remapped_dimensions(data: xr.Dataset) -> xr.Dataset:
    """Return remapped data with ``across``/``along`` horizontal dimensions.

    CDO output keeps the target-grid dimension names, but tools renaming them
    are handled by falling back to the dimensions of a two-dimensional
    ``lon`` variable.
    """
    present = [dim in data.dims for dim in HORIZONTAL_DIMS]
    if all(present):
        return data
    if any(present):
        raise ContractError(
            "Remapped data must contain both 'across' and 'along' dimensions, "
            "or neither."
        )
    if "lon" not in data or data["lon"].ndim != 2:
        raise ContractError(
            "Remapped data must have 'across'/'along' dimensions or a "
            "two-dimensional 'lon' variable."
        )
    spatial_dims = data["lon"].dims
    if spatial_dims[0] == spatial_dims[1]:
        raise ContractError("Remapped spatial dimensions must be distinct.")
    conflicting = [name for name in HORIZONTAL_DIMS if name in data.variables]
    if conflicting:
        raise ContractError(
            "Cannot normalise remapped dimensions because variables use the "
            f"target names: {conflicting}."
        )
    return data.rename(dict(zip(spatial_dims, HORIZONTAL_DIMS, strict=True)))


def _window_attr_names(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}{dim}_{bound}"
        for dim in HORIZONTAL_DIMS
        for bound in ("start", "stop")
    )


def remapped_window(
    scene: xr.Dataset, remapped: xr.Dataset, window: Window | None = None
) -> Window:
    """Return the scene window a remapped dataset covers.

    Args:
        scene: Sensor scene with ``across``/``along`` dimensions.
        remapped: Remapped dataset on the window.
        window: Explicit window; when omitted, the ``window_*`` attributes of
            ``remapped`` are read with :meth:`Window.from_attrs`.

    Returns:
        The window, whose shape matches the remapped dimensions.

    Raises:
        ContractError: If no window is given and the ``window_*`` attributes
            are incomplete, or if the window does not match the remapped
            dimensions or exceeds the scene.
    """
    _require_horizontal_dimensions(scene, "Scene")
    _require_horizontal_dimensions(remapped, "Remapped")
    if window is None:
        names = _window_attr_names(WINDOW_ATTR_PREFIX)
        missing = [name for name in names if name not in remapped.attrs]
        if missing:
            raise ContractError(
                "The embedding window must be given explicitly or carried as "
                f"'{WINDOW_ATTR_PREFIX}*' attributes of the remapped data; "
                f"missing attributes: {missing}."
            )
        window = Window.from_attrs(remapped.attrs, prefix=WINDOW_ATTR_PREFIX)
    for dim, selection in window.indexers.items():
        if selection.stop > scene.sizes[dim]:
            raise ContractError(
                f"The {dim} window [{selection.start}, {selection.stop}) exceeds "
                f"the scene dimension of size {scene.sizes[dim]}."
            )
        if selection.stop - selection.start != remapped.sizes[dim]:
            raise ContractError(
                f"Remapped dimension '{dim}' has size {remapped.sizes[dim]}, but "
                f"the window [{selection.start}, {selection.stop}) has size "
                f"{selection.stop - selection.start}."
            )
    return window


def _suffix_conflicts(
    scene: xr.Dataset, remapped: xr.Dataset, suffix: str
) -> xr.Dataset:
    """Suffix remapped variable names that conflict with scene variables."""
    conflicts = (set(scene.variables) & set(remapped.variables)) - set(HORIZONTAL_DIMS)
    replacements = {name: f"{name}{suffix}" for name in conflicts}
    occupied = set(scene.variables) | set(remapped.variables)
    blocked = sorted(
        replacement for replacement in replacements.values() if replacement in occupied
    )
    if blocked:
        raise ContractError(
            "Cannot suffix conflicting remapped variables because these names "
            f"already exist: {blocked}."
        )
    return remapped.rename(replacements)


def _open_input(value: xr.Dataset | str | Path) -> tuple[xr.Dataset, bool]:
    if isinstance(value, xr.Dataset):
        return value, False
    return xr.open_dataset(value, chunks="auto"), True


def _close_all(datasets: Iterable[xr.Dataset]) -> None:
    for dataset in datasets:
        dataset.close()


def merge_scene_with_column(
    scene: xr.Dataset,
    remapped: xr.Dataset | str | Path,
    *,
    window: Window | None = None,
    suffix: str = "_plume",
) -> xr.Dataset:
    """Merge a scene window with a column remapped onto that window.

    The window is either given explicitly or read from the ``window_*``
    attributes written by the regridders; the remapped dimensions must match
    its shape. A remapped dataset opened from a path stays lazy and is closed
    with the returned dataset.

    Args:
        scene: Sensor scene with ``across``/``along`` dimensions.
        remapped: Remapped dataset or its NetCDF path.
        window: Scene window the remapped data covers; the ``window_*``
            attributes of ``remapped`` when omitted.
        suffix: Suffix for remapped variable names that conflict with the scene.

    Returns:
        The scene cropped to the window and merged with the remapped
        variables; scene attributes take precedence.

    Raises:
        ContractError: If dimensions, the window or variable names are
            incompatible, or if no window is given and the remapped data does
            not carry the ``window_*`` attributes.
    """
    opened: list[xr.Dataset] = []
    try:
        remapped_data, close_remapped = _open_input(remapped)
        if close_remapped:
            opened.append(remapped_data)
        normalised = normalise_remapped_dimensions(remapped_data)
        window = remapped_window(scene, normalised, window)
        scene_window = scene.isel(window.indexers)
        coordinates = {
            dim: _dimension_coordinate(scene, dim)[selection]
            for dim, selection in window.indexers.items()
        }
        scene_window = scene_window.assign_coords(coordinates)
        normalised = normalised.assign_coords(coordinates)
        normalised = _suffix_conflicts(scene_window, normalised, suffix)
        merged = xr.merge(
            (scene_window, normalised),
            compat="no_conflicts",
            join="exact",
            combine_attrs="override",
        )
        merged.attrs = dict(scene.attrs)
        for name, value in remapped_data.attrs.items():
            if name not in merged.attrs:
                merged.attrs[name] = value
    except Exception:
        _close_all(opened)
        raise
    else:
        if opened:
            merged.set_close(lambda: _close_all(opened))
        return merged


def expand_window(
    data: xr.Dataset,
    window: Window,
    sizes: tuple[int, int] | SceneGeometry,
    fill_value: float = 0.0,
) -> xr.Dataset:
    """Expand a windowed result onto the full ``(across, along)`` index grid.

    Args:
        data: Dataset on the window, with ``across``/``along`` dimensions or a
            two-dimensional ``lon`` variable naming them.
        window: Index window the data covers.
        sizes: Full ``(n_across, n_along)`` grid size or the scene geometry;
            dimension coordinates of the geometry are used when present.
        fill_value: Value assigned outside the window.

    Returns:
        Dataset reindexed to the full dimensions, without the CDO auxiliary
        coordinate variables.
    """
    output = normalise_remapped_dimensions(data)
    if isinstance(sizes, SceneGeometry):
        n_across, n_along = sizes.shape
        dim0, dim1 = sizes.spatial_dims
        full = {
            "across": _dimension_coordinate(sizes.x.to_dataset(name="x"), dim0),
            "along": _dimension_coordinate(sizes.x.to_dataset(name="x"), dim1),
        }
    else:
        n_across, n_along = (int(sizes[0]), int(sizes[1]))
        full = {
            "across": np.arange(n_across, dtype=np.int32),
            "along": np.arange(n_along, dtype=np.int32),
        }
    if window.across.stop > n_across or window.along.stop > n_along:
        raise ContractError("Window exceeds the full grid size.")
    if (output.sizes["across"], output.sizes["along"]) != window.shape:
        raise ContractError("Data dimensions do not match the window.")
    output = output.assign_coords(
        across=full["across"][window.across], along=full["along"][window.along]
    )
    output = output.drop_vars([name for name in _AUXILIARY_VARIABLES if name in output])
    return output.reindex(
        across=full["across"], along=full["along"], fill_value=fill_value
    )


__all__ = [
    "expand_window",
    "merge_scene_with_column",
    "normalise_remapped_dimensions",
    "remapped_window",
]
