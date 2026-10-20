"""Conformance checks for sensor adapters.

The checks raise ``AssertionError`` with a message naming the adapter and the
violated requirement, so they run inside pytest or ad hoc on a real adapter
and real files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.contracts.sensor_scene import (
    PLATFORMS,
    SENSOR_DIMS,
    SPATIAL_DIMS,
    validate_sensor_scene,
)
from spectra.protocols import SensorAdapter
from spectra.sensors.native import split_variable
from spectra.types import SpectralResponse, ViewingGeometry, Window


def _check(condition: bool, message: str) -> None:
    """Raise ``AssertionError(message)`` unless ``condition`` holds."""
    if not condition:
        raise AssertionError(message)


def _native_pixels(
    path: Path, adapter: SensorAdapter, pixels: list[tuple[int, int]]
) -> np.ndarray:
    """Read spectra of ``(across, along)`` pixels straight from the native file."""
    layout = adapter.LAYOUT
    group, name = split_variable(layout.variable)
    with netCDF4.Dataset(path) as dataset:
        node = dataset if group is None else dataset[group]
        variable = node.variables[name]
        common = [layout.dims[dim] for dim in variable.dimensions]
        spectra = []
        for across, along in pixels:
            index = {"across": across, "along": along, "band": slice(None)}
            values = variable[tuple(index[dim] for dim in common)]
            spectra.append(np.ma.filled(values.astype(float), np.nan))
    return np.asarray(spectra, dtype=float)


def run_adapter_conformance(
    adapter: object,
    files: Mapping[str, str | os.PathLike[str]],
    *,
    n_pixels: int = 16,
    seed: int = 0,
) -> xr.Dataset:
    """Check that an adapter opens one scene according to the contracts.

    Besides the ``SensorScene`` contract, sampled radiance spectra of the
    scene must equal a direct read of the native file through the adapter's
    ``LAYOUT``, which proves import and export use the same mapping.

    Args:
        adapter: Candidate sensor adapter.
        files: Files of one scene by role.
        n_pixels: Number of pixels sampled.
        seed: Seed of the pixel sample.

    Returns:
        The opened scene.

    Raises:
        AssertionError: Naming the first violated requirement.
    """
    _check(
        isinstance(adapter, SensorAdapter),
        f"{type(adapter).__name__} does not implement the SensorAdapter protocol.",
    )
    assert isinstance(adapter, SensorAdapter)
    tag = f"[{adapter.name}]"
    _check(adapter.platform in PLATFORMS, f"{tag} platform must be in {PLATFORMS}.")
    _check(
        bool(adapter.ROLES.get(adapter.LAYOUT.role)),
        f"{tag} the LAYOUT role must be a required role.",
    )
    scene = adapter.open_scene({role: Path(path) for role, path in files.items()})
    try:
        validate_sensor_scene(scene, require_lazy=True)
    except ContractError as error:
        raise AssertionError(
            f"{tag} violates the SensorScene contract: {error}"
        ) from error
    _check(scene.attrs["sensor"] == adapter.name, f"{tag} attr 'sensor' != name.")
    _check(
        scene.attrs["platform"] == adapter.platform,
        f"{tag} attr 'platform' != adapter.platform.",
    )
    rng = np.random.default_rng(seed)
    n_across, n_along = scene.sizes["across"], scene.sizes["along"]
    pixels = [
        (int(rng.integers(0, n_across)), int(rng.integers(0, n_along)))
        for _ in range(n_pixels)
    ]
    expected = _native_pixels(Path(files[adapter.LAYOUT.role]), adapter, pixels)
    radiance = scene["radiance"].transpose(*SENSOR_DIMS)
    for (across, along), native in zip(pixels, expected, strict=True):
        opened = np.asarray(
            radiance.isel(across=across, along=along).values, dtype=float
        )
        _check(
            np.array_equal(opened, native, equal_nan=True),
            f"{tag} scene radiance at (across={across}, along={along}) differs "
            "from the native file read through LAYOUT.",
        )
    valid = scene["valid"].transpose(*SPATIAL_DIMS)
    for across, along in pixels:
        if not bool(valid.isel(across=across, along=along)):
            continue
        for name in ("sza", "vza", "surface_elevation", "sensor_altitude"):
            value = scene[name]
            if value.ndim:
                value = value.isel(across=across, along=along)
            _check(
                bool(np.isfinite(float(value))),
                f"{tag} {name} is not finite at a valid pixel.",
            )
        area = float(scene["pixel_area"].isel(across=across, along=along))
        _check(np.isfinite(area) and area > 0, f"{tag} pixel_area must be positive.")
    window = Window(across=slice(0, min(4, n_across)), along=slice(0, min(4, n_along)))
    try:
        SpectralResponse.from_scene(scene)
        ViewingGeometry.from_scene(scene, window=window)
    except ContractError as error:
        raise AssertionError(
            f"{tag} the scene does not describe a spectral response and viewing "
            f"geometry: {error}"
        ) from error
    return scene


__all__ = ["run_adapter_conformance"]
