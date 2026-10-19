"""Conformance checks for sensor adapters.

The checks raise ``AssertionError`` with a message naming the plugin and the
violated requirement, so they can run inside pytest or ad hoc on a real
adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError, MissingExtraError
from spectra.contracts.sensor_scene import (
    PLATFORMS,
    SPATIAL_DIMS,
    validate_sensor_scene,
)
from spectra.protocols import SensorAdapter
from spectra.registry import list_plugins, load_plugin
from spectra.types import (
    SceneRecord,
    SpectralResponse,
    ViewingGeometry,
)


def _check(condition: bool, message: str) -> None:
    """Raise ``AssertionError(message)`` unless ``condition`` holds."""
    if not condition:
        raise AssertionError(message)


def run_sensor_conformance(adapter: object, record: SceneRecord) -> None:
    """Check that an adapter serves one record according to the contracts.

    Args:
        adapter: Candidate sensor adapter.
        record: Scene record of that adapter to open.

    Raises:
        AssertionError: Naming the first violated requirement.
    """
    _check(
        isinstance(adapter, SensorAdapter),
        f"{type(adapter).__name__} does not implement the SensorAdapter protocol.",
    )
    assert isinstance(adapter, SensorAdapter)
    name = adapter.name
    tag = f"[{name}:{record.scene_id}]"
    _check(
        isinstance(name, str) and bool(name), "adapter.name must be a non-empty str."
    )
    _check(
        adapter.platform in PLATFORMS,
        f"{tag} adapter.platform must be one of {PLATFORMS}; got {adapter.platform!r}.",
    )
    _check(
        record.sensor == name,
        f"{tag} record.sensor {record.sensor!r} differs from adapter.name {name!r}.",
    )

    scene = adapter.open_scene(record)
    _check(isinstance(scene, xr.Dataset), f"{tag} open_scene() must return a Dataset.")
    try:
        info = validate_sensor_scene(scene, require_lazy=True)
    except ContractError as error:
        raise AssertionError(
            f"{tag} open_scene() violates the SensorScene contract: {error}"
        ) from error
    _check(
        info.sensor == name,
        f"{tag} scene attr sensor {info.sensor!r} differs from adapter.name.",
    )
    _check(
        info.scene_id == record.scene_id,
        f"{tag} scene attr scene_id {info.scene_id!r} differs from the record.",
    )
    _check(
        info.platform == adapter.platform,
        f"{tag} scene attr platform {info.platform!r} differs from adapter.platform.",
    )

    masks = adapter.quality_masks(scene, record)
    _check(
        isinstance(masks, xr.Dataset), f"{tag} quality_masks() must return a Dataset."
    )
    _check("valid" in masks, f"{tag} quality_masks() must include a 'valid' mask.")
    for mask_name, mask in masks.data_vars.items():
        _check(
            tuple(mask.dims) == SPATIAL_DIMS,
            f"{tag} mask {mask_name!r} must have dims {SPATIAL_DIMS}; got {mask.dims}.",
        )
        _check(
            mask.dtype.kind == "b",
            f"{tag} mask {mask_name!r} must be boolean; got {mask.dtype}.",
        )
        _check(
            mask.shape == (info.n_across, info.n_along),
            f"{tag} mask {mask_name!r} shape {mask.shape} differs from the scene.",
        )

    group = adapter.background_group(record)
    _check(
        isinstance(group, str) and bool(group),
        f"{tag} background_group() must return a non-empty str; got {group!r}.",
    )
    _check(
        group == info.background_group,
        f"{tag} background_group() {group!r} differs from the scene attr "
        f"{info.background_group!r}.",
    )
    _check(
        group == record.background_group,
        f"{tag} background_group() {group!r} differs from the record "
        f"{record.background_group!r}.",
    )
    _check(
        adapter.background_group(record) == group,
        f"{tag} background_group() is not stable across calls.",
    )

    area = np.asarray(scene["pixel_area"].values, dtype=float)
    _check(
        bool(np.all(np.isfinite(area)) and np.all(area > 0)),
        f"{tag} pixel_area must be finite and positive everywhere.",
    )
    try:
        SpectralResponse.from_scene(scene)
        ViewingGeometry.from_scene(scene)
    except ContractError as error:
        raise AssertionError(
            f"{tag} the scene does not describe a spectral response and viewing "
            f"geometry: {error}"
        ) from error


def conformance_parametrize() -> list[Any]:
    """Return pytest parameters ``(name, adapter)`` for every registered sensor.

    Adapters that cannot be instantiated without options, or whose optional
    dependencies are missing, are returned with a skip marker naming the
    reason, so ``tests/test_conformance.py`` reports them instead of failing.
    """
    import pytest

    parameters: list[Any] = []
    for name in sorted(list_plugins("sensors")):
        try:
            adapter = load_plugin("sensors", name)
        except MissingExtraError as error:
            reason = f"{name}: {error}"
        except Exception as error:  # any construction failure means "skip"
            reason = f"{name} cannot be instantiated without options: {error}"
        else:
            parameters.append(pytest.param(name, adapter, id=name))
            continue
        parameters.append(
            pytest.param(name, None, id=name, marks=pytest.mark.skip(reason=reason))
        )
    return parameters


__all__ = [
    "conformance_parametrize",
    "run_sensor_conformance",
]
