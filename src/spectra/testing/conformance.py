"""Conformance checks for sensor adapters and retrievers.

The checks raise ``AssertionError`` with a message naming the plugin and the
violated requirement, so they can run inside pytest or ad hoc on a real
adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from spectra.contracts.column_product import validate_column_product
from spectra.contracts.errors import ContractError, MissingExtraError
from spectra.contracts.sensor_scene import (
    PLATFORMS,
    SPATIAL_DIMS,
    validate_sensor_scene,
)
from spectra.protocols import Retriever, SensorAdapter, SupportsProduct
from spectra.registry import list_plugins, load_plugin
from spectra.types import (
    RetrievalSettings,
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

    if isinstance(adapter, SupportsProduct) and record.has_product:
        product = adapter.open_product(record)
        _check(
            isinstance(product, xr.Dataset),
            f"{tag} open_product() must return a Dataset.",
        )
        try:
            product_info = validate_column_product(product)
        except ContractError as error:
            raise AssertionError(
                f"{tag} open_product() violates the ColumnProduct contract: {error}"
            ) from error
        _check(
            product_info.sensor == name and product_info.scene_id == record.scene_id,
            f"{tag} product attrs sensor/scene_id differ from the record.",
        )
        if product_info.geometry_space == "sensor":
            for dim in SPATIAL_DIMS:
                _check(
                    product.sizes[dim] == scene.sizes[dim],
                    f"{tag} product size along {dim!r} differs from the scene.",
                )


def run_retriever_conformance(
    retriever: object,
    scene: xr.Dataset,
    target: xr.Dataset,
    settings: RetrievalSettings,
) -> xr.Dataset:
    """Check that a retriever turns a scene into a valid column product.

    Args:
        retriever: Candidate retriever.
        scene: ``SensorScene`` dataset.
        target: Unit absorption spectrum used as retrieval target.
        settings: Retrieval settings.

    Returns:
        The retrieved product.

    Raises:
        AssertionError: Naming the first violated requirement.
    """
    _check(
        isinstance(retriever, Retriever),
        f"{type(retriever).__name__} does not implement the Retriever protocol.",
    )
    assert isinstance(retriever, Retriever)
    tag = f"[{retriever.name}]"
    product = retriever.retrieve(scene, target, settings=settings)
    _check(isinstance(product, xr.Dataset), f"{tag} retrieve() must return a Dataset.")
    try:
        info = validate_column_product(product)
    except ContractError as error:
        raise AssertionError(
            f"{tag} retrieve() violates the ColumnProduct contract: {error}"
        ) from error
    _check(
        info.geometry_space == "sensor",
        f"{tag} retrieve() must return a sensor-space product.",
    )
    for dim in SPATIAL_DIMS:
        _check(
            product.sizes[dim] == scene.sizes[dim],
            f"{tag} product size along {dim!r} differs from the scene.",
        )
    for coordinate in ("lat", "lon", "x", "y"):
        _check(
            coordinate in product, f"{tag} product lacks the {coordinate!r} coordinate."
        )
        _check(
            bool(
                np.allclose(
                    np.asarray(product[coordinate].values, dtype=float),
                    np.asarray(scene[coordinate].values, dtype=float),
                    equal_nan=True,
                )
            ),
            f"{tag} product coordinate {coordinate!r} differs from the scene.",
        )
    for attribute in ("sensor", "scene_id", "background_group"):
        _check(
            product.attrs.get(attribute) == scene.attrs.get(attribute),
            f"{tag} product attr {attribute!r} differs from the scene.",
        )
    _check(
        product.attrs.get("retriever") == retriever.name,
        f"{tag} product attr 'retriever' must equal retriever.name.",
    )
    valid = np.asarray(product["retrieval_valid"].values, dtype=bool)
    _check(bool(valid.any()), f"{tag} retrieval_valid is False everywhere.")
    uncertainty = np.asarray(product["ch4_mass_uncertainty"].values, dtype=float)
    _check(
        bool(np.all(np.isfinite(uncertainty[valid]) & (uncertainty[valid] > 0))),
        f"{tag} ch4_mass_uncertainty must be finite and positive where valid.",
    )
    mass = np.asarray(product["ch4_mass"].values, dtype=float)
    _check(
        bool(np.all(np.isfinite(mass[valid]))),
        f"{tag} ch4_mass must be finite wherever retrieval_valid.",
    )

    stats = retriever.background_statistics(scene, settings=settings)
    _check(
        isinstance(stats, xr.Dataset),
        f"{tag} background_statistics() must return a Dataset.",
    )
    if settings.statistics_scope == "cached_scene":
        cached = retriever.retrieve(
            scene, target, settings=settings, background_stats=stats
        )
        try:
            xr.testing.assert_allclose(cached["ch4_mass"], product["ch4_mass"])
        except AssertionError as error:
            raise AssertionError(
                f"{tag} retrieve() with cached background statistics differs from "
                f"the scene-computed result: {error}"
            ) from error
    return product


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
    "run_retriever_conformance",
    "run_sensor_conformance",
]
