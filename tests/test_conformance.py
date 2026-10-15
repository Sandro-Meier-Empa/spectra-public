"""Adapter and retriever conformance tests over the registered plugins."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from spectra.contracts import ContractError, validate_column_product
from spectra.protocols import Retriever, SensorAdapter, SupportsProduct
from spectra.registry import get_retriever, get_sensor
from spectra.retrievers.toy import ToyRetriever
from spectra.sensors.toy import ToySensorAdapter
from spectra.testing import (
    conformance_parametrize,
    make_sensor_scene,
    make_unit_absorption_spectrum,
    run_retriever_conformance,
    run_sensor_conformance,
)
from spectra.types import RetrievalSettings, SceneRecord

PPM_M_TO_KG_M2 = 6.78e-7


@pytest.mark.parametrize(("name", "adapter"), conformance_parametrize())
def test_sensor_conformance(name: str, adapter: SensorAdapter) -> None:
    # Real adapters read their scenes from an explicit scene table named by
    # SPECTRA_TEST_<NAME>_SCENES; the synthetic toy scenes need no files.
    variable = f"SPECTRA_TEST_{name.upper()}_SCENES"
    scenes = os.environ.get(variable)
    if isinstance(adapter, ToySensorAdapter):
        # pyright keeps the protocol signature after narrowing; the toy takes None.
        records = cast(ToySensorAdapter, adapter).list_scenes(None, n=1)
    elif not scenes or not Path(scenes).exists():
        pytest.skip(f"{name}: {variable} is not set or does not exist")
    else:
        records = adapter.list_scenes(scenes)
    if not records:
        pytest.skip(f"{name}: scene table {scenes} lists no scenes")
    run_sensor_conformance(adapter, records[0])


def test_toy_adapter_from_registry() -> None:
    plugin = get_sensor("toy", n_across=16, n_along=24, noise_sigma=0.01)
    assert isinstance(plugin, ToySensorAdapter)
    assert isinstance(plugin, SensorAdapter)
    assert isinstance(plugin, SupportsProduct)
    # pyright keeps the protocol signature after narrowing; the toy takes None.
    adapter = cast(ToySensorAdapter, plugin)
    records = adapter.list_scenes(None, n=2)
    assert [record.scene_id for record in records] == ["toy-0001", "toy-0002"]
    assert records[1].acquisition_time.day == records[0].acquisition_time.day + 1
    assert records[0].has_product and records[0].uris == {}
    for record in records:
        run_sensor_conformance(adapter, record)
    scene = adapter.open_scene(records[0], chunks={"across": 8})
    assert scene.sizes == {"across": 16, "along": 24, "band": 54}
    assert scene["radiance"].chunks is not None
    assert scene["radiance"].chunks[0] == (8, 8)
    # The same record always opens to the same data.
    again = adapter.open_scene(records[0])
    np.testing.assert_array_equal(scene["radiance"].values, again["radiance"].values)
    assert adapter.list_scenes(None, scene_id="toy-0002")[0].scene_id == "toy-0002"
    assert [r.scene_id for r in adapter.list_scenes("ignored.csv", n=1)] == ["toy-0001"]
    masks = adapter.quality_masks(scene, records[0])
    assert set(masks.data_vars) == {"valid", "cloud", "shadow"}
    assert not bool(masks["cloud"].any())
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.list_scenes(None, flight_line=1)
    with pytest.raises(ValueError, match="per scene record"):
        ToySensorAdapter(scene_id="x")
    with pytest.raises(ValueError, match="Unknown"):
        ToySensorAdapter(bogus=1)


def test_sensor_conformance_reports_violations() -> None:
    adapter = ToySensorAdapter(n_across=8, n_along=8)
    record = adapter.list_scenes()[0]
    wrong_sensor = SceneRecord(
        sensor="other",
        scene_id=record.scene_id,
        platform=record.platform,
        acquisition_time=record.acquisition_time,
        background_group=record.background_group,
        uris={},
        center_lonlat=record.center_lonlat,
    )
    with pytest.raises(AssertionError, match=r"record\.sensor"):
        run_sensor_conformance(adapter, wrong_sensor)
    with pytest.raises(AssertionError, match="SensorAdapter"):
        run_sensor_conformance(object(), record)


@pytest.fixture(scope="module")
def clean_scene() -> xr.Dataset:
    return make_sensor_scene(noise_sigma=0.005, seed=3)


@pytest.fixture(scope="module")
def target(clean_scene: xr.Dataset) -> xr.Dataset:
    return make_unit_absorption_spectrum(clean_scene["wavelength"].values)


def inject(
    scene: xr.Dataset, column_ppm_m: np.ndarray, target: xr.Dataset
) -> xr.Dataset:
    """Return the scene with a Lambert-Beer enhancement multiplied into the radiance."""
    kernel = np.asarray(target["unit_abs_spect"].values, dtype=float)
    transmission = np.exp(column_ppm_m[:, :, None] * kernel[None, None, :])
    injected = scene.copy()
    radiance = (scene["radiance"] * transmission.astype(np.float32)).astype(np.float32)
    injected["radiance"] = radiance.assign_attrs(scene["radiance"].attrs)
    return injected


def test_toy_retriever_conformance(clean_scene: xr.Dataset, target: xr.Dataset) -> None:
    retriever = get_retriever("toy")
    assert isinstance(retriever, ToyRetriever)
    assert isinstance(retriever, Retriever)
    settings = RetrievalSettings()
    product = run_retriever_conformance(retriever, clean_scene, target, settings)
    assert product.attrs["retrieval_n_bands"] == clean_scene.sizes["band"]
    # A plume-free scene retrieves zero enhancement within the noise.
    assert abs(float(product["ch4_mass"].mean())) < 3e-6
    assert float(product["ch4_mass"].std()) < 1e-4
    stats = retriever.background_statistics(clean_scene, settings=settings)
    assert stats["background_mean"].dims == ("across", "band")
    assert bool(np.all(stats["n_samples"].values == clean_scene.sizes["along"]))


def test_toy_retriever_recovers_injected_enhancement(
    clean_scene: xr.Dataset, target: xr.Dataset
) -> None:
    retriever = ToyRetriever()
    settings = RetrievalSettings()
    across, along = np.meshgrid(
        np.arange(clean_scene.sizes["across"]),
        np.arange(clean_scene.sizes["along"]),
        indexing="ij",
    )
    blob = 2000.0 * np.exp(
        -0.5 * (((across - 30) / 5.0) ** 2 + ((along - 70) / 9.0) ** 2)
    )
    injected_kg_m2 = blob * PPM_M_TO_KG_M2
    scene = inject(clean_scene, blob, target)
    # Idealised background: statistics from the plume-free scene.
    stats = retriever.background_statistics(clean_scene, settings=settings)
    product = retriever.retrieve(
        scene, target, settings=settings, background_stats=stats
    )
    validate_column_product(product)
    retrieved = product["ch4_mass"].values.astype(float)
    correlation = np.corrcoef(retrieved.ravel(), injected_kg_m2.ravel())[0, 1]
    assert correlation > 0.95
    assert retrieved.max() == pytest.approx(injected_kg_m2.max(), rel=0.1)
    assert bool(np.all(product["ch4_mass_uncertainty"].values > 0))
    assert float(product["ch4_mass_uncertainty"].mean()) < 1e-4
    # Statistics from the contaminated scene itself still locate the plume.
    own = retriever.retrieve(scene, target, settings=settings)
    assert (
        np.corrcoef(own["ch4_mass"].values.ravel(), injected_kg_m2.ravel())[0, 1] > 0.8
    )
    # Interpolating a finer target onto the scene bands gives the same answer.
    fine = make_unit_absorption_spectrum(np.arange(2090.0, 2500.0, 0.7))
    resampled = retriever.retrieve(
        scene, fine, settings=settings, background_stats=stats
    )
    assert resampled["ch4_mass"].values.max() == pytest.approx(
        retrieved.max(), rel=0.05
    )
    # A grid sharing no node with the scene bands needs interpolation.
    shifted = make_unit_absorption_spectrum(np.arange(2100.37, 2500.0, 7.5))
    with pytest.raises(ContractError, match="interpolation is off"):
        retriever.retrieve(
            scene,
            shifted,
            settings=RetrievalSettings(options={"interpolate_target": False}),
        )
    with pytest.raises(ContractError, match="bands"):
        retriever.retrieve(
            scene,
            make_unit_absorption_spectrum(np.arange(1500.0, 1700.0, 5.0)),
            settings=settings,
        )


def test_toy_retriever_marks_invalid_pixels(
    clean_scene: xr.Dataset, target: xr.Dataset
) -> None:
    scene = clean_scene.copy()
    valid = np.ones((scene.sizes["across"], scene.sizes["along"]), dtype=bool)
    valid[:, :10] = False
    valid[3, :] = False
    scene["valid"] = (("across", "along"), valid)
    product = run_retriever_conformance(
        ToyRetriever(), scene, target, RetrievalSettings()
    )
    assert not bool(product["retrieval_valid"].values[:, :10].any())
    assert not bool(product["retrieval_valid"].values[3].any())
    assert bool(product["retrieval_valid"].values[4, 10:].all())
    assert np.all(np.isnan(product["ch4_mass"].values[3]))
    assert bool(np.all(product["ch4_mass_uncertainty"].values > 0))
    with pytest.raises(ContractError, match="background_mean"):
        ToyRetriever().retrieve(
            scene, target, settings=RetrievalSettings(), background_stats=xr.Dataset()
        )
    with pytest.raises(ValueError):
        ToyRetriever(uncertainty_floor_kg_m2=0.0)
