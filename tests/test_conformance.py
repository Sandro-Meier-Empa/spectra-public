"""Adapter conformance tests over the registered plugins."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from spectra.protocols import SensorAdapter
from spectra.registry import get_sensor
from spectra.sensors.toy import ToySensorAdapter
from spectra.testing import conformance_parametrize, run_sensor_conformance
from spectra.types import SceneRecord


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
    # pyright keeps the protocol signature after narrowing; the toy takes None.
    adapter = cast(ToySensorAdapter, plugin)
    records = adapter.list_scenes(None, n=2)
    assert [record.scene_id for record in records] == ["toy-0001", "toy-0002"]
    assert records[1].acquisition_time.day == records[0].acquisition_time.day + 1
    assert records[0].uris == {}
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


def test_toy_adapter_accepts_scene_keyword_options() -> None:
    adapter = ToySensorAdapter(n_across=8, n_along=8, platform="airborne")
    record = adapter.list_scenes(None, n=1)[0]
    assert adapter.open_scene(record).sizes["across"] == 8
    with pytest.raises(ValueError, match="per scene record"):
        ToySensorAdapter(lazy=False)


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
