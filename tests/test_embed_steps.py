"""Step-wise radiance embedding (locate/regrid -> embed) on toy plugins."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectra.contracts import validate_sensor_scene
from spectra.contracts.errors import ContractError
from spectra.pipeline import (
    EmbeddingSpec,
    build_lut,
    embed_radiance_sample,
    place_and_regrid,
    scene_window,
    snapshot_column,
)
from spectra.regrid.polygon import PolygonRegridder
from spectra.sensors.toy import ToySensorAdapter
from spectra.spectral.providers.toy import ToyUnitAbsorption
from spectra.testing import make_plume_column
from spectra.types import Placement

SPEC = EmbeddingSpec(enhancements_ppm=(1.0, 10.0, 100.0, 1000.0))


@pytest.fixture(scope="module")
def toy_inputs() -> dict:
    adapter = ToySensorAdapter(n_across=32, n_along=48, noise_sigma=0.0)
    scene = adapter.open_scene(adapter.list_scenes(None, n=1)[0]).load()
    snapshot = snapshot_column(
        make_plume_column(nx=24, ny=24, dx=2.0, n_time=1, seed=1), time_index=0
    )
    placement = Placement(anchor_along=10, anchor_across=10)
    injected = place_and_regrid(
        scene, snapshot, placement, regridder=PolygonRegridder()
    )
    return {
        "scene": scene,
        "column": snapshot,
        "placement": placement,
        "injected": injected,
    }


def test_embed_radiance_sample_returns_attenuated_window(toy_inputs: dict) -> None:
    scene, injected = toy_inputs["scene"], toy_inputs["injected"]
    embedded = embed_radiance_sample(scene, injected, ToyUnitAbsorption(), spec=SPEC)
    validate_sensor_scene(embedded)
    assert "injected_ch4_mass" in embedded
    assert "spectra_version" in embedded.attrs

    window, _ = scene_window(scene, injected)
    original = window["radiance"].sel(band=embedded["band"])
    plume = (embedded["injected_ch4_mass"] > 0).transpose(*original.dims[:2])
    ratio = embedded["radiance"] / original
    assert bool(plume.any())
    assert float(ratio.where(plume).min()) < 1.0
    assert float(ratio.where(plume).max()) <= 1.0 + 1e-12
    np.testing.assert_allclose(
        ratio.where(~plume).values[~np.isnan(ratio.where(~plume).values)], 1.0
    )


def test_embed_radiance_sample_uses_given_lut(toy_inputs: dict) -> None:
    scene, injected = toy_inputs["scene"], toy_inputs["injected"]
    with pytest.raises(ContractError):
        embed_radiance_sample(scene, injected, spec=SPEC)
    from spectra.contracts.sensor_scene import select_bands

    window, _ = scene_window(scene, injected)
    window = select_bands(window, ((1480.0, 1800.0), (2080.0, 2500.0)))
    lut = build_lut(ToyUnitAbsorption(), window, SPEC)
    from_lut = embed_radiance_sample(scene, injected, spec=SPEC, lut=lut)
    from_provider = embed_radiance_sample(
        scene, injected, ToyUnitAbsorption(), spec=SPEC
    )
    xr.testing.assert_allclose(from_lut["radiance"], from_provider["radiance"])
