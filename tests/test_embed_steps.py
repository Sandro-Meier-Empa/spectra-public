"""Step-wise radiance embedding (locate/regrid -> embed -> retrieve) on toy plugins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from spectra import cli
from spectra.contracts import validate_column_product, validate_sensor_scene
from spectra.contracts.errors import ContractError
from spectra.pipeline import (
    EmbeddingSpec,
    build_lut,
    embed_radiance_sample,
    embed_sample,
    place_and_regrid,
    retrieve_embedded,
    scene_window,
    snapshot_column,
)
from spectra.plume.io import write_plume_column
from spectra.regrid.polygon import PolygonRegridder
from spectra.retrievers.toy import ToyRetriever
from spectra.sensors.toy import ToySensorAdapter
from spectra.spectral.providers.toy import ToyUnitAbsorption
from spectra.testing import make_plume_column
from spectra.types import Placement, RetrievalSettings

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
    window = select_bands(window, RetrievalSettings().band_windows_nm)
    lut = build_lut(ToyUnitAbsorption(), window, SPEC)
    from_lut = embed_radiance_sample(scene, injected, spec=SPEC, lut=lut)
    from_provider = embed_radiance_sample(
        scene, injected, ToyUnitAbsorption(), spec=SPEC
    )
    xr.testing.assert_allclose(from_lut["radiance"], from_provider["radiance"])


def test_steps_compose_to_embed_sample(toy_inputs: dict) -> None:
    scene = toy_inputs["scene"]
    provider, retriever = ToyUnitAbsorption(), ToyRetriever()
    settings = RetrievalSettings(statistics_scope="window", min_n_samples=10)
    embedded = embed_radiance_sample(
        scene,
        toy_inputs["injected"],
        provider,
        spec=SPEC,
        band_windows_nm=settings.band_windows_nm,
    )
    stepwise = retrieve_embedded(
        embedded,
        retriever=retriever,
        settings=settings,
        provider=provider,
        spec=SPEC,
    )
    with pytest.raises(ContractError):
        retrieve_embedded(embedded, retriever=retriever, settings=settings)
    combined = embed_sample(
        scene,
        toy_inputs["column"],
        toy_inputs["placement"],
        method="radiance",
        regridder=PolygonRegridder(),
        spec=SPEC,
        provider=provider,
        retriever=retriever,
        settings=settings,
    )
    validate_column_product(stepwise)
    xr.testing.assert_allclose(stepwise, combined)
    assert stepwise.attrs == combined.attrs


def test_cli_lists_no_retrieval(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["embed", "radiance", "--help"])
    assert info.value.code == 0
    assert "--no-retrieval" in capsys.readouterr().out


def _cli_args(plume: Path, save_path: Path) -> list[str]:
    return [
        "embed",
        "radiance",
        "--sensor",
        "toy",
        "--adapter-option",
        "n_across=32",
        "--adapter-option",
        "n_along=48",
        "--file",
        "radiance=unused",
        "--plume",
        str(plume),
        "--anchor-along",
        "10",
        "--anchor-across",
        "10",
        "--provider",
        "toy",
        "--retriever",
        "toy",
        "--enhancement-min",
        "1",
        "--enhancement-max",
        "1000",
        "--enhancement-nodes",
        "4",
        "--save-path",
        str(save_path),
        "--no-retrieval",
    ]


def test_cli_no_retrieval_writes_embedded_radiance(tmp_path: Path) -> None:
    plume = write_plume_column(
        make_plume_column(nx=24, ny=24, dx=2.0, n_time=1, seed=1),
        tmp_path / "toy.plumecolumn.nc",
    )
    save_path = tmp_path / "embedded_radiance.nc"
    with pytest.raises(SystemExit):
        cli.main([*_cli_args(plume, save_path), "--prism"])
    assert not save_path.exists()

    cli.main(_cli_args(plume, save_path))
    with xr.open_dataset(save_path) as written:
        validate_sensor_scene(written)
        assert "injected_ch4_mass" in written
        assert written.attrs["embedding_method"] == "radiance"
