"""Pipeline integration on the golden window (no regridding involved)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectra.contracts import validate_column_product
from spectra.export.prism import to_prism_product, write_product
from spectra.pipeline import (
    EmbeddingSpec,
    embed_product,
    embed_radiance,
    retrieve_embedded,
    scene_window,
)
from spectra.types import RetrievalSettings, Window

pytestmark = pytest.mark.golden


def _scene(golden_radiance: xr.Dataset, golden_metadata: xr.Dataset) -> xr.Dataset:
    from spectra.geometry.crs import projected_crs_from_xy, spatial_ref_coordinate

    scene = golden_radiance.copy()
    scene = scene.set_coords(
        [n for n in ("wavelength", "fwhm") if n in scene.data_vars]
    )
    scene = scene.assign_coords(
        {n: golden_metadata[n] for n in ("lon", "lat", "x", "y")}
    )
    crs = projected_crs_from_xy(golden_metadata["x"], golden_metadata["y"])
    scene = scene.assign_coords(spatial_ref=spatial_ref_coordinate(crs))
    scene["pixel_area"] = xr.ones_like(golden_metadata["x"]) * 0.2
    scene.attrs.update(
        {
            "sensor": "aviris4",
            "platform": "airborne",
            "scene_id": "golden",
            "background_group": "aviris4:golden",
        }
    )
    return scene


def _injected(column: xr.DataArray, scene: xr.Dataset) -> xr.DataArray:
    window = Window(
        across=slice(0, scene.sizes["across"]), along=slice(0, scene.sizes["along"])
    )
    injected = column.copy()
    injected.name = "injected_ch4_mass"
    injected.attrs.update(window.attrs("window_"))
    injected.attrs.update(
        {"regrid_method": "none", "anchor_along": 0, "anchor_across": 0}
    )
    return injected


def test_radiance_branch_matches_golden_and_retrieves(
    golden_radiance,
    golden_metadata,
    golden_lut,
    golden_column_on_grid,
    golden_expected,
    golden_inputs,
) -> None:
    pytest.importorskip("aviris4_ch4")
    from spectra.retrievers.aviris4_mf import Aviris4MatchedFilter

    scene = _scene(golden_radiance, golden_metadata)
    injected = _injected(golden_column_on_grid, scene)
    window_scene, window = scene_window(scene, injected)
    assert window.shape == (scene.sizes["across"], scene.sizes["along"])
    spec = EmbeddingSpec(enhancements_ppm=tuple(golden_lut["enhancement"].values))
    embedded = embed_radiance(window_scene, injected, golden_lut, spec=spec)
    expected = xr.open_dataset(golden_expected / "embedded_radiance_vertical.nc").load()
    np.testing.assert_array_equal(
        embedded["radiance"].values, expected["radiance"].values
    )
    assert embedded.attrs["embedding_method"] == "radiance"
    assert "injected_ch4_mass" in embedded

    target = xr.open_dataset(golden_inputs / "target_10ppm.nc").load()
    settings = RetrievalSettings(
        statistics_scope="window", band_windows_nm=((2080.0, 2500.0),), min_n_samples=1
    )
    retriever = Aviris4MatchedFilter(
        noise_model_file=str(golden_inputs / "emit_noise.txt"),
        mask_shadows=False,
        parallel=False,
    )
    product = retrieve_embedded(
        embedded, target, retriever=retriever, settings=settings
    )
    info = validate_column_product(product)
    assert info.has_injected
    assert product.attrs["embedding_method"] == "radiance"
    assert product.attrs["embedding_version"] == "lambert-beer-v1"
    # The retrieval must see the plume: core enhancement well above the background.
    core = product["ch4_mass"].where(product["injected_ch4_mass"] > 0.01)
    background = product["ch4_mass"].where(product["injected_ch4_mass"] < 1e-5)
    assert float(core.mean()) > float(background.mean()) + 3 * float(background.std())

    prism_product = to_prism_product(product)
    assert prism_product["ch4_mass"].dims == ("y", "x")
    assert set(prism_product.data_vars) >= {
        "ch4_mass",
        "ch4_mass_uncertainty",
        "retrieval_valid",
        "injected_ch4_mass",
    }


def test_product_branch_matches_golden(
    golden_radiance,
    golden_metadata,
    golden_column_on_grid,
    golden_inputs,
    golden_expected,
    tmp_path,
) -> None:
    from spectra.contracts.column_product import convert_column_units

    scene = _scene(golden_radiance, golden_metadata)
    injected = _injected(golden_column_on_grid, scene)
    observed = xr.open_dataset(golden_inputs / "observed_product.nc").load()["CH4"]
    product = xr.Dataset(coords={"across": scene["across"], "along": scene["along"]})
    product["ch4_mass"] = convert_column_units(observed, "kg m-2", gas="CH4")
    product["ch4_mass_uncertainty"] = xr.full_like(product["ch4_mass"], 1e-4)
    product["ch4_mass_uncertainty"].attrs["units"] = "kg m-2"
    product["retrieval_valid"] = xr.ones_like(product["ch4_mass"], dtype=bool)
    product = product.assign_coords(
        {n: scene[n] for n in ("lon", "lat", "x", "y", "spatial_ref")}
    )
    product.attrs.update(
        {"sensor": "aviris4", "scene_id": "golden", "geometry_space": "sensor"}
    )
    embedded = embed_product(product, injected)
    expected = xr.open_dataset(golden_expected / "product_embedded_kg.nc").load()["CH4"]
    np.testing.assert_allclose(embedded["ch4_mass"].values, expected.values, rtol=1e-6)
    assert embedded.attrs["embedding_method"] == "product"
    prism_product = to_prism_product(embedded)
    path = write_product(prism_product, tmp_path / "sample.zarr")
    reopened = xr.open_zarr(path)
    assert reopened["injected_ch4_mass"].dims == ("y", "x")
