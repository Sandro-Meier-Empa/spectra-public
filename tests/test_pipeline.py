"""Pipeline integration on the golden window (no regridding involved)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectra.pipeline import EmbeddingSpec, embed_radiance, scene_window
from spectra.types import Window

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


def test_radiance_branch_matches_golden(
    golden_radiance,
    golden_metadata,
    golden_lut,
    golden_column_on_grid,
    golden_expected,
) -> None:
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
