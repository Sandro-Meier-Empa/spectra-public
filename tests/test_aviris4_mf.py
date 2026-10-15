"""Equivalence of the AVIRIS-4 retriever wrapper with a direct matched filter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectra.contracts import validate_column_product
from spectra.contracts.column_product import convert_column_units
from spectra.types import RetrievalSettings

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def mf_module():
    pytest.importorskip("aviris4_ch4")
    import aviris4_ch4.mf as mf

    return mf


def _scene_from_golden(radiance: xr.Dataset, metadata: xr.Dataset) -> xr.Dataset:
    from spectra.geometry.crs import projected_crs_from_xy, spatial_ref_coordinate

    scene = radiance.copy()
    scene = scene.set_coords(
        [n for n in ("wavelength", "fwhm") if n in scene.data_vars]
    )
    scene = scene.assign_coords({n: metadata[n] for n in ("lon", "lat", "x", "y")})
    crs = projected_crs_from_xy(metadata["x"], metadata["y"])
    scene = scene.assign_coords(spatial_ref=spatial_ref_coordinate(crs))
    scene.attrs.update(
        {
            "sensor": "aviris4",
            "platform": "airborne",
            "scene_id": "golden",
            "background_group": "aviris4:golden",
        }
    )
    return scene


def test_wrapper_matches_direct_matched_filter(
    mf_module, golden_radiance, golden_metadata, golden_inputs
) -> None:
    from spectra.retrievers.aviris4_mf import Aviris4MatchedFilter

    target = xr.open_dataset(golden_inputs / "target_10ppm.nc").load()
    noise = str(golden_inputs / "emit_noise.txt")
    scene = _scene_from_golden(golden_radiance, golden_metadata)
    settings = RetrievalSettings(
        statistics_scope="window",
        band_windows_nm=((2080.0, 2500.0),),
        target_enhancement_ppm=10.0,
        min_n_samples=1,
    )
    retriever = Aviris4MatchedFilter(
        noise_model_file=noise, mask_shadows=False, parallel=False
    )
    product = retriever.retrieve(scene, target, settings=settings)
    info = validate_column_product(product)
    assert info.geometry_space == "sensor"

    direct = mf_module.MatchedFilter(
        unit_abs_spect_dataset=target,
        min_n_samples=1,
        noise_model_file=noise,
        mask_shadows=False,
        mean_function="median",
        albedo_correction=True,
    ).apply_matched_filter(
        data=scene[["radiance"]], enhancement=10.0, parallel=False, save=False
    )
    expected = convert_column_units(direct["CH4"], "kg m-2", gas="CH4")
    np.testing.assert_array_equal(product["ch4_mass"].values, expected.values)
    expected_unc = convert_column_units(direct["uncertainty"], "kg m-2", gas="CH4")
    np.testing.assert_array_equal(
        product["ch4_mass_uncertainty"].values, expected_unc.values
    )
    assert product.attrs["retriever"] == "aviris4_mf"
    assert product["retrieval_valid"].dtype.kind == "b"
    assert np.isfinite(product["ch4_mass"].values).mean() > 0.9
