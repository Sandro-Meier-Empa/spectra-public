"""Golden regression tests of the Lambert-Beer embedding port."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import spectra
from spectra.spectral import (
    COLUMN_UNITS,
    EMBEDDING_VERSION,
    PPM_METRE_UNITS,
    embed_column_in_product,
    embed_column_in_radiance,
    embedding_provenance,
    injected_column,
)

pytestmark = pytest.mark.golden


def sha256(values: np.ndarray) -> str:
    """Digest of an array's shape, dtype and bytes (scripts/build_golden_fixture)."""
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values)
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_expected(root: Path, name: str, variable: str) -> xr.DataArray:
    return xr.open_dataset(root / name).load()[variable]


@pytest.fixture(scope="module")
def observed_product(golden_inputs: Path) -> xr.DataArray:
    return xr.open_dataset(golden_inputs / "observed_product.nc").load()["CH4"]


def assert_same_array(result: xr.DataArray, expected: xr.DataArray) -> None:
    assert result.dtype == expected.dtype
    assert np.array_equal(result.values, expected.values)
    xr.testing.assert_identical(result.load(), expected)


@pytest.mark.parametrize("convention", ["vertical", "slant"])
def test_embed_column_in_radiance_matches_golden(
    convention,
    golden_radiance,
    golden_column_on_grid,
    golden_lut,
    golden_expected,
    golden_record,
):
    result = embed_column_in_radiance(
        golden_radiance["radiance"],
        golden_column_on_grid,
        golden_lut,
        column_convention=convention,
    )
    expected = load_expected(
        golden_expected, f"embedded_radiance_{convention}.nc", "radiance"
    )
    assert_same_array(result, expected)
    assert (
        sha256(result.values)
        == golden_record["hashes"][f"embedded_radiance_{convention}"]
    )


def test_embed_column_in_radiance_dask_matches_golden(
    golden_radiance, golden_column_on_grid, golden_lut, golden_expected
):
    radiance = golden_radiance["radiance"].chunk({"across": 8})
    result = embed_column_in_radiance(
        radiance, golden_column_on_grid, golden_lut, column_convention="vertical"
    )
    assert result.chunks is not None
    expected = load_expected(
        golden_expected, "embedded_radiance_vertical.nc", "radiance"
    )
    computed = result.compute()
    assert computed.dtype == expected.dtype
    assert np.array_equal(computed.values, expected.values)


def test_embedded_radiance_keeps_name_and_attrs(
    golden_radiance, golden_column_on_grid, golden_lut
):
    radiance = golden_radiance["radiance"]
    result = embed_column_in_radiance(radiance, golden_column_on_grid, golden_lut)
    assert result.name == radiance.name
    assert result.attrs == radiance.attrs
    assert result.dims == radiance.dims


def test_injected_column_matches_golden(
    golden_column_on_grid, golden_expected, golden_record
):
    result = injected_column(golden_column_on_grid, convention="vertical")
    expected = load_expected(
        golden_expected, "injected_ch4_mass.nc", "injected_ch4_mass"
    )
    assert result.name == "injected_ch4_mass"
    assert_same_array(result, expected)
    assert sha256(result.values) == golden_record["hashes"]["injected_ch4_mass"]


def test_injected_column_without_convention(golden_column_on_grid):
    result = injected_column(golden_column_on_grid)
    assert result.attrs == {"units": "kg m-2", "long_name": "injected CH4 column"}
    assert result.dtype == np.float32


def test_injected_column_requires_units(golden_column_on_grid):
    bare = golden_column_on_grid.copy()
    bare.attrs = {}
    with pytest.raises(ValueError, match="units"):
        injected_column(bare)


def test_embed_column_in_product_matches_golden(
    observed_product, golden_column_on_grid, golden_expected, golden_record
):
    result = embed_column_in_product(observed_product, golden_column_on_grid)
    expected = load_expected(golden_expected, "product_embedded_ppm.nc", "CH4")
    assert result.attrs["units"] == PPM_METRE_UNITS
    assert_same_array(result, expected)
    assert sha256(result.values) == golden_record["hashes"]["product_embedded_ppm"]


def test_embed_column_in_product_kg_matches_golden(
    observed_product, golden_column_on_grid, golden_expected, golden_record
):
    result = embed_column_in_product(
        observed_product, golden_column_on_grid, output_units=COLUMN_UNITS
    )
    expected = load_expected(golden_expected, "product_embedded_kg.nc", "CH4")
    assert result.attrs["units"] == COLUMN_UNITS
    assert_same_array(result, expected)
    assert sha256(result.values) == golden_record["hashes"]["product_embedded_kg"]


def test_embedding_provenance_matches_golden(golden_lut, golden_record):
    attrs = embedding_provenance(
        method="radiance",
        geometric_amf=golden_lut.attrs["geometric_air_mass_factor"],
        lut=golden_lut,
        column_convention="vertical",
    )
    for key, value in golden_record["provenance_radiance"].items():
        assert key in attrs
        assert attrs[key] == value
    assert attrs["embedding_version"] == EMBEDDING_VERSION
    assert attrs["embedding_spectra_version"] == spectra.__version__

    product = embedding_provenance(method="product")
    for key, value in golden_record["provenance_product"].items():
        assert product[key] == value
    assert "embedding_column_convention" not in product


def test_embedding_provenance_records_sensor_and_regridder():
    attrs = embedding_provenance(method="product", sensor="aviris4", regridder="cdo")
    assert attrs["sensor"] == "aviris4"
    assert attrs["embedding_regridder"] == "cdo"


def test_golden_lut_hash(golden_lut, golden_record):
    assert (
        sha256(golden_lut["optical_depth"].values)
        == golden_record["hashes"]["lut_optical_depth"]
    )


def test_invalid_convention_raises(golden_radiance, golden_column_on_grid, golden_lut):
    with pytest.raises(ValueError, match="column_convention"):
        embed_column_in_radiance(
            golden_radiance["radiance"],
            golden_column_on_grid,
            golden_lut,
            column_convention="diagonal",  # type: ignore[arg-type]
        )


def test_mismatched_grid_raises(golden_radiance, golden_column_on_grid, golden_lut):
    column = golden_column_on_grid.rename({"along": "x"})
    with pytest.raises(ValueError, match="spatial"):
        embed_column_in_radiance(golden_radiance["radiance"], column, golden_lut)
