"""Golden regression tests of the Lambert-Beer embedding port."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from spectra.spectral import (
    embed_column_in_radiance,
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


def assert_same_array(result: xr.DataArray, expected: xr.DataArray) -> None:
    assert result.dtype == expected.dtype
    assert np.array_equal(result.values, expected.values)
    xr.testing.assert_identical(result.load(), expected)


def test_embed_column_in_radiance_matches_golden(
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
    )
    expected = load_expected(
        golden_expected, "embedded_radiance_vertical.nc", "radiance"
    )
    assert_same_array(result, expected)
    assert (
        sha256(result.values) == golden_record["hashes"]["embedded_radiance_vertical"]
    )


def test_embed_column_in_radiance_dask_matches_golden(
    golden_radiance, golden_column_on_grid, golden_lut, golden_expected
):
    radiance = golden_radiance["radiance"].chunk({"across": 8})
    result = embed_column_in_radiance(radiance, golden_column_on_grid, golden_lut)
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


def test_golden_lut_hash(golden_lut, golden_record):
    assert (
        sha256(golden_lut["optical_depth"].values)
        == golden_record["hashes"]["lut_optical_depth"]
    )


def test_mismatched_grid_raises(golden_radiance, golden_column_on_grid, golden_lut):
    column = golden_column_on_grid.rename({"along": "x"})
    with pytest.raises(ValueError, match="spatial"):
        embed_column_in_radiance(golden_radiance["radiance"], column, golden_lut)
