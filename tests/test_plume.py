"""Tests for plume-column I/O and partial columns on a synthetic plume column."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.contracts.plume_column import (
    plume_column_attrs,
    select_time,
    validate_plume_column,
)
from spectra.plume import (
    column_below_altitude,
    open_plume_column,
    plume_fraction_below,
    write_plume_column,
)

NX, NY, NT = 32, 24, 3
DX = DY = 2.0
SOURCE_X, SOURCE_Y = 11.0, 23.0
EMISSION_KG_S = 0.1
ALTITUDES = np.array([10.0, 50.0, 100.0])
ALTITUDE_FRACTIONS = np.array([0.3, 0.7, 1.0])


def make_plume_column() -> xr.Dataset:
    """Build a Gaussian blob drifting along x over three snapshots."""
    x = DX / 2 + DX * np.arange(NX)
    y = DY / 2 + DY * np.arange(NY)
    seconds = np.array([0.0, 60.0, 120.0])
    xx, yy = np.meshgrid(x, y)
    column = np.stack(
        [
            np.exp(-(((xx - 20 - 6 * i) / 8) ** 2) - ((yy - SOURCE_Y) / 5) ** 2)
            for i in range(NT)
        ]
    ).astype("float32")
    column[column < 1e-3] = 0.0
    partial = column[:, None, :, :] * ALTITUDE_FRACTIONS[None, :, None, None]
    reference = pd.Timestamp("2024-09-18T12:15:00")
    dataset = xr.Dataset(
        {
            "column": (("time", "y", "x"), column, {"units": "kg m-2"}),
            "emission_rate": (
                ("time",),
                np.full(NT, EMISSION_KG_S),
                {"units": "kg s-1"},
            ),
            "column_below_altitude": (
                ("time", "altitude", "y", "x"),
                partial.astype("float32"),
                {"units": "kg m-2"},
            ),
            "wind_u": (("time",), np.array([2.0, 2.5, 3.0]), {"units": "m s-1"}),
            "wind_v": (("time",), np.array([0.5, 0.0, -0.5]), {"units": "m s-1"}),
        },
        coords={
            "time": reference + pd.to_timedelta(seconds, unit="s"),
            "simulation_time": ("time", seconds, {"units": "s"}),
            "altitude": ("altitude", ALTITUDES, {"units": "m"}),
            "y": ("y", y, {"units": "m"}),
            "x": ("x", x, {"units": "m"}),
        },
        attrs=plume_column_attrs(
            gas="CH4",
            les_model="synthetic",
            case_id="blob",
            tracer="ch4",
            source_id="blob:ch4",
            source_x=SOURCE_X,
            source_y=SOURCE_Y,
            source_z=3.0,
            dx=DX,
            dy=DY,
            reference_time="2024-09-18T12:15:00Z",
            passive_tracer=1,
            emission_rate_is_constant=1,
        ),
    )
    return dataset


@pytest.fixture
def plume() -> xr.Dataset:
    return make_plume_column()


def test_synthetic_dataset_is_valid(plume):
    info = validate_plume_column(plume)
    assert info.n_time == NT
    assert info.has_partial


def test_io_round_trip(plume, tmp_path):
    target = write_plume_column(plume, tmp_path / "blob.nc")
    assert target.exists()
    reopened = open_plume_column(target)
    assert reopened["column"].dtype == np.float32
    np.testing.assert_array_equal(reopened["column"].values, plume["column"].values)
    np.testing.assert_array_equal(
        reopened["column_below_altitude"].values, plume["column_below_altitude"].values
    )
    np.testing.assert_array_equal(reopened["time"].values, plume["time"].values)
    np.testing.assert_array_equal(
        reopened["simulation_time"].values, plume["simulation_time"].values
    )
    for key in ("contract", "case_id", "source_x", "dx", "reference_time"):
        assert reopened.attrs[key] == plume.attrs[key]
    reopened.close()


def test_write_refuses_to_overwrite(plume, tmp_path):
    target = write_plume_column(plume, tmp_path / "blob.nc")
    with pytest.raises(FileExistsError):
        write_plume_column(plume, target)
    reopened = open_plume_column(target)
    rewritten = write_plume_column(reopened, target, overwrite=True)
    reopened.close()
    assert rewritten == target
    assert open_plume_column(target).sizes["time"] == NT


def test_open_rejects_invalid_file(plume, tmp_path):
    broken = plume.drop_vars("emission_rate")
    broken.to_netcdf(tmp_path / "broken.nc")
    with pytest.raises(ContractError):
        open_plume_column(tmp_path / "broken.nc")


def test_column_below_altitude(plume):
    column = plume["column"]
    exact = column_below_altitude(plume, 50.0)
    assert exact.dims == ("time", "y", "x")
    np.testing.assert_allclose(exact.values, column.values * 0.7, rtol=1e-6)
    assert exact.attrs["altitude"] == 50.0
    interpolated = column_below_altitude(plume, 30.0)
    np.testing.assert_allclose(interpolated.values, column.values * 0.5, rtol=1e-6)
    floored = column_below_altitude(plume, 30.0, method="floor")
    np.testing.assert_allclose(floored.values, column.values * 0.3, rtol=1e-6)
    above = column_below_altitude(plume, 500.0)
    np.testing.assert_allclose(above.values, column.values, rtol=1e-6)
    with pytest.raises(ContractError):
        column_below_altitude(plume, 5.0)
    with pytest.raises(ContractError):
        column_below_altitude(plume.drop_vars("column_below_altitude"), 50.0)


def test_plume_fraction_below(plume):
    fraction = plume_fraction_below(plume, 10.0)
    assert fraction.dims == ("time",)
    np.testing.assert_allclose(fraction.values, 0.3, rtol=1e-6)
    np.testing.assert_allclose(plume_fraction_below(plume, 30.0).values, 0.5, rtol=1e-6)
    np.testing.assert_allclose(
        plume_fraction_below(plume, 100.0).values, 1.0, rtol=1e-6
    )
    snapshot = select_time(plume, 2)
    assert float(plume_fraction_below(snapshot, 50.0)) == pytest.approx(0.7, rel=1e-6)
    empty = plume.copy()
    empty["column"] = xr.zeros_like(empty["column"])
    empty["column_below_altitude"] = xr.zeros_like(empty["column_below_altitude"])
    assert np.isnan(plume_fraction_below(empty, 50.0).values).all()
