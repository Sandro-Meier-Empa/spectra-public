"""Validate the MicroHH processing chain's plume-column exporter on a real case."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from spectra.contracts.plume_column import select_time, validate_plume_column
from spectra.plume import open_plume_column

pytestmark = pytest.mark.data

mpc = pytest.importorskip("microhh_processing_chain")

from microhh_processing_chain.export import export_plume_column  # noqa: E402
from microhh_processing_chain.initialisation import read_case  # noqa: E402
from microhh_processing_chain.sources import get_source_strength  # noqa: E402

# UTC time of model second zero when the .ini has no ``[time] datetime_utc``.
# The Pau release cases start 1 h 15 min before the release (notebook convention).
START_TIME_ENV = "SPECTRA_TEST_MICROHH_START_TIME"
DEFAULT_START_TIME = "2024-09-18T12:15:00"


@pytest.fixture(scope="module")
def case(microhh_case_dir: Path):
    return read_case(microhh_case_dir, scratch_path=microhh_case_dir.parent)


@pytest.fixture(scope="module")
def start_time(case) -> str | None:
    if case.config.get("time", "datetime_utc") is not None:
        return None
    return os.environ.get(START_TIME_ENV, DEFAULT_START_TIME)


@pytest.fixture(scope="module")
def exported(case, start_time, tmp_path_factory) -> tuple[xr.Dataset, Path]:
    output = tmp_path_factory.mktemp("export") / "col.nc"
    dataset = export_plume_column(
        case, "ch4", start_time=start_time, output_path=output
    )
    return dataset, output


@pytest.fixture(scope="module")
def raw_path(case) -> xr.Dataset:
    return xr.open_dataset(case.directory / "ch4_path.xy.nc", decode_times=False)


def test_export_validates(exported):
    dataset, output = exported
    info = validate_plume_column(dataset)
    assert output.exists()
    assert info.gas == "CH4"
    assert info.tracer == "ch4"
    assert info.source_id == f"{dataset.attrs['case_id']}:ch4"
    assert dataset.attrs["contract"] == "spectra.PlumeColumn"
    assert dataset.attrs["les_model"] == "MicroHH"
    assert dataset.attrs["producer"] == "microhh_processing_chain.export"
    assert "ch4_path.xy.nc" in json.loads(dataset.attrs["source_files"])


def test_column_equals_raw_path(exported, raw_path):
    dataset, _ = exported
    column = dataset["column"]
    assert column.dtype == np.float32
    assert column.dims == ("time", "y", "x")
    assert column.sizes == raw_path["ch4_path"].sizes
    raw = raw_path["ch4_path"]
    for index in range(dataset.sizes["time"]):
        np.testing.assert_array_equal(
            column.isel(time=index).values,
            raw.isel(time=index).values.astype("float32"),
        )
    np.testing.assert_allclose(dataset["x"].values, raw_path["x"].values)
    np.testing.assert_allclose(dataset["y"].values, raw_path["y"].values)


def test_emission_rate_matches_source_strength(exported, case):
    dataset, _ = exported
    rates = dataset["emission_rate"]
    assert rates.attrs["units"] == "kg s-1"
    timedep = bool(case.config["source"]["swtimedep_strength"])
    expected = np.array(
        [
            float(
                get_source_strength(
                    case, to_unit="kg/s", time_s=float(seconds) if timedep else None
                )[0]
            )
            for seconds in dataset["simulation_time"].values
        ]
    )
    np.testing.assert_allclose(rates.values, expected)
    assert bool(dataset.attrs["emission_rate_is_constant"]) == (not timedep)


def test_time_axis(exported, raw_path):
    dataset, _ = exported
    reference = pd.Timestamp(dataset.attrs["reference_time"]).tz_convert(None)
    seconds = dataset["simulation_time"].values
    np.testing.assert_array_equal(seconds, raw_path["time"].values.astype(float))
    expected = (reference + pd.to_timedelta(seconds, unit="s")).to_numpy()
    np.testing.assert_array_equal(dataset["time"].values, expected)
    assert dataset.attrs["dx"] == 1.0
    assert dataset.attrs["dy"] == 1.0


def test_wind_at_source_height(exported):
    dataset, _ = exported
    for name in ("wind_u", "wind_v"):
        assert dataset[name].dims == ("time",)
        assert dataset[name].attrs["units"] == "m s-1"
        assert dataset[name].attrs["height"] == pytest.approx(dataset.attrs["source_z"])
        assert np.isfinite(dataset[name].values).all()


def test_select_time_and_round_trip(exported):
    dataset, output = exported
    snapshot = select_time(dataset, 120)
    assert snapshot.attrs["time_index"] == 120
    assert snapshot["column"].dims == ("y", "x")
    reopened = open_plume_column(output)
    assert reopened.sizes == dataset.sizes
    np.testing.assert_array_equal(reopened["time"].values, dataset["time"].values)
    np.testing.assert_array_equal(
        reopened["column"].isel(time=120).values, snapshot["column"].values
    )
    reopened.close()


def test_times_selection_and_overwrite(case, start_time, tmp_path):
    output = tmp_path / "subset.nc"
    subset = export_plume_column(
        case, "ch4", start_time=start_time, times=[4500, 4520], output_path=output
    )
    np.testing.assert_array_equal(subset["simulation_time"].values, [4500.0, 4520.0])
    with pytest.raises(FileExistsError):
        export_plume_column(
            case, "ch4", start_time=start_time, times=[4500], output_path=output
        )
    with pytest.raises(ValueError, match="not in the output"):
        export_plume_column(
            case, "ch4", start_time=start_time, times=[4510], output_path=False
        )
