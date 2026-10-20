"""Sensor adapters on synthetic native files (and real ones when listed)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

import netCDF4
import numpy as np
import pytest
import xarray as xr
from conftest import native_files

from spectra.contracts.errors import ContractError
from spectra.protocols import SensorAdapter
from spectra.sensors import SENSORS, Aviris4Adapter, EmitAdapter, get_sensor
from spectra.sensors.native import resolve_files
from spectra.testing import run_adapter_conformance
from spectra.testing.native import write_aviris4_files, write_emit_files


def test_get_sensor_resolves_names_and_objects() -> None:
    assert set(SENSORS) == {"aviris3", "aviris4", "emit"}
    assert isinstance(get_sensor("emit"), EmitAdapter)
    custom = Aviris4Adapter(chunks={"across": 4})
    assert get_sensor(custom) is custom
    for name in SENSORS:
        assert isinstance(get_sensor(name), SensorAdapter)
    with pytest.raises(ContractError, match="Unknown sensor"):
        get_sensor("hyspex")
    with pytest.raises(ContractError, match="SensorAdapter"):
        get_sensor(object())  # type: ignore[arg-type]


def test_resolve_files(tmp_path: Path) -> None:
    roles = {"radiance": True, "obs": True, "mask": False}
    first, second = tmp_path / "a.nc", tmp_path / "b.nc"
    first.write_bytes(b"")
    second.write_bytes(b"")
    resolved = resolve_files(roles, {"radiance": "a.nc", "obs": second}, base=tmp_path)
    assert resolved == {"radiance": first.resolve(), "obs": second.resolve()}
    with pytest.raises(ContractError, match="Unknown file roles"):
        resolve_files(roles, {"radiance": first, "obs": second, "rgb": first})
    with pytest.raises(ContractError, match="Missing required"):
        resolve_files(roles, {"radiance": first})
    with pytest.raises(ContractError, match="does not exist"):
        resolve_files(roles, {"radiance": first, "obs": tmp_path / "c.nc"})
    with pytest.raises(ContractError, match="same file"):
        resolve_files(roles, {"radiance": first, "obs": first})


# -- AVIRIS-4 -----------------------------------------------------------------


@pytest.mark.parametrize("schema", ["2025", "2026"])
def test_aviris4_scene(tmp_path: Path, schema: Literal["2025", "2026"]) -> None:
    files = write_aviris4_files(
        tmp_path, schema=schema, sun_zenith=41.5, sun_azimuth=133.0
    )
    adapter = Aviris4Adapter(chunks={"across": 8, "along": 10})
    scene = run_adapter_conformance(adapter, files)
    assert scene["radiance"].chunks is not None
    assert float(scene["sza"]) == 41.5 and float(scene["saa"]) == 133.0
    with netCDF4.Dataset(files["metadata"]) as metadata:
        scan = metadata["scan_zenith"][:].astype(float)
        altitude = metadata["aircraft_altitude"][:].astype(float)
    np.testing.assert_allclose(scene["vza"].values, np.abs(scan) / 100.0)
    np.testing.assert_array_equal(scene["sensor_altitude"].values, altitude)
    assert scene["time"].values == np.datetime64("2024-09-28T07:29:27")
    assert scene.attrs["acquisition_time"] == "2024-09-28T07:29:27"
    assert scene.attrs["scene_id"] == files["radiance"].stem
    assert scene["radiance"].attrs["units"] == "W m-2 nm-1 sr-1"
    assert not bool(scene["valid"].isel(across=0, along=0))
    assert bool(scene["valid"].isel(across=1, along=1))
    area = scene["pixel_area"].values
    np.testing.assert_allclose(area, 4.0, rtol=1e-6)
    assert scene.attrs["crs"].endswith("32634")


def test_aviris4_rejects_inconsistent_files(tmp_path: Path) -> None:
    files = write_aviris4_files(tmp_path / "a")
    adapter = Aviris4Adapter()
    with netCDF4.Dataset(files["metadata"], "r+") as ds:
        ds["cast_shadow"].delncattr("zenith")
    with pytest.raises(ContractError, match="solar angles"):
        adapter.open_scene(files)

    other = write_aviris4_files(tmp_path / "b", time="2024-09-28T08:00:00")
    good = write_aviris4_files(tmp_path / "c")
    with pytest.raises(ContractError, match="another acquisition"):
        adapter.open_scene(
            {"radiance": good["radiance"], "metadata": other["metadata"]}
        )
    shorter = write_aviris4_files(tmp_path / "d", n_along=10)
    with pytest.raises(ContractError, match="pixels"):
        adapter.open_scene(
            {"radiance": good["radiance"], "metadata": shorter["metadata"]}
        )
    timeless = tmp_path / "timeless.nc"
    with xr.open_dataset(good["radiance"]) as ds:
        ds.drop_vars("time").to_netcdf(timeless)
    with pytest.raises(ContractError, match="scalar 'time'"):
        adapter.open_scene({"radiance": timeless, "metadata": good["metadata"]})
    with netCDF4.Dataset(good["radiance"], "r+") as ds:
        ds.setncattr("spectra_version", "0.2.0")
    with pytest.raises(ContractError, match="already embedded"):
        adapter.open_scene(good)


# -- EMIT ---------------------------------------------------------------------


@pytest.mark.parametrize("build", ["010635", "010617"])
def test_emit_scene(tmp_path: Path, build: Literal["010635", "010617"]) -> None:
    files = write_emit_files(tmp_path, build=build, fill_pixels=[(3, 7)])
    scene = run_adapter_conformance(EmitAdapter(), files)
    assert scene.sizes == {"across": 20, "along": 24, "band": 40}
    with netCDF4.Dataset(files["radiance"]) as ds:
        ds["radiance"].set_auto_maskandscale(False)
        native = ds["radiance"][5, 9, :]
        lon = ds["location"]["lon"][5, 9]
        elevation = ds["location"]["elev"][5, 9]
    # Downtrack is along, crosstrack is across.
    np.testing.assert_array_equal(scene["radiance"].isel(along=5, across=9), native)
    assert float(scene["lon"].isel(along=5, across=9)) == lon
    assert scene["radiance"].attrs["units"] == "uW/cm^2/SR/nm"
    # The fill value is masked and the pixel invalid.
    assert bool(scene["radiance"].isel(along=3, across=7).isnull().all())
    assert not bool(scene["valid"].isel(along=3, across=7))
    with netCDF4.Dataset(files["obs"]) as ds:
        obs = ds["obs"][5, 9, :].astype(float)
    assert float(scene["sza"].isel(along=5, across=9)) == pytest.approx(obs[4])
    assert float(scene["vaa"].isel(along=5, across=9)) == pytest.approx(obs[1])
    expected_altitude = obs[0] * np.cos(np.deg2rad(obs[2])) + elevation
    assert float(scene["sensor_altitude"].isel(along=5, across=9)) == pytest.approx(
        expected_altitude
    )
    assert scene.attrs["crs"] == "EPSG:32613"
    assert scene.attrs["scene_id"] == "emit20260819t180247_o23112_s001"
    assert scene["time"].values == np.datetime64("2026-08-19T18:02:47")
    np.testing.assert_allclose(scene["pixel_area"].values, 3600.0, rtol=1e-3)


def test_emit_southern_hemisphere_uses_south_zone(tmp_path: Path) -> None:
    files = write_emit_files(tmp_path, centre_lonlat=(-70.5, -33.4))
    scene = EmitAdapter().open_scene(files)
    assert scene.attrs["crs"] == "EPSG:32719"


def test_emit_rejects_mismatched_obs(tmp_path: Path) -> None:
    files = write_emit_files(tmp_path / "a")
    other = write_emit_files(tmp_path / "b", time="2026-08-19T18:03:00")
    with pytest.raises(ContractError, match="not the OBS granule"):
        EmitAdapter().open_scene({"radiance": files["radiance"], "obs": other["obs"]})
    with netCDF4.Dataset(files["obs"], "r+") as ds:
        ds["sensor_band_parameters"]["observation_bands"][4] = "Something else"
    with pytest.raises(ContractError, match="To-sun zenith"):
        EmitAdapter().open_scene(files)


# -- real files ---------------------------------------------------------------


@pytest.mark.data
@pytest.mark.parametrize("sensor", ["aviris4", "emit"])
def test_real_scene_conformance(sensor: str) -> None:
    files = native_files(sensor)
    scene = run_adapter_conformance(get_sensor(sensor), files, n_pixels=8)
    assert scene.sizes["band"] > 200


SCENE_TWICE = """
import sys
from pathlib import Path
import xarray as xr
from spectra.sensors import EmitAdapter
files = {"radiance": Path(sys.argv[1]), "obs": Path(sys.argv[2])}
first = EmitAdapter().open_scene(files)
second = EmitAdapter().open_scene(files)
with xr.open_dataset(files["obs"]) as obs:
    obs["obs"].isel(bands=0).values
print(float(first["sza"].mean()), float(second["sza"].mean()))
"""


def _open_scene_twice(files: dict[str, Path]) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", SCENE_TWICE, str(files["radiance"]), str(files["obs"])],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_emit_scene_reopens_obs(tmp_path: Path) -> None:
    # The OBS band names are variable-length strings with a string fill value;
    # keeping a real granule open while it is reopened crashes HDF5 1.14, so
    # the adapter reads the OBS bands eagerly. The crash only reproduces on
    # real granules (test_real_emit_scene_then_embed).
    _open_scene_twice(write_emit_files(tmp_path))


SCENE_THEN_EMBED = """
import sys
from pathlib import Path
import spectra
from spectra.plume.io import write_plume_column
from spectra.sensors import EmitAdapter
from spectra.testing import make_plume_column
files = {"radiance": sys.argv[1], "obs": sys.argv[2]}
out = Path(sys.argv[3])
plume = out.parent / "plume.nc"
write_plume_column(make_plume_column(nx=24, ny=24, dx=20.0, n_time=1), plume)
scene = EmitAdapter().open_scene({k: Path(v) for k, v in files.items()})
spectra.embed({
    "sensor": "emit", "files": files, "output": str(out), "seed": 1,
    "forward_model": {"model": "toy"},
    "plumes": [{"file": str(plume), "time_index": 0, "anchor": "random"}],
})
"""


@pytest.mark.data
def test_real_emit_scene_then_embed(tmp_path: Path) -> None:
    # Reproduces the HDF5 crash of a lazily opened OBS granule on real data.
    files = native_files("emit")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            SCENE_THEN_EMBED,
            str(files["radiance"]),
            str(files["obs"]),
            str(tmp_path / "out.nc"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
