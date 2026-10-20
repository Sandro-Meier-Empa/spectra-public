"""AVIRIS-3 adapter and export on synthetic RDN/OBS granules (and a real one)."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pyproj
import pytest
import xarray as xr
from conftest import native_files

import spectra
import spectra.export.native as native_export
from spectra.contracts.errors import ContractError
from spectra.geometry.crs import crs_from_dataset, to_lonlat
from spectra.plume.io import write_plume_column
from spectra.sensors import Aviris3Adapter, aviris3, get_sensor
from spectra.sensors.aviris3 import LAYOUT
from spectra.testing import make_plume_column, run_adapter_conformance
from spectra.testing.native import (
    assert_native_preserved,
    native_snapshot,
    write_aviris3_files,
)


def native_radiance(path: Path) -> np.ndarray:
    """Return the raw radiance in native ``(wavelength, lines, samples)`` order."""
    with netCDF4.Dataset(path) as ds:
        variable = ds[LAYOUT.variable]
        variable.set_auto_maskandscale(False)
        return np.asarray(variable[:])


def test_aviris3_scene(tmp_path: Path) -> None:
    files = write_aviris3_files(tmp_path, fill_pixels=[(3, 7)])
    scene = run_adapter_conformance(Aviris3Adapter(chunks={"along": 8}), files)
    assert scene.sizes == {"across": 20, "along": 24, "band": 40}
    assert scene["radiance"].chunks is not None
    native = native_radiance(files["radiance"])
    # Lines are along, samples across; the band index replaces the wavelengths.
    np.testing.assert_array_equal(
        scene["radiance"].isel(along=5, across=9).values, native[:, 5, 9]
    )
    np.testing.assert_array_equal(scene["band"].values, np.arange(40))
    with netCDF4.Dataset(files["radiance"]) as ds:
        wavelength = ds["radiance/wavelength"][:]
        lon = ds["lon"][5, 9]
        elevation = ds["elev"][5, 9]
    np.testing.assert_array_equal(scene["wavelength"].values, wavelength)
    assert float(scene["lon"].isel(along=5, across=9)) == lon
    assert scene["radiance"].attrs["units"] == "uW nm-1 cm-2 sr-1"
    assert bool(scene["radiance"].isel(along=3, across=7).isnull().all())
    assert not bool(scene["valid"].isel(along=3, across=7))
    assert bool(scene["valid"].isel(along=4, across=7))
    with netCDF4.Dataset(files["obs"]) as ds:
        obs = {
            name: float(ds[f"observation_parameters/{name}"][5, 9])
            for name in ("path_length", "to_sensor_zenith", "to_sun_zenith")
        }
    assert float(scene["sza"].isel(along=5, across=9)) == pytest.approx(
        obs["to_sun_zenith"]
    )
    expected_altitude = (
        obs["path_length"] * np.cos(np.deg2rad(obs["to_sensor_zenith"])) + elevation
    )
    assert float(scene["sensor_altitude"].isel(along=5, across=9)) == pytest.approx(
        expected_altitude
    )
    assert scene.attrs["crs"] == "EPSG:32613"
    assert scene.attrs["scene_id"] == "AV320231006t181013_005"
    assert scene["time"].values == np.datetime64("2023-10-06T18:13:29")
    assert scene.attrs["acquisition_time"] == "2023-10-06T18:13:29"
    # Float32 lon/lat limit the centre spacing to a few decimetres.
    assert float(scene["pixel_area"].median()) == pytest.approx(16.0, rel=0.05)


def test_aviris3_crs_comes_from_the_grid_mapping(tmp_path: Path) -> None:
    # The granule's own UTM zone is used, not the zone of the scene centre.
    files = write_aviris3_files(tmp_path)
    wkt = pyproj.CRS.from_epsg(32614).to_wkt("WKT1_GDAL")
    with netCDF4.Dataset(files["radiance"], "r+") as ds:
        ds["transverse_mercator"].crs_wkt = wkt
        ds["transverse_mercator"].spatial_ref = wkt
    scene = Aviris3Adapter().open_scene(files)
    assert scene.attrs["crs"] == "EPSG:32614"
    with netCDF4.Dataset(files["radiance"]) as ds:
        lon, lat = ds["lon"][0, 0], ds["lat"][0, 0]
    x, y = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:32614", always_xy=True
    ).transform(lon, lat)
    assert float(scene["x"][0, 0]) == pytest.approx(x)
    assert float(scene["y"][0, 0]) == pytest.approx(y)


def test_aviris3_rejects_inconsistent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Aviris3Adapter()
    good = write_aviris3_files(tmp_path / "good")
    for other in (
        write_aviris3_files(tmp_path / "scene", scene="006"),
        write_aviris3_files(tmp_path / "time", time="2023-10-06T18:20:00"),
    ):
        with pytest.raises(ContractError, match="not the OBS granule"):
            adapter.open_scene({"radiance": good["radiance"], "obs": other["obs"]})
    shorter = write_aviris3_files(tmp_path / "short", n_lines=10)
    with pytest.raises(ContractError, match="differ in size"):
        adapter.open_scene({"radiance": good["radiance"], "obs": shorter["obs"]})
    with monkeypatch.context() as patch:
        patch.setitem(aviris3.OBS_VARIABLES, "sza", "solar_zenith")
        with pytest.raises(ContractError, match="solar_zenith"):
            adapter.open_scene(good)

    unmapped = write_aviris3_files(tmp_path / "unmapped")
    with netCDF4.Dataset(unmapped["radiance"], "r+") as ds:
        ds["transverse_mercator"].delncattr("grid_mapping_name")
    with pytest.raises(ContractError, match="grid-mapping"):
        adapter.open_scene(unmapped)
    embedded = write_aviris3_files(tmp_path / "embedded")
    with netCDF4.Dataset(embedded["radiance"], "r+") as ds:
        ds.setncattr("spectra_version", "0.2.0")
    with pytest.raises(ContractError, match="already embedded"):
        adapter.open_scene(embedded)


@pytest.mark.parametrize("block_bytes", [None, 1], ids=["one-block", "band-blocks"])
def test_aviris3_embed_writes_native_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_bytes: int | None
) -> None:
    if block_bytes is not None:
        # Band-major radiance is patched in blocks of bands once a plume window
        # is large; tiny blocks force that split on the small synthetic granule.
        monkeypatch.setattr(native_export, "_BLOCK_BYTES", block_bytes)
    files = write_aviris3_files(
        tmp_path / "in", n_lines=40, n_samples=30, fill_pixels=[(11, 10)]
    )
    plume = tmp_path / "plume.nc"
    write_plume_column(make_plume_column(nx=24, ny=24, dx=1.0, n_time=2, seed=1), plume)
    scene = get_sensor("aviris3").open_scene(files)
    x = float(scene["x"][20, 28])
    y = float(scene["y"][20, 28])
    lon, lat = (float(value) for value in to_lonlat(crs_from_dataset(scene), x, y))
    run = {
        "sensor": "aviris3",
        "files": {role: str(path) for role, path in files.items()},
        "output": str(tmp_path / "out" / "embedded.nc"),
        "seed": 11,
        "forward_model": {"model": "toy", "width_nm": 60.0, "saturation_ppm_m": 5e4},
        "plumes": [
            {
                "file": str(plume),
                "time_index": 1,
                "anchor": {"along": 10, "across": 10},
                "emission_scale": 0.5,
            },
            {
                "file": str(plume),
                "time_index": 0,
                "anchor": {"lon": lon, "lat": lat},
                "emission_scale": 0.5,
            },
        ],
    }
    before = native_radiance(files["radiance"])

    outcome = spectra.embed(run)

    result = outcome.result
    assert [cluster.sources for cluster in result.clusters] == [(0,), (1,)]
    after = native_radiance(outcome.output)
    changed = after != before
    assert changed.any()
    assert not changed[:, 11, 10].any()
    # The exporter and the in-memory view apply the same attenuation.
    expected = result.to_dataset(scene)["radiance"]
    expected_native = expected.transpose("band", "along", "across").values
    valid = np.isfinite(expected_native)
    np.testing.assert_array_equal(after[valid], expected_native[valid])
    mask = {LAYOUT.variable: changed}
    output_snapshot = native_snapshot(outcome.output, masks=mask)
    input_snapshot = native_snapshot(files["radiance"], masks=mask)
    new_objects = sorted(set(output_snapshot) - set(input_snapshot))
    assert {"injected_ch4_mass", "source_lines", "source_samples"} <= set(new_objects)
    assert not any(name.startswith("radiance/") for name in new_objects)
    with xr.open_dataset(outcome.output) as dataset:
        root_attrs = [name for name in dataset.attrs if name.startswith("spectra_")]
        assert dataset["injected_ch4_mass"].dims == ("source", "lines", "samples")
        np.testing.assert_array_equal(dataset["source_lines"].values, [10, 28])
        np.testing.assert_array_equal(dataset["source_samples"].values, [10, 20])
    assert_native_preserved(
        input_snapshot,
        output_snapshot,
        new_objects=new_objects,
        new_root_attrs=root_attrs,
    )
    with pytest.raises(ContractError, match="already embedded"):
        get_sensor("aviris3").open_scene(
            {"radiance": outcome.output, "obs": files["obs"]}
        )
    # Mass is conserved per plume on the sensor grid. The regridder's cell
    # polygons and the centre-spacing pixel areas agree only to the float32
    # precision of the AVIRIS-3 lon/lat (decimetres on 4 m pixels).
    area = np.asarray(scene["pixel_area"].values)
    for source in result.sources:
        window_area = area[source.window.across, source.window.along]
        assert float((source.injected * window_area).sum()) == pytest.approx(
            source.injected_mass_kg, rel=1e-2
        )


@pytest.mark.data
def test_real_aviris3_scene_conformance() -> None:
    scene = run_adapter_conformance(
        get_sensor("aviris3"), native_files("aviris3"), n_pixels=8
    )
    assert scene.sizes["band"] > 200
