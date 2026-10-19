"""Tests of the synthetic generators in ``spectra.testing.synthetic``."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectra.contracts import (
    ContractError,
    plume_support_bounds,
    scale_emission,
    select_time,
    validate_plume_column,
    validate_sensor_scene,
    validate_unit_absorption_spectrum,
)
from spectra.testing import (
    make_plume_column,
    make_sensor_scene,
    make_unit_absorption_spectrum,
    scene_pixel_centres,
    scene_seed,
)
from spectra.types import SceneGeometry, SpectralResponse, ViewingGeometry


@pytest.fixture(scope="module")
def plume() -> xr.Dataset:
    return make_plume_column()


@pytest.fixture(scope="module")
def scene() -> xr.Dataset:
    return make_sensor_scene()


# --- plume columns ---------------------------------------------------------


def test_plume_column_validates(plume: xr.Dataset) -> None:
    info = validate_plume_column(plume)
    assert info.n_time == 3
    assert info.gas == "CH4"
    assert info.source_id == "toy_case:ch4"
    assert info.emission_rate_is_constant
    assert info.dx == info.dy == 2.0
    assert plume["column"].dtype == np.float32
    assert plume.attrs["les_model"] == "toy"
    assert plume.attrs["producer"] == "spectra.testing"


def test_plume_total_mass_matches_attr(plume: xr.Dataset) -> None:
    masses = np.asarray(plume.attrs["toy_total_mass_kg"], dtype=float)
    assert masses.shape == (3,)
    area = plume.attrs["dx"] * plume.attrs["dy"]
    for index, expected in enumerate(masses):
        gridded = float(plume["column"].isel(time=index).astype(np.float64).sum())
        assert gridded * area == pytest.approx(expected, rel=1e-5)
    # The plume segment inside the domain grows with time until it reaches the edge.
    assert np.all(np.diff(masses) >= 0)
    assert masses[0] == pytest.approx(1.0 * 60.0)


def test_plume_zero_outside_support(plume: xr.Dataset) -> None:
    column = plume["column"].values
    assert np.all(column >= 0)
    assert np.mean(column == 0) > 0.3
    # Nothing upwind of the source for a wind blowing towards +x.
    upwind = plume["column"].sel(x=slice(None, plume.attrs["source_x"] - 2.0))
    assert float(upwind.sum()) == 0.0


def test_plume_wind_and_advection(plume: xr.Dataset) -> None:
    assert np.allclose(plume["wind_u"].values, 3.0)
    assert np.allclose(plume["wind_v"].values, 0.0)
    assert plume["wind_u"].attrs["height"] == 10.0
    weights = plume["column"].isel(time=-1)
    centroid_x = float((weights * weights["x"]).sum() / weights.sum())
    centroid_y = float((weights * weights["y"]).sum() / weights.sum())
    assert centroid_x > plume.attrs["source_x"] + 50.0
    assert centroid_y == pytest.approx(plume.attrs["source_y"], abs=10.0)


def test_plume_support_inside_domain(plume: xr.Dataset) -> None:
    xmin, ymin, xmax, ymax = plume_support_bounds(plume)
    width = plume.sizes["x"] * plume.attrs["dx"]
    height = plume.sizes["y"] * plume.attrs["dy"]
    assert 0.0 <= xmin < xmax <= width
    assert 0.0 < ymin < ymax < height
    assert xmin == pytest.approx(plume.attrs["source_x"], abs=plume.attrs["dx"])


def test_select_time_and_scale_emission(plume: xr.Dataset) -> None:
    snapshot = select_time(plume, 1)
    assert "time" not in snapshot.dims
    assert snapshot.attrs["time_index"] == 1
    assert snapshot.attrs["simulation_time_s"] == pytest.approx(120.0)
    assert snapshot.attrs["snapshot_time"] == "2026-01-01T12:02:00"

    scaled = scale_emission(plume, 3.0)
    xr.testing.assert_allclose(scaled["column"], plume["column"] * 3.0)
    assert np.allclose(scaled["emission_rate"].values, 3.0)
    assert scaled.attrs["emission_scale"] == 3.0
    assert scaled["column"].attrs["units"] == "kg m-2"
    assert scaled["emission_rate"].attrs["units"] == "kg s-1"
    validate_plume_column(scaled)
    assert scale_emission(scaled, 2.0).attrs["emission_scale"] == 6.0
    with pytest.raises(ContractError):
        select_time(plume, 3)


def test_plume_column_options() -> None:
    rotated = make_plume_column(
        nx=64, ny=96, dx=4.0, n_time=2, dt_s=20.0, wind_dir_deg=180.0, seed=5
    )
    info = validate_plume_column(rotated)
    assert info.n_time == 2
    assert np.allclose(rotated["wind_u"].values, 0.0)
    assert np.allclose(rotated["wind_v"].values, 3.0)
    weights = rotated["column"].isel(time=-1)
    centroid_y = float((weights * weights["y"]).sum() / weights.sum())
    assert centroid_y > rotated.attrs["source_y"]
    assert rotated.attrs["dy"] == 4.0
    with pytest.raises(ValueError, match="inside the domain"):
        make_plume_column(source_xy=(-10.0, 10.0))
    with pytest.raises(ValueError):
        make_plume_column(emission_rate_kg_s=0.0)


def test_plume_column_is_deterministic() -> None:
    first = make_plume_column(nx=32, ny=32, n_time=1, seed=1)
    second = make_plume_column(nx=32, ny=32, n_time=1, seed=1)
    other = make_plume_column(nx=32, ny=32, n_time=1, seed=2)
    xr.testing.assert_identical(first["column"], second["column"])
    assert not np.array_equal(first["column"].values, other["column"].values)


# --- sensor scenes ---------------------------------------------------------


@pytest.mark.parametrize("rotation", [0.0, 30.0])
def test_sensor_scene_validates(rotation: float) -> None:
    scene = make_sensor_scene(grid_rotation_deg=rotation, pixel_along_m=5.0)
    info = validate_sensor_scene(scene, require_lazy=True)
    assert info.is_lazy
    assert (info.n_across, info.n_along, info.n_band) == (64, 128, 54)
    assert info.crs.to_epsg() == 32632
    assert scene["radiance"].dtype == np.float32
    assert scene["radiance"].attrs["units"] == "W m-2 nm-1 sr-1"
    assert scene["radiance"].chunks is not None
    assert scene["valid"].dtype == bool and bool(scene["valid"].all())
    assert np.allclose(scene["pixel_area"].values, 50.0)
    assert scene.attrs["background_group"] == "toy:toy-0001"
    assert scene.attrs["geometry_space"] == "sensor"
    assert scene.attrs["acquisition_time"] == "2026-06-01T10:30:00"
    assert scene["time"].dims == ()
    # Pixel centres form a regular lattice with the requested spacing.
    x, y = scene["x"].values, scene["y"].values
    along = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    across = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    assert np.allclose(along, 5.0) and np.allclose(across, 10.0)
    direction = np.degrees(np.arctan2(y[0, 1] - y[0, 0], x[0, 1] - x[0, 0]))
    assert direction == pytest.approx(rotation, abs=1e-6)
    assert scene["x"].attrs["epsg"] == 32632
    assert 8.0 < float(scene["lon"].mean()) < 10.0
    assert 46.0 < float(scene["lat"].mean()) < 48.0
    SpectralResponse.from_scene(scene)
    SceneGeometry.from_scene(scene)
    geometry = ViewingGeometry.from_scene(scene)
    assert geometry.sza == 30.0 and geometry.platform == "airborne"


def test_sensor_scene_eager_fails_lazy_check() -> None:
    scene = make_sensor_scene(n_across=8, n_along=16, lazy=False)
    info = validate_sensor_scene(scene)
    assert not info.is_lazy
    with pytest.raises(ContractError, match="dask-backed"):
        validate_sensor_scene(scene, require_lazy=True)


def test_sensor_scene_is_deterministic() -> None:
    first = make_sensor_scene(n_across=8, n_along=16, noise_sigma=0.01, seed=4)
    second = make_sensor_scene(n_across=8, n_along=16, noise_sigma=0.01, seed=4)
    other = make_sensor_scene(n_across=8, n_along=16, noise_sigma=0.01, seed=5)
    np.testing.assert_array_equal(first["radiance"].values, second["radiance"].values)
    assert not np.array_equal(first["radiance"].values, other["radiance"].values)
    assert scene_seed("toy-0001") == scene_seed("toy-0001")
    assert scene_seed("toy-0001") != scene_seed("toy-0002")


def test_sensor_scene_options() -> None:
    scene = make_sensor_scene(
        n_across=4,
        n_along=6,
        wavelengths=[1600.0, 1650.0, 1700.0],
        fwhm_nm=[5.0, 6.0, 7.0],
        platform="spaceborne",
        background_group="site:a",
        acquisition_time="2025-03-04T05:06:07",
        crs=32631,
    )
    info = validate_sensor_scene(scene, require_lazy=True)
    assert info.platform == "spaceborne"
    assert info.background_group == "site:a"
    assert info.crs.to_epsg() == 32631
    assert scene.attrs["acquisition_time"] == "2025-03-04T05:06:07"
    assert np.allclose(scene["fwhm"].values, [5.0, 6.0, 7.0])
    x, y = scene_pixel_centres(n_across=2, n_along=3, pixel_across_m=1.0)
    assert x.shape == y.shape == (2, 3)
    with pytest.raises(ValueError, match="platform"):
        make_sensor_scene(platform="balloon")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="increasing"):
        make_sensor_scene(wavelengths=[2.0, 1.0])
    with pytest.raises(ContractError):
        make_sensor_scene(crs="EPSG:4326")


# --- unit absorption -------------------------------------------------------


def test_unit_absorption_spectrum(scene: xr.Dataset) -> None:
    target = make_unit_absorption_spectrum(scene["wavelength"].values)
    validate_unit_absorption_spectrum(target)
    values = target["unit_abs_spect"].values
    assert np.all(values <= 0) and values.min() == pytest.approx(-5e-5, rel=0.2)
    assert target["unit_abs_spect"].attrs["units"] == "ppm-1 m-1"
    with pytest.raises(ValueError, match="negative"):
        make_unit_absorption_spectrum(scene["wavelength"].values, peak=1e-5)
