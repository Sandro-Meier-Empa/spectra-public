"""Negative tests of the data contracts, shared types and unit conversion."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyproj
import pytest
import xarray as xr

from spectra.contracts import (
    ContractError,
    convert_column_units,
    scale_emission,
    validate_lut,
    validate_plume_column,
    validate_sensor_scene,
)
from spectra.contracts.lut import LUT_REQUIRED_ATTRS, LUT_VARIABLES
from spectra.contracts.plume_column import REQUIRED_ATTRS as PLUME_ATTRS
from spectra.contracts.sensor_scene import GEOMETRY_VARIABLES
from spectra.contracts.sensor_scene import REQUIRED_ATTRS as SCENE_ATTRS
from spectra.geometry.crs import spatial_ref_coordinate
from spectra.testing import (
    make_plume_column,
    make_sensor_scene,
)
from spectra.types import (
    Placement,
    SpectralResponse,
    Window,
)

Mutation = Callable[[xr.Dataset], xr.Dataset]
PPM_M_TO_KG_M2 = 6.78e-7


# --- helpers ---------------------------------------------------------------


def without_attr(dataset: xr.Dataset, name: str) -> xr.Dataset:
    mutated = dataset.copy()
    del mutated.attrs[name]
    return mutated


def with_attr(dataset: xr.Dataset, name: str, value: object) -> xr.Dataset:
    mutated = dataset.copy()
    mutated.attrs[name] = value
    return mutated


def with_units(dataset: xr.Dataset, name: str, units: str | None) -> xr.Dataset:
    mutated = dataset.copy()
    variable = mutated[name].copy()
    if units is None:
        variable.attrs.pop("units", None)
    else:
        variable.attrs["units"] = units
    mutated[name] = variable
    return mutated


def with_value(dataset: xr.Dataset, name: str, value: float) -> xr.Dataset:
    mutated = dataset.copy(deep=True)
    values = mutated[name].values
    values[(0,) * values.ndim] = value
    return mutated


def as_dtype(dataset: xr.Dataset, name: str, dtype: str) -> xr.Dataset:
    mutated = dataset.copy()
    mutated[name] = mutated[name].astype(dtype)
    return mutated


def assert_contract_error(
    validate: Callable[[xr.Dataset], object], dataset: xr.Dataset
) -> None:
    with pytest.raises(ContractError):
        validate(dataset)


# --- Contract A: PlumeColumn ------------------------------------------------


@pytest.fixture(scope="module")
def plume() -> xr.Dataset:
    return make_plume_column(nx=48, ny=40, n_time=2, dt_s=10.0)


def partial_column(
    dataset: xr.Dataset, altitude: list[float], units: str
) -> xr.Dataset:
    mutated = dataset.copy()
    column = dataset["column"]
    stacked = np.stack([column.values * 0.5, column.values], axis=1)
    mutated["column_below_altitude"] = (
        ("time", "altitude", "y", "x"),
        stacked,
        {"units": units},
    )
    mutated = mutated.assign_coords(altitude=("altitude", altitude))
    return mutated


def concentration(
    dataset: xr.Dataset, dims: tuple[str, ...], units: str, z: bool
) -> xr.Dataset:
    mutated = dataset.copy()
    column = dataset["column"].values
    values = np.stack([column, column], axis=1) if len(dims) == 4 else column
    mutated["concentration"] = (dims, values, {"units": units})
    if z:
        mutated = mutated.assign_coords(z=("z", [50.0, 150.0]))
    return mutated


PLUME_MUTATIONS: dict[str, Mutation] = {
    "declares_other_contract": lambda ds: with_attr(ds, "contract", "spectra.Other"),
    "x_not_coordinate": lambda ds: ds.drop_vars("x"),
    "x_non_uniform": lambda ds: ds.assign_coords(
        x=np.sort(np.random.default_rng(0).random(ds.sizes["x"]))
    ),
    "x_decreasing": lambda ds: ds.assign_coords(x=ds["x"].values[::-1]),
    "y_single_cell": lambda ds: ds.isel(y=slice(0, 1)),
    "time_not_datetime": lambda ds: ds.assign_coords(time=np.arange(ds.sizes["time"])),
    "missing_simulation_time": lambda ds: ds.drop_vars("simulation_time"),
    "simulation_time_scalar": lambda ds: ds.drop_vars("simulation_time").assign_coords(
        simulation_time=0.0
    ),
    "column_convention_slant": lambda ds: with_attr(ds, "column_convention", "slant"),
    "emission_rate_units_attr": lambda ds: with_attr(
        ds, "emission_rate_units", "kg h-1"
    ),
    "dx_attr_disagrees": lambda ds: with_attr(ds, "dx", 3.0),
    "dy_attr_disagrees": lambda ds: with_attr(ds, "dy", 3.0),
    "missing_column": lambda ds: ds.drop_vars("column"),
    "column_wrong_dims": lambda ds: ds.assign(
        column=ds["column"].transpose("y", "x", "time")
    ),
    "column_units": lambda ds: with_units(ds, "column", "ppm-m"),
    "column_no_units": lambda ds: with_units(ds, "column", None),
    "column_integer": lambda ds: as_dtype(ds, "column", "int32"),
    "column_nan": lambda ds: with_value(ds, "column", np.nan),
    "column_negative": lambda ds: with_value(ds, "column", -1.0),
    "column_empty_support": lambda ds: ds.assign(column=xr.zeros_like(ds["column"])),
    "missing_emission_rate": lambda ds: ds.drop_vars("emission_rate"),
    "emission_rate_scalar": lambda ds: ds.assign(
        emission_rate=ds["emission_rate"].isel(time=0)
    ),
    "emission_rate_units": lambda ds: with_units(ds, "emission_rate", "kg h-1"),
    "emission_rate_negative": lambda ds: with_value(ds, "emission_rate", -1.0),
    "wind_u_without_v": lambda ds: ds.drop_vars("wind_v"),
    "wind_units": lambda ds: with_units(ds, "wind_u", "km h-1"),
    "wind_wrong_dims": lambda ds: ds.assign(
        wind_u=("y", np.zeros(ds.sizes["y"]), {"units": "m s-1"})
    ),
    "partial_wrong_dims": lambda ds: ds.assign(column_below_altitude=ds["column"]),
    "partial_units": lambda ds: partial_column(ds, [100.0, 200.0], "ppm-m"),
    "altitude_not_increasing": lambda ds: partial_column(ds, [200.0, 100.0], "kg m-2"),
    "concentration_wrong_dims": lambda ds: concentration(
        ds, ("time", "y", "x"), "kg m-3", z=False
    ),
    "concentration_units": lambda ds: concentration(
        ds, ("time", "z", "y", "x"), "kg m-2", z=True
    ),
    "concentration_without_z": lambda ds: concentration(
        ds, ("time", "z", "y", "x"), "kg m-3", z=False
    ),
}


def test_plume_fixture_and_optional_variables_validate(plume: xr.Dataset) -> None:
    validate_plume_column(plume)
    both = concentration(
        partial_column(plume, [100.0, 200.0], "kg m-2"),
        ("time", "z", "y", "x"),
        "kg m-3",
        z=True,
    )
    info = validate_plume_column(both, require_partial=True, require_3d=True)
    assert info.has_partial and info.has_3d
    with pytest.raises(ContractError, match="column_below_altitude"):
        validate_plume_column(plume, require_partial=True)
    with pytest.raises(ContractError, match="concentration"):
        validate_plume_column(plume, require_3d=True)


@pytest.mark.parametrize("mutation", list(PLUME_MUTATIONS), ids=list(PLUME_MUTATIONS))
def test_plume_column_rejects(plume: xr.Dataset, mutation: str) -> None:
    assert_contract_error(validate_plume_column, PLUME_MUTATIONS[mutation](plume))


@pytest.mark.parametrize("name", PLUME_ATTRS)
def test_plume_column_requires_attr(plume: xr.Dataset, name: str) -> None:
    assert_contract_error(validate_plume_column, without_attr(plume, name))


def test_scale_emission_rejects(plume: xr.Dataset) -> None:
    with pytest.raises(ContractError):
        scale_emission(plume, 0.0)
    with pytest.raises(ContractError):
        scale_emission(plume, np.inf)
    with pytest.raises(ContractError, match="passive"):
        scale_emission(with_attr(plume, "passive_tracer", False), 2.0)


# --- Contract B: SensorScene -------------------------------------------------


@pytest.fixture(scope="module")
def scene() -> xr.Dataset:
    return make_sensor_scene(
        n_across=6, n_along=10, wavelengths=np.arange(2100.0, 2200.0, 10.0)
    )


def without_crs(dataset: xr.Dataset) -> xr.Dataset:
    mutated = dataset.drop_vars("spatial_ref")
    for name in ("x", "y"):
        coordinate = mutated[name].copy()
        coordinate.attrs.pop("epsg", None)
        mutated = mutated.assign_coords({name: coordinate})
    return mutated


SCENE_MUTATIONS: dict[str, Mutation] = {
    "missing_dim_band": lambda ds: ds.drop_dims("band"),
    "missing_dim_along": lambda ds: ds.isel(along=0),
    "missing_wavelength": lambda ds: ds.drop_vars("wavelength"),
    "wavelength_not_increasing": lambda ds: ds.assign_coords(
        wavelength=("band", ds["wavelength"].values[::-1])
    ),
    "wavelength_wrong_dims": lambda ds: ds.drop_vars("wavelength").assign_coords(
        wavelength=2100.0
    ),
    "fwhm_nonpositive": lambda ds: ds.assign_coords(
        fwhm=("band", np.zeros(ds.sizes["band"]))
    ),
    "missing_lat": lambda ds: ds.drop_vars("lat"),
    "missing_lon": lambda ds: ds.drop_vars("lon"),
    "missing_x": lambda ds: without_crs(ds).drop_vars("x"),
    "missing_y": lambda ds: ds.drop_vars("y"),
    "x_not_2d": lambda ds: ds.drop_vars("x").assign_coords(
        x=("along", np.arange(ds.sizes["along"], dtype=float))
    ),
    "no_crs": without_crs,
    "geographic_crs": lambda ds: ds.assign_coords(
        spatial_ref=spatial_ref_coordinate(pyproj.CRS.from_epsg(4326))
    ),
    "missing_time": lambda ds: ds.drop_vars("time"),
    "time_not_datetime": lambda ds: ds.drop_vars("time").assign_coords(time=0.0),
    "time_wrong_dims": lambda ds: ds.drop_vars("time").assign_coords(
        time=(
            "across",
            np.array(["2026-01-01"] * ds.sizes["across"], dtype="datetime64[ns]"),
        )
    ),
    "missing_radiance": lambda ds: ds.drop_vars("radiance"),
    "radiance_wrong_dims": lambda ds: ds.assign(radiance=ds["radiance"].isel(band=0)),
    "radiance_no_units": lambda ds: with_units(ds, "radiance", None),
    "sza_wrong_dims": lambda ds: ds.assign(sza=("band", np.zeros(ds.sizes["band"]))),
    "missing_valid": lambda ds: ds.drop_vars("valid"),
    "valid_not_bool": lambda ds: as_dtype(ds, "valid", "int8"),
    "valid_wrong_dims": lambda ds: ds.assign(valid=ds["valid"].isel(across=0)),
    "missing_pixel_area": lambda ds: ds.drop_vars("pixel_area"),
    "pixel_area_wrong_dims": lambda ds: ds.assign(pixel_area=100.0),
    "platform_invalid": lambda ds: with_attr(ds, "platform", "balloon"),
    "geometry_space_map": lambda ds: with_attr(ds, "geometry_space", "map"),
    "radiance_wrong_order": lambda ds: ds.assign(
        radiance=ds["radiance"].transpose("band", "across", "along")
    ),
}


def test_scene_fixture_validates(scene: xr.Dataset) -> None:
    info = validate_sensor_scene(scene, require_lazy=True)
    assert info.n_band == 10
    along_time = scene.drop_vars("time").assign_coords(
        time=(
            "along",
            np.array(
                ["2026-06-01T10:30:00"] * scene.sizes["along"], dtype="datetime64[ns]"
            ),
        )
    )
    validate_sensor_scene(along_time)
    # Variables beyond the contract are carried along unchecked.
    validate_sensor_scene(scene.assign(cloud=xr.zeros_like(scene["valid"])))


@pytest.mark.parametrize("mutation", list(SCENE_MUTATIONS), ids=list(SCENE_MUTATIONS))
def test_sensor_scene_rejects(scene: xr.Dataset, mutation: str) -> None:
    assert_contract_error(validate_sensor_scene, SCENE_MUTATIONS[mutation](scene))


@pytest.mark.parametrize("name", GEOMETRY_VARIABLES)
def test_sensor_scene_requires_geometry(scene: xr.Dataset, name: str) -> None:
    assert_contract_error(validate_sensor_scene, scene.drop_vars(name))


@pytest.mark.parametrize("name", SCENE_ATTRS)
def test_sensor_scene_requires_attr(scene: xr.Dataset, name: str) -> None:
    assert_contract_error(validate_sensor_scene, without_attr(scene, name))


def test_sensor_scene_require_lazy(scene: xr.Dataset) -> None:
    eager = scene.assign(radiance=scene["radiance"].compute())
    validate_sensor_scene(eager)
    with pytest.raises(ContractError, match="dask-backed"):
        validate_sensor_scene(eager, require_lazy=True)


# --- unit conversion -------------------------------------------


def test_convert_column_units_ppm_m_to_kg_m2() -> None:
    column = xr.DataArray(
        [1.0, 10.0], dims="n", attrs={"units": "ppm-m", "long_name": "x"}, name="ch4"
    )
    mass = convert_column_units(column, "kg m-2", gas="CH4")
    assert np.allclose(mass.values, [PPM_M_TO_KG_M2, 10 * PPM_M_TO_KG_M2], rtol=0.01)
    assert mass.attrs["units"] == "kg m-2"
    assert mass.attrs["long_name"] == "x" and mass.name == "ch4"
    back = convert_column_units(mass, "ppm m", gas="CH4")
    assert np.allclose(back.values, column.values, rtol=1e-9)
    assert back.attrs["units"] == "ppm-m"
    explicit = convert_column_units(
        column.drop_attrs(), "kg/m2", gas="CH4", from_unit="ppm-m"
    )
    assert np.allclose(explicit.values, mass.values)
    with pytest.raises(ContractError, match="units"):
        convert_column_units(column.drop_attrs(), "kg m-2", gas="CH4")


def test_convert_column_units_identity() -> None:
    column = xr.DataArray([1e-4, 2e-4], dims="n", attrs={"units": "kg m-2"})
    same = convert_column_units(column, "kg m-2", gas="CH4")
    assert same is not column
    np.testing.assert_array_equal(same.values, column.values)
    alias = convert_column_units(
        column.assign_attrs(units="kg/m2"), "kg m^-2", gas="CH4"
    )
    np.testing.assert_array_equal(alias.values, column.values)
    assert alias.attrs["units"] == "kg m-2"


# --- LUT ---------------------------------------


@pytest.fixture(scope="module")
def lut() -> xr.Dataset:
    enhancement = np.array([1.0, 10.0, 100.0])
    wavelength = np.array([2100.0, 2200.0, 2300.0, 2400.0])
    values = -1e-5 * np.ones((enhancement.size, wavelength.size))
    return xr.Dataset(
        {
            name: (("enhancement", "wavelength"), values.copy(), {"units": "ppm-1 m-1"})
            for name in LUT_VARIABLES
        },
        coords={
            "enhancement": ("enhancement", enhancement, {"units": "ppm"}),
            "wavelength": ("wavelength", wavelength, {"units": "nm"}),
            "vertical_amount": (
                "enhancement",
                enhancement * 1000.0,
                {"units": "ppm-m"},
            ),
            "slant_amount": ("enhancement", enhancement * 2333.0, {"units": "ppm-m"}),
        },
        attrs={
            "gas": "CH4",
            "layer_height": 1000.0,
            "effective_layer_height": 900.0,
            "radiance_leg": "both",
            "sza": 30.0,
            "vza": 0.0,
            "geometric_air_mass_factor": 2.1547,
            "embedding_version": "lambert-beer-v1",
        },
    )


LUT_MUTATIONS: dict[str, Mutation] = {
    "variable_wrong_dims": lambda ds: ds.assign(
        sigma_slant=ds["sigma_slant"].transpose("wavelength", "enhancement")
    ),
    "variable_not_finite": lambda ds: with_value(ds, "optical_depth", np.inf),
    "enhancement_nonpositive": lambda ds: ds.assign_coords(
        enhancement=[0.0, 10.0, 100.0]
    ),
    "enhancement_not_increasing": lambda ds: ds.assign_coords(
        enhancement=[1.0, 100.0, 10.0]
    ),
    "wavelength_not_increasing": lambda ds: ds.assign_coords(
        wavelength=ds["wavelength"].values[::-1]
    ),
    "missing_vertical_amount": lambda ds: ds.drop_vars("vertical_amount"),
    "missing_slant_amount": lambda ds: ds.drop_vars("slant_amount"),
    "slant_amount_not_increasing": lambda ds: ds.assign_coords(
        slant_amount=("enhancement", [3.0, 2.0, 1.0])
    ),
    "vertical_amount_wrong_dims": lambda ds: ds.drop_vars(
        "vertical_amount"
    ).assign_coords(vertical_amount=1.0),
}


def test_lut_fixture_validates(lut: xr.Dataset) -> None:
    validate_lut(lut)


@pytest.mark.parametrize("mutation", list(LUT_MUTATIONS), ids=list(LUT_MUTATIONS))
def test_lut_rejects(lut: xr.Dataset, mutation: str) -> None:
    assert_contract_error(validate_lut, LUT_MUTATIONS[mutation](lut))


@pytest.mark.parametrize("name", LUT_VARIABLES)
def test_lut_requires_variable(lut: xr.Dataset, name: str) -> None:
    assert_contract_error(validate_lut, lut.drop_vars(name))


@pytest.mark.parametrize("name", LUT_REQUIRED_ATTRS)
def test_lut_requires_attr(lut: xr.Dataset, name: str) -> None:
    assert_contract_error(validate_lut, without_attr(lut, name))


# --- shared types ------------------------------------------------------------


def test_window_validation() -> None:
    window = Window(across=slice(2, 6), along=slice(0, 10))
    assert window.shape == (4, 10)
    assert window.indexers == {"across": slice(2, 6), "along": slice(0, 10)}
    assert Window.from_attrs(window.attrs()) == window
    for across in (
        slice(None, 6),
        slice(2, None),
        slice(2, 6, 2),
        slice(-1, 6),
        slice(6, 6),
        slice(7, 6),
    ):
        with pytest.raises(ContractError):
            Window(across=across, along=slice(0, 10))
    with pytest.raises(ContractError):
        Window(across=slice(0, 4), along=slice(3, 3))
    with pytest.raises(ContractError, match="Missing window attribute"):
        Window.from_attrs({})


def test_placement_validation() -> None:
    placement = Placement(rotation_deg=37, flip="x", emission_scale=2.0, time_index=1)
    assert placement.attrs()["rotation_deg"] == 37
    assert "anchor_x" not in placement.attrs()
    exact = Placement(anchor_corner="source", anchor_xy=(500000, 5200000.5))
    assert exact.anchor_xy == (500000.0, 5200000.5)
    assert exact.attrs()["anchor_x"] == 500000.0
    assert exact.attrs()["anchor_corner"] == "source"
    invalid: list[dict[str, object]] = [
        {"anchor_corner": "centre"},
        {"flip": "xy"},
        {"rotation_deg": 1.5},
        {"rotation_deg": True},
        {"rotation_deg": 360},
        {"rotation_deg": -360},
        {"anchor_along": -1},
        {"anchor_across": 2.0},
        {"anchor_across": True},
        {"time_index": -1},
        {"emission_scale": 0.0},
        {"emission_scale": -1.0},
        {"emission_scale": np.inf},
        {"anchor_xy": (1.0, np.nan)},
        {"anchor_xy": (1.0, 2.0, 3.0)},
    ]
    for kwargs in invalid:
        with pytest.raises(ContractError):
            Placement(**kwargs)  # type: ignore[arg-type]


def test_spectral_response_validation() -> None:
    response = SpectralResponse(
        wavelength=np.array([2100.0, 2150.0, 2200.0]), fwhm=np.array([8.0, 8.0, 8.0])
    )
    assert response.size == 3
    np.testing.assert_array_equal(response.select([(2140.0, 2210.0)]), [1, 2])
    with pytest.raises(ContractError):
        SpectralResponse(wavelength=np.array([2100.0, 2150.0]), fwhm=np.array([8.0]))
    with pytest.raises(ContractError):
        SpectralResponse(
            wavelength=np.array([[2100.0, 2150.0]]), fwhm=np.array([[8.0, 8.0]])
        )
    with pytest.raises(ContractError, match="increasing"):
        SpectralResponse(
            wavelength=np.array([2150.0, 2100.0]), fwhm=np.array([8.0, 8.0])
        )
    with pytest.raises(ContractError, match="positive"):
        SpectralResponse(
            wavelength=np.array([2100.0, 2150.0]), fwhm=np.array([8.0, 0.0])
        )
