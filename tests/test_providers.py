"""Unit-absorption providers: toy analytics, libRadtran/ARTS golden, registry."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import xarray as xr

from spectra.contracts.errors import MissingExtraError
from spectra.contracts.lut import validate_lut, validate_unit_absorption_spectrum
from spectra.geometry.amf import effective_layer_height, path_amf
from spectra.protocols import UnitAbsorptionProvider
from spectra.registry import get_unit_absorption_provider
from spectra.spectral import (
    LibRadtranArtsProvider,
    ToyUnitAbsorption,
    UnitAbsorptionModel,
    build_slant_absorption_lut,
    embed_column_in_radiance,
    model_from_provider,
)
from spectra.spectral.providers.base import SPECTRUM_ATTRS
from spectra.types import SpectralResponse, ViewingGeometry

NODES_PPM = [0.01, 0.1, 1.0, 10.0, 100.0]


@pytest.fixture
def response() -> SpectralResponse:
    return SpectralResponse(
        wavelength=np.linspace(2100.0, 2500.0, 40), fwhm=np.full(40, 10.0)
    )


@pytest.fixture
def geometry() -> ViewingGeometry:
    return ViewingGeometry(
        sza=30.0,
        saa=150.0,
        vza=5.0,
        vaa=90.0,
        surface_elevation=200.0,
        sensor_altitude=3000.0,
        lat=47.0,
        lon=8.0,
        time_utc=np.datetime64("2024-06-01T10:00:00"),
        platform="airborne",
    )


@pytest.fixture
def toy_lut(response, geometry) -> xr.Dataset:
    return ToyUnitAbsorption().lut(response, geometry, enhancements_ppm=NODES_PPM)


@pytest.fixture
def flat_radiance(response) -> xr.DataArray:
    n_across, n_along = 12, 20
    return xr.DataArray(
        np.full((n_across, n_along, response.size), 0.05, dtype=np.float32),
        dims=("across", "along", "band"),
        coords={
            "across": np.arange(n_across),
            "along": np.arange(n_along),
            "band": np.arange(response.size),
            "wavelength": ("band", response.wavelength),
            "fwhm": ("band", response.fwhm),
        },
        name="radiance",
        attrs={"units": "W m-2 nm-1 sr-1"},
    )


def gaussian_column(radiance: xr.DataArray, peak_ppm_m: float) -> xr.DataArray:
    across = np.arange(radiance.sizes["across"])[:, None]
    along = np.arange(radiance.sizes["along"])[None, :]
    values = peak_ppm_m * np.exp(
        -0.5 * (((across - 6) / 3.0) ** 2 + ((along - 10) / 5.0) ** 2)
    )
    return xr.DataArray(
        values.astype(np.float32),
        dims=("across", "along"),
        coords={"across": radiance["across"], "along": radiance["along"]},
        name="column",
        attrs={"units": "ppm-m"},
    )


def invert_linear(
    embedded: xr.DataArray, radiance: xr.DataArray, lut: xr.Dataset
) -> np.ndarray:
    """Least-squares path amount per pixel from ln(L_emb / L0) = s * amount."""
    depth = np.log(embedded.values.astype(float) / radiance.values.astype(float))
    spectrum = lut["unit_abs_spect"].isel(enhancement=0).values
    return np.tensordot(depth, spectrum, axes=([2], [0])) / np.dot(spectrum, spectrum)


# --- toy provider -----------------------------------------------------------


def test_toy_lut_and_spectrum_pass_contracts(toy_lut, response, geometry):
    validate_lut(toy_lut)
    spectrum = ToyUnitAbsorption().spectrum(response, geometry, enhancement_ppm=1.0)
    validate_unit_absorption_spectrum(spectrum)
    assert np.all(spectrum["unit_abs_spect"].values <= 0)
    assert tuple(spectrum["fwhm"].dims) == ("wavelength",)
    for key in SPECTRUM_ATTRS:
        assert key in spectrum.attrs
    assert spectrum.attrs["provider"] == "toy"
    assert toy_lut.attrs["provider"] == "toy"
    assert "scenario" not in toy_lut.attrs


def test_toy_spectrum_formula(response, geometry):
    provider = ToyUnitAbsorption()
    spectrum = provider.spectrum(
        response, geometry, enhancement_ppm=1000.0, radiance_leg="solar"
    )
    height = effective_layer_height(geometry.surface_elevation, 1000.0)
    amount = 1000.0 * height * path_amf(geometry.sza, geometry.vza, "solar")
    expected = -provider.line_shape(response.wavelength) / (
        1.0 + amount / provider.saturation_ppm_m
    )
    np.testing.assert_allclose(spectrum["unit_abs_spect"].values, expected, rtol=1e-12)
    assert spectrum.attrs["geometric_amf"] == path_amf(
        geometry.sza, geometry.vza, "solar"
    )
    assert spectrum.attrs["effective_layer_height_m"] == 800.0


def test_toy_lut_geometry(toy_lut, geometry):
    amf = path_amf(geometry.sza, geometry.vza, "both")
    assert toy_lut.attrs["geometric_air_mass_factor"] == amf
    assert toy_lut.attrs["effective_layer_height"] == 800.0
    np.testing.assert_allclose(
        toy_lut["vertical_amount"].values, np.asarray(NODES_PPM) * 800.0
    )
    np.testing.assert_allclose(
        toy_lut["slant_amount"].values, toy_lut["vertical_amount"].values * amf
    )
    # Saturation: rows weaken with the node amount.
    strength = -toy_lut["unit_abs_spect"].values.min(axis=1)
    assert np.all(np.diff(strength) < 0)


def test_toy_linear_inversion_recovers_column(flat_radiance, toy_lut):
    # Stay well below the saturation amount so the table is linear.
    column = gaussian_column(flat_radiance, peak_ppm_m=50.0)
    embedded = embed_column_in_radiance(flat_radiance, column, toy_lut)
    assert embedded.dtype == np.float32
    assert np.all(embedded.values <= flat_radiance.values)
    recovered = invert_linear(embedded, flat_radiance, toy_lut)
    mask = column.values > 5.0
    relative = np.abs(recovered[mask] - column.values[mask]) / column.values[mask]
    assert relative.max() < 1e-3


def test_toy_slant_convention_differs_by_amf(flat_radiance, toy_lut):
    amf = toy_lut.attrs["geometric_air_mass_factor"]
    column = gaussian_column(flat_radiance, peak_ppm_m=50.0)
    slant = embed_column_in_radiance(
        flat_radiance, column, toy_lut, column_convention="slant"
    )
    vertical = embed_column_in_radiance(
        flat_radiance, column, toy_lut, column_convention="vertical"
    )
    assert not np.array_equal(slant.values, vertical.values)
    recovered = invert_linear(slant, flat_radiance, toy_lut)
    mask = column.values > 5.0
    np.testing.assert_allclose(recovered[mask], column.values[mask] / amf, rtol=1e-3)
    equivalent = column / amf
    equivalent.attrs = column.attrs
    vertical_scaled = embed_column_in_radiance(flat_radiance, equivalent, toy_lut)
    np.testing.assert_allclose(slant.values, vertical_scaled.values, rtol=1e-5)


def test_plume_above_largest_node_raises(flat_radiance, toy_lut):
    column = gaussian_column(flat_radiance, peak_ppm_m=1.0e6)
    with pytest.raises(ValueError, match="beyond the tabulated maximum"):
        embed_column_in_radiance(flat_radiance, column, toy_lut)


def test_wavelengths_outside_lut_raise(flat_radiance, toy_lut):
    shifted = flat_radiance.assign_coords(
        wavelength=("band", flat_radiance["wavelength"].values + 1.0)
    )
    column = gaussian_column(flat_radiance, peak_ppm_m=1.0)
    with pytest.raises(ValueError, match="absent from the LUT"):
        embed_column_in_radiance(shifted, column, toy_lut)


def test_lut_rejects_bad_nodes(response, geometry):
    provider = ToyUnitAbsorption()
    with pytest.raises(ValueError, match="positive"):
        provider.lut(response, geometry, enhancements_ppm=[0.0, 1.0])
    with pytest.raises(ValueError, match="increasing"):
        provider.lut(response, geometry, enhancements_ppm=[1.0, 1.0])
    with pytest.raises(ValueError, match="non-empty"):
        provider.lut(response, geometry, enhancements_ppm=[])


def test_model_from_provider_satisfies_protocol(response, geometry):
    provider = ToyUnitAbsorption()
    model = model_from_provider(provider, response, geometry)
    assert isinstance(model, UnitAbsorptionModel)
    assert model.sza == geometry.sza
    assert model.elevation == geometry.surface_elevation
    table = build_slant_absorption_lut(model, NODES_PPM)
    direct = provider.lut(response, geometry, enhancements_ppm=NODES_PPM)
    assert set(direct.attrs) - set(table.attrs) == {"provider", "provider_version"}
    stripped = direct.copy()
    del stripped.attrs["provider"], stripped.attrs["provider_version"]
    xr.testing.assert_identical(table, stripped)


def test_toy_rejects_bad_parameters():
    with pytest.raises(ValueError):
        ToyUnitAbsorption(lines_nm=(2200.0,), strength=(1.0e-5, 2.0e-5))
    with pytest.raises(ValueError):
        ToyUnitAbsorption(width_nm=0.0)


# --- registry ---------------------------------------------------------------


def test_registry_returns_toy_provider():
    provider = get_unit_absorption_provider("toy")
    assert isinstance(provider, UnitAbsorptionProvider)
    assert isinstance(provider, ToyUnitAbsorption)
    assert provider.name == "toy"


def test_registry_lists_libradtran_provider():
    provider = get_unit_absorption_provider(
        "libradtran_arts", uvspec_path="/x", sol_reference_path="/x", arts_path="/x"
    )
    assert isinstance(provider, UnitAbsorptionProvider)
    assert provider.name == "libradtran_arts"


# --- libRadtran/ARTS provider ----------------------------------------------


def test_libradtran_missing_extra(monkeypatch, response, geometry):
    monkeypatch.setitem(sys.modules, "aviris4_ch4", None)
    monkeypatch.setitem(sys.modules, "aviris4_ch4.mf", None)
    provider = LibRadtranArtsProvider("/x", "/x", "/x")
    with pytest.raises(MissingExtraError) as info:
        provider.spectrum(response, geometry, enhancement_ppm=1.0)
    assert info.value.extra == "aviris4"
    assert info.value.package == "aviris4_ch4"


def test_libradtran_window_bounds_match_file_names():
    provider = LibRadtranArtsProvider("/x", "/x", "/x", wmin=1000.0, wmax=2500.0)
    assert provider.wmin == 1000 and isinstance(provider.wmin, int)
    assert provider.wmax == 2500 and isinstance(provider.wmax, int)
    assert LibRadtranArtsProvider("/x", "/x", "/x", wmin=1000.5).wmin == 1000.5


@pytest.fixture(scope="module")
def golden_geometry(golden_record) -> ViewingGeometry:
    geometry = golden_record["geometry"]
    return ViewingGeometry(
        sza=geometry["sza"],
        saa=0.0,
        vza=geometry["vza"],
        vaa=0.0,
        surface_elevation=geometry["elevation"],
        sensor_altitude=geometry["altitude"],
        lat=geometry["lat"],
        lon=geometry["lon"],
        time_utc=np.datetime64(geometry["time_utc"]),
        platform="airborne",
    )


@pytest.fixture(scope="module")
def libradtran_provider(resources_root, golden_record) -> LibRadtranArtsProvider:
    return LibRadtranArtsProvider(
        resources_root / "spectrum",
        resources_root / "spectrum",
        resources_root / "arts",
        xch4_bg=golden_record["geometry"]["xch4_bg"],
    )


@pytest.mark.resources
@pytest.mark.golden
def test_libradtran_lut_matches_golden(
    libradtran_provider, golden_radiance, golden_geometry, golden_record, golden_lut
):
    geometry = golden_record["geometry"]
    table = libradtran_provider.lut(
        SpectralResponse.from_scene(golden_radiance),
        golden_geometry,
        enhancements_ppm=geometry["nodes_ppm"],
        layer_height_m=geometry["layer_height"],
    )
    validate_lut(table)
    assert table.attrs["provider"] == "libradtran_arts"
    assert table.attrs["sza"] == geometry["sza"]
    stripped = table.copy()
    del stripped.attrs["provider"], stripped.attrs["provider_version"]
    xr.testing.assert_identical(stripped, golden_lut)
    assert np.array_equal(
        table["optical_depth"].values, golden_lut["optical_depth"].values
    )


@pytest.mark.resources
@pytest.mark.golden
def test_libradtran_spectrum_matches_golden(
    libradtran_provider, golden_radiance, golden_geometry, golden_inputs
):
    response = SpectralResponse.from_scene(golden_radiance)
    spectrum = libradtran_provider.spectrum(
        response, golden_geometry, enhancement_ppm=10.0
    )
    validate_unit_absorption_spectrum(spectrum)
    target = xr.open_dataset(golden_inputs / "target_10ppm.nc").load()
    assert np.array_equal(
        spectrum["unit_abs_spect"].values, target["unit_abs_spect"].values
    )
    # The aviris4_ch4 variables are kept next to the schema attributes.
    assert float(spectrum["enhancement"]) == 10.0
    assert spectrum.attrs["sza"] == float(target["sza"].squeeze())
    for key in SPECTRUM_ATTRS:
        assert key in spectrum.attrs
    # The model is cached per geometry and band arrays.
    assert libradtran_provider.model(response, golden_geometry) is (
        libradtran_provider.model(response, golden_geometry)
    )
