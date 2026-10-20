"""Tests of the ported libRadtran/ARTS forward model and the toy model.

The resource-gated tests compare the port bit for bit against tables the
original ``aviris4_ch4.mf.UnitAbsorption`` produced (``tests/golden``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import spectra.spectral.libradtran as libradtran
from spectra.contracts.errors import ContractError
from spectra.contracts.lut import validate_lut
from spectra.protocols import ForwardModel
from spectra.spectral.libradtran import (
    LibRadtranArts,
    LibRadtranArtsConfig,
    LibRadtranArtsFiles,
    RadiativeTransferResources,
    UnitAbsorption,
)
from spectra.spectral.toy import ToyForwardModel
from spectra.types import SpectralResponse, ViewingGeometry

GOLDEN = Path(__file__).parent / "golden"
ARTS = GOLDEN / "inputs" / "arts" / "midlatitude-summer"
REFERENCE = GOLDEN / "reference"
SCENARIO = "midlatitude-summer"
PROFILE_ROLES = ("z", "t", "CH4", "CO2", "H2O")


def arts_profiles(root: Path) -> dict[str, Path]:
    """Return the ARTS profile files of the scenario by role."""
    return {role: root / f"{SCENARIO}.{role}.xml" for role in PROFILE_ROLES}


def resource_files(root: Path) -> LibRadtranArtsFiles:
    """Return the real resource bundle below ``root`` (spectrum/, arts/)."""
    spectrum = root / "spectrum"
    return LibRadtranArtsFiles(
        solar_reference=spectrum
        / "hybrid_reference_spectrum_p005nm_resolution_c2021-03-04_with_unc.nc",
        tau={
            name: spectrum / f"UVSPEC_moltau_{SCENARIO}_{name}_1000-2500nm-0.001nm.nc"
            for name in ("all", "CH4", "CO2", "H2O")
        },
        profiles=arts_profiles(root / "arts" / SCENARIO),
    )


def geometry(
    sza: float, vza: float, elevation: float, altitude: float
) -> ViewingGeometry:
    """Return a viewing geometry."""
    return ViewingGeometry(
        sza=sza, vza=vza, surface_elevation=elevation, sensor_altitude=altitude
    )


@pytest.fixture(scope="module")
def synthetic_files(tmp_path_factory: pytest.TempPathFactory) -> LibRadtranArtsFiles:
    """Write a tiny resource bundle around 2300 nm with the real ARTS profiles."""
    root = tmp_path_factory.mktemp("rt")
    wavelength = np.linspace(2290.0, 2310.0, 2001)
    # libRadtran orders levels and layers from the top of the atmosphere down.
    z_km = libradtran._parse_tensor(ARTS / f"{SCENARIO}.z.xml")[1:][::-1] / 1e3
    n_layers = z_km.size - 1
    line = np.exp(-((wavelength - 2300.0) ** 2) / (2 * 0.5**2))
    layers = np.linspace(0.1, 1.0, n_layers)[:, None]
    tau_gas = 2e-3 * layers * (0.2 + line)
    taus = {"CH4": tau_gas, "all": tau_gas + 1e-3 * layers}
    tau_paths = {}
    for name, tau in taus.items():
        dataset = xr.Dataset(
            {
                "wvlmin": (("none",), [wavelength[0]]),
                "wvlmax": (("none",), [wavelength[-1]]),
                "z": (("nlev",), z_km),
                "wvl": (("nwvl",), wavelength),
                "tau": (("nlyr", "nwvl", "nstk", "nstk2"), tau[:, :, None, None]),
            }
        )
        tau_paths[name] = root / f"tau_{name}.nc"
        dataset.to_netcdf(tau_paths[name])
    solar = xr.Dataset(
        {
            "SSI": (("wavelength",), 0.1 + 0.01 * np.cos(wavelength / 3.0)),
            "Vacuum Wavelength": (("wavelength",), wavelength),
        }
    )
    solar_path = root / "solar.nc"
    solar.to_netcdf(solar_path)
    return LibRadtranArtsFiles(
        solar_reference=solar_path, tau=tau_paths, profiles=arts_profiles(ARTS)
    )


@pytest.fixture
def synthetic_model(synthetic_files: LibRadtranArtsFiles) -> LibRadtranArts:
    return LibRadtranArts(
        LibRadtranArtsConfig(
            files=synthetic_files, backgrounds={"CH4": 1.9}, n_enhancements=5
        )
    )


SYNTHETIC_BANDS = SpectralResponse(
    wavelength=np.array([2296.0, 2300.0, 2304.0]), fwhm=np.array([2.0, 2.0, 2.0])
)


def test_models_satisfy_protocol(synthetic_model):
    assert isinstance(synthetic_model, ForwardModel)
    assert isinstance(ToyForwardModel(), ForwardModel)


def test_arts_parser_matches_lxml_reference():
    profiles = json.loads((REFERENCE / "profiles.json").read_text())
    files = LibRadtranArtsFiles(
        solar_reference=Path("unused"), tau={}, profiles=arts_profiles(ARTS)
    )
    resources = RadiativeTransferResources(files, wmin=1000, wmax=2500)
    for gas, expected in profiles.items():
        for name, values in zip("zcpt", resources.profile(gas), strict=True):
            assert np.array_equal(values, np.asarray(expected[name])), (gas, name)
        assert float(resources.column_average(gas)) == expected["xgas_ref_ppmv"]


def test_window_defaults_to_tau_coverage(synthetic_model):
    assert synthetic_model.spectral_range_nm == (2290.0, 2310.0)
    assert synthetic_model.resources.wmin == 2290


def test_synthetic_lut_reads_each_file_once(synthetic_model, monkeypatch):
    opened: list[Path] = []
    original = libradtran._open_tau

    def counting(path: Path) -> xr.Dataset:
        opened.append(path)
        return original(path)

    monkeypatch.setattr(libradtran, "_open_tau", counting)
    model = LibRadtranArts(synthetic_model.config)
    first = model.lut(SYNTHETIC_BANDS, geometry(30.0, 5.0, 100.0, 4000.0), gas="ch4")
    second = model.lut(SYNTHETIC_BANDS, geometry(40.0, 10.0, 1500.0, 4000.0), gas="CH4")
    validate_lut(first)
    validate_lut(second)
    assert np.all(first["optical_depth"].values < 0)
    assert first.attrs["sza"] == 30.0 and second.attrs["sza"] == 40.0
    # wvlmin/wvlmax of "all", then the "all" and "CH4" optical depths once.
    assert len(opened) == 3


def test_memoised_radiance_is_unchanged(synthetic_files):
    resources = RadiativeTransferResources(synthetic_files)

    def model() -> UnitAbsorption:
        return UnitAbsorption(
            "CH4",
            sza=30.0,
            vza=5.0,
            cw=SYNTHETIC_BANDS.wavelength,
            fwhm=SYNTHETIC_BANDS.fwhm,
            elevation=100.0,
            altitude=4000.0,
            resources=resources,
            backgrounds={"CH4": 1.9},
        )

    cached = model()
    first = cached.compute_unit_absorption_spectrum(5.0)
    again = cached.compute_unit_absorption_spectrum(5.0)
    fresh = model().compute_unit_absorption_spectrum(5.0)
    assert np.array_equal(first["unit_abs_spect"], again["unit_abs_spect"])
    assert np.array_equal(first["unit_abs_spect"], fresh["unit_abs_spect"])


def test_negative_elevation_is_clamped(synthetic_model, caplog):
    bands, view = SYNTHETIC_BANDS, geometry(30.0, 5.0, -28.0, 400_000.0)
    with caplog.at_level("WARNING", logger="spectra.spectral.libradtran"):
        clamped = synthetic_model.lut(bands, view, gas="CH4")
    assert "below the lowest model level" in caplog.text
    assert clamped.attrs["surface_elevation_clamped_from"] == -28.0
    assert clamped.attrs["effective_layer_height"] == 1000.0
    at_sea_level = synthetic_model.lut(
        bands, geometry(30.0, 5.0, 0.0, 400_000.0), gas="CH4"
    )
    del clamped.attrs["surface_elevation_clamped_from"]
    xr.testing.assert_identical(clamped, at_sea_level)


@pytest.mark.parametrize(
    ("angles", "match"),
    [
        ((30.0, 5.0, 100.0, float("nan")), "altitude is not finite"),
        ((95.0, 5.0, 100.0, 4000.0), r"\[0, 90\)"),
    ],
)
def test_invalid_geometry_is_rejected(synthetic_model, angles, match):
    with pytest.raises(ContractError, match=match):
        synthetic_model.lut(SYNTHETIC_BANDS, geometry(*angles), gas="CH4")


def test_bands_outside_window_are_rejected(synthetic_model):
    bands = SpectralResponse(wavelength=np.array([2200.0, 2300.0]), fwhm=np.ones(2))
    with pytest.raises(ContractError, match="outside the modelled"):
        synthetic_model.lut(bands, geometry(30.0, 5.0, 100.0, 4000.0), gas="CH4")


def test_check_reports_missing_background_and_files(synthetic_files, tmp_path):
    model = LibRadtranArts(
        LibRadtranArtsConfig(files=synthetic_files, backgrounds={"CO2": 420.0})
    )
    with pytest.raises(ContractError, match="lacks the background of CH4"):
        model.check("ch4")
    with pytest.raises(ContractError, match="not supported"):
        model.check("N2O")
    without_ch4 = LibRadtranArtsFiles(
        solar_reference=synthetic_files.solar_reference,
        tau={"all": synthetic_files.tau["all"]},
        profiles=synthetic_files.profiles,
    )
    with pytest.raises(ContractError, match=r"tau\.CH4"):
        without_ch4.required("CH4", {"CH4": 1.9})
    missing = LibRadtranArtsFiles(
        solar_reference=tmp_path / "absent.nc",
        tau=synthetic_files.tau,
        profiles=synthetic_files.profiles,
    )
    model = LibRadtranArts(
        LibRadtranArtsConfig(files=missing, backgrounds={"CH4": 1.9})
    )
    with pytest.raises(ContractError, match="not found"):
        model.check("CH4")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backgrounds": {"N2O": 0.3}}, "not supported"),
        ({"backgrounds": {"CH4": -1.0}}, "positive"),
        ({"backgrounds": {"CH4": 1.9}, "wmin": 1000.0}, "both"),
        ({"backgrounds": {"CH4": 1.9}, "wmin": 2000.0, "wmax": 1000.0}, "smaller"),
        ({"backgrounds": {"CH4": 1.9}, "n_enhancements": 1}, "at least 2"),
        ({"backgrounds": {"CH4": 1.9}, "enhancement_min_ppm": 0.0}, "enhancement"),
    ],
)
def test_config_validation(synthetic_files, kwargs, match):
    with pytest.raises(ContractError, match=match):
        LibRadtranArtsConfig(files=synthetic_files, **kwargs)


def test_config_nodes_and_provenance(synthetic_model):
    config = synthetic_model.config
    assert config.backgrounds == {"CH4": 1.9}
    assert len(config.enhancements_ppm) == 5
    assert config.enhancements_ppm[0] == pytest.approx(0.01)
    record = synthetic_model.provenance()
    assert record["name"] == "libradtran_arts"
    assert record["background_ch4_ppm"] == 1.9
    assert record["tau_ch4"].endswith("tau_CH4.nc")
    assert record["profile_z"].endswith(f"{SCENARIO}.z.xml")


def test_toy_lut_is_nonlinear():
    model = ToyForwardModel(saturation_ppm_m=500.0)
    bands = SpectralResponse(
        wavelength=np.linspace(2100.0, 2450.0, 30), fwhm=np.full(30, 8.0)
    )
    table = model.lut(bands, geometry(30.0, 5.0, 100.0, 4000.0), gas="ch4")
    validate_lut(table)
    depth = -table["optical_depth"].values.max(axis=1)
    amount = table["vertical_amount"].values
    assert np.all(np.diff(depth / amount) < 0)
    assert model.provenance()["name"] == "toy"


# Resource-gated comparisons against the recorded aviris4_ch4 output.


@pytest.fixture(scope="module")
def real_files(resources_root: Path) -> LibRadtranArtsFiles:
    return resource_files(resources_root)


@pytest.mark.resources
@pytest.mark.golden
def test_lut_matches_golden(real_files, golden_record, golden_radiance, golden_lut):
    record = golden_record["geometry"]
    nodes = record["nodes_ppm"]
    model = LibRadtranArts(
        LibRadtranArtsConfig(
            files=real_files,
            backgrounds={"CH4": record["xch4_bg"]},
            scenario=SCENARIO,
            wmin=1000,
            wmax=2500,
            enhancement_min_ppm=nodes[0],
            enhancement_max_ppm=nodes[-1],
            n_enhancements=len(nodes),
            layer_height_m=record["layer_height"],
        )
    )
    assert np.array_equal(model.enhancements_ppm, nodes)
    view = geometry(
        record["sza"], record["vza"], record["elevation"], record["altitude"]
    )
    response = SpectralResponse.from_scene(golden_radiance)
    table = model.lut(response, view, gas="CH4")
    xr.testing.assert_identical(table, golden_lut)
    target = xr.open_dataset(GOLDEN / "inputs" / "target_10ppm.nc").load()
    spectrum = model.model(response, view, gas="CH4").compute_unit_absorption_spectrum(
        10.0
    )
    assert np.array_equal(spectrum["unit_abs_spect"], target["unit_abs_spect"])


@pytest.mark.resources
@pytest.mark.golden
@pytest.mark.parametrize("case", ["emit", "aviris4", "aviris4_high"])
def test_lut_matches_reference(real_files, case):
    record = json.loads((REFERENCE / "record.json").read_text())["cases"][case]
    expected = xr.open_dataset(REFERENCE / f"{case}_lut.nc").load()
    target = xr.open_dataset(REFERENCE / f"{case}_target_10ppm.nc").load()
    nodes = record["nodes_ppm"]
    model = LibRadtranArts(
        LibRadtranArtsConfig(
            files=real_files,
            backgrounds=record["backgrounds"],
            scenario=record["scenario"],
            enhancement_min_ppm=nodes[0],
            enhancement_max_ppm=nodes[-1],
            n_enhancements=len(nodes),
            layer_height_m=record["layer_height"],
        )
    )
    response = SpectralResponse(
        wavelength=target["wavelength"].values, fwhm=target["fwhm"].values
    )
    view = geometry(
        record["sza"], record["vza"], record["elevation"], record["altitude"]
    )
    table = model.lut(response, view, gas="CH4")
    xr.testing.assert_identical(table, expected)
    spectrum = model.model(response, view, gas="CH4").compute_unit_absorption_spectrum(
        10.0
    )
    assert np.array_equal(spectrum["unit_abs_spect"], target["unit_abs_spect"])
