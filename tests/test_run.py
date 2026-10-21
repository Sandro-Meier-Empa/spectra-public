"""Run files: parsing, validation and the resolved round trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectra.contracts.errors import ContractError
from spectra.run import AnchorSpec, PlumeSpec, RunConfig, dump_run, load_run
from spectra.spectral.libradtran import LibRadtranArtsConfig
from spectra.spectral.toy import ToyForwardModel

RUN = """
sensor: emit
files:
  radiance: in/RAD.nc
  obs: /abs/OBS.nc
output: out/embedded.nc
seed: 7
forward_model:
  solar_reference: rt/solar.nc
  tau: {all: rt/tau_all.nc, CH4: rt/tau_ch4.nc}
  profiles: {z: rt/z.xml, t: rt/t.xml, CH4: rt/ch4.xml}
  backgrounds: {CH4: 1.93}
  scenario: midlatitude-summer
  n_enhancements: 8
plumes:
  - file: plumes/a.nc
    time_index: 120
    anchor: {lon: -102.9, lat: 32.1}
    rotation: 30
    emission_scale: 2.0
  - file: plumes/b.nc
    time_index: 90
    anchor: {along: 812, across: 233, corner: lower left}
    flip: x
  - file: plumes/b.nc
    time_index: 60
    anchor: {random: true, max_outside_fraction: 0.05}
  - file: plumes/b.nc
    time_index: 61
    anchor: random
"""


@pytest.fixture
def run_file(tmp_path: Path) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(RUN)
    return path


def test_load_run_resolves_paths_and_types(run_file: Path, tmp_path: Path) -> None:
    config = load_run(run_file)
    root = tmp_path.resolve()
    assert config.sensor == "emit"
    assert config.files == {"radiance": root / "in/RAD.nc", "obs": Path("/abs/OBS.nc")}
    assert config.output == root / "out/embedded.nc"
    assert config.seed == 7 and config.overwrite is False
    model = config.forward_model
    assert isinstance(model, LibRadtranArtsConfig)
    assert model.files.tau["CH4"] == root / "rt/tau_ch4.nc"
    assert model.files.profiles["z"] == root / "rt/z.xml"
    assert model.backgrounds == {"CH4": 1.93} and model.n_enhancements == 8
    first, second, third, fourth = config.plumes
    assert first.anchor.mode == "lonlat" and first.anchor.corner == "source"
    assert first.rotation == 30 and first.emission_scale == 2.0
    assert first.file == root / "plumes/a.nc"
    assert second.anchor == AnchorSpec(along=812, across=233, corner="lower left")
    assert second.anchor.mode == "pixel" and second.flip == "x"
    assert third.anchor.mode == "random"
    assert third.anchor.max_outside_fraction == 0.05
    assert fourth.anchor == AnchorSpec(random=True)


def test_dump_and_reload_round_trip(run_file: Path) -> None:
    config = load_run(run_file)
    drawn = PlumeSpec(
        file=config.plumes[2].file,
        time_index=60,
        anchor=AnchorSpec(random=True, along=5, across=6, max_outside_fraction=0.05),
    )
    resolved = config.replace(plumes=(*config.plumes[:2], drawn, config.plumes[3]))
    again = load_run(yaml.safe_load(dump_run(resolved)))
    assert again == resolved
    # A drawn random anchor replays as a pixel anchor.
    assert again.plumes[2].anchor.mode == "pixel"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([2000, 2500], ((2000.0, 2500.0),)),
        ([[2000, 2500], [1400, 1800.5]], ((1400.0, 1800.5), (2000.0, 2500.0))),
    ],
)
def test_wavelength_ranges_in_run(tmp_path: Path, value, expected) -> None:
    mapping = yaml.safe_load(RUN)
    assert load_run(mapping, base=tmp_path).wavelength_ranges_nm is None
    assert "wavelength_ranges_nm" not in yaml.safe_load(
        dump_run(load_run(mapping, base=tmp_path))
    )
    mapping["wavelength_ranges_nm"] = value
    config = load_run(mapping, base=tmp_path)
    assert config.wavelength_ranges_nm == expected
    assert load_run(yaml.safe_load(dump_run(config))) == config


def test_toy_forward_model_in_run(tmp_path: Path) -> None:
    mapping = yaml.safe_load(RUN)
    mapping["forward_model"] = {"model": "toy", "width_nm": 30.0}
    config = load_run(mapping, base=tmp_path)
    assert isinstance(config.forward_model, ToyForwardModel)
    dumped = yaml.safe_load(dump_run(config))
    assert dumped["forward_model"]["model"] == "toy"
    assert dumped["forward_model"]["width_nm"] == 30.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda m: m.update(colour="red"), "Unknown keys"),
        (lambda m: m.pop("output"), "needs 'output'"),
        (lambda m: m.update(plumes=[]), "non-empty"),
        (lambda m: m["plumes"][0].update(speed=3), r"plumes\[0\]"),
        (lambda m: m["plumes"][0].pop("time_index"), "time_index"),
        (lambda m: m["plumes"][0]["anchor"].pop("lat"), "both lon and lat"),
        (
            lambda m: m["plumes"][0]["anchor"].update(along=1, across=2),
            "either lon/lat",
        ),
        (lambda m: m["plumes"][0].update(anchor={}), "needs lon/lat"),
        (lambda m: m["plumes"][0].update(rotation=1.5), "integer"),
        (lambda m: m["plumes"][0].update(rotation=400), "rotation"),
        (lambda m: m["plumes"][0].update(flip="z"), "flip"),
        (lambda m: m["plumes"][0].update(emission_scale=0), "emission_scale"),
        (lambda m: m["plumes"][0]["anchor"].update(corner="centre"), "corner"),
        (lambda m: m.update(overwrite="yes"), "overwrite"),
        (lambda m: m.update(seed=-1), "seed"),
        (lambda m: m["forward_model"].pop("backgrounds"), "backgrounds"),
        (lambda m: m["forward_model"].update(model="modtran"), "model"),
        (lambda m: m["forward_model"]["backgrounds"].update(N2O=0.3), "N2O"),
        (lambda m: m.update(output="in/RAD.nc"), "never modifies"),
        (lambda m: m.update(wavelength_ranges_nm=[]), "pair"),
        (lambda m: m.update(wavelength_ranges_nm="2000-2500"), "pair"),
        (lambda m: m.update(wavelength_ranges_nm=[1400, 1800, 2000]), "pair"),
        (lambda m: m.update(wavelength_ranges_nm=[["a", 2500]]), "number"),
        (lambda m: m.update(wavelength_ranges_nm=[2500, 2000]), "lower < upper"),
        (
            lambda m: m.update(wavelength_ranges_nm=[[1400, 1800], [1800, 2500]]),
            "overlap",
        ),
    ],
)
def test_load_run_rejects(tmp_path: Path, change, match: str) -> None:
    mapping = yaml.safe_load(RUN)
    change(mapping)
    with pytest.raises(ContractError, match=match):
        load_run(mapping, base=tmp_path)


def test_missing_run_file(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="does not exist"):
        load_run(tmp_path / "absent.yaml")


def test_run_config_requires_plumes(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="at least one plume"):
        RunConfig(
            sensor="emit",
            files={"radiance": tmp_path / "a.nc"},
            output=tmp_path / "b.nc",
            plumes=(),
            forward_model=ToyForwardModel(),
        )
