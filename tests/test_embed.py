"""End to end: run file -> native output, through the API and the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pytest
import xarray as xr
import yaml

import spectra
from spectra.cli import main
from spectra.contracts.errors import ContractError
from spectra.geometry.crs import crs_from_dataset, to_lonlat
from spectra.plume.io import write_plume_column
from spectra.sensors import get_sensor
from spectra.testing import make_plume_column
from spectra.testing.native import (
    assert_native_preserved,
    native_snapshot,
    write_aviris4_files,
    write_emit_files,
)

pytest.importorskip("h5py")

TOY = {"model": "toy", "width_nm": 60.0, "saturation_ppm_m": 5e4}
SENSORS: dict[str, dict[str, Any]] = {
    "aviris4": {"dx": 0.5, "pixel": 2.0},
    "emit": {"dx": 20.0, "pixel": 60.0},
}


def write_scene(sensor: str, directory: Path) -> dict[str, Path]:
    if sensor == "aviris4":
        return write_aviris4_files(directory, n_across=30, n_along=40)
    return write_emit_files(directory, n_downtrack=40, n_crosstrack=30)


def make_run(sensor: str, tmp_path: Path, **changes: Any) -> dict[str, Any]:
    files = write_scene(sensor, tmp_path / "in")
    plume = tmp_path / "plume.nc"
    write_plume_column(
        make_plume_column(nx=24, ny=24, dx=SENSORS[sensor]["dx"], n_time=2, seed=1),
        plume,
    )
    run = {
        "sensor": sensor,
        "files": {role: str(path) for role, path in files.items()},
        "output": str(tmp_path / "out" / "embedded.nc"),
        "seed": 11,
        "forward_model": dict(TOY),
        "plumes": [
            {
                "file": str(plume),
                "time_index": 1,
                "anchor": {"along": 10, "across": 10},
            },
            {
                "file": str(plume),
                "time_index": 0,
                "anchor": {"along": 13, "across": 12},
                "rotation": 45,
                "flip": "x",
            },
            {"file": str(plume), "time_index": 1, "anchor": "random"},
        ],
    }
    run.update(changes)
    return run


def native_radiance(path: Path) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        ds["radiance"].set_auto_maskandscale(False)
        return np.asarray(ds["radiance"][:])


@pytest.mark.parametrize("sensor", list(SENSORS))
def test_embed_writes_native_file(sensor: str, tmp_path: Path) -> None:
    run = make_run(sensor, tmp_path)
    # A third plume well apart from the first two forms its own cluster.
    run["plumes"][2]["anchor"] = {"along": 28, "across": 20}
    adapter = get_sensor(sensor)
    files = {role: Path(path) for role, path in run["files"].items()}
    before = native_radiance(files["radiance"])

    outcome = spectra.embed(run)

    result = outcome.result
    assert outcome.output == Path(run["output"]).resolve()
    assert [cluster.sources for cluster in result.clusters] == [(0, 1), (2,)]
    after = native_radiance(outcome.output)
    changed = after != before
    assert changed.any()
    # The exporter and the in-memory view apply the same attenuation.
    scene = adapter.open_scene(files)
    expected = result.to_dataset(scene)["radiance"]
    native_dims = list(xr.open_dataset(files["radiance"])["radiance"].dims)
    common = [adapter.LAYOUT.dims[str(dim)] for dim in native_dims]
    expected_native = expected.transpose(*common).values
    valid = np.isfinite(expected_native)
    np.testing.assert_array_equal(after[valid], expected_native[valid])
    # Everything but the attenuated elements is untouched at the storage level.
    mask = {"radiance": changed}
    assert_native_preserved(
        native_snapshot(files["radiance"], masks=mask),
        native_snapshot(outcome.output, masks=mask),
        new_objects=[
            name
            for name in native_snapshot(outcome.output)
            if name not in native_snapshot(files["radiance"])
        ],
        new_root_attrs=[
            name
            for name in xr.open_dataset(outcome.output).attrs
            if name.startswith("spectra_")
        ],
    )
    # Mass is conserved per plume on the sensor grid.
    area = np.asarray(scene["pixel_area"].values)
    for source in result.sources:
        window_area = area[source.window.across, source.window.along]
        assert float((source.injected * window_area).sum()) == pytest.approx(
            source.injected_mass_kg, rel=1e-4
        )
        assert source.outside_fraction == pytest.approx(0.0, abs=1e-9)


def test_run_config_in_output_reproduces_it(tmp_path: Path) -> None:
    run = make_run("emit", tmp_path)
    first = spectra.embed(run)
    with xr.open_dataset(first.output) as dataset:
        stored = yaml.safe_load(dataset.attrs["spectra_run_config"])
        source_file = dataset.attrs["spectra_source_radiance_path"]
    assert stored["seed"] == 11
    drawn = stored["plumes"][2]["anchor"]
    assert drawn["random"] is True and {"along", "across"} <= set(drawn)
    assert source_file == str(Path(run["files"]["radiance"]).resolve())
    stored["output"] = str(tmp_path / "replay.nc")
    stored["seed"] = 99  # the drawn anchor no longer depends on the seed
    replay = spectra.embed(stored)
    np.testing.assert_array_equal(
        native_radiance(replay.output), native_radiance(first.output)
    )
    with xr.open_dataset(first.output) as a, xr.open_dataset(replay.output) as b:
        np.testing.assert_array_equal(a["injected_ch4_mass"], b["injected_ch4_mass"])
        np.testing.assert_array_equal(a["source_downtrack"], b["source_downtrack"])


def test_seed_controls_random_anchors(tmp_path: Path) -> None:
    run = make_run("aviris4", tmp_path)
    run["plumes"] = run["plumes"][2:]
    anchors = []
    for seed in (5, 5, 6):
        outcome = spectra.embed(
            run, seed=seed, output=tmp_path / f"seed{seed}.nc", overwrite=True
        )
        source = outcome.result.sources[0]
        anchors.append((source.anchor_along, source.anchor_across))
    assert anchors[0] == anchors[1]
    assert anchors[0] != anchors[2]


def test_lonlat_anchor_places_the_source(tmp_path: Path) -> None:
    run = make_run("emit", tmp_path)
    files = {role: Path(path) for role, path in run["files"].items()}
    scene = get_sensor("emit").open_scene(files)
    x = 0.5 * float(scene["x"][14, 20] + scene["x"][15, 21])
    y = 0.5 * float(scene["y"][14, 20] + scene["y"][15, 21])
    lon, lat = (float(value) for value in to_lonlat(crs_from_dataset(scene), x, y))
    run["plumes"] = [
        {
            "file": run["plumes"][0]["file"],
            "time_index": 1,
            "anchor": {"lon": lon, "lat": lat},
            "rotation": 120,
        }
    ]
    outcome = spectra.embed(run)
    (source,) = outcome.result.sources
    assert source.anchor_mode == "lonlat"
    assert (source.lon, source.lat) == pytest.approx((lon, lat), abs=1e-9)
    with xr.open_dataset(outcome.output) as dataset:
        assert float(dataset["source_lon"][0]) == pytest.approx(lon, abs=1e-9)
        assert int(dataset["source_anchor_mode"].values[0]) == 0
        assert dataset["source_anchor_mode"].attrs["flag_meanings"].split()[0] == (
            "lonlat"
        )


def test_embed_errors(tmp_path: Path) -> None:
    run = make_run("aviris4", tmp_path)
    other = tmp_path / "co2.nc"
    plume = make_plume_column(nx=24, ny=24, dx=0.5, n_time=2, seed=1)
    write_plume_column(plume.assign_attrs(gas="CO2"), other)
    mixed = dict(
        run, plumes=[*run["plumes"][:1], dict(run["plumes"][0], file=str(other))]
    )
    with pytest.raises(ContractError, match="same gas"):
        spectra.embed(mixed)
    tight = dict(run, forward_model={**TOY, "enhancements_ppm": [0.001, 0.002]})
    with pytest.raises(ContractError, match="raise enhancement_max_ppm"):
        spectra.embed(tight)
    spectra.embed(run)
    with pytest.raises(FileExistsError):
        spectra.embed(run)
    spectra.embed(run, overwrite=True)


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = make_run("emit", tmp_path)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(run))

    main(["embed", str(path), "--dry-run"])
    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["seed"] == 11
    assert {"along", "across"} <= set(resolved["plumes"][2]["anchor"])
    assert not Path(run["output"]).exists()

    main(["embed", str(path)])
    assert "Wrote" in capsys.readouterr().out
    assert Path(run["output"]).exists()
    with pytest.raises(SystemExit):
        main(["embed", str(path)])
    assert "exists" in capsys.readouterr().err
    main(["embed", str(path), "--overwrite", "--seed", "3"])

    main(["sensors"])
    listing = capsys.readouterr().out
    assert "emit" in listing and "obs" in listing and "aviris3" in listing

    main(["validate", run["plumes"][0]["file"], "--contract", "plume-column"])
    assert "valid plume-column" in capsys.readouterr().out
