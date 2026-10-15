"""CDO preparation, mass validation and the CDO regridder."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pyproj
import pytest
import xarray as xr

from spectra.contracts.errors import ContractError, RegridderUnavailableError
from spectra.geometry.swath import place_in_scene
from spectra.protocols import Regridder
from spectra.regrid.cdo import (
    CdoPreparation,
    CdoRegridder,
    _cell_areas_from_lonlat,
    georeferenced_source,
    prepare_cdo_files,
    prepare_regular_target_grid,
    run_cdo_remapcon,
    validate_conservative_mass,
)
from spectra.scene import merge_scene_with_column
from spectra.types import MapGrid, Placement, SceneGeometry, Window

PLACEMENTS = ("rot", "flip", "partial")
# Global attributes of the raw MicroHH dataset that aviris4-ch4 copied onto the
# CDO source; a DataArray column does not carry dataset attributes.
RAW_DATASET_ATTRS = ("golden_case", "golden_time_index", "simulation_time_s")


def sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values)
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def placement_from_record(record: dict) -> Placement:
    return Placement(
        anchor_along=record["along_index"],
        anchor_across=record["across_index"],
        anchor_corner=record["anchor"],
        rotation_deg=record["rotation"],
        flip=record["flip_axis"] or "none",
    )


def assert_same_storage(written: Path, expected: Path) -> None:
    """Compare raw NetCDF layout: variable order, dtypes, chunking, filters."""
    with netCDF4.Dataset(written) as a, netCDF4.Dataset(expected) as b:
        assert list(a.variables) == list(b.variables)
        assert list(a.dimensions) == list(b.dimensions)
        for name, variable in b.variables.items():
            mine = a.variables[name]
            assert mine.dtype == variable.dtype, name
            assert mine.dimensions == variable.dimensions, name
            assert mine.chunking() == variable.chunking(), name
            assert mine.filters() == variable.filters(), name


@pytest.fixture(scope="module")
def geometry(golden_metadata: xr.Dataset) -> SceneGeometry:
    return SceneGeometry.from_scene(golden_metadata)


@pytest.fixture(scope="module")
def column(golden_microhh_raw: xr.Dataset) -> xr.DataArray:
    column = golden_microhh_raw["ch4_path"].copy()
    column.attrs["units"] = "kg m-2"
    return column


@pytest.mark.golden
@pytest.mark.parametrize("name", PLACEMENTS)
def test_prepare_cdo_files_matches_golden(
    golden_record: dict,
    golden_expected: Path,
    geometry: SceneGeometry,
    column: xr.DataArray,
    tmp_path: Path,
    name: str,
) -> None:
    record = golden_record["placements"][name]
    scene_placement = place_in_scene(
        column, geometry, placement=placement_from_record(record)
    )
    preparation = prepare_cdo_files(
        scene_placement,
        geometry,
        tmp_path / "source.nc",
        tmp_path / "target.nc",
        variable="ch4_path",
        metadata_source="metadata_window.nc",
    )
    assert preparation.along_index == record["along_index"]
    assert preparation.across_index == record["across_index"]
    assert preparation.along_slice == slice(
        record["along_slice"]["start"], record["along_slice"]["stop"]
    )
    assert preparation.across_slice == slice(
        record["across_slice"]["start"], record["across_slice"]["stop"]
    )
    assert preparation.longitude == pytest.approx(record["longitude"], abs=1e-12)
    assert preparation.latitude == pytest.approx(record["latitude"], abs=1e-12)
    assert preparation.x_offset == pytest.approx(record["x_offset"], abs=1e-12)
    assert preparation.y_offset == pytest.approx(record["y_offset"], abs=1e-12)
    assert preparation.rotation == record["rotation"]
    assert preparation.flip_axis == record["flip_axis"]
    assert preparation.outside_fraction == pytest.approx(
        record["outside_fraction"], abs=1e-12
    )
    assert preparation.attempts is None
    assert preparation.window == scene_placement.window
    assert preparation.command == (
        f"cdo remapcon,{tmp_path / 'target.nc'} {tmp_path / 'source.nc'} <output.nc>"
    )

    with xr.open_dataset(preparation.source_path) as dataset:
        source = dataset.load()
    with xr.open_dataset(golden_expected / f"placement_{name}_cdo_source.nc") as ds:
        expected_source = ds.load()
    # Accepted differences: SPECTRA requires and writes the column units; the
    # golden source carries the raw MicroHH dataset attributes.
    assert source["ch4_path"].attrs.pop("units") == "kg m-2"
    for key in RAW_DATASET_ATTRS:
        expected_source.attrs.pop(key)
    xr.testing.assert_identical(source, expected_source)
    assert_same_storage(
        preparation.source_path, golden_expected / f"placement_{name}_cdo_source.nc"
    )
    hashes = golden_record["hashes"]
    assert (
        sha256(source["ch4_path"].values) == hashes[f"placement_{name}_source_ch4_path"]
    )
    assert sha256(source["lonc"].values) == hashes[f"placement_{name}_source_lonc"]

    with xr.open_dataset(preparation.target_grid_path) as dataset:
        target = dataset.load()
    with xr.open_dataset(golden_expected / f"placement_{name}_cdo_target.nc") as ds:
        expected_target = ds.load()
    xr.testing.assert_identical(target, expected_target)
    assert_same_storage(
        preparation.target_grid_path,
        golden_expected / f"placement_{name}_cdo_target.nc",
    )
    assert sha256(target["lonc"].values) == hashes[f"placement_{name}_target_lonc"]
    with netCDF4.Dataset(preparation.target_grid_path) as grid:
        assert grid.variables["lonc"].dtype == np.dtype("float64")
        assert grid.variables["dummy"].dtype == np.dtype("float32")
        assert grid.variables["across"].dtype == np.dtype("int32")
        n_across, n_along = scene_placement.window.shape
        assert grid.variables["lonc"].chunking() == [min(8, n_across), n_along, 4]
        assert grid.variables["lon"].chunking() == [min(8, n_across), n_along]


@pytest.mark.golden
def test_prepare_cdo_files_output_policy(
    geometry: SceneGeometry, column: xr.DataArray, tmp_path: Path
) -> None:
    scene_placement = place_in_scene(
        column, geometry, placement=Placement(anchor_along=15, anchor_across=2)
    )
    with pytest.raises(ContractError, match="different"):
        prepare_cdo_files(
            scene_placement,
            geometry,
            tmp_path / "a.nc",
            tmp_path / "a.nc",
            variable="c",
        )
    first = prepare_cdo_files(
        scene_placement, geometry, tmp_path / "s.nc", tmp_path / "t.nc", variable="c"
    )
    assert first.source_path.exists() and first.target_grid_path.exists()
    with pytest.raises(FileExistsError):
        prepare_cdo_files(
            scene_placement,
            geometry,
            tmp_path / "s.nc",
            tmp_path / "t.nc",
            variable="c",
        )
    second = prepare_cdo_files(
        scene_placement,
        geometry,
        tmp_path / "s.nc",
        tmp_path / "t.nc",
        variable="c",
        overwrite=True,
    )
    with xr.open_dataset(second.source_path) as source:
        assert "c" in source
        assert source.attrs["embedding_plume_variable"] == "c"
        # The default metadata source is the geometry source file name.
        assert source.attrs["embedding_metadata_source"] == "metadata_window.nc"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.golden
def test_georeferenced_source_keeps_scalar_time(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    timed = column.assign_coords(time=np.datetime64("2024-09-16T11:33:13", "ns"))
    scene_placement = place_in_scene(
        timed, geometry, placement=Placement(anchor_along=15, anchor_across=2)
    )
    source = georeferenced_source(scene_placement, geometry, variable="ch4_path")
    assert source["time"].ndim == 0
    assert source["time"].encoding["units"] == "seconds since 1970-01-01 00:00:00"
    assert source["ch4_path"].dims == ("nrows", "nobs")
    assert source["lonc"].dims == ("nrows", "nobs", "vertices")
    assert source["ch4_path"].dtype == column.dtype


@pytest.mark.golden
def test_validate_conservative_mass_on_prepared_files(
    golden_record: dict, geometry: SceneGeometry, column: xr.DataArray, tmp_path: Path
) -> None:
    # The fully contained golden placement, so the whole mass lies inside.
    scene_placement = place_in_scene(
        column,
        geometry,
        placement=placement_from_record(golden_record["placements"]["rot"]),
    )
    preparation = prepare_cdo_files(
        scene_placement,
        geometry,
        tmp_path / "s.nc",
        tmp_path / "t.nc",
        variable="ch4_path",
    )
    with xr.open_dataset(preparation.target_grid_path) as target:
        grid = target.load()
    # A perfectly conservative fake remap spreads the in-window mass uniformly.
    with xr.open_dataset(preparation.source_path) as source:
        total = float(
            (
                source["ch4_path"].astype("float64") * scene_placement.placed.cell_area
            ).sum()
        )
    area = grid["dummy"].copy(data=np.zeros(grid["dummy"].shape))
    target_area = _cell_areas_from_lonlat(grid)
    uniform = total / float(target_area.sum())
    remapped = xr.Dataset(
        {
            "ch4_path": area + uniform,
            "lon": grid["lon"],
            "lat": grid["lat"],
        }
    )
    result = validate_conservative_mass(
        preparation.source_path, remapped, preparation.target_grid_path, "ch4_path"
    )
    # Source cell areas from the lon/lat corners are 1 m2 to about 1e-9.
    assert result.attrs["maximum_relative_error"] < 1e-6
    assert float(result["inside_mass_fraction"]) == pytest.approx(1.0, abs=1e-6)
    assert float(result["source_mass"]) == pytest.approx(total, rel=1e-6)
    with pytest.raises(ContractError, match="exceeds"):
        validate_conservative_mass(
            preparation.source_path,
            remapped * 1.01,
            preparation.target_grid_path,
            "ch4_path",
        )
    with pytest.raises(ContractError, match="rtol"):
        validate_conservative_mass(
            preparation.source_path,
            remapped,
            preparation.target_grid_path,
            "ch4_path",
            rtol=-1,
        )


@pytest.mark.data
def test_validate_conservative_mass_real_data(data_root: Path) -> None:
    result = validate_conservative_mass(
        data_root / "microhh_cdo.nc",
        data_root / "microhh_remapped.nc",
        data_root / "microhh_grid_aviris.nc",
        "ch4_path",
    )
    assert result.attrs["maximum_relative_error"] < 1e-3
    assert float(result["inside_mass_fraction"]) == pytest.approx(1.0, abs=1e-3)
    assert float(result["remapped_mass"]) == pytest.approx(
        float(result["source_mass_inside"]), rel=1e-3
    )


def test_run_cdo_remapcon_unavailable(tmp_path: Path) -> None:
    preparation = CdoPreparation(
        source_path=tmp_path / "s.nc",
        target_grid_path=tmp_path / "t.nc",
        along_index=0,
        across_index=0,
        along_slice=slice(0, 1),
        across_slice=slice(0, 1),
        longitude=0.0,
        latitude=0.0,
        x_offset=0.0,
        y_offset=0.0,
    )
    with pytest.raises(RegridderUnavailableError, match="not found"):
        run_cdo_remapcon(
            preparation, tmp_path / "o.nc", executable="spectra-missing-cdo-binary"
        )


def test_cdo_regridder_protocol_and_to_map() -> None:
    regridder = CdoRegridder()
    assert isinstance(regridder, Regridder)
    assert regridder.name == "cdo"
    grid = MapGrid(x=np.arange(3.0), y=np.arange(2.0), crs=pyproj.CRS.from_epsg(32632))
    with pytest.raises(NotImplementedError, match="to_map"):
        regridder.to_map(xr.DataArray(), grid, Placement())


@pytest.mark.golden
def test_cdo_regridder_unavailable(
    golden_metadata: xr.Dataset, column: xr.DataArray
) -> None:
    regridder = CdoRegridder(executable="spectra-missing-cdo-binary")
    with pytest.raises(RegridderUnavailableError):
        regridder.to_scene(
            column,
            golden_metadata,
            Placement(anchor_along=15, anchor_across=2, anchor_corner="upper left"),
        )


@pytest.mark.golden
@pytest.mark.skipif(shutil.which("cdo") is None, reason="cdo is not installed")
def test_cdo_regridder_to_scene(
    golden_record: dict,
    golden_metadata: xr.Dataset,
    column: xr.DataArray,
    tmp_path: Path,
) -> None:
    record = golden_record["placements"]["rot"]
    regridder = CdoRegridder(workdir=tmp_path, keep_files=True)
    result = regridder.to_scene(column, golden_metadata, placement_from_record(record))
    assert result.dims == ("across", "along")
    assert result.shape == (
        record["across_slice"]["stop"] - record["across_slice"]["start"],
        record["along_slice"]["stop"] - record["along_slice"]["start"],
    )
    assert result.attrs["units"] == "kg m-2"
    assert result.attrs["regrid_method"] == "cdo-remapcon"
    assert result.attrs["regrid_max_relative_mass_error"] < 1e-3
    assert result.attrs["window_across_start"] == record["across_slice"]["start"]
    assert result.attrs["embedding_along_index"] == record["along_index"]
    assert list(tmp_path.glob("cdo-*/cdo_remapped.nc"))
    window = Window(across=slice(0, 40), along=slice(0, 160))
    full = regridder.to_scene(
        column, golden_metadata, placement_from_record(record), window=window
    )
    assert full.shape == (40, 160)
    np.testing.assert_allclose(
        full.values[
            record["across_slice"]["start"] : record["across_slice"]["stop"],
            record["along_slice"]["start"] : record["along_slice"]["stop"],
        ],
        result.values,
    )


FAKE_CDO = """#!{python}
# Stand-in for ``cdo remapcon``: a uniform, mass-conserving fake remap laid
# out like a CDO output file (float coordinates, lon/lat bounds, crs).
import sys

import numpy as np
import xarray as xr

from spectra.regrid.cdo import (
    _area_integral,
    _cell_areas_from_lonlat,
    _source_window_overlap,
    _target_window_outline,
)

operator, source_path, output_path = sys.argv[1:4]
target_path = operator.split(",", 1)[1]
with xr.open_dataset(source_path, decode_times=False) as dataset:
    source = dataset.load()
with xr.open_dataset(target_path) as dataset:
    grid = dataset.load()
variable = source.attrs["embedding_plume_variable"]
outline = _target_window_outline(grid[["lonc", "latc"]])
inside = float(
    _area_integral(source, variable, weights=_source_window_overlap(source, outline))
)
area = _cell_areas_from_lonlat(grid)
values = np.full(area.shape, inside / float(area.sum()), dtype="float32")
remapped = xr.Dataset(
    {{
        variable: (("across", "along"), values, dict(source[variable].attrs)),
        "lon": (("across", "along"), grid["lon"].values),
        "lat": (("across", "along"), grid["lat"].values),
        "lon_bnds": (("across", "along", "vertices"), grid["lonc"].values),
        "lat_bnds": (("across", "along", "vertices"), grid["latc"].values),
        "crs": grid["crs"],
    }},
    coords={{
        "across": grid["across"].values.astype("float64"),
        "along": grid["along"].values.astype("float64"),
    }},
    attrs={{**source.attrs, "history": "fake cdo remapcon"}},
)
remapped.to_netcdf(output_path)
"""


def write_fake_cdo(directory: Path) -> Path:
    """Write the fake ``cdo`` executable and return its path."""
    fake = directory / "cdo"
    fake.write_text(FAKE_CDO.format(python=sys.executable))
    fake.chmod(0o755)
    return fake


@pytest.mark.golden
def test_cdo_regridder_with_fake_executable(
    golden_record: dict,
    golden_metadata: xr.Dataset,
    column: xr.DataArray,
    tmp_path: Path,
) -> None:
    fake = write_fake_cdo(tmp_path)
    record = golden_record["placements"]["rot"]
    placement = placement_from_record(record)
    regridder = CdoRegridder(
        executable=str(fake), workdir=tmp_path / "work", keep_files=True
    )
    result = regridder.to_scene(column, golden_metadata, placement)
    across = slice(record["across_slice"]["start"], record["across_slice"]["stop"])
    along = slice(record["along_slice"]["start"], record["along_slice"]["stop"])
    assert result.dims == ("across", "along")
    assert result.shape == (across.stop - across.start, along.stop - along.start)
    assert result.name == "ch4_path"
    assert result.attrs["units"] == "kg m-2"
    assert result.attrs["regrid_method"] == "cdo-remapcon"
    assert result.attrs["window_across_start"] == across.start
    assert result.attrs["window_along_stop"] == along.stop
    assert result.attrs["embedding_along_index"] == record["along_index"]
    assert result.attrs["embedding_placement"] == "explicit"
    assert result.attrs["regrid_inside_mass_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert result.attrs["regrid_max_relative_mass_error"] < 1e-6
    assert result.attrs["regrid_target_mass"] == pytest.approx(
        result.attrs["regrid_source_mass"], rel=1e-6
    )
    np.testing.assert_array_equal(
        result["across"].values, np.arange(across.start, across.stop)
    )
    np.testing.assert_array_equal(
        result["along"].values, np.arange(along.start, along.stop)
    )
    assert float(result.min()) > 0
    kept = list((tmp_path / "work").glob("cdo-*/cdo_remapped.nc"))
    assert len(kept) == 1
    # The kept CDO output carries the ``embedding_*`` provenance only, so
    # merging it into the scene needs the window explicitly.
    with pytest.raises(ContractError, match="'window_\\*' attributes"):
        merge_scene_with_column(golden_metadata, kept[0])
    merged = merge_scene_with_column(
        golden_metadata, kept[0], window=Window(across=across, along=along)
    )
    assert (merged.sizes["across"], merged.sizes["along"]) == result.shape
    assert merged.sizes["vertices"] == 4
    np.testing.assert_array_equal(merged["ch4_path"].values, result.values)
    assert "lon_plume" in merged and "lat_plume" in merged
    assert merged.attrs["embedding_along_index"] == record["along_index"]
    merged.close()

    full = regridder.to_scene(
        column,
        golden_metadata,
        placement,
        window=Window(across=slice(0, 40), along=slice(0, 160)),
    )
    assert full.shape == (40, 160)
    np.testing.assert_array_equal(full.values[across, along], result.values)
    assert float(np.abs(full.values[across.stop :, :]).sum()) == 0.0
    assert float(np.abs(full.values[:, along.stop :]).sum()) == 0.0
    assert full.attrs["window_across_stop"] == 40

    disjoint = CdoRegridder(executable=str(fake)).to_scene(
        column,
        golden_metadata,
        placement,
        window=Window(across=slice(38, 40), along=slice(100, 160)),
    )
    assert disjoint.shape == (2, 60)
    assert float(np.abs(disjoint.values).sum()) == 0.0
    # The default regridder without a workdir cleans its temporary files.
    assert len(list((tmp_path / "work").glob("cdo-*/cdo_source.nc"))) == 2


@pytest.mark.golden
def test_prepared_window_merges_fake_remap(
    golden_record: dict,
    geometry: SceneGeometry,
    golden_metadata: xr.Dataset,
    column: xr.DataArray,
    tmp_path: Path,
) -> None:
    fake = write_fake_cdo(tmp_path)
    record = golden_record["placements"]["partial"]
    scene_placement = place_in_scene(
        column, geometry, placement=placement_from_record(record)
    )
    preparation = prepare_cdo_files(
        scene_placement,
        geometry,
        tmp_path / "source.nc",
        tmp_path / "target.nc",
        variable="ch4_path",
    )
    # Both CDO files keep the ``embedding_*`` window bounds as provenance.
    for path in (preparation.source_path, preparation.target_grid_path):
        with xr.open_dataset(path, decode_times=False) as dataset:
            attrs = dict(dataset.attrs)
        assert Window.from_attrs(attrs, prefix="embedding_") == preparation.window
        assert not any(key.startswith("window_") for key in attrs)
    remapped_path = run_cdo_remapcon(
        preparation, tmp_path / "remapped.nc", executable=str(fake)
    )
    with pytest.raises(ContractError, match="'window_\\*' attributes"):
        merge_scene_with_column(golden_metadata, remapped_path)
    merged = merge_scene_with_column(
        golden_metadata, remapped_path, window=preparation.window
    )
    n_across, n_along = preparation.window.shape
    assert merged.sizes["across"] == n_across
    assert merged.sizes["along"] == n_along
    np.testing.assert_array_equal(
        merged["across"].values,
        golden_metadata["across"].values[preparation.across_slice],
    )
    np.testing.assert_array_equal(
        merged["along"].values,
        golden_metadata["along"].values[preparation.along_slice],
    )
    assert float(merged["ch4_path"].min()) > 0
    assert merged.attrs["embedding_across_start"] == preparation.across_slice.start
    assert (
        merged.attrs["golden_source_file"]
        == golden_metadata.attrs["golden_source_file"]
    )
    merged.close()
    with pytest.raises(ContractError, match="exceeds the scene"):
        merge_scene_with_column(
            golden_metadata,
            remapped_path,
            window=Window(
                across=slice(40 - n_across + 1, 41), along=preparation.along_slice
            ),
        )


def test_run_cdo_remapcon_reports_failure(tmp_path: Path) -> None:
    failing = tmp_path / "cdo"
    failing.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    failing.chmod(0o755)
    preparation = CdoPreparation(
        source_path=tmp_path / "s.nc",
        target_grid_path=tmp_path / "t.nc",
        along_index=0,
        across_index=0,
        along_slice=slice(0, 1),
        across_slice=slice(0, 1),
        longitude=0.0,
        latitude=0.0,
        x_offset=0.0,
        y_offset=0.0,
    )
    with pytest.raises(RuntimeError, match=r"status 3.*boom"):
        run_cdo_remapcon(preparation, tmp_path / "o.nc", executable=str(failing))


@pytest.mark.golden
def test_prepare_regular_target_grid(geometry: SceneGeometry, tmp_path: Path) -> None:
    path = prepare_regular_target_grid(geometry, 2.0, None, tmp_path / "grid.nc")
    with xr.open_dataset(path) as grid:
        assert grid["lonc"].dims == ("y", "x", "vertices")
        assert grid.attrs["resolution_m"] == 2.0
        assert grid["x"].attrs["epsg"] == geometry.crs.to_epsg()
        xmin, ymin, xmax, ymax = grid.attrs["projected_bounds"]
        assert xmin <= float(geometry.x.min()) and xmax >= float(geometry.x.max())
        assert ymin <= float(geometry.y.min()) and ymax >= float(geometry.y.max())
        assert float(grid["x"][0]) == xmin + 1.0
        assert float(grid["y"][0]) == ymax - 1.0
    with pytest.raises(ContractError):
        prepare_regular_target_grid(geometry, 0.0, None, tmp_path / "bad.nc")
