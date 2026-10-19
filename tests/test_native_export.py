"""Copy-and-patch export into synthetic AVIRIS-4 and EMIT files."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr
import yaml

import spectra.export.native as native
from spectra.contracts.embedded import SOURCE_VARIABLES, check_not_embedded
from spectra.contracts.errors import ContractError
from spectra.export.native import write_native
from spectra.result import ClusterResult, EmbedResult, SourceResult
from spectra.spectral.embedding import embed_column_in_radiance
from spectra.spectral.toy import ToyForwardModel
from spectra.testing.native import (
    assert_native_preserved,
    native_snapshot,
    write_aviris4_files,
    write_emit_files,
)
from spectra.types import NativeLayout, SpectralResponse, ViewingGeometry, Window

pytest.importorskip("h5py")

AVIRIS4 = NativeLayout(
    "radiance", "radiance", {"across": "across", "along": "along", "band": "band"}
)
EMIT = NativeLayout(
    "radiance",
    "radiance",
    {"downtrack": "along", "crosstrack": "across", "bands": "band"},
)
# The fill pixel (across 5, along 6) lies inside the first plume.
FILL = (5, 6)
FILL_EMIT = (6, 5)  # (downtrack, crosstrack) of the same pixel
WRITERS: dict[str, tuple[Callable[..., dict[str, Path]], NativeLayout]] = {
    "aviris4-2025": (partial(write_aviris4_files, fill_pixels=[FILL]), AVIRIS4),
    "aviris4-2026": (
        partial(write_aviris4_files, schema="2026", fill_pixels=[FILL]),
        AVIRIS4,
    ),
    "emit-010635": (partial(write_emit_files, fill_pixels=[FILL_EMIT]), EMIT),
    "emit-010617": (
        partial(write_emit_files, build="010617", fill_pixels=[FILL_EMIT]),
        EMIT,
    ),
}
NEW_OBJECTS = (
    "source",
    "injected_ch4_mass",
    *SOURCE_VARIABLES,
    "source_across",
    "source_along",
    "source_downtrack",
    "source_crosstrack",
)
RUN_CONFIG = "sensor: test\nplumes: []\n"


def blob(window: Window, peak: float, centre: tuple[float, float]) -> np.ndarray:
    """Return a Gaussian column in kg m-2 on a window, zero at its border."""
    across = np.arange(window.across.start, window.across.stop)[:, None]
    along = np.arange(window.along.start, window.along.stop)[None, :]
    values = peak * np.exp(
        -((across - centre[0]) ** 2 + (along - centre[1]) ** 2) / 4.0
    )
    values[values < 1e-3 * peak] = 0.0
    return values


def geometry() -> ViewingGeometry:
    return ViewingGeometry(
        sza=30.0,
        saa=150.0,
        vza=5.0,
        vaa=100.0,
        surface_elevation=300.0,
        sensor_altitude=4000.0,
        lat=0.0,
        lon=0.0,
        time_utc=np.datetime64("2025-01-01T00:00:00"),
        platform="airborne",
    )


def make_result(files: dict[str, Path], layout: NativeLayout) -> EmbedResult:
    """Two clusters: two overlapping plumes and one alone."""
    with netCDF4.Dataset(files["radiance"]) as ds:
        if layout is EMIT:
            group = ds.groups["sensor_band_parameters"]
            wavelength = group["wavelengths"][:].data
            fwhm = group["fwhm"][:].data
        else:
            wavelength = ds["wavelength"][:].data
            fwhm = ds["fwhm"][:].data
    first = int(np.argmax(wavelength >= 1000.0))
    bands = slice(first, wavelength.size)
    model = ToyForwardModel(width_nm=60.0, saturation_ppm_m=5e4)
    lut = model.lut(
        SpectralResponse(wavelength=wavelength[bands], fwhm=fwhm[bands]),
        geometry(),
        gas="ch4",
    )
    windows = [
        Window(across=slice(2, 8), along=slice(3, 9)),
        Window(across=slice(4, 10), along=slice(6, 12)),
        Window(across=slice(12, 18), along=slice(15, 22)),
    ]
    columns = [
        blob(windows[0], 4e-3, (5.0, 6.0)),
        blob(windows[1], 3e-3, (7.0, 9.0)),
        blob(windows[2], 5e-3, (15.0, 18.0)),
    ]
    sources = tuple(
        SourceResult(
            index=index,
            plume_file=f"/path/to/plume_{index}.nc",
            case_id=f"case-{index}",
            time_index=10 * index,
            emission_rate_kg_s=0.1 * (index + 1),
            emission_scale=1.0 + index,
            lon=10.0 + index,
            lat=45.0 + index,
            anchor_along=window.along.start + 1,
            anchor_across=window.across.start + 1,
            anchor_mode=("lonlat", "pixel", "random")[index],
            rotation_deg=15 * index,
            flip=("none", "x", "y")[index],
            outside_fraction=0.0,
            injected_mass_kg=float(column.sum()),
            cluster=0 if index < 2 else 1,
            window=window,
            injected=column.astype(np.float32),
        )
        for index, (window, column) in enumerate(zip(windows, columns, strict=True))
    )
    union = Window(across=slice(2, 10), along=slice(3, 12))
    summed = np.zeros(union.shape)
    for window, column in zip(windows[:2], columns[:2], strict=True):
        summed[
            window.across.start - 2 : window.across.stop - 2,
            window.along.start - 3 : window.along.stop - 3,
        ] += column
    clusters = (
        ClusterResult(0, (0, 1), union, bands, summed, geometry(), lut),
        ClusterResult(1, (2,), windows[2], bands, columns[2], geometry(), lut),
    )
    return EmbedResult(
        gas="ch4",
        sources=sources,
        clusters=clusters,
        attrs={"spectra_seed": 7, "spectra_run_config": RUN_CONFIG},
    )


def native_radiance(path: Path, layout: NativeLayout) -> np.ndarray:
    """Return the raw radiance in common ``(across, along, band)`` order."""
    with netCDF4.Dataset(path) as ds:
        variable = ds["radiance"]
        variable.set_auto_maskandscale(False)
        raw = variable[:]
        common = [layout.dims[dim] for dim in variable.dimensions]
    return np.asarray(raw).transpose(
        [common.index(d) for d in ("across", "along", "band")]
    )


def expected_radiance(
    radiance: np.ndarray, result: EmbedResult
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the clusters independently; return the radiance and change mask."""
    expected = radiance.copy()
    changed = np.zeros(radiance.shape, dtype=bool)
    for cluster in result.clusters:
        window = cluster.window
        block = radiance[window.across, window.along, cluster.bands]
        attenuated = embed_column_in_radiance(
            xr.DataArray(
                block,
                dims=("across", "along", "band"),
                coords={"wavelength": ("band", cluster.lut["wavelength"].values)},
            ),
            xr.DataArray(
                cluster.column, dims=("across", "along"), attrs={"units": "kg m-2"}
            ),
            cluster.lut,
            gas="ch4",
        ).values
        valid = np.isfinite(block) & (block != -9999.0)
        patch = (cluster.column > 0)[..., None] & valid
        region = expected[window.across, window.along, cluster.bands]
        region[patch] = attenuated[patch]
        expected[window.across, window.along, cluster.bands] = region
        mask_region = changed[window.across, window.along, cluster.bands]
        mask_region[patch] = True
        changed[window.across, window.along, cluster.bands] = mask_region
    return expected, changed


def to_native(mask: np.ndarray, path: Path, layout: NativeLayout) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        common = [layout.dims[dim] for dim in ds["radiance"].dimensions]
    return mask.transpose([("across", "along", "band").index(d) for d in common])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(params=list(WRITERS))
def case(request: pytest.FixtureRequest, tmp_path: Path):
    writer, layout = WRITERS[request.param]
    files = writer(tmp_path / "in")
    return files, layout, make_result(files, layout)


def test_export_patches_only_plume_pixels(case, tmp_path: Path) -> None:
    files, layout, result = case
    radiance = native_radiance(files["radiance"], layout)
    expected, changed = expected_radiance(radiance, result)
    assert changed.any()
    # The fill pixel inside the first plume stays untouched.
    assert not changed[FILL[0], FILL[1]].any()
    digest, mtime = sha256(files["radiance"]), files["radiance"].stat().st_mtime_ns
    before = native_snapshot(
        files["radiance"],
        masks={"radiance": to_native(changed, files["radiance"], layout)},
    )

    output = write_native(layout, files, tmp_path / "out" / "embedded.nc", result)

    assert output == (tmp_path / "out" / "embedded.nc").resolve()
    after = native_snapshot(
        output, masks={"radiance": to_native(changed, files["radiance"], layout)}
    )
    root_attrs = [name for name in after["/"]["attrs"] if name.startswith("spectra_")]
    assert_native_preserved(
        before,
        after,
        new_objects=[name for name in NEW_OBJECTS if name in after],
        new_root_attrs=root_attrs,
    )
    patched = native_radiance(output, layout)
    assert np.array_equal(patched, expected, equal_nan=True)
    assert np.all(patched[changed] <= radiance[changed])
    assert np.any(patched[changed] < radiance[changed])
    # The input file is neither modified nor touched.
    assert sha256(files["radiance"]) == digest
    assert files["radiance"].stat().st_mtime_ns == mtime
    assert not list(output.parent.glob(".*.partial"))


def test_export_writes_metadata_schema(case, tmp_path: Path) -> None:
    files, layout, result = case
    output = write_native(layout, files, tmp_path / "embedded.nc", result)
    spatial = [dim for dim in layout.dims if layout.dims[dim] != "band"]
    with netCDF4.Dataset(output) as ds:
        variable = ds["radiance"]
        y_dim, x_dim = [d for d in variable.dimensions if layout.dims[d] != "band"]
        field = ds["injected_ch4_mass"]
        assert field.dimensions == ("source", y_dim, x_dim)
        assert field.dtype == np.float32 and field.units == "kg m-2"
        assert field.filters()["zlib"] and field.filters()["shuffle"]
        assert field.chunking()[0] == 1
        values = field[:].data
        assert len(ds.dimensions["source"]) == 3
        for dim in spatial:
            assert f"source_{dim}" in ds.variables
        assert list(ds["source_plume_file"][:]) == [
            f"/path/to/plume_{i}.nc" for i in range(3)
        ]
        assert list(ds["source_anchor_mode"][:]) == ["lonlat", "pixel", "random"]
        np.testing.assert_array_equal(ds["source_flip"][:], [0, 1, 2])
        assert ds["source_flip"].flag_meanings == "none x y"
        np.testing.assert_array_equal(ds["source_cluster"][:], [0, 0, 1])
        assert ds["source_lut_sza"][0] == 30.0
        assert ds["source_lut_surface_elevation"][0] == 300.0
        attrs = {name: ds.getncattr(name) for name in ds.ncattrs()}
    assert yaml.safe_load(attrs["spectra_run_config"]) == yaml.safe_load(RUN_CONFIG)
    assert attrs["spectra_seed"] == 7
    assert attrs["spectra_gas"] == "ch4"
    assert attrs["spectra_n_sources"] == 3
    assert attrs["spectra_source_radiance_path"] == str(files["radiance"].resolve())
    assert attrs["spectra_source_radiance_size"] == files["radiance"].stat().st_size
    assert attrs["spectra_attenuated_wavelength_min_nm"] >= 1000.0
    # Per-source fields in common order: zero outside the window, the column inside.
    common = values if layout.dims[y_dim] == "across" else values.transpose(0, 2, 1)
    for position, source in enumerate(result.sources):
        inside = common[position][source.window.across, source.window.along]
        np.testing.assert_array_equal(inside, source.injected)
        outside = common[position].copy()
        outside[source.window.across, source.window.along] = 0
        assert not outside.any()
    for cluster in result.clusters:
        summed = common[list(cluster.sources)].sum(axis=0)
        np.testing.assert_allclose(
            summed[cluster.window.across, cluster.window.along],
            cluster.column,
            rtol=1e-6,
        )
    with xr.open_dataset(output) as dataset:
        assert "radiance" in dataset and "injected_ch4_mass" in dataset


def test_export_refusals(tmp_path: Path) -> None:
    files = write_aviris4_files(tmp_path / "in")
    result = make_result(files, AVIRIS4)
    output = tmp_path / "out.nc"
    write_native(AVIRIS4, files, output, result)
    with pytest.raises(FileExistsError):
        write_native(AVIRIS4, files, output, result)
    first = output.stat().st_ino
    write_native(AVIRIS4, files, output, result, overwrite=True)
    assert output.stat().st_ino != first
    for target in (files["radiance"], files["metadata"]):
        with pytest.raises(ContractError, match="never modifies"):
            write_native(AVIRIS4, files, target, result, overwrite=True)
    link = tmp_path / "link.nc"
    link.symlink_to(files["radiance"])
    with pytest.raises(ContractError, match="never modifies"):
        write_native(AVIRIS4, files, link, result, overwrite=True)
    hard = tmp_path / "hard.nc"
    os.link(files["radiance"], hard)
    with pytest.raises(ContractError, match="never modifies"):
        write_native(AVIRIS4, files, hard, result, overwrite=True)
    with pytest.raises(ContractError, match="directory"):
        write_native(AVIRIS4, files, tmp_path, result, overwrite=True)
    # An embedded output is refused as input.
    with pytest.raises(ContractError, match="already embedded"):
        write_native(
            AVIRIS4, {**files, "radiance": output}, tmp_path / "again.nc", result
        )
    with pytest.raises(ContractError, match="already embedded"):
        check_not_embedded(output)


def test_export_rejects_colliding_and_classic_inputs(tmp_path: Path) -> None:
    files = write_aviris4_files(tmp_path / "in")
    result = make_result(files, AVIRIS4)
    with netCDF4.Dataset(files["radiance"], "r+") as ds:
        ds.createVariable("source_x", "f8", ())
    with pytest.raises(ContractError, match="collide"):
        write_native(AVIRIS4, files, tmp_path / "out.nc", result)
    classic = tmp_path / "classic.nc"
    with netCDF4.Dataset(classic, "w", format="NETCDF4_CLASSIC") as ds:
        for name, size in (("across", 20), ("along", 24), ("band", 40)):
            ds.createDimension(name, size)
        ds.createVariable("radiance", "f4", ("across", "along", "band"))
    with pytest.raises(ContractError, match="NETCDF4_CLASSIC"):
        write_native(AVIRIS4, {"radiance": classic}, tmp_path / "out.nc", result)


def test_export_leaves_nothing_on_failure(tmp_path: Path, monkeypatch) -> None:
    files = write_emit_files(tmp_path / "in")
    result = make_result(files, EMIT)

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(native, "_write_metadata", broken)
    output = tmp_path / "out" / "embedded.nc"
    with pytest.raises(RuntimeError, match="metadata failed"):
        write_native(EMIT, files, output, result)
    assert not output.exists()
    assert not list(output.parent.iterdir())


def test_export_rejects_results_outside_the_scene(tmp_path: Path) -> None:
    files = write_aviris4_files(tmp_path / "in", n_along=10)
    result = make_result(files, AVIRIS4)
    with pytest.raises(ContractError, match="outside the scene"):
        write_native(AVIRIS4, files, tmp_path / "out.nc", result)
